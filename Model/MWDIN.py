import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.io import read_image
from typing import Type
from Model.Transformer_Block import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-6,
                 data_format: str = "channels_first") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, high):  # b,c,h,w -> b,(c n), h,w -> b, n, c, (h, w)
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        v = v * high

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        #save_fea(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class DSA(nn.Module):
    def __init__(self,
                 in_channels: int,
                 shuffle: bool = False,
                 square_kernel_size: int = 3,
                 band_kernel_size: int = 11,
                 branch_ratio: float = 0.25) -> None:
        super(DSA, self).__init__()

        gc = int(in_channels // 2 * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = (in_channels // 2 - 3 * gc, gc, gc, gc)
        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.SiLU(True)
            # nn.Sigmoid()
        )
        self.shuffle = shuffle

        self.sa = XCA(in_channels // 2, 4, True)

        ### For low-high-guide
        self.high2low = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 1),
            nn.SiLU(True)

        )
        self.low2high = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 1),
            nn.SiLU(True)

        )

    def forward(self, x):
        # TODO: list
        # [1] multiple x: 2023.10.24 17:17  38.0110
        # [2] not multiple x
        # [3] low guides high
        # [4] high guides low

        high, low = torch.chunk(x, 2, 1)

        x_id, x_hw, x_w, x_h = torch.split(high, self.split_indexes, dim=1)
        high = torch.cat(
                (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)
                 ), dim=1
            )

        low = self.sa(low, self.high2low(high))

        out = torch.cat([self.low2high(low) * high, low], dim=1)
        out = self.conv_act(out) * x
        return rearrange(out, 'b (g d) h w -> b (d g) h w', g=64) if self.shuffle else out



class DCA(nn.Module):
    def __init__(self,
                 in_features: int,
                 mlp_ratio: int = 4,
                 act_layer: Type[nn.Module] = nn.GELU,
                 drop: float = 0.,
                 use_attn: bool = True) -> None:
        super(DCA, self).__init__()
        hidden_features = int(in_features * mlp_ratio)
        self.fc1_ = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv_ = nn.Conv2d(hidden_features,hidden_features, 3, 1, 1, groups=hidden_features)
        self.act_ = act_layer()
        self.fc2_ = nn.Conv2d(hidden_features, in_features, 1)
        self.drop_ = nn.Dropout(drop)

        self.ca = BiAttn(in_features) if use_attn else nn.Identity()

    def forward(self, x):
        x = self.fc1_(x)
        x = self.dwconv_(x)
        x = self.act_(x)
        x = self.drop_(x)
        x = self.fc2_(x)
        x = self.drop_(x)
        x = self.ca(x)
        return x


class BiAttn(nn.Module):
    def __init__(self,
                 in_channels: int,
                 act_ratio: float = 0.25,
                 act_fn: Type[nn.Module] = nn.GELU,
                 gate_fn: Type[nn.Module] = nn.Sigmoid) -> None:
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        # self.norm = LayerNorm(in_channels, data_format='channels_first')
        self.global_reduce = nn.Conv2d(in_channels, reduce_channels, 1) #Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Conv2d(in_channels, reduce_channels, 1)
        self.act_fn = act_fn()
        self.channel_select = nn.Conv2d(reduce_channels, in_channels, 1)
        self.spatial_select = nn.Conv2d(reduce_channels * 2, 1, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        """
        :param x: [B, C, H, W]
        :return: [B, C, H, W]
        """
        ori_x = x
        # x = self.norm(x)
        x_global = F.adaptive_avg_pool2d(x, 1) #x.mean(1, keepdim=True)   # [B, C, 1, 1]
        x_global = self.act_fn(self.global_reduce(x_global))   # [B, C/4, 1, 1]
        x_local = self.act_fn(self.local_reduce(x))   # [B, C/4, H, W]

        c_attn = self.channel_select(x_global)  # [B, C, 1, 1]
        c_attn = self.gate_fn(c_attn)  # [B, C, 1, 1]

        # temp = torch.ones((b, int(c/4), h, w), dtype=x_global.dtype, device=x_global.device) * x_global #(torch.cat([x_local, temp], dim=1)) #)
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(x_global.shape[0], x_global.shape[1], x.shape[2], x.shape[3])], dim=1))
        s_attn = self.gate_fn(s_attn)  # [B, 1, H, W]

        attn = c_attn * s_attn  # [B, N, C]
        return ori_x * attn


class CALayer(nn.Module):
    # TODO: GELU -> LeakReLU
    def __init__(self,
                 channel: int,
                 reduction: int = 16) -> None:
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.GELU(), #nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class MetaNeXtBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 shuffle: bool = False,
                 square_kernel_size: int = 3,
                 band_kernel_size: int = 11,
                 branch_ratio: float = 0.25,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 mlp_ratio: int = 4,
                 act_layer: Type[nn.Module] = nn.GELU,
                 drop: float = 0.,
                 ls_init_value: float = 1e-6,
                 drop_path: float = 0.,
                 use_attn: bool = True) -> None:
        super().__init__()
        self.token_mixer = DSA(dim, shuffle=shuffle,
                               square_kernel_size=square_kernel_size,
                               band_kernel_size=band_kernel_size,
                               branch_ratio=branch_ratio)
        self.norm = norm_layer(dim)
        self.mlp = DCA(dim, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop, use_attn=use_attn)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()



    def forward(self, x):

        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, x.shape[1], 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class MetaNeXt(nn.Module):
    def __init__(self,
                 stage: int = 16,
                 embed_dims: int = 64,
                 scale: int = 2,
                 shuffle: bool = False,
                 square_kernel_size: int = 3,
                 band_kernel_size: int = 11,
                 branch_ratio: float = 0.25,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 mlp_ratio: int = 2,
                 act_layer: Type[nn.Module] = nn.GELU,
                 drop: float = 0.,
                 ls_init_value: float = 1e-6,
                 drop_path: float = 0.,
                 use_attn: bool = True,
                 **kwargs) -> None:
        super().__init__()

        self.head = nn.Conv2d(3, embed_dims, 3, 1, 1)

        self.tail_X4 = nn.Sequential(
            nn.Conv2d(embed_dims, 3 * scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale)
        )

        self.stages = nn.Sequential()
        stages = []
        for i in range(stage):
            stages.append(MetaNeXtBlock(
                dim=embed_dims,
                shuffle=shuffle,
                square_kernel_size=square_kernel_size,
                band_kernel_size=band_kernel_size,
                branch_ratio=branch_ratio,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop,
                ls_init_value=ls_init_value,
                drop_path=drop_path,
                use_attn=use_attn
            ))
        self.stages = nn.Sequential(*stages)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.head(x)
        x = self.stages(x) + x
        x = self.tail_X4(x)
        return x

    def _grad(self):
        for name, param in self.named_parameters():
            # print(name, param.shape)
            if name.split(".")[0] == 'tail':  # tail.0.weight torch.Size([12, 64, 3, 3])
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from Metrics.ntire.model_summary import get_model_flops

    scale = 2
    model = MetaNeXt(scale=scale, stage=16).cuda()
    h, w = 720//scale, 1280//scale
    x = torch.randn((1, 3, h, w)).cuda()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.shape)

    with torch.no_grad():
        out = model(x)
        print(out.shape)

        input_dim = (3, h, w)  # set the input dimension
        flops = get_model_flops(model, input_dim, False)
        flops = flops / 10 ** 9
        print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters / 10 ** 6
        print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))

