import cv2
import numpy as np
import torch
import torchvision.transforms.functional as ff
from torch.nn import functional as F
import albumentations
import os


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.

    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I


    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


def tensor2torchuint8(img):
    """
    Args:
        img: BCHW 0-1 Tensor
    Returns: BCHW 0-255 torch.uint8

    """
    return torch.round(img.detach().clamp(0., 1.) * 255.).byte()


def hist_equal_loss(out, gt, criterion, device, clip_limit=4.0, tile_grid_size=(8,8), p=0.5):
    """
    Args:
        out: BCHW Tensor 0-1
        gt: BCHW Tensor 0-1. The tensor dtype must be ``torch.uint8`` and values are expected to be in ``[0, 255]``.
        criterion: loss function
    Returns:
    """

    gt_ = ff.equalize(tensor2torchuint8(gt)).to(device)
    out_ = tensor2torchuint8(out).to(device)
    loss = criterion(gt_.float(), out_.float())

    return loss


def subimg_div(imgs: list):
    """
    Args:
        imgs: 25 subimages BCHW 0-1 Tensor: [img1, img2, img3, ...], shape: 25,B,C,H,W
    Returns: 1 subimage BCHW 0-1 Tensor
    """
    img_stack = torch.stack(imgs, dim=1)  # B, 25, C, H, W
    img_mean = torch.mean(img_stack, dim=1)  # B, C, H, W
    return img_mean


if __name__ == '__main__':
    img1 = torch.ones((8, 3, 9, 9), device='cuda')
    # img2 = torch.ones(8, 3, 9, 9)+1
    # img3 = torch.ones(8, 3, 9, 9)+2
    # img4 = torch.ones(8, 3, 9, 9)+3
    #
    # img_mean = subimg_div([img1, img2, img3, img4])
    # print(img_mean.shape)
    # print(img_mean)

    img_ = tensor2torchuint8(img1)
    print(img_.shape, img_.dtype)
    print(img1.device, img_.device)

