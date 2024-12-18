import cv2
import numpy as np
import torchvision
import torch
from Metrics.psnr_ssim.utils import reorder_image,to_y_channel
from Metrics.psnr_ssim.niqe import calculate_niqe

#from basicsr.metrics.metric_util import reorder_image, to_y_channel
#from basicsr.utils.registry import METRIC_REGISTRY
__all__=['_calculate_psnr_ssim_niqe','calculate_psnr','calculate_ssim','calculate_niqe']
'''
psnr,ssim script based on opencv, which the image format is HWC and the channel order can BGR
'''
def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        # psnr_val += compare_psnr(clean[i], recoverd[i])
        # ssim += compare_ssim(clean[i], recoverd[i], multichannel=True)
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]

def single2uint(img):
    return np.uint8((img.clip(0.,1.) * 255).round())

def uint2single(img):
    # torch.from_numpy(img.astype(np.float32) / 255.).float().permute(2,0,1)
    return img.astype(np.float32) / 255.

def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if type(img) == np.ndarray:  # (h,w,c) numpy (0,255)
        # img *= std
        # img += mean
        mean = np.array(mean).reshape((1,3,1,1))
        std = np.array(std).reshape((1,3,1,1))
        img = (img * std) + mean #img.mul_(std).add_(mean)
    if type(img) == torch.Tensor:  # (c,h,w) tensor
        dtype, device = img.dtype, img.device
        mean = torch.as_tensor(mean,dtype,device)[None, :, None, None]
        std = torch.as_tensor(std,dtype,device)[None, :, None, None]
        img = img.mul_(std).add_(mean)
    return img

def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if type(img) == np.ndarray:  # (h,w,c) numpy (0,255)
        img /= 255.
        img -= mean
        img /= std
    if type(img) == torch.Tensor:  # (c,h,w) tensor
        mean = torch.as_tensor(mean)[:, None, None]
        std = torch.as_tensor(std)[:, None, None]
        img = img.sub_(mean).div_(std)
    return img

def _calculate_psnr_ssim_niqe(output,target,cal_psnr=True,cal_ssim=True,cal_niqe=False, rgb_range=1,crop_border=0, input_order='CHW', test_y_channel=True,mean=(0,0,0), std=(1,1,1)):
    # denormalize:
    assert rgb_range == 1, 'RGB_range should be 0~1 tensor [BCHW]'
    output_np = single2uint(denormalize(output.data.cpu().numpy().astype(np.float32),mean=mean,std=std))
    target_np = single2uint(denormalize(target.data.cpu().numpy().astype(np.float32),mean=mean,std=std))
    # else:
    #     output_denormalize = output.data.cpu().numpy().astype(np.float64)
    #     target_denormalize = target.data.cpu().numpy().astype(np.float64)

    b = output.shape[0]
    psnr, ssim, niqe = 0.,0.,0.
    for i in range(b):
        if cal_psnr: psnr += calculate_psnr(output_np[i,:,:,:],target_np[i,:,:,:],crop_border=crop_border,input_order=input_order,test_y_channel=test_y_channel)
        if cal_ssim: ssim += calculate_ssim(output_np[i,:,:,:],target_np[i,:,:,:],crop_border=crop_border,input_order=input_order,test_y_channel=test_y_channel)
        if cal_niqe: niqe += calculate_niqe(target_np[i,:,:,:],crop_border=crop_border,input_order=input_order,convert_to='y')
    return psnr/b , ssim/b, niqe/b, b


def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()

if __name__ == '__main__':
    print(np.array([1,2,3,4,5]).mean())
    img1=cv2.imread('QJ.png')
    img2=cv2.imread('dog.jpg')
    img1=torchvision.transforms.ToTensor()(img1)
    img2 = torchvision.transforms.ToTensor()(img2)
    img1,img2=img1.unsqueeze(0),img2.unsqueeze(0)
    # img1,img2=torch_to_numpy(img1,img2)
    # print(img1.shape)

    # img2=cv2.imread('../dog.jpg')
    img1=img1[:,:,0:100,0:100]
    img2 = img2[:,:,0:100, 0:100]
    psnr=calculate_psnr(img1,img2)
    ssim=calculate_ssim(img1,img2)
    print(psnr,ssim)

    # img=Image.open('../val.png')
    # print(img)

    #requests.get('https://api.day.app/SjMonLD9o37FLn6GS67TKT/model finished!')
    # x=torch.rand(4,3,24,24)+125
    # y=torch.rand((4,3,24,24))+200
    # #print(x.shape[0],x[0].shape)
    # psnr=calculate_psnr(x,y,test_y_channel=True)
    # ssim=calculate_ssim(x,y,test_y_channel=True)
    # print(psnr,ssim)



