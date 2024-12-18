import torch
import cv2
import os
from Model.SR_SAM_arch_RealSR import EfficientViT
# from Data.data_utils import _get_paths_from_images
from Demo.demo import uint2tensor, tensor2uint8
from natsort import os_sorted
from Metrics.psnr_ssim import calculate_psnr, _calculate_psnr_ssim_niqe
from Utils.msic import get_time

def _get_paths_from_images(path, suffix=''):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname) and suffix in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return os_sorted(images, reverse=True)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.pth', '.pt']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def load_func(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    ###
    model_dict = model.state_dict()
    overlap_ = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(overlap_)
    print(f'{(len(overlap_) * 1.0 / len(checkpoint["state_dict"]) * 100):.4f}% weights is loaded!')
    print([k for k, v in checkpoint['state_dict'].items() if k not in model_dict])
    ###
    model.load_state_dict(model_dict)

if __name__ == '__main__':

    weight_paths = r'/mnt/data/617/wsy/SR/experiments/ViT_RealSR_RealSR/192_192_0.0005_cos_l1_ViT_RealSR_x2/20231102-153847/weights'#r'/mnt/data/617/wsy/SR/experiments/ViT_RealSR_RealSR/192_192_0.0005_cos_l1_ViT_RealSR_x2/20231031-220429/weights'
    weights = _get_paths_from_images(weight_paths)
    image_path = r'/mnt/data/617/wsy/SR/Datasets/RealSR/version3/RealSR(V3)/Canon_Nikon/Test/2'
    lqs = _get_paths_from_images(image_path, suffix='_LR2.png')
    gts = _get_paths_from_images(image_path, suffix='_HR.png')
    print(f'weight:{len(weights)} | image:{len(lqs)}')

    device = torch.device('cuda:0')
    model = EfficientViT(scale=2, num_stages=16, num_heads=1, mlp_ratios=2, use_sam=False, use_dw=True).to(device)

    model.eval()
    with torch.no_grad():
        for w in weights:
            print(f'======{get_time()}=======')
            load_func(model, w)
            psnr = 0.
            for lqp, gtp in zip(lqs, gts):
                # base, ext = os.path.splitext(os.path.basename(im))
                lq = uint2tensor(cv2.imread(lqp)[:, :, ::-1]).to(device)
                gt = uint2tensor(cv2.imread(gtp)[:, :, ::-1]).to(device)
                output = model(lq)
                # sr = tensor2uint8(output)
                # psnr += calculate_psnr(sr, gt, test_y_channel=True, crop_border=2)
                p, _, _, _ = _calculate_psnr_ssim_niqe(output, gt, crop_border=2, test_y_channel=True)
                psnr += p
            avg_psnr = psnr / len(lqs)
            print(f'{os.path.basename(w)}: {avg_psnr}')
            with open('realsr.txt', 'a+') as f:
                f.write(f'{os.path.basename(w)}: {avg_psnr}\n')


