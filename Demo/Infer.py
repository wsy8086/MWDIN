import sys
sys.path.append(r'/media/sr6/codes/wsy/SR2')
from Metrics.psnr_ssim import _calculate_psnr_ssim_niqe
import torch
from utils_func import AverageMeter
import os
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils_func import _get_paths_from_images


class Test_Dataset(Dataset):
    def __init__(self,
                 hr_path: str = '',
                 huawei_degraded_path: str = '',
                 huawei_gt_path: str = '',
                 scale: int = 2,
                 mode: str = 'benchmark') -> None:
        super(Dataset, self).__init__()

        if mode == 'benchmark':
            self.lr_path = os.path.join(os.path.abspath(os.path.join(hr_path, os.path.pardir)), f'LRbicx{scale}')
            self.hr_path = hr_path
            self.hr_imgs = _get_paths_from_images(self.hr_path)
            self.lr_imgs = _get_paths_from_images(self.lr_path)

        elif mode == 'realsr':  # Canon | Nikon | Canon_Nikon
            self.lr_path = rf'/media/sr6/codes/wsy/SR2/Datasets/RealSR/version3/RealSR(V3)/Canon_Nikon/Test/{scale}'
            self.hr_path = rf'/media/sr6/codes/wsy/SR2/Datasets/RealSR/version3/RealSR(V3)/Canon_Nikon/Test/{scale}'
            #self.hr_path = rf'/mnt/data/617/wsy/SR/Datasets/RealSR/version3/RealSR(V3)/Canon_Nikon/Test/{scale}'
            self.hr_imgs = _get_paths_from_images(self.hr_path, suffix=f'_HR.png')
            self.lr_imgs = _get_paths_from_images(self.lr_path, suffix=f'_LR{scale}.png')

        elif mode == 'huawei':
            self.lr_path = huawei_degraded_path  # E:\Dataset\Misc\HUAWEI\degraded_on_GT
            self.hr_path = huawei_gt_path  # E:\Dataset\Misc\HUAWEI\GT
            self.hr_imgs = _get_paths_from_images(self.hr_path)
            self.lr_imgs = _get_paths_from_images(self.lr_path)

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, item):
        hr_p, lr_p = self.hr_imgs[item], self.lr_imgs[item]
        base = os.path.basename(hr_p)

        hr = np.array(Image.open(hr_p)).astype(np.float32) / 255.
        lr = np.array(Image.open(lr_p)).astype(np.float32) / 255.
        return self.np2tensor(hr), self.np2tensor(lr), base

    def np2tensor(self, img):
        return torch.from_numpy(img.transpose(2, 0, 1)).float()


@torch.no_grad()
def Test(dataloader, model, device, record=False, save=False, dataset=None, scale=2,
         save_path='', record_path=''):
    if save:
        save_path = os.path.join(save_path, f'{dataset}{os.sep}x{scale}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    psnr, ssim = AverageMeter(), AverageMeter()

    for i, (hr, lr, basename) in enumerate(tqdm(dataloader)):
        hr, lr = hr.to(device), lr.to(device)
        output = model(lr)

        psnr_temp, ssim_temp, _, batch = _calculate_psnr_ssim_niqe(output, hr, crop_border=scale,
                                                                           input_order='CHW',
                                                                           test_y_channel=True, mean=(0, 0, 0),
                                                                           std=(1, 1, 1))
        psnr.update(psnr_temp, batch)
        ssim.update(ssim_temp, batch)
        # print(f'{basename[0]}, PSNR:{psnr_temp}, SSIM:{ssim_temp}')

        if save:
            output_copy = output.data.cpu().numpy().squeeze(0).transpose(1, 2, 0).astype(np.float32)
            output_copy = np.uint8((output_copy.clip(0, 1) * 255.).round())
            cv2.imwrite(os.path.join(save_path, basename[0]), output_copy[:,:,::-1])

    avg_psnr = psnr.avg
    avg_ssim = ssim.avg

    if record:
        with open(f'{record_path}', 'a+') as f:  # {record_path}{os.sep}record.txt
            f.write(f'Scale:{scale} Dataset:{dataset} Avg_psnr:{avg_psnr} Avg_ssim:{avg_ssim}\n')

    return avg_psnr, avg_ssim


if __name__ == '__main__':

    # ------------ Configuration ------------#
    set5_hr = r'/media/sr6/codes/wsy/SR2/Datasets/Benchmark/Set5/GTmod12'
    set14_hr = r'/media/sr6/codes/wsy/SR2/Datasets/Benchmark/Set14/GTmod12'
    urban_hr = r'/media/sr6/codes/wsy/SR2/Datasets/Benchmark/Urban100/GTmod12'
    manga_hr = r'/media/sr6/codes/wsy/SR2/Datasets/Benchmark/Manga109/GTmod12'
    bsds_hr = r'/media/sr6/codes/wsy/SR2/Datasets/Benchmark/BSDS100/GTmod12'


    save = False
    save_path = ''
    record = False
    record_path = ''

    # ----------- device ---------- #
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ----------- model ----------- #
    from Model.MWDIN import MetaNeXt
    model = MetaNeXt(scale=4, stage=16, use_attn=True, shuffle=True)
    weight = r''
    checkpoint = torch.load(weight, map_location='cpu')
    load_func(model, checkpoint['state_dict'])
    print(f'{model_name}_x{scale}_ckpt_pred: {checkpoint["pred"]}')

    # ---------- Dataloader ------- #
    set5_dataloader = DataLoader(Test_Dataset(set5_hr, scale=scale), batch_size=1)
    set14_dataloader = DataLoader(Test_Dataset(set14_hr, scale=scale), batch_size=1)
    urban_dataloader = DataLoader(Test_Dataset(urban_hr, scale=scale), batch_size=1)
    manga_dataloader = DataLoader(Test_Dataset(manga_hr, scale=scale), batch_size=1)
    bsds_dataloader = DataLoader(Test_Dataset(bsds_hr, scale=scale), batch_size=1)
    # realsr_dataloader = DataLoader(Test_Dataset('', scale=scale, mode='realsr'), batch_size=1)

    # ------------ test ----------- #
    psnr, ssim = Test(set5_dataloader, model, device, dataset='Set5', scale=scale, save=save, record=record,
                      save_path=save_path, record_path=record_path)
    print(f'\nScale:{scale} Set5: psnr:{psnr}, ssim:{ssim}')

    psnr, ssim = Test(set14_dataloader, model, device, dataset='Set14', scale=scale, save=save, record=record,
                      save_path=save_path, record_path=record_path)
    print(f'\nScale:{scale} Set14: psnr:{psnr}, ssim:{ssim}')

    psnr, ssim = Test(urban_dataloader, model, device, dataset='Urban100', scale=scale, save=save, record=record,
                      save_path=save_path, record_path=record_path)
    print(f'\nScale:{scale} Urban100: psnr:{psnr}, ssim:{ssim}')

    psnr, ssim = Test(manga_dataloader, model, device, dataset='Manga109', scale=scale, save=save, record=record,
                      save_path=save_path, record_path=record_path)
    print(f'\nScale:{scale} Manga: psnr:{psnr}, ssim:{ssim}')

    psnr, ssim = Test(bsds_dataloader, model, device, dataset='BSDS100', scale=scale, save=save, record=record,
                      save_path=save_path, record_path=record_path)
    print(f'\nScale:{scale} BSDS: psnr:{psnr}, ssim:{ssim}')
    #
    # psnr, ssim = Test(realsr_dataloader, model, device, dataset='realsr', scale=scale, save=save, record=record,
    #                   save_path=save_path, record_path=record_path)
    # print(f'\nScale:{scale} RealSR: psnr:{psnr}, ssim:{ssim}')


