import os
import random
import cv2
import numpy as np
from PIL import Image
from natsort import os_sorted as natsorted
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from Utils.imresize import imresize
# from Tricks.mosaic import Mosaic
# from Tricks.mosaic_process import mosaic_dataset, multi_thread_mosaic_dataset
from Data.data_utils import _get_paths_from_images, paired_random_crop_, random_augmentation
import albumentations


def Dataloader(scale, gt_size, train_batchsize, val_batchsize, num_worker, mode='train_small', pin_memory=False, repeat=1):
    train_dataset = Dataset_PairedImage(scale, mode=mode, gt_size=gt_size, repeat=repeat)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=num_worker, drop_last=True, pin_memory=pin_memory)

    set5 = Benchmark(dataset='Set5', scale=scale)
    set14 = Benchmark(dataset='Set14', scale=scale)
    urban100 = Benchmark(dataset='Urban100', scale=scale)
    manga109 = Benchmark(dataset='Manga109', scale=scale)

    set5_dataloader = DataLoader(set5, batch_size=val_batchsize, shuffle=False, num_workers=num_worker, drop_last=False, pin_memory=pin_memory)
    set14_dataloader = DataLoader(set14, batch_size=val_batchsize, shuffle=False, num_workers=num_worker, drop_last=False, pin_memory=pin_memory)
    urban100_dataloader = DataLoader(urban100, batch_size=val_batchsize, shuffle=False, num_workers=num_worker, drop_last=False, pin_memory=pin_memory)
    manga109_dataloader = DataLoader(manga109, batch_size=val_batchsize, shuffle=False, num_workers=num_worker, drop_last=False, pin_memory=pin_memory)

    return train_dataloader, set5_dataloader, set14_dataloader, urban100_dataloader, manga109_dataloader


def Dataloader_RealSR(scale, gt_size, train_batchsize, val_batchsize, num_worker, mode='realsr', pin_memory=False, repeat=10):
    train_dataset = Dataset_PairedImage(scale,mode=mode, gt_size=gt_size, repeat=repeat)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=num_worker, drop_last=True, pin_memory=pin_memory)

    realsr = Realsr(dataset='Realsr', scale=scale)
    realsr_dataloader = DataLoader(realsr, batch_size=val_batchsize, shuffle=False, num_workers=num_worker, drop_last=False, pin_memory=pin_memory)

    return train_dataloader, realsr_dataloader


class Dataset_PairedImage(Dataset):
    def __init__(self,scale=2,mode='train_small', gt_size=128, geometric_augs=True, repeat=1):
        super(Dataset_PairedImage, self).__init__()
        self.scale = scale
        self.mode = mode
        assert self.mode in ('train_big','train_small', 'realsr')

        root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

        if self.mode == 'train_big':  # multiscale_df2k_sub_p480 + ost_sub  # 221120
            self.hr_p1 = os.path.join(root, f'Datasets{os.sep}DF2K{os.sep}DF2K_multiscale_sub_p480')
            self.hr1 = _get_paths_from_images(self.hr_p1)

            self.hr_p2 = os.path.join(root, f'Datasets{os.sep}OST{os.sep}OST_sub_p480')
            self.hr2 = _get_paths_from_images(self.hr_p2)
            self.hr = natsorted(self.hr1 + self.hr2)

            self.lr_p1 = os.path.join(root, f'Datasets{os.sep}DF2K{os.sep}DF2K_multiscale_sub_p480_x{self.scale}m')
            self.lr1 = _get_paths_from_images(self.lr_p1)

            self.lr_p2 = os.path.join(root, f'Datasets{os.sep}OST{os.sep}OST_sub_p480_x{self.scale}m')
            self.lr2 = _get_paths_from_images(self.lr_p2)
            self.lr = natsorted(self.lr1 + self.lr2)

        elif self.mode == 'train_small':  # df2k #3450
            self.hr_p = r'/media/sr6/datasets/SR/liux/DF2K/DF2K_HR_train'
            #self.hr_p = os.path.join(root, f'Datasets{os.sep}DF2K{os.sep}DF2K_HR_train')  # _p512
            self.hr = _get_paths_from_images(self.hr_p)
            self.lr_p = r'/media/sr6/datasets/SR/liux/DF2K/DF2K_HR_train_x2m'
            #self.lr_p = os.path.join(root, f'Datasets{os.sep}DF2K{os.sep}DF2K_HR_train_x{self.scale}m')  # _p512  _H264
            self.lr = _get_paths_from_images(self.lr_p)

        elif mode == 'realsr':   # repeat=10
            self.hr_p = os.path.join(root, f'/media/sr6/datasets/SR/liux/RealSR/version3/RealSR(V3)/Canon_Nikon/Train/{scale}')  # _p512
            self.hr = _get_paths_from_images(self.hr_p, suffix='_HR.png')

            #/media/sr6/datasets/SR/liux/RealSR/version3/RealSR(V3)/Canon_Nikon
            self.lr_p = os.path.join(root, f'/media/sr6/datasets/SR/liux/RealSR/version3/RealSR(V3)/Canon_Nikon/Train/{scale}')  # _p512  _H264
            self.lr = _get_paths_from_images(self.lr_p, suffix=f'_LR{scale}.png')

            # self.hr = _get_paths_from_images(rf'E:\Dataset\SR\RealSR\version3\RealSR(V3)\Canon_Nikon\Train\{scale}', suffix='HR.png')
            # self.lr = _get_paths_from_images(rf'E:\Dataset\SR\RealSR\version3\RealSR(V3)\Canon_Nikon\Train\{scale}', suffix=f'LR{scale}.png')

        self.mean = torch.tensor([0.,0.,0.])#.reshape(3,1,1)
        self.std = torch.tensor([1.,1.,1.])#.reshape(3,1,1)

        self.gt_size = gt_size

        self.geometric_augs = geometric_augs

        self.repeat = repeat

    def __getitem__(self, index):
        index = self._get_index(index)
        gt_path = self.hr[index]
        lr_path = self.lr[index]

        # assert os.path.basename(gt_path) == os.path.basename(lr_path), f'gt_path:{os.path.basename(gt_path)}, lr_path:{os.path.basename(lr_path)}'
        img_gt = np.array(Image.open(gt_path), dtype=np.uint8)
        img_lq = np.array(Image.open(lr_path), dtype=np.uint8)
        img_gt, img_lq = paired_random_crop_(img_gt,img_lq,self.gt_size,self.scale) #_random_crop_patch(img_gt, patch_size=self.gt_size,flag='np')

        if self.geometric_augs:
            img_gt, img_lq = random_augmentation(img_gt, img_lq)

        ### hist equalization
        # img_gt = albumentations.CLAHE(clip_limit=1,
        #                                     tile_grid_size=(8, 8),
        #                                     always_apply=True,
        #                                     p=1)(image=img_gt)["image"]
        #
        # ###

        img_gt, img_lq = self.np2tensor(img_gt), self.np2tensor(img_lq)
        return img_lq, img_gt

    def __len__(self):
        return len(self.hr) * self.repeat

    def _get_index(self, idx):
        return idx % len(self.hr)

    def np2tensor(self, imgs):
        return torch.from_numpy(imgs.astype(np.float32) / 255.).float().permute(2,0,1)

    def tensor2np(self, imgs):
        imgs_np = np.uint8((imgs.data.cpu().numpy().squeeze(0).transpose(1,2,0).astype(np.float32).clip(0,1) * 255.).round())
        return imgs_np


class Benchmark(Dataset):
    def __init__(self, dataset='Set5', scale=2):
        super(Benchmark, self).__init__()
        assert dataset in ('Set5','Set14','Urban100', 'Manga109')
        self.scale = scale

        root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

        self.hr_p = os.path.join(root, f'Datasets/Benchmark/{dataset}/GTmod12')
        self.hr = _get_paths_from_images(self.hr_p)

        self.lr_p = os.path.join(root, f'Datasets/Benchmark/{dataset}/LRbicx{self.scale}')  #_H264
        self.lr = _get_paths_from_images(self.lr_p)

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, index):
        gt_path = self.hr[index]
        # hr_basename = os.path.basename(gt_path)
        # lr_path = os.path.join(self.lr_p, hr_basename)
        lr_path = self.lr[index]

        img_gt = np.array(Image.open(gt_path), dtype=np.uint8)
        img_lq = np.array(Image.open(lr_path), dtype=np.uint8)

        return self.np2tensor(img_lq), self.np2tensor(img_gt)

    def np2tensor(self, imgs):
        return torch.from_numpy(imgs.astype(np.float32) / 255.).float().permute(2,0,1)


class Realsr(Dataset):
    def __init__(self, dataset='Set5', scale=2):
        super(Realsr, self).__init__()
        self.scale = scale
        # TODO: Here, Train -> Test !!! THis is eval process, why use trainset?
        self.hr = _get_paths_from_images(rf'/media/sr6/datasets/SR/liux/RealSR/version3/RealSR(V3)/Canon_Nikon/Test/{scale}',
                                         suffix='_HR.png')
        self.lr = _get_paths_from_images(rf'/media/sr6/datasets/SR/liux/RealSR/version3/RealSR(V3)/Canon_Nikon/Test/{scale}',
                                         suffix=f'_LR{scale}.png')

    def __len__(self):
        return len(self.hr)

    def __getitem__(self, index):
        gt_path = self.hr[index]
        lr_path = self.lr[index]

        img_gt = np.array(Image.open(gt_path), dtype=np.uint8)
        img_lq = np.array(Image.open(lr_path), dtype=np.uint8)

        return self.np2tensor(img_lq), self.np2tensor(img_gt)

    def np2tensor(self, imgs):
        return torch.from_numpy(imgs.astype(np.float32) / 255.).float().permute(2,0,1)



if __name__ == '__main__':
    # opt={}
    # opt['scale'] = 2
    # opt['phase'] = 'train'
    # opt['gt_size'] = 128
    # opt['geometric_augs'] = True
    #
    # dataset=Dataset_PairedImage()
    # dataset2=Dataset_PairedImage_val()
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    train_dataloader, set5_dataloader, set14_dataloader, urban100_dataloader, manga109_dataloader = Dataloader(2,256,4,1,1,'train_small')
    for i, (im, gt) in enumerate(manga109_dataloader):
        print(f'{im.shape} {gt.shape}')


    # data=Dataloader(2,256,'sr1',1,1,1)
    # for ii, (im,gt) in enumerate(data):
    #     for jj in range(im.shape[0]):
    #         img = im.numpy()
    #         gt = gt.numpy()
    #         print(img.shape, gt.shape)  # (2, 3, 1024, 2048) (2, 1, 1024, 2048)
    #     if ii == 1:
    #         break

    # print( os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))

    # x=1
    # y=2
    # assert x==y, f'{x} != {y}'

    # z=torch.ones(1,3,4,4)
    # z=z.type(torch.int)
    # print(z.dtype)