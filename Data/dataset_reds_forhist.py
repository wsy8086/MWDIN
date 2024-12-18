import os
import random
import cv2
import numpy as np
from PIL import Image
from natsort import os_sorted as natsorted
import torch
import albumentations
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from Utils.imresize import imresize
# from Tricks.mosaic import Mosaic
# from Tricks.mosaic_process import mosaic_dataset, multi_thread_mosaic_dataset
from Data.data_utils import _get_paths_from_images, paired_random_crop_, random_augmentation


def Dataloader(scale, gt_size, train_batchsize, val_batchsize, num_worker, mode='reds', pin_memory=False, repeat=1, vedio='000'):
    train_dataset = Dataset_PairedImage(scale, mode=mode, gt_size=gt_size, repeat=repeat)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=num_worker, drop_last=True, pin_memory=pin_memory)

    reds = Benchmark(video=vedio, scale=scale)

    test_dataloader = DataLoader(reds, batch_size=val_batchsize, shuffle=False, num_workers=num_worker, drop_last=False, pin_memory=pin_memory)

    return train_dataloader, test_dataloader


class Dataset_PairedImage(Dataset):
    def __init__(self, scale=2, mode='reds', gt_size=128, geometric_augs=True, repeat=1):
        """
        Args:
            scale: 2 | 3 | 4
            mode: reds
            gt_size: gt size
            geometric_augs:
            repeat: reds has 240 videos, per video has 100 frames. repest = 15 (3600)
        """
        super(Dataset_PairedImage, self).__init__()
        self.scale = scale
        self.mode = mode
        assert self.mode in ('reds', 'vim90k')

        root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

        if self.mode == 'reds':  # df2k #3450
            self.hr_p = os.path.join(root, f'Datasets{os.sep}Reds{os.sep}GT')
            self.lr_p = os.path.join(root, f'Datasets{os.sep}Reds{os.sep}DR_675_1125_30')
            self.hr_dirs = os.listdir(self.hr_p)
            self.lr_dirs = os.listdir(self.lr_p)

        self.mean = torch.tensor([0., 0., 0.])#.reshape(3,1,1)
        self.std = torch.tensor([1., 1., 1.])#.reshape(3,1,1)

        self.gt_size = gt_size
        self.geometric_augs = geometric_augs
        self.repeat = repeat

    def __getitem__(self, index):
        # print(f'===> {self.__len__()}')  #  1170
        index = self._get_index(index)
        # Per Video has 100 Frames: 00000000 - 00000099
        random_frame = random.randint(0, 97)
        start, middle, end = f'{random_frame:08d}', f'{random_frame+1:08d}', f'{random_frame+2:08d}'

        # Dir: 000 - 239
        start_frame = os.path.join(self.lr_p, f'{index:03d}', f'{start}.png')
        middle_frame = os.path.join(self.lr_p, f'{index:03d}', f'{middle}.png')
        end_frame = os.path.join(self.lr_p, f'{index:03d}', f'{end}.png')
        gt_frame = os.path.join(self.hr_p, f'{index:03d}', f'{end}.png')

        start_frame_img = np.array(Image.open(start_frame), dtype=np.uint8)
        middle_frame_img = np.array(Image.open(middle_frame), dtype=np.uint8)
        end_frame_img = np.array(Image.open(end_frame), dtype=np.uint8)
        gt_frame_img = np.array(Image.open(gt_frame), dtype=np.uint8)

        ### hist equalization
        gt_frame_img = albumentations.CLAHE(clip_limit=1,
                                            tile_grid_size=(8, 8),
                                            always_apply=True,
                                            p=1)(image=gt_frame_img)["image"]

        ###

        gt_frame_img, (start_frame_img, middle_frame_img, end_frame_img) = \
            paired_random_crop_([gt_frame_img], [start_frame_img, middle_frame_img, end_frame_img], self.gt_size,self.scale)

        if self.geometric_augs:
            gt_frame_img, start_frame_img, middle_frame_img, end_frame_img = \
                random_augmentation(gt_frame_img, start_frame_img, middle_frame_img, end_frame_img)

        gt_frame_img, start_frame_img, middle_frame_img, end_frame_img = \
            self.np2tensor(gt_frame_img), self.np2tensor(start_frame_img), self.np2tensor(middle_frame_img), self.np2tensor(end_frame_img)
        return gt_frame_img, start_frame_img, middle_frame_img, end_frame_img

    def __len__(self):
        return len(self.hr_dirs) * self.repeat

    def _get_index(self, idx):
        return idx % len(self.hr_dirs)

    def np2tensor(self, imgs):
        return torch.from_numpy(imgs.astype(np.float32) / 255.).float().permute(2,0,1)

    def tensor2np(self, imgs):
        imgs_np = np.uint8((imgs.data.cpu().numpy().squeeze(0).transpose(1,2,0).astype(np.float32).clip(0,1) * 255.).round())
        return imgs_np


class Benchmark(Dataset):
    def __init__(self, video='000', scale=2):
        """
        Args:
            video: 000 - 239
            scale: 2 | 3 | 4
        """
        super(Benchmark, self).__init__()
        # assert dataset in ('Set5','Set14','Urban100', 'Manga109')
        self.scale = scale

        root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

        self.hr_p = os.path.join(root, f'Datasets{os.sep}Reds{os.sep}GT{os.sep}{video}')
        self.lr_p = os.path.join(root, f'Datasets{os.sep}Reds{os.sep}DR_675_1125_30{os.sep}{video}')

        self.hr_dirs = os.listdir(self.hr_p)
        self.lr_dirs = os.listdir(self.lr_p)

    def __len__(self):
        return len(self.hr_dirs)

    def __getitem__(self, index):

        if index == 0:
            frame_index = [0, 0, 0]  # [-2, -1, 0]
        elif index == 1:
            frame_index = [0, 0, 1]  # [-1, 0, 1]
        else:
            frame_index = [index-2, index-1, index]

        gt_frame = np.array(Image.open(f'{self.hr_p}{os.sep}{index:08d}.png'), dtype=np.uint8)

        frame0 = np.array(Image.open(f'{self.hr_p}{os.sep}{frame_index[0]:08d}.png'), dtype=np.uint8)
        frame1 = np.array(Image.open(f'{self.hr_p}{os.sep}{frame_index[1]:08d}.png'), dtype=np.uint8)
        frame2 = np.array(Image.open(f'{self.hr_p}{os.sep}{frame_index[2]:08d}.png'), dtype=np.uint8)

        return self.np2tensor(gt_frame), self.np2tensor(frame0), self.np2tensor(frame1), self.np2tensor(frame2)

    def np2tensor(self, imgs):
        return torch.from_numpy(imgs.astype(np.float32) / 255.).float().permute(2, 0, 1)



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

    # train_dataloader, set5_dataloader, set14_dataloader, urban100_dataloader, manga109_dataloader = Dataloader(2,256,4,1,1,'train_small')
    # for i, (im, gt) in enumerate(manga109_dataloader):
    #     print(f'{im.shape} {gt.shape}')

    random_frame = random.randint(96, 97)
    start_frame, end_frame = f'{random_frame:08d}', f'{random_frame + 2:08d}'
    print(start_frame, end_frame)

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