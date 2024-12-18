import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import warnings
import wandb
import requests
from tqdm import tqdm
import numpy as np
from PIL import ImageFile

from Metrics.flops_compute import compute_parameters,compute_flops
from Utils.print_utils import print_log_message, print_info_message
from Utils.lr_scheduler_real import EMA, LR_Scheduler
from Utils.Saver import MySaver
from Utils.msic import AverageMeter, set_seed, delete_state_module

from Metrics.psnr_ssim import _calculate_psnr_ssim_niqe
from Tricks.process import hist_equal_loss

'''Data'''

'''Ignore Warnings'''
warnings.filterwarnings("ignore")
'''Drop problem images'''
ImageFile.LOAD_TRUNCATED_IMAGES=True
np.seterr(divide='ignore', invalid='ignore')
'''Wandb'''
os.environ["WANDB_API_KEY"] = ''
os.environ["WANDB_MODE"] = "dryrun"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

'''
ssh ssh sr6@192.168.123.11 -p 22

CUDA_VISIBLE_DEVICES=0 python train_sr_convnet.py  --use_cuda --gpu_ids 0 --exp_name van_12  --service 614 --train_batchsize 16 --val_batchsize 1 --patch_size 128 128  --scale_factor 2 --workers 4 --dataset df2k --dataset_mode train_big  --lr 0.0001 --lr_scheduler cos  --step_size 200 --step_gamma 0.5  --warmup_epochs 0 --start_epoch 0 --epochs 30 --model van_12  --resume   --finetune

--use_cuda --gpu_ids 0 --exp_name DDDAN_L_XCA_x2_6_big --train_batchsize 16 --val_batchsize 1 --patch_size 192 192 --scale_factor 2 --workers 4 --dataset DF2K --dataset_mode train_big  --lr 0.00005 --lr_scheduler c
os  --step_size 200 --step_gamma 0.5  --warmup_epochs 0 --start_epoch 0 --epochs 15 --model DDDAN_L_XCA --resume /media/sr6/codes/wsy/SR2/experiments/DDDAN_L_XCA_DF2K/192_192_0.0001_cos_l1_DDDAN_L_XCA_x2_5_big/20231106-104747/weights/best.pth  --finetune  --iter_eval

'''

class Trainer():
    def __init__(self, args):

        """----------------------------------------------- opt -----------------------------------------------"""
        self.args = args

        """----------------------------------------------- Seed ----------------------------------------------"""
        set_seed(self.args.seed)

        """---------------------------------------------- Device ---------------------------------------------"""
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        ''' ---------------------------------------------- Saver -------------------------------------------- '''
        self.saver = MySaver(directory=self.args.directory, exp_name=self.args.exp_name)

        """----------------------------------------------- Model ---------------------------------------------"""
        from Model.model_import import model_import
        self.G = model_import(self.args.model, scale=self.args.scale_factor)
        with torch.no_grad():
            h, w = 720 // self.args.scale_factor, 1280 // self.args.scale_factor
            parameters = compute_parameters(self.G)
            flops = compute_flops(self.G, input=torch.Tensor(1, 3, h, w))
            print_info_message(f'Model Parameters:{parameters:,}, Flops:{flops:,} of input size:{h}x{w}')
        self.saver.save_configs(args, parameters, flops, h, w)

        """----------------------------------------------- Loss ----------------------------------------------"""
        from Loss.basic_loss import FFTLoss, CharbonnierLoss

        self.l1, self.mse, self.charbonnier = nn.L1Loss(), nn.MSELoss(), CharbonnierLoss()
        self.fft = FFTLoss()

        """--------------------------------------------- Optimizer -------------------------------------------"""
        self.optim_net = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.args.lr)

        """----------------------------------------------- Wandb ---------------------------------------------"""
        if self.args.wandb:
            wandb.init(project=self.args.wandb_project, entity="")
            print_info_message('Use Wandb!')
        else:
            print_info_message('Not use Wandb!')

        ''' ----------------------------------------------- Cuda -------------------------------------------- '''
        if self.args.use_cuda:
            print_info_message('Use cuda!')
            self.G = nn.DataParallel(self.G, device_ids=self.args.gpu_ids).to(self.device)
            self.mse, self.l1 = self.mse.to(self.device), self.l1.to(self.device)
            self.fft = self.fft.to(self.device)

        """---------------------------------------------- Dataset --------------------------------------------"""
        from Data.dataset_bicubic_pair import Dataloader
        self.train_dataloader, self.set5_dataloader, self.set14_dataloader, self.urban100_dataloader, self.manga109_dataloader = \
            Dataloader(self.args.scale_factor, self.args.patch_size[0], self.args.train_batchsize, self.args.val_batchsize,
                       self.args.workers, mode=self.args.dataset_mode, pin_memory=self.args.pin_memory)
        self.train_dataloader_len = len(self.train_dataloader)
        print_info_message('Train:      Train_dataloader_len:{} | Train images:{}'.format(self.train_dataloader_len, self.args.train_batchsize * self.train_dataloader_len))
        print_info_message(f'Benchmark: set5:{len(self.set5_dataloader)} | set14:{len(self.set14_dataloader)} | urban100:{len(self.urban100_dataloader)} | manga109:{len(self.manga109_dataloader)}')

        """-------------------------------------------- LR_scheduler -----------------------------------------"""
        if self.args.lr_scheduler == 'poly' or 'cos' or 'step' or 'multistep':
            self.lr_scheduler_net = LR_Scheduler(self.args.lr_scheduler, base_lr=self.args.lr, num_epochs=self.args.epochs, iters_per_epoch=self.train_dataloader_len,
                                                 lr_step=self.args.step_size,warmup_epochs=self.args.warmup_epochs, step_gamma=self.args.step_gamma, eta_min=1e-7)
        elif self.args.lr_scheduler == 'reducelr':
            self.lr_scheduler_net2 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim_net,'min',0.5,6)
        print_info_message('Use {} scheduler mode'.format(self.args.lr_scheduler))

        """---------------------------------------------- EMA ---------------------------------------------"""
        if self.args.ema:
            self.ema_g = EMA(self.G, 0.999)
            self.ema_g.register()
            print_info_message('Use EMA!')

        """------------------------------------------ Resume checkpoint --------------------------------------"""
        self.best_pred = 0.
        if self.args.resume is not None:
            checkpoint = torch.load(self.args.resume,map_location=self.device)
            self.args.start_epoch = checkpoint['epoch'] + 1
            if self.args.delete_module:
                checkpoint = delete_state_module(checkpoint)
            ###
            model_dict = self.G.module.state_dict()
            overlap_ = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(overlap_)
            print_info_message(f'{(len(overlap_) * 1.0 / len(checkpoint["state_dict"]) * 100):.4f}% weights is loaded!')
            print([k for k, v in checkpoint['state_dict'].items() if k not in model_dict])
            ###
            self.G.module.load_state_dict(model_dict)

            if not self.args.finetune:
                self.optim_net.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']
                print_info_message('Not Finetune!')
            elif self.args.finetune:
                self.args.start_epoch = 0
                self.best_pred = 0.0  # checkpoint['best_pred']
                print_info_message(f'Finetune!')

            print_info_message(f"Loading checkpoint:'{self.args.resume}' epoch:{checkpoint['epoch']} previous_best_pred:{checkpoint['best_pred']}")
            keys = [k for k, v in checkpoint.items()]
            print_info_message(keys)

    def net_train(self,now_epoch):
        self.G.train()
        train_losses = AverageMeter()
        tbar = tqdm(self.train_dataloader, unit='image')
        for i, (im, gt) in enumerate(tbar):
            self.lr_scheduler_net(self.optim_net, i, now_epoch, self.best_pred)
            if i == 0 and now_epoch == 0: print(im.shape, gt.shape)
            im, gt = im.to(self.device), gt.to(self.device)

            out = self.G(im)
            loss = self.l1(out, gt) + 0.1 * self.fft(out, gt)

            tbar.set_description(f'[Train Epoch:{now_epoch}] [train_loss: fft_loss: {0} SR_loss: {loss.item():.4f} Total_loss:{loss.item():.4f}]')
            train_losses.update(loss.item(), n=im.shape[0])
            if self.args.wandb: wandb.log({"train_iter_loss": loss.item()})

            self.optim_net.zero_grad()
            loss.backward()
            self.optim_net.step()

            if self.args.ema:
                self.ema_g.update()

            if self.args.iter_eval:
                if i % self.args.eval_freq == 0 or i == -1:
                    self.Benchmark_iter(now_epoch, i, self.set5_dataloader)

            if self.args.noval:
                self.saver.save_checkpoint_noval(
                    {
                        'epoch': now_epoch,
                        'iter': i,
                        'state_dict': self.G.module.state_dict(),
                        'optimizer': self.optim_net.state_dict(),
                    }, now_epoch, i, self.args, print_=False, iter_save=True, save_freq=200
                )

        # epoch / iter mode test and save
        if self.args.epoch_eval:
            self.Benchmark(now_epoch, self.set5_dataloader)

        print_info_message('Epoch:{}, numImages:{}, Train_loss:{:.3f}'.format(epoch, i * self.args.train_batchsize + im.data.shape[0], train_losses.avg))
        if self.args.wandb: wandb.log({"total_train_loss": train_losses.avg})
        self.saver.save_record_train(now_epoch,train_losses.avg)

    @torch.no_grad()
    def net_eval(self,now_epoch):
        self.G.eval()
        lr_rate = self.optim_net.param_groups[0]['lr']
        val_losses, psnr, ssim, niqe = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        tbar = tqdm(self.val_dataloader,unit='image')
        with torch.no_grad():
            for i, (im,gt) in enumerate(tbar):
                if i == 0 and now_epoch == 0: print(im.shape,gt.shape)
                # save input
                # self.saver.save_sr(i,500,gt,now_epoch,denormal=False,flag='hr')
                # self.saver.save_sr(i,500,im,now_epoch,denormal=False,flag='lr')
                im, gt = im.to(self.device), gt.to(self.device)
                output = self.G(im)
                # save sr
                # self.saver.save_sr(i,500,output,now_epoch,denormal=False,flag='sr')
                loss = self.l1(output, gt)

                val_losses.update(loss.item(), n=im.shape[0])
                tbar.set_description(f'[Val Epoch:{now_epoch}] [val_loss:{loss.item():.4f}]')
                if self.args.wandb: wandb.log({"val_iter_loss": loss.item()})

                psnr_temp,ssim_temp,niqe_temp,batch = _calculate_psnr_ssim_niqe(output,gt,crop_border=self.args.scale_factor,input_order='CHW',test_y_channel=True,mean=(0,0,0),std=(1,1,1))
                psnr.update(psnr_temp, batch)
                ssim.update(ssim_temp, batch)
                niqe.update(niqe_temp, batch)

            avg_psnr = psnr.avg
            avg_ssim = ssim.avg
            avg_niqe = niqe.avg
            if self.args.wandb:
                wandb.log({"total_val_loss": val_losses.avg,
                           "Avg_PSNR": avg_psnr,
                           "Avg_SSIM": avg_ssim,
                           "Avg_niqe": avg_niqe})
            self.saver.save_record_val(now_epoch,lr_rate,val_losses.avg,avg_psnr,avg_ssim,avg_niqe)
            print_log_message(f'val loss_script:{val_losses.avg:.4f}  Avg_psnr:{avg_psnr:.6f} Avg_ssim:{avg_ssim:.6f} Avg_niqe:{avg_niqe:.6f}')

        new_pred=avg_psnr
        self.saver.save_checkpoint_override(
            {
                'epoch': now_epoch,
                'state_dict': self.G.module.state_dict(),  #
                'optimizer': self.optim_net.state_dict(),
                'pred': new_pred,
                'ssim': avg_ssim,
                'niqe': avg_niqe
            }, now_epoch, new_pred, self.best_pred, self.args, print_=True
        )
        if new_pred > self.best_pred:
            self.best_pred = new_pred

    @torch.no_grad()
    def Benchmark(self, now_epoch, dataloader):
        if self.args.ema:
            self.ema_g.apply_shadow()

        self.G.eval()
        lr_rate = self.optim_net.param_groups[0]['lr']
        val_losses, psnr, ssim, niqe = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        tbar = tqdm(dataloader, unit='image')
        with torch.no_grad():
            for i, (im, gt) in enumerate(tbar):
                if i == 0 and now_epoch == 0: print(im.shape, gt.shape)
                # save input
                # self.saver.save_sr(i, 500, gt, now_epoch, denormal=False, flag='hr')
                # self.saver.save_sr(i, 500, im, now_epoch, denormal=False, flag='lr')
                im, gt = im.to(self.device), gt.to(self.device)
                output = self.G(im)
                # save sr
                # self.saver.save_sr(i, 500, output, now_epoch, denormal=False, flag='sr')
                loss = self.l1(output, gt)

                val_losses.update(loss.item(), n=im.shape[0])
                tbar.set_description(f'[Val Epoch:{now_epoch}] [val_loss:{loss.item():.4f}]')
                if self.args.wandb: wandb.log({"val_iter_loss": loss.item()})

                psnr_temp, ssim_temp, niqe_temp, batch = _calculate_psnr_ssim_niqe(output, gt, crop_border=0,
                                                                                   input_order='CHW',
                                                                                   test_y_channel=True, mean=(0, 0, 0),
                                                                                   std=(1, 1, 1))
                psnr.update(psnr_temp, batch)
                ssim.update(ssim_temp, batch)
                niqe.update(niqe_temp, batch)

            avg_psnr = psnr.avg
            avg_ssim = ssim.avg
            avg_niqe = niqe.avg
            if self.args.wandb:
                wandb.log({"total_val_loss": val_losses.avg,
                           "Avg_PSNR": avg_psnr,
                           "Avg_SSIM": avg_ssim,
                           "Avg_niqe": avg_niqe})
            self.saver.save_record_val(now_epoch, lr_rate, val_losses.avg, avg_psnr, avg_ssim, avg_niqe)
            print_log_message(
                f'val loss:{val_losses.avg:.4f}  Avg_psnr:{avg_psnr:.6f} Avg_ssim:{avg_ssim:.6f} Avg_niqe:{avg_niqe:.6f}')

        new_pred = avg_psnr
        self.saver.save_checkpoint_override(
            {
                'epoch': now_epoch,
                'state_dict': self.G.module.state_dict(),  #
                'optimizer': self.optim_net.state_dict(),
                'pred': new_pred,
                'ssim': avg_ssim,
                'niqe': avg_niqe
            }, now_epoch, new_pred, self.best_pred, self.args, print_=True
        )
        if new_pred > self.best_pred:
            self.best_pred = new_pred

        if self.args.ema:
            self.ema_g.restore()

    @torch.no_grad()
    def Benchmark_iter(self, now_epoch, iter, dataloader):
        if self.args.ema:
            self.ema_g.apply_shadow()

        self.G.eval()
        lr_rate = self.optim_net.param_groups[0]['lr']
        val_losses, psnr, ssim, niqe = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        # tbar = tqdm(self.set5_dataloader, unit='image')
        with torch.no_grad():
            for i, (im, gt) in enumerate(dataloader):
                # if i == 0 and now_epoch == 0: print(im.shape, gt.shape)
                # save input
                # self.saver.save_sr(i, 500, gt, now_epoch, denormal=False, flag='hr')
                # self.saver.save_sr(i, 500, im, now_epoch, denormal=False, flag='lr')
                im, gt = im.to(self.device), gt.to(self.device)
                output = self.G(im)
                # save sr
                # self.saver.save_sr(i, 500, output, now_epoch, denormal=False, flag='sr')
                # loss = self.l1(output, gt)
                #
                # val_losses.update(loss.item(), n=im.shape[0])
                # tbar.set_description(f'[Val Epoch:{now_epoch}] [val_loss:{loss.item():.4f}]')
                # if self.args.wandb: wandb.log({"val_iter_loss": loss.item()})
                psnr_temp, ssim_temp, niqe_temp, batch = _calculate_psnr_ssim_niqe(output, gt, crop_border=0,
                                                                                   input_order='CHW',
                                                                                   test_y_channel=True, mean=(0, 0, 0),
                                                                                   std=(1, 1, 1))
                psnr.update(psnr_temp, batch)
                ssim.update(ssim_temp, batch)
                niqe.update(niqe_temp, batch)

            avg_psnr = psnr.avg
            avg_ssim = ssim.avg
            avg_niqe = niqe.avg
            if self.args.wandb:
                wandb.log({"total_val_loss": val_losses.avg,
                           "Avg_PSNR": avg_psnr,
                           "Avg_SSIM": avg_ssim,
                           "Avg_niqe": avg_niqe})
            self.saver.save_record_val(now_epoch, lr_rate, val_losses.avg, avg_psnr, avg_ssim, avg_niqe)

        new_pred = avg_psnr
        self.saver.save_checkpoint_override(
            {
                'epoch': now_epoch,
                'iter': iter,
                'state_dict': self.G.module.state_dict(),  #
                'optimizer': self.optim_net.state_dict(),
                'pred': new_pred,
                'ssim': avg_ssim,
                'niqe': avg_niqe
            }, now_epoch, new_pred, self.best_pred, self.args, print_=False
        )
        if new_pred > self.best_pred:
            self.best_pred = new_pred

        if self.args.ema:
            self.ema_g.restore()


if __name__ == '__main__':
    from option import args
    trainer = Trainer(args)

    for epoch in range(trainer.args.start_epoch, args.epochs):
        trainer.net_train(epoch)
    print(f'The last best predict is: {trainer.best_pred}')
    # requests.get('[url]/Train Finished!')



