
import os
import cv2
from typing import Dict
import time
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import DatasetFromHdf5
from torchvision.utils import save_image

from TrunDiffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import UNet
from Scheduler import GradualWarmupScheduler

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset =DatasetFromHdf5(r"E:\2Spectral SR\HSRnet\data\Sen2OHS.h5")
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], input_ch=modelConfig["channel"],ch=modelConfig["feature"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        for i, (data) in enumerate(dataloader):
            # train
            starttime = time.time()
            optimizer.zero_grad()
            x_0 = data.to(device)
            loss = trainer(x_0).sum() / 1000.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), modelConfig["grad_clip"])
            optimizer.step()
            endtime = time.time()
            print("\r ===> Epoch[{}]({}/{}): train_losses={:.6f} Time:{:.6f}s lr={:.9f}".format(e,i,len(dataloader),loss.item(),endtime - starttime,
                                                                                            optimizer.state_dict()['param_groups'][0]["lr"]), end='')
        warmUpScheduler.step()

        if e%50==49:
            torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt'+str(e)+".pt"))

def eval(noisyImage,modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], input_ch=modelConfig["channel"],ch=modelConfig["feature"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],20).to(device)
        sampledImgs = sampler(noisyImage,modelConfig["trun_cut"])
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]

        # Sampled from standard normal distribution
        # import numpy as np
        # from torchvision.utils import save_image
        # from evaluate import analysis_accu
        # from store2tiff import writeTiff as wtiff
        # noise = torch.randn(size=[modelConfig["batch_size"], 3, 512, 512], device=device)
        # sampledImgs = sampler(noise, modelConfig["T"])
        # sampledImgs = sampledImgs * 0.5 + 0.5
        # save_image(sampledImgs[:, [14, 7, 2], :, :], os.path.join("E:\Diffusion\SampledImgs\HS\generation\output.png"), nrow=modelConfig["nrow"])
        # sampledImgs = sampledImgs.permute(2, 3, 1, 0).cpu()
        # output = sampledImgs.numpy().astype(np.float32)
        # wtiff(output[:, :, :, 0], output.shape[2], output.shape[0], output.shape[1],os.path.join("E:\Diffusion\SampledImgs\HS\generation", "ID" + str(i + 1) + 'output0.tiff'))
        # wtiff(output[:, :, :, 1], output.shape[2], output.shape[0], output.shape[1],
        #       os.path.join("E:\Diffusion\SampledImgs\HS\generation", "ID" + str(i + 1) + 'output1.tiff'))
        # wtiff(output[:, :, :, 2], output.shape[2], output.shape[0], output.shape[1],
        #       os.path.join("E:\Diffusion\SampledImgs\HS\generation", "ID" + str(i + 1) + 'output2.tiff'))
        # wtiff(output[:, :, :, 3], output.shape[2], output.shape[0], output.shape[1],
        #       os.path.join("E:\Diffusion\SampledImgs\HS\generation", "ID" + str(i + 1) + 'output3.tiff'))
        # img=cv2.imread('horse.png')
        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # baseImage=torch.from_numpy(np.transpose(img,(2,0,1))).float().unsqueeze(dim=0).cuda()
        # baseImage=baseImage/torch.max(baseImage)*2-1
        # noisyImage=baseImage*0.9+noise*0.5
        #
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
        #     modelConfig["sampled_dir"], 'noised.png'), nrow=modelConfig["nrow"])

        # sampledImgs = sampler(noisyImage,modelConfig["trun_cut"])
        # sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # save_image(sampledImgs, os.path.join(
        #     modelConfig["sampled_dir"], 'denoised.png'), nrow=modelConfig["nrow"])
    return sampledImgs