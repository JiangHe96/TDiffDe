from Train import train, eval
import os
import scipy.io as io
from torch.autograd import Variable
import torch
import numpy as np
from torchvision.utils import save_image
from evaluate import analysis_accu
from store2tiff import writeTiff

def main(model_config = None):
    modelConfig = {
        "state": "eval", # train or eval
        "epoch": 400,
        "batch_size": 4,
        "T": 1000,
        "channel": 32,
        "feature": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/"+"HS"+"/",
        "test_load_weight": "ckpt.pt",
        "sampled_dir": "./SampledImgs/"+"HS"+"/",
        "sampledImgName": "TDiffDe.png",
        "nrow": 2,
        "trun_cut": 90
        }

    print(modelConfig["save_weight_dir"])
    if not os.path.exists(modelConfig["save_weight_dir"]):
        os.makedirs(modelConfig["save_weight_dir"])
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        savepath=modelConfig["sampled_dir"]+str(modelConfig["trun_cut"])+'/'
        print(savepath)
        data = io.loadmat(r'data/ohs_real2.mat')
        img_noise = data['noise']

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        j=modelConfig["trun_cut"]
        print(j)
        img_nois=img_noise

        # image_transpose
        img_in = np.transpose(img_nois, [2, 0, 1])
        # input_data_construction
        img_in = img_in
        with torch.no_grad():
            img = Variable(torch.from_numpy(img_in).float()).view(1, -1, img_in.shape[1],img_in.shape[2]).cuda()
            noiseimg = 2 * img - 1
            out=eval(noiseimg.cuda(),modelConfig)

            # save_image(ana_refference[:, :, [14, 7, 2]].permute(2,0,1), os.path.join(savepath, "ID"+str(i+1)+"GT.png"), nrow=modelConfig["nrow"])
            # save_image(img[:, [14, 7, 2], :, :], os.path.join(savepath, "ID" + str(i+1) + "noise.png"),nrow=modelConfig["nrow"])
            save_image(out[:, [14, 7, 2], :, :], os.path.join(savepath, "Trunc" + str(j) + "output.png"),nrow=modelConfig["nrow"])

            # out_temp=out.permute(2, 3, 1, 0)
            # out_ana=out_temp[:, :, :, 0]
            # index[:, :, i]=analysis_accu(ana_refference,out_ana,1)

            # output = out_ana.cpu()
            # output = output.numpy().astype(np.float32)
            # output=output[20,:,:,:]
            # writeTiff(output, output.shape[2], output.shape[0], output.shape[1],
            #       os.path.join(savepath, "ID" + str(i + 1) + 'output.tiff'))
            # writeTiff(img_refference, img_refference.shape[2], img_refference.shape[0], img_refference.shape[1],
            #       os.path.join(savepath, "ID" + str(i + 1) + 'GT.tiff'))
            # writeTiff(img_nois, img_nois.shape[2], img_nois.shape[0], img_nois.shape[1],
            #       os.path.join(savepath, "ID" + str(i + 1) + 'noise.tiff'))

            # print("image "+str(i+1)+": ",index[:, 0, i])

if __name__ == '__main__':
    main()
