# %%
import torch.nn as nn
import torch
# import torch.nn.functional as F
# from torchvision.models import vgg16_bn
from torchvision.models import vgg16


# %%
class Perceptual_Loss(nn.Module):
    def __init__(self):
        super(Perceptual_Loss, self).__init__()
        dtype = torch.cuda.FloatTensor
        vgg = vgg16(pretrained=True).features
        vgg = vgg[0:29]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval().cuda()
        # self.loss = torch.nn.L1Loss().type(dtype) 
        self.loss = torch.nn.MSELoss().type(dtype)

    def forward(self, x, y):
        loss_total = 0
        for i in range(x.shape[1]):
            input_img_x = x[:, [i, i, i], :, :];
            input_img_y = y[:, [i, i, i], :, :]
            out_img_x = self.vgg(input_img_x);
            out_img_y = self.vgg(input_img_y)
            loss_here = self.loss(out_img_x, out_img_y) / 512
            loss_total = loss_total + loss_here
        return loss_total
