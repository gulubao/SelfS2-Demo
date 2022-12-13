from numpy.core.fromnumeric import shape
import torch.nn as nn
import torch
from torch import autograd


class Enconding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Enconding_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), padding_mode='reflect', padding=(1,1))
        self.down = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),  padding=(1, 1), padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), padding_mode='reflect', padding=(1,1))
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.LeakyReLU()


    def forward(self, input):
        output = self.conv1(input)
        output = self.down(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        return output



class Deconding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Deconding_Block, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_ch)

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), padding_mode='reflect',  padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(1, 1), padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.LeakyReLU()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, input):
        output = self.bn0(input)

        output = self.conv1(output)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        output = self.up(output)
        return output




class Skip_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Skip_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output



class Unet(nn.Module):
    def __init__(self, in_ch, middle_ch, out_ch):
        super(Unet, self).__init__()
        self.start_conv = nn.Conv2d(in_channels=in_ch, out_channels=middle_ch, kernel_size=(1, 1))
        self.start_bn = nn.BatchNorm2d(middle_ch)
        self.start_relu = nn.LeakyReLU()


        self.encoding = Enconding_Block(in_ch=middle_ch, out_ch=middle_ch)
        self.skip = Skip_Block(in_ch=middle_ch, out_ch=out_ch)

        self.decoding_first = Deconding_Block(in_ch=out_ch, out_ch=middle_ch)
        self.decoding_other = Deconding_Block(in_ch=middle_ch+out_ch, out_ch=middle_ch)


        self.end_conv = nn.Conv2d(in_channels=middle_ch, out_channels=out_ch, kernel_size=(1, 1))
        self.end_bn = nn.BatchNorm2d(out_ch)
        self.end_relu = nn.LeakyReLU()
        
    def forward(self, input):
        start = self.start_conv(input)
        start = self.start_bn(start)
        start = self.start_relu(start)

        d1 = self.encoding(start)
        d2 = self.encoding(d1)
        d3 = self.encoding(d2)
        d4 = self.encoding(d3)
        d5 = self.encoding(d4)

        s1 = self.skip(d1)
        s2 = self.skip(d2)
        s3 = self.skip(d3)
        s4 = self.skip(d4)
        s5 = self.skip(d5)

        u5 = self.decoding_first(s5)
        u5 = torch.cat((u5, s4), dim = 1)

        u4 = self.decoding_other(u5)
        u4 = torch.cat((u4, s3), dim = 1)

        u3 = self.decoding_other(u4)
        u3 = torch.cat((u3, s2), dim = 1)

        u2 = self.decoding_other(u3)
        u2 = torch.cat((u2, s1), dim = 1)

        u1 = self.decoding_other(u2)

        out = self.end_conv(u1)
        out = self.end_bn(out)
        out = self.end_relu(out)

        return out





