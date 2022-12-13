from numpy.core.fromnumeric import shape
import torch.nn as nn
import torch
from torch import autograd



class SeperateConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SeperateConv3D, self).__init__()
        self.conv3d_133 = nn.Conv3d(in_channels=in_ch,    out_channels=out_ch, kernel_size=(
            1, 3, 3), stride=1,  padding=(0, 1, 1), padding_mode='replicate')
        self.conv3d_311 = nn.Conv3d(in_channels=in_ch,    out_channels=out_ch, kernel_size=(
            3, 1, 1), stride=1,  padding=(1, 0, 0), padding_mode='replicate')
        self.conv3d_111 = nn.Conv3d(in_channels=out_ch*2, out_channels=out_ch,
                                    kernel_size=(1, 1, 1), stride=1,  padding=0, padding_mode='replicate')
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        e1 = self.conv3d_133(input)
        e1 = self.bn(e1)
        e1 = self.relu(e1)

        e2 = self.conv3d_311(input)
        e2 = self.bn(e2)
        e2 = self.relu(e2)

        e3 = torch.cat([e1, e2], dim=1)
        e3 = self.conv3d_111(e3)
        e3 = self.bn(e3)
        e3 = self.relu(e3)

        out = e3 + input
        out = self.bn(out)
        out = self.relu(out)

        return out



class SeperateConv3D_Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SeperateConv3D_Up, self).__init__()
        self.conv3d_133 = nn.Conv3d(in_channels=in_ch,    out_channels=out_ch, kernel_size=(
            1, 3, 3), stride=1,  padding=(0, 1, 1), padding_mode='replicate')
        self.conv3d_311 = nn.Conv3d(in_channels=in_ch,    out_channels=out_ch, kernel_size=(
            3, 1, 1), stride=1,  padding=(1, 0, 0), padding_mode='replicate')
        self.conv3d_111 = nn.Conv3d(in_channels=out_ch*2, out_channels=out_ch,
                                    kernel_size=(1, 1, 1), stride=1,  padding=0, padding_mode='replicate')
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        e1 = self.conv3d_133(input)
        e1 = self.bn(e1)
        e1 = self.relu(e1)

        e2 = self.conv3d_311(input)
        e2 = self.bn(e2)
        e2 = self.relu(e2)

        e3 = torch.cat([e1, e2], dim=1)
        e3 = self.conv3d_111(e3)
        e3 = self.bn(e3)
        e3 = self.relu(e3)

        return e3



class SeperateConv3D_Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SeperateConv3D_Down, self).__init__()
        self.conv3d_133 = nn.Conv3d(in_channels=in_ch,    out_channels=out_ch, kernel_size=(
            1, 3, 3), stride=1,  padding=(0, 1, 1), padding_mode='replicate')
        self.conv3d_311 = nn.Conv3d(in_channels=in_ch,    out_channels=out_ch, kernel_size=(
            3, 1, 1), stride=1,  padding=(1, 0, 0), padding_mode='replicate')
        self.conv3d_111 = nn.Conv3d(in_channels=out_ch*2, out_channels=out_ch,
                                    kernel_size=(1, 1, 1), stride=1,  padding=0, padding_mode='replicate')
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        e1 = self.conv3d_133(input)
        e1 = self.bn(e1)
        e1 = self.relu(e1)

        e2 = self.conv3d_311(input)
        e2 = self.bn(e2)
        e2 = self.relu(e2)

        e3 = torch.cat([e1, e2], dim=1)
        e3 = self.conv3d_111(e3)
        e3 = self.bn(e3)
        e3 = self.relu(e3)

        return e3



class Enconding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Enconding_Block, self).__init__()
        self.seperateConv3D = SeperateConv3D(in_ch, out_ch)

        self.conv3d_133 = self.seperateConv3D.conv3d_133
        self.conv3d_311 = self.seperateConv3D.conv3d_311
        self.conv3d_111 = self.seperateConv3D.conv3d_111
        self.bn = self.seperateConv3D.bn
        self.relu = self.seperateConv3D.relu
        self.down_conv3d = nn.Conv3d(in_channels=out_ch, out_channels=out_ch, kernel_size=(
            1, 3, 3), stride=(1, 2, 2),  padding=(0, 1, 1), padding_mode='replicate')

    def forward(self, input):
        for i in range(5):
            e1 = self.conv3d_133(input)
            e1 = self.bn(e1)
            e1 = self.relu(e1)

            e2 = self.conv3d_311(input)
            e2 = self.bn(e2)
            e2 = self.relu(e2)

            e3 = torch.cat([e1, e2], dim=1)
            e3 = self.conv3d_111(e3)
            e3 = self.bn(e3)
            e3 = self.relu(e3)

            input = e3 + input

        down = self.down_conv3d(input)
        return input, down



class Deconding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Deconding_Block, self).__init__()
        # self.up = nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0,1,1))
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='nearest')

        self.relu = nn.LeakyReLU(inplace=True)
        self.conv3d_133 = nn.Conv3d(in_channels=in_ch,    out_channels=out_ch, kernel_size=(
            1, 3, 3), stride=1,  padding=(0, 1, 1), padding_mode='replicate')
        self.conv3d_311 = nn.Conv3d(in_channels=in_ch,    out_channels=out_ch, kernel_size=(
            3, 1, 1), stride=1,  padding=(1, 0, 0), padding_mode='replicate')
        self.conv3d_111 = nn.Conv3d(in_channels=out_ch*2, out_channels=out_ch,
                                    kernel_size=(1, 1, 1), stride=1,  padding=0, padding_mode='replicate')
        self.bn = nn.BatchNorm3d(out_ch)

    def forward(self, f, d):
        d = self.up(d)
        d = self.relu(d)
        d = torch.cat([f, d], dim=1)
        d1 = self.conv3d_111(d)
        d1 = self.relu(d1)

        for i in range(5):
            d2 = self.conv3d_311(d1)
            d2 = self.relu(d2)

            d3 = self.conv3d_133(d1)
            d3 = self.relu(d3)

            d4 = torch.cat([d2, d3], dim=1)

            d4 = self.conv3d_111(d4)
            d4 = self.relu(d4)

            d1 = d4 + d1

        return d1




class Skip_Block(nn.Module):
    def __init__(self, in_ch_3, out_ch_3):
        super(Skip_Block, self).__init__()
        self.ifskip = True if in_ch_3 == out_ch_3 else False
        self.conv3d = SeperateConv3D(in_ch=in_ch_3, out_ch=out_ch_3)
        self.conv3d2 = SeperateConv3D(in_ch=out_ch_3, out_ch=out_ch_3)
        self.bn = nn.BatchNorm3d(out_ch_3)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        output = self.conv3d(input)
        output = self.bn(output)
        output = self.relu(output)
        output = self.conv3d2(output)
        output = self.bn(output)
        output = self.relu(output)
        if self.ifskip == True:
            output = output + input
        return output




class Unet_3D(nn.Module):
    def __init__(self, in_ch, middle_ch, out_ch):
        super(Unet_3D, self).__init__()
        self.start_conv3d = nn.Conv3d(in_channels=in_ch, out_channels=middle_ch,
                                      kernel_size=1, stride=1,  padding=0, padding_mode='replicate')
        self.enconding_block = Enconding_Block(middle_ch, middle_ch)
        self.deconding_block = Deconding_Block(middle_ch, middle_ch)
        self.end_conv3d = nn.Conv3d(in_channels=middle_ch, out_channels=out_ch,
                                    kernel_size=1, stride=1,  padding=0, padding_mode='replicate')

        self.relu = nn.LeakyReLU(inplace=True)
        
        self.skip = Skip_Block(in_ch_3 = middle_ch, out_ch_3 = middle_ch)

    def forward(self, x):
        z0 = self.start_conv3d(x)
        z0 = self.relu(z0)
        # Enconding
        s0,down0 = self.enconding_block(z0)  # name='encoding_0'
        s1,down1 = self.enconding_block(down0)  # name='encoding_1'
        s2,down2 = self.enconding_block(down1)  # name='encoding_2'
        s3,down3 = self.enconding_block(down2)  # name='encoding_3'
        
        s0 = self.skip(s0)
        s1 = self.skip(s1)
        s2 = self.skip(s2)
        s3 = self.skip(s3)
#         s3 = self.final_skip(s3)
             
        # decoding
        deco3 = self.deconding_block(s3, down3)  # name='decoding_3'
        deco2 = self.deconding_block(s2, deco3)  # name='decoding_2'
        deco1 = self.deconding_block(s1, deco2)  # name='decoding_1'
        deco0 = self.deconding_block(s0, deco1)  # name='decoding_0'

        out = self.end_conv3d(deco0)
        out = out+x

        return out


class Unet_3D_5(nn.Module):
    def __init__(self, in_ch, middle_ch, out_ch):
        super(Unet_3D_5, self).__init__()
        self.start_conv3d = nn.Conv3d(in_channels=in_ch, out_channels=middle_ch,
                                      kernel_size=1, stride=1,  padding=0, padding_mode='replicate')
        self.enconding_block = Enconding_Block(middle_ch, middle_ch)
        self.deconding_block = Deconding_Block(middle_ch, middle_ch)
        self.end_conv3d = nn.Conv3d(in_channels=middle_ch, out_channels=out_ch,
                                    kernel_size=1, stride=1,  padding=0, padding_mode='replicate')

        self.relu = nn.LeakyReLU(inplace=True)
        
        self.skip = Skip_Block(in_ch_3 = middle_ch, out_ch_3 = middle_ch)
        self.endskip = Skip_Block(in_ch_3 = out_ch, out_ch_3 = out_ch)

    def forward(self, x):
        z0 = self.start_conv3d(x)
        z0 = self.relu(z0)
        # Enconding
        s0,down0 = self.enconding_block(z0)  # name='encoding_0'
        s1,down1 = self.enconding_block(down0)  # name='encoding_1'
        s2,down2 = self.enconding_block(down1)  # name='encoding_2'
        s3,down3 = self.enconding_block(down2)  # name='encoding_3'
        s4,down4 = self.enconding_block(down3)  # name='encoding_3'
        
        s0 = self.skip(s0)
        s1 = self.skip(s1)
        s2 = self.skip(s2)
        s3 = self.skip(s3)
        s4 = self.skip(s4)
#         s3 = self.final_skip(s3)
             
        # decoding
        deco4 = self.deconding_block(s4, down4)  # name='decoding_3'
        deco3 = self.deconding_block(s3, deco4)  # name='decoding_3'
        deco2 = self.deconding_block(s2, deco3)  # name='decoding_2'
        deco1 = self.deconding_block(s1, deco2)  # name='decoding_1'
        deco0 = self.deconding_block(s0, deco1)  # name='decoding_0'

        out = self.end_conv3d(deco0)
        out = out+self.endskip(x)

        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

