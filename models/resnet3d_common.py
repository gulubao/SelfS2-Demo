# %%
import torch.nn as nn
import torch



class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1      = conv1x1(inplanes, width)
        self.bn1        = norm_layer(width)
        self.conv2      = conv3x3(width, width, stride, groups, dilation)
        self.bn2        = norm_layer(width)
        self.conv3      = conv1x1(width, planes * self.expansion)
        self.bn3        = norm_layer(planes * self.expansion)
        self.relu       = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: 
            identity = self.downsample(x)

        out += identity
        out  = self.relu(out)

        return out

class SeperateConv3D(nn.Module):
    # no used here
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1, padding=1, bias = False):
        super(SeperateConv3D, self).__init__()
        self.conv3d_133 = nn.Conv3d(in_channels=in_planes,    
                                    out_channels=in_planes,
                                    kernel_size=(1, kernel_size, kernel_size),
                                    padding=(0, int((kernel_size-1)/2), int((kernel_size-1)/2)), 
                                    padding_mode='replicate')

        self.conv3d_311 = nn.Conv3d(in_channels=in_planes,
                                    out_channels=in_planes,
                                    kernel_size=(kernel_size, 1, 1), 
                                    padding=(int((kernel_size-1)/2), 0, 0), 
                                    padding_mode='replicate')

        self.conv3d_111 = nn.Conv3d(in_channels=in_planes*2, 
                                    out_channels=out_planes,
                                    kernel_size=(1, 1, 1),
                                    stride=stride,
                                    groups=groups, 
                                    padding=0, 
                                    padding_mode='replicate', 
                                    dilation=dilation, 
                                    bias=bias)

        self.bn1 = nn.BatchNorm3d(in_planes)
        self.bn2 = nn.BatchNorm3d(out_planes)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        e1 = self.conv3d_133(input)
        e1 = self.bn1(e1)  # 添加了BN层
        e1 = self.relu(e1)

        e2 = self.conv3d_311(input)
        e2 = self.bn1(e2)  # 添加了BN层
        e2 = self.relu(e2)

        e3 = torch.cat([e1, e2], dim=1)
        e3 = self.conv3d_111(e3)
        e3 = self.bn2(e3)  # 添加了BN层
        out = self.relu(e3)

        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # return SeperateConv3D(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False, groups=groups, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,groups=1,base_width=None,dilation=None,norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = norm_layer(planes)
        self.relu       = nn.LeakyReLU(inplace=True)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = norm_layer(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None: 
            identity = self.downsample(x)

        out += identity
        out  = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, input_chunnel, middle_chunnel, out_chunnel,zero_init_residual=False,
                 groups=1, width_per_group=32*2, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.middle_chunnel = middle_chunnel
        self.out_chunnel = out_chunnel
        self.inplanes = middle_chunnel
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups     = groups
        self.base_width = width_per_group

        self.conv1 = conv3x3(input_chunnel, self.middle_chunnel)

        self.bn1     = norm_layer(self.middle_chunnel)
        self.relu    = nn.LeakyReLU(inplace=True)

        self.layer1  = self._make_layer(block, self.middle_chunnel, layers[0])
        self.layer2  = self._make_layer(block, self.middle_chunnel, layers[1])
        self.layer3 = self._make_layer(block, self.middle_chunnel, layers[2])
        self.layer4 = self._make_layer(block, self.middle_chunnel, layers[3])

        self.conv2 = conv3x3(self.inplanes, self.out_chunnel)
        self.bn1 = norm_layer(self.out_chunnel)
                                       
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer        = self._norm_layer
        downsample        = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride         = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(block, layers, input_chunnel, middle_chunnel, out_chunnel, **kwargs):
    model = ResNet(block, layers, input_chunnel, middle_chunnel, out_chunnel, **kwargs)
    return model

def resnet18(input_chunnel,middle_chunnel, out_chunnel,  **kwargs):
    return _resnet(BasicBlock, [1, 1, 1, 1], input_chunnel,middle_chunnel, out_chunnel,  **kwargs)

def resnet34(input_chunnel,middle_chunnel, out_chunnel,  **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], input_chunnel, middle_chunnel, out_chunnel, **kwargs)

def resnet50(input_chunnel, middle_chunnel, out_chunnel, **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], input_chunnel,middle_chunnel, out_chunnel,  **kwargs)

def resnet101(input_chunnel,middle_chunnel, out_chunnel,  **kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], input_chunnel,middle_chunnel, out_chunnel,  **kwargs)

def resnet152(input_chunnel,middle_chunnel, out_chunnel,  **kwargs):
    return _resnet( Bottleneck, [3, 8, 36, 3], input_chunnel, middle_chunnel, out_chunnel, **kwargs)