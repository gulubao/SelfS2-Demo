# %%
# from numpy.core.fromnumeric import shape

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.nn.modules import conv
import scipy
import scipy.io as scio

dtype = torch.cuda.FloatTensor
import scipy.stats as st

import scipy.stats as st


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


def GaussianMatrix(sigma, kernelSize):
    end = int(kernelSize / 2) - 0.5;
    start = -end
    GassVector = np.arange(start, end + 1, 1)
    X = np.asarray(GassVector)
    i = 0
    for v_i in X:
        GassVector[i] = Gaussian(v_i, sigma)
        i += 1
    kernel = np.outer(GassVector, GassVector.T)
    kernel = kernel / np.sum(kernel)
    return kernel


def Gaussian(x, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-((x - 0) ** 2)) / (2 * sigma ** 2))


class Corr2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.corr2d(x, self.kernel) + self.bias

    @staticmethod
    def corr2d(x, kernel):
        H, W = x.shape
        h, w = kernel.shape
        ret = torch.zeros(H - h + 1, W - w + 1)
        for i in range(ret.shape[0]):
            for j in range(ret.shape[1]):
                ret[i, j] = (x[i:(i + h), j:(j + w)] * kernel).sum()
        return ret

    def chech_coor(self):
        # test corr2d
        # import torch
        # import numpy as np
        x = torch.tensor(np.reshape(np.arange(1, 16 + 1), (4, 4)))
        y = np.flip(np.flip(np.reshape(np.arange(1, 9 + 1), (3, 3)), 0), 1)
        y = torch.tensor(y)
        coor = Corr2D.corr2d(x, y)
        print(coor)

        from scipy import signal
        true_res = signal.correlate(x, y, mode='valid')
        print(true_res)

        kernel = y.unsqueeze(0).unsqueeze(0).float()
        weight = torch.nn.Parameter(data=kernel, requires_grad=False)
        conv = torch.nn.functional.conv2d(x.unsqueeze(0).unsqueeze(0).float(), weight, padding=0)
        print(conv)


class S2_GaussianBlur(nn.Module):
    def __init__(self, nr, nc, kernelSize=10, scale=2, RealData=False):
        super(S2_GaussianBlur, self).__init__()
        if RealData == False:  # simulate
            mtf = [.32, .26, .28, .24, .38, .34, .34, .26, .33, .26, .22, .23]
        elif RealData == True:  # realdata
            mtf = [.32, .26, .28, .24, .38, .34, .34, .26, .23, .33, .26, .22]  # realdata
        self.kernelSize = kernelSize
        self.scale = scale
        self.nr = int(nr)
        self.nc = int(nc)
        self.middlel = int(nr / 2)
        self.middlec = int(nc / 2)

        if self.scale == 'all':
            self.mtf = mtf
        elif self.scale == 1:
            self.mtf = [mtf[1], mtf[2], mtf[3], mtf[7]]
        elif self.scale == 2:
            self.mtf = [mtf[4], mtf[5], mtf[6], mtf[8], mtf[10], mtf[11]]
        elif self.scale == 6:
            self.mtf = [mtf[0], mtf[9]]

        self.sigmas = self._calculateSigma()
        self.channels = len(self.sigmas)

        self.kernel = np.array(self._getKernels(
            self.kernelSize, self.sigmas)).astype(np.float32)

    def forward(self, x_):
        """
        Args:
                x ([Tensor|Variable]): [(1,C,L,W)]

            Returns:
                [Tensor|Variable]: [(1,C,L,W)]
        """
        x = x_.clone()
        x_new = None
        for i in range(x.shape[1]):
            temp_kernel = self.kernel[i]
            temp_kernel = torch.FloatTensor(
                temp_kernel).unsqueeze(0).unsqueeze(0)
            temp_weight = nn.Parameter(
                data=temp_kernel, requires_grad=False).cuda()
            temp_x = x[:, i, :, :].unsqueeze(dim=0)
            temp_x = F.pad(temp_x, pad=[self.middlec, self.middlec, self.middlel, self.middlel], mode='circular')
            temp_x = F.conv2d(temp_x, temp_weight, padding=0, groups=1).squeeze(0)
            temp_x = temp_x[:, 0:-1, 0:-1]
            if x_new is None:
                x_new = temp_x
            else:
                x_new = torch.cat((x_new, temp_x), axis=0)

        x_new = x_new.unsqueeze(dim=0)
        return x_new

    def _getKernels(self, kernelSize, sigmas):
        """
            Args:
                kernelSize ([int])
                sigmas ([list])

            Returns:
                [list[numpy]]: []
        """

        def _getKernel(sigma):
            """
                Args:
                    kernelSize ([int])
                    sigma ([float])

                Returns:
                    [numpy]: []
            """
            twoDimKernel = GaussianMatrix(sigma, kernelSize)
            return twoDimKernel

        extra_bias = -int(self.scale / 2)
        row_start = int(self.middlel - self.kernelSize / 2 + extra_bias)
        row_end = int(self.middlel + self.kernelSize / 2 + extra_bias)
        col_start = int(self.middlec - self.kernelSize / 2 + extra_bias)
        col_end = int(self.middlec + self.kernelSize / 2 + extra_bias)
        Kernels = []
        for sigma in sigmas:
            B = np.zeros([self.nr, self.nc])
            kernel = _getKernel(sigma)
            if self.scale == 2:
                B[row_start:row_end, col_start:col_end] = kernel
            elif self.scale == 6:
                B[row_start:row_end, col_start:col_end] = kernel
            B = np.flip(B, 1)
            B = np.flip(B, 0)
            Kernels.append(B)
        return Kernels

    def _calculateSigma(self):
        # matlab sdf
        # % Sequence of bands
        # % [B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12]
        # % subsampling factors (in pixels)
        # d = [6 1 1 1 2 2 2 1 2 6 2 2]'; %
        # % convolution  operators (Gaussian convolution filters), taken from ref [5]
        # mtf = [ .32 .26 .28 .24 .38 .34 .34 .26 .33 .26 .22 .23];
        # sdf = d.*sqrt(-2*log(mtf)/pi^2)';
        mtf = np.array(self.mtf)
        sdf = self.scale * np.sqrt(-2 * np.log(mtf) / np.power(np.pi, 2))
        return sdf.tolist()


class S2_SpatialDownSampler(nn.Module):
    def __init__(self, downMethod='angle', choosePart=(0, 0), scale=2):
        super(S2_SpatialDownSampler, self).__init__()
        self.choosePart = choosePart
        self.scale = scale
        self.SpatialDownSampler = self._MeanDownSampler if downMethod == 'mean' else self._AngleDownSampler

    def forward(self, x_):
        x = x_.clone()
        x = self.SpatialDownSampler(x)
        return x

    def _AngleDownSampler(self, x):
        def choose(shape):
            '''
            (0,0) (0,1)
            (1,0) (1,1)
            '''

            L_point = np.arange(start=0, stop=shape[-1], step=self.scale)  # 列数/行的长度
            W_point = np.arange(start=0, stop=shape[-2], step=self.scale)  # 行数/宽的长度

            W_point = W_point + self.choosePart[0]
            L_point = L_point + self.choosePart[1]

            return W_point, L_point

        x = x.squeeze(0)
        W_point, L_point = choose(x.shape)
        x = x[:, :, L_point]
        x = x[:, W_point, :]
        x = x.unsqueeze(0)
        return x

    def _MeanDownSampler(self, x):
        kernel = torch.ones((x.shape[1], 1, self.scale, self.scale), device="cuda:0") / self.scale * self.scale
        kernel = kernel.type(dtype)
        weight = nn.Parameter(data=kernel, requires_grad=False)
        return F.conv2d(x, weight, padding=0, stride=self.scale, groups=x.shape[1])


class S2_GaussianNoise(nn.Module):
    def __init__(self, mean=0, SNR=40):
        super(S2_GaussianNoise, self).__init__()
        self.mean = mean
        self.SNR = SNR

    def forward(self, x_):
        x = x_.clone()
        noise = self._GaussianNoiseFunciton(x)
        x = x + noise
        return x

    def _GaussianNoiseFunciton(self, x_):
        gaussianNoise = torch.zeros_like(x_)
        x = x_.clone()
        x = x.squeeze(dim=0)
        channel = x.shape[0]
        for i in range(channel):
            tempx = x[i].clone()
            nr, nc = tempx.shape
            noise = torch.randn(nr, nc, device="cuda:0") * \
                    self.variance(tempx, nr, nc)[None,]
            gaussianNoise[:, i, :, :] = noise
        return gaussianNoise

    def variance(self, x, nr, nc, SNR=40):
        '''
        x.shape = [nr, nc]
        '''
        sigma = torch.sqrt(torch.pow(torch.norm(x), 2) /
                           (nr * nc * np.power(10, int(SNR / 10))))
        return sigma


class S2_Downsampler(nn.Module):
    def __init__(self, scale, ifNoise=True, downMethod='angle', SNR=40, ifBlur_10=False, ifNoise_10=True, kernelSize=10,
                 nr='', nc='', RealData=''):
        super(S2_Downsampler, self).__init__()
        self.scale = scale
        self.gausBlur = S2_GaussianBlur(nr=nr, nc=nc, kernelSize=kernelSize, scale=scale,
                                        RealData=RealData).cuda()
        self.spaDown = S2_SpatialDownSampler(scale=scale,
                                             downMethod='mean').cuda() if downMethod == 'mean' else S2_SpatialDownSampler(
            scale=scale, downMethod='angle').cuda()
        self.gausNoise = S2_GaussianNoise(SNR=SNR).cuda()
        self.ifNoise = ifNoise
        self.ifBlur_10 = ifBlur_10
        self.ifNoise_10 = ifNoise_10

    def forward(self, X):
        out = X.clone()
        if self.scale == 1 and self.ifBlur_10 == False:
            pass
        else:
            out = self.gausBlur(out)

        if self.scale == 1 and self.ifNoise_10 == False:
            pass
        elif self.ifNoise:
            out = self.gausNoise(out)

        if self.scale != 1:
            out = self.spaDown(out)

        return out
