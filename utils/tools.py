# %%

from sys import path
import os
import numpy as np
from numpy.linalg import norm

import torch
import torch.nn as nn

import h5py

import scipy
import scipy.io as scio
from skimage import transform

path.append('../')
try:
    from ..models.S2Downsampler import S2_Downsampler
    from ..models.S2Downsampler import S2_GaussianNoise
except:
    from models.S2Downsampler import S2_Downsampler
    from models.S2Downsampler import S2_GaussianNoise


class GTV2D_Loss(nn.Module):
    def __init__(self):
        super(GTV2D_Loss, self).__init__()

    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        gradient_a_z = torch.abs(a[:, :-1, :, :] - a[:, 1:, :, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y) + torch.mean(gradient_a_z)


def my_regular(input_pic_, regular_strategy, needRe=False):
    """Normalize the image or image list according to the [regular_strategy]

    Args:
        input_pic ([numpy/list]): [Pictures or list of pictures that need to be normalized]
        regular_strategy ([str]): [all|per|no]

    Returns:
        [numpy/list]: [A normalized picture or list.]
    """

    def max_min_regular(input_pic):  # 0/1
        """According to the global maximum and minimum, all the spectra are normalized.

        Args:
            input_pic ([numpy/list]): [Pictures or list of pictures that need to be normalized]

        Returns:
            [numpy/list]: [A normalized picture or list.]
        """
        if type(input_pic) == list:
            input_max = np.array([item.max() for item in input_pic]).max()
            input_min = np.array([item.min() for item in input_pic]).min()
            return [(item - input_min) / (input_max - input_min) for item in input_pic]
        else:
            return (input_pic - input_pic.min()) / (input_pic.max() - input_pic.min())

    def per_regular(input_pic, needRe=needRe):
        """Normalization per spectrum

        Args:
            input_pic ([numpy/list]): [Pictures or list of pictures that need to be normalized]

        Returns:
            [numpy/list]: [A normalized picture or list.]
        """

        def _per_regular(pic, needRe=needRe):
            if needRe == True:
                Re_range = []
                Re_min = []
                for i in range(pic.shape[0]):
                    range_ = pic[i, :, :].max() - pic[i, :, :].min()
                    min_ = pic[i, :, :].min()

                    pic[i, :, :] = (pic[i, :, :] - min_) / range_

                    Re_range.append(range_)
                    Re_min.append(min_)

                return pic, Re_range, Re_min

            elif needRe == False:
                for i in range(pic.shape[0]):
                    range_ = pic[i, :, :].max() - pic[i, :, :].min()
                    min_ = pic[i, :, :].min()
                    pic[i, :, :] = (pic[i, :, :] - min_) / range_

                return pic

        if type(input_pic) == list:
            for i in range(len(input_pic)):
                input_pic[i] = _per_regular(input_pic[i])

            return input_pic

        else:
            return _per_regular(input_pic)

    input_pic = input_pic_.copy()

    if regular_strategy == 'all':
        return max_min_regular(input_pic)
    elif regular_strategy == 'per':
        return per_regular(input_pic)
    elif regular_strategy == 'no':
        return input_pic
    elif regular_strategy == '255':
        return (per_regular(input_pic) * 255).astype(np.int16)
    else:
        import os
        print('regular_strategy = \'all\' | \'per\'')
        os.exit()


def my_reRegular(input_pic_, Re_range_, Re_min_):
    pic = input_pic_.copy()
    Re_range = Re_range_.copy()
    Re_min = Re_min_.copy()
    for i in range(pic.shape[0]):
        range_ = Re_range[i]
        min_ = Re_min[i]
        pic[i, :, :] = pic[i, :, :] * range_ + min_

    return pic


def fill_noise(x, noise_type, db):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x = x.uniform_()
        x *= db
    elif noise_type == 'n':
        x = x.normal_()
        x *= db
    elif noise_type == 's':
        gausNoise = S2_GaussianNoise(SNR=db).cuda()
        x = torch.from_numpy(x.astype(np.float32))[None,].cuda()
        x = gausNoise(x)
        x = x.detach().cpu().squeeze(0)
    else:
        assert False
    return x


def get_noise(x, input_depth, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    shape = [input_depth, spatial_size[0], spatial_size[1]]
    net_input = x if noise_type == 's' else torch.zeros(shape)
    net_input = fill_noise(net_input, noise_type, db=var)
    return net_input.numpy()


def get_input(input_type, *args):
    """According to the given [input_type], get the input required by the network

    Args:
        input_type ([str]): [bicubic，random，sharp，pre ]

    Returns:
        [numpy]: [Input required by the network]
    """
    bicubic = args[0]
    if input_type == 'bicubic':
        return bicubic
    elif input_type == 'random':
        size = args[2]
        d_20_random = get_noise(
            size, (bicubic.shape[1], bicubic.shape[2]), noise_type='n', var=1. / 10)
        return d_20_random
    elif input_type == 'sharp':
        import cv2
        # # get high-frequency

        def get_edge(data):
            rs = np.zeros_like(data)
            N = data.shape[0]
            for i in range(N):
                rs[i, :, :] = data[i, :, :] - \
                              cv2.boxFilter(data[i, :, :], -1, (20, 20))

            return rs

        return get_edge(bicubic)
    elif input_type == 'pre':
        path = args[1]
        if 'ini' in path:
            data = scio.loadmat(path)['X_init']
        else:
            data = scio.loadmat(path)['pred']
        return data


def process_new_input(input_HR, If60, img_d10_np, NormMethod):
    if input_HR.shape[0] > input_HR.shape[2]:
        input_HR = input_HR.transpose(2, 0, 1)
    if input_HR.shape[0] == 12:
        TwoScaleList = [4, 5, 6, 8, 10, 11]
        SixScaleList = [0, 9]
        OneScaleList = [1, 2, 3, 7]
        if If60 == True:
            bands = np.array(TwoScaleList + SixScaleList + OneScaleList)
        else:
            bands = np.array(TwoScaleList + OneScaleList)
        input_HR = input_HR[bands, :, :]
    if input_HR.shape[1] != img_d10_np.shape[1] or input_HR.shape[2] != img_d10_np.shape[2]:
        if input_HR.shape[1] > img_d10_np.shape[1]:
            boundary_2side = input_HR.shape[1] - img_d10_np.shape[1]
            if boundary_2side % 2 == 0:
                boundary = int(boundary_2side / 2)
                input_HR = input_HR[:, boundary:-boundary, :]
            elif boundary_2side % 2 == 1:
                boundary = int(boundary_2side / 2)
                input_HR = input_HR[:, boundary:-(boundary + 1), :]

        if input_HR.shape[2] > img_d10_np.shape[2]:
            boundary_2side = input_HR.shape[2] - img_d10_np.shape[2]
            if boundary_2side % 2 == 0:
                boundary = int(boundary_2side / 2)
                input_HR = input_HR[:, :, boundary:-boundary]
            elif boundary_2side % 2 == 1:
                boundary = int(boundary_2side / 2)
                input_HR = input_HR[:, :, boundary:-(boundary + 1)]

        if input_HR.shape[1] < img_d10_np.shape[1] or input_HR.shape[2] < img_d10_np.shape[
            2]:
            boundary_dim1 = img_d10_np.shape[1] - input_HR.shape[1]
            boundary_dim2 = img_d10_np.shape[2] - input_HR.shape[2]
            if boundary_dim1 % 2 == 0:
                boundary_dim1_left = boundary_dim1_right = int(boundary_dim1 / 2)
            else:
                boundary_dim1_left = int(boundary_dim1 / 2)
                boundary_dim1_right = boundary_dim1_left + 1
            if boundary_dim2 % 2 == 0:
                boundary_dim2_left = boundary_dim2_right = int(boundary_dim2 / 2)
            else:
                boundary_dim2_left = int(boundary_dim2 / 2)
                boundary_dim2_right = boundary_dim2_left + 1

            input_HR = np.pad(input_HR, (
                (0, 0), (boundary_dim1_left, boundary_dim1_right), (boundary_dim2_left, boundary_dim2_right)),
                              'linear_ramp')
    if input_HR.max() > 1000:
        input_HR, _ = normaliseData(input_HR, method=NormMethod)
    return input_HR


def normaliseData(input_data, av=None, method='maxkeep'):
    Yim = input_data.copy()
    channel = Yim.shape[0]

    def method_UnitPower(Yim, channel, av):
        if av is None:
            av = []
            for i in range(channel):
                av.append(np.mean(np.power(Yim[i, :, :], 2)))
                Yim[i, :, :] = np.sqrt(np.power(Yim[i, :, :], 2) / av[i])
        else:
            for i in range(channel):
                Yim[i, :, :] = np.sqrt(np.power(Yim[i, :, :], 2) / av[i])

        return Yim, av

    def method_01(Yim, channel, av):
        if av is None:
            av = []
            for i in range(channel):
                range_ = Yim[i, :, :].max() - Yim[i, :, :].min()
                min_ = Yim[i, :, :].min()
                Yim[i, :, :] = (Yim[i, :, :] - min_) / range_
                av.append([range_, min_])
        else:
            for i in range(channel):
                Yim[i, :, :] = (Yim[i, :, :] - av[i][1]) / av[i][0]

        return Yim, av

    def method_max(Yim, channel, av):
        if av is None:
            av = []
            for i in range(channel):
                max_ = Yim[i, :, :].max()
                Yim[i, :, :] = Yim[i, :, :] / max_
                av.append([max_])
        else:
            for i in range(channel):
                Yim[i, :, :] = Yim[i, :, :] / av[i][0]

        return Yim, av

    if method == '01':
        Yim, av = method_01(Yim, channel, av)
    elif method == 'UnitPower':
        Yim, av = method_UnitPower(Yim, channel, av)
    elif method == 'max' or method == 'maxkeep':
        Yim, av = method_max(Yim, channel, av)

    else:
        print("Please enter the correct [method] parameters.")

    return Yim, av


def unnormaliseData(input_data, av=None, method='maxkeep'):
    Yim = input_data.copy()

    def method_UnitPower(Yim, channel, av):
        for i in range(channel):
            Yim[i, :, :] = np.sqrt(np.power(Yim[i, :, :], 2) * av[i])
        return Yim

    def method_01(Yim, channel, av):
        for i in range(channel):
            Yim[i, :, :] = Yim[i, :, :] * av[i][0] + av[i][1]
        return Yim

    def method_max(Yim, channel, av):
        for i in range(channel):
            Yim[i, :, :] = Yim[i, :, :] * av[i][0]
        return Yim

    if av is None:
        return Yim
    else:
        channel = Yim.shape[0]
        if method == '01':
            Yim = method_01(Yim, channel, av)
        elif method == 'UnitPower':
            Yim = method_UnitPower(Yim, channel, av)
        elif method == 'max':
            Yim = method_max(Yim, channel, av)
        elif method == 'maxkeep':
            Yim = Yim
        else:
            print("Please enter the correct [method] parameters.")
        return Yim


def groupNorm(img_d10_np_, img_LR_np_, d_bicubic_, ifNorm=True, method='maxkeep', If60=False):
    """Norm 10m and 20m bicubic
    """
    av, av_bic = None, None
    img_d10_np = img_d10_np_.copy()
    img_LR_np = img_LR_np_.copy()
    d_bicubic = d_bicubic_.copy()
    if ifNorm == True:
        img_d10_np, _ = normaliseData(img_d10_np, method=method)
        d_bicubic, av_bic = normaliseData(d_bicubic, method=method)

        if If60 == True:
            temp_LR_list = []
            temp_av_list = []
            for item in img_LR_np:
                temp_LR, temp_av = normaliseData(item, method=method)
                temp_LR_list.append(temp_LR)
                temp_av_list.append(temp_av)
            img_LR_np = temp_LR_list;
            av = temp_av_list
        else:
            img_LR_np, av = normaliseData(img_LR_np, method=method)

    return img_d10_np, img_LR_np, d_bicubic, av, av_bic


def process_data(mat_name, data_name='', Blur_10=False, Noise_10=True, RealData=False, kernelSize=10, method='maxkeep'):
    """
    Process loading the original downsampled data and resampling
    """

    def read_data(mat_name, RealData):
        fname = ''
        if mat_name == 'avirisLowCity':
            fname = '../data/data_SSSS/avirisLowCity.mat'
        elif mat_name == 'avirisLowCoast':
            fname = '../data/data_SSSS/avirisLowCoast.mat'
        elif mat_name == 'avirisLowCrops':
            fname = '../data/data_SSSS/avirisLowCrops.mat'
        elif mat_name == 'avirisLowMontain':
            fname = '../data/data_SSSS/avirisLowMontain.mat'
        elif mat_name == 'hydiceSample':
            fname = '../data/data_za/hydiceSample.mat'
        elif mat_name == 'Malmo':
            fname = '../data/realdata/Malmo.mat'
        elif mat_name == 'Verona':
            fname = '../data/realdata/Verona.mat'
        elif mat_name == 'APEX':
            fname = '../data/data_SSSS/APEX_10_11.mat'
        elif mat_name == 'dc':
            fname = '../data/data_SSSS/dc_10_11.mat'
        elif mat_name == 'Terrain':
            fname = '../data/data_SSSS/Terrain_10_11_2.mat'

        else:
            import sys
            print("Data does not exist")
            sys.exit()

        data = scio.loadmat(fname)
        if RealData == False:
            Xm_im = data['Xm_im']
            if mat_name == 'hydiceSample' or mat_name == 'dc' or mat_name == 'Terrain' or mat_name == 'APEX':
                Xm_im = transform.resize(Xm_im, (96, 96, 12),
                                         order=3)
            Yim = data['Yim']
        else:
            Yim = data['Yim']
            Xm_im = ''
        return Xm_im, Yim

    def _process_realdata(Yim, method):
        OneScaleList = [1, 2, 3, 7]
        TwoScaleList = [4, 5, 6, 8, 11, 12] if Yim.shape[1] == 13 else [4, 5, 6, 8, 10, 11]
        SixScaleList = [0, 9]

        d10 = None
        for i in range(len(OneScaleList)):
            place = OneScaleList[i]
            if i == 0:
                temp = np.expand_dims(Yim[0][place], axis=0)
                d10 = temp
            else:
                temp = np.expand_dims(Yim[0][place], axis=0)
                d10 = np.concatenate([d10, temp], 0)
        d20 = None
        for i in range(len(TwoScaleList)):
            place = TwoScaleList[i]
            if i == 0:
                temp = np.expand_dims(Yim[0][place], axis=0)
                d20 = temp
            else:
                temp = np.expand_dims(Yim[0][place], axis=0)
                d20 = np.concatenate([d20, temp], 0)
        d60 = None
        for i in range(len(SixScaleList)):
            place = SixScaleList[i]
            if i == 0:
                temp = np.expand_dims(Yim[0][place], axis=0)
                d60 = temp
            else:
                temp = np.expand_dims(Yim[0][place], axis=0)
                d60 = np.concatenate([d60, temp], 0)

        if method == 'maxkeep':
            d10, _ = normaliseData(d10, method=method)
            d20, _ = normaliseData(d20, method=method)
            d60, _ = normaliseData(d60, method=method)

        d20_Bicubic = transform.resize(d20, (d20.shape[0], d10.shape[1], d10.shape[2]), order=3)
        d60_Bicubic = transform.resize(d60, (d60.shape[0], d10.shape[1], d10.shape[2]), order=3)
        for i in range(d20.shape[0]):
            d20_Bicubic[i, :, :] = np.clip(d20_Bicubic[i, :, :], d20[i, :, :].min(), d20[i, :, :].max())
        for i in range(d60.shape[0]):
            d60_Bicubic[i, :, :] = np.clip(d60_Bicubic[i, :, :], d60[i, :, :].min(), d60[i, :, :].max())

        pic_new_size_60_w, pic_new_size_60_l = enforce_div8_new_size(d60[0])
        pic_new_size_20_w, pic_new_size_20_l = pic_new_size_60_w * 3, pic_new_size_60_l * 3
        pic_new_size_10_w, pic_new_size_10_l = pic_new_size_60_w * 6, pic_new_size_60_l * 6

        d10_crop = np.zeros((d10.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d10.shape[0]):
            d10_crop[i] = enforse_div8(d10[i], (pic_new_size_10_w, pic_new_size_10_l))

        d20_crop_bicubic = np.zeros((d20_Bicubic.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d20_Bicubic.shape[0]):
            d20_crop_bicubic[i] = enforse_div8(d20_Bicubic[i], (pic_new_size_10_w, pic_new_size_10_l))

        d60_crop_bicubic = np.zeros((d60_Bicubic.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d60_Bicubic.shape[0]):
            d60_crop_bicubic[i] = enforse_div8(d60_Bicubic[i], (pic_new_size_10_w, pic_new_size_10_l))

        d20_crop = np.zeros((d20.shape[0], pic_new_size_20_w, pic_new_size_20_l))
        for i in range(d20.shape[0]):
            d20_crop[i] = enforse_div8(d20[i], (pic_new_size_20_w, pic_new_size_20_l))

        d60_crop = np.zeros((d60.shape[0], pic_new_size_60_w, pic_new_size_60_l))
        for i in range(d60.shape[0]):
            d60_crop[i] = enforse_div8(d60[i], (pic_new_size_60_w, pic_new_size_60_l))

        d10gt_crop = None
        d20gt_crop = None
        d60gt_crop = None
        return d10_crop, d10gt_crop, d20_crop, d20gt_crop, d20_crop_bicubic, d60_crop, d60gt_crop, d60_crop_bicubic

    def enforce_div8_new_size(img_orig_np, size=int(np.power(2, 4))):
        new_size = (img_orig_np.shape[0] - img_orig_np.shape[0] % size,
                    img_orig_np.shape[1] - img_orig_np.shape[1] % size)

        return new_size

    def enforse_div8(img_orig_np, new_size):
        bbox = [
            int((img_orig_np.shape[0] - new_size[0]) / 2),
            int((img_orig_np.shape[1] - new_size[1]) / 2),
            int((img_orig_np.shape[0] + new_size[0]) / 2),
            int((img_orig_np.shape[1] + new_size[1]) / 2),
        ]
        img_HR_np = img_orig_np[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        return img_HR_np

    def _process_readedData(Xm_im, data_name, mat_name, Blur_10, Noise_10, method, RealData):

        d10_gt = np.zeros((4, Xm_im.shape[0], Xm_im.shape[1]))  # Yim  [1,2,3,7]
        d20_gt = np.zeros((6, Xm_im.shape[0], Xm_im.shape[1]))
        d60_gt = np.zeros((2, Xm_im.shape[0], Xm_im.shape[1]))

        Xm_im_list = [1, 2, 3, 7]
        for i in range(len(Xm_im_list)):
            d10_gt[i, :, :] = Xm_im[:, :, Xm_im_list[i]]

        Xm_im_list = [4, 5, 6, 8, 10, 11]
        for i in range(len(Xm_im_list)):
            d20_gt[i, :, :] = Xm_im[:, :, Xm_im_list[i]]

        Xm_im_list = [0, 9]
        for i in range(len(Xm_im_list)):
            d60_gt[i, :, :] = Xm_im[:, :, Xm_im_list[i]]

        d10_gt_norm, av_10 = normaliseData(d10_gt, method=method)
        if Blur_10 == True or Noise_10 == True:
            down_10 = S2_Downsampler(scale=1, nr=Xm_im.shape[0], nc=Xm_im.shape[1], ifNoise=True, downMethod='angle',
                                     SNR=40, ifBlur_10=Blur_10, ifNoise_10=Noise_10, kernelSize=kernelSize,
                                     RealData=RealData).cuda()
            d10 = down_10(torch.from_numpy(d10_gt_norm.astype(np.float32))[
                              None,].cuda()).squeeze().detach().cpu().numpy()
            if method != 'maxkeep':
                d10 = unnormaliseData(d10, av_10, method=method)
        else:
            d10 = d10_gt.copy()

        d20_gt_norm, av_20 = normaliseData(d20_gt, method=method)
        d60_gt_norm, av_60 = normaliseData(d60_gt, method=method)
        down_20 = S2_Downsampler(scale=2, nr=Xm_im.shape[0], nc=Xm_im.shape[1], ifNoise=True, downMethod='angle',
                                 SNR=40, kernelSize=kernelSize, RealData=RealData).cuda()
        down_60 = S2_Downsampler(scale=6, nr=Xm_im.shape[0], nc=Xm_im.shape[1], ifNoise=True, downMethod='angle',
                                 SNR=40, kernelSize=kernelSize, RealData=RealData).cuda()
        d20 = down_20(torch.from_numpy(d20_gt_norm.astype(np.float32))[
                          None,].cuda()).squeeze().detach().cpu().numpy()
        d60 = down_60(torch.from_numpy(d60_gt_norm.astype(np.float32))[
                          None,].cuda()).squeeze().detach().cpu().numpy()
        if method != 'maxkeep':
            d20 = unnormaliseData(d20, av_20, method=method)
            d60 = unnormaliseData(d60, av_60, method=method)

        new_Yim = np.array([
            d60[0],
            d10[0],
            d10[1],
            d10[2],
            d20[0],
            d20[1],
            d20[2],
            d10[3],
            d20[3],
            d60[1],
            d20[4],
            d20[5]
        ])

        d20_Bicubic = transform.resize(d20, (d20.shape[0], d10.shape[1], d10.shape[2]), order=3)
        d60_Bicubic = transform.resize(d60, (d60.shape[0], d10.shape[1], d10.shape[2]), order=3)

        for i in range(d20.shape[0]):
            d20_Bicubic[i, :, :] = np.clip(d20_Bicubic[i, :, :], d20[i, :, :].min(), d20[i, :, :].max())
        for i in range(d60.shape[0]):
            d60_Bicubic[i, :, :] = np.clip(d60_Bicubic[i, :, :], d60[i, :, :].min(), d60[i, :, :].max())

        Bicubic = np.array([
            d60_Bicubic[0],
            d10[0],
            d10[1],
            d10[2],
            d20_Bicubic[0],
            d20_Bicubic[1],
            d20_Bicubic[2],
            d10[3],
            d20_Bicubic[3],
            d60_Bicubic[1],
            d20_Bicubic[4],
            d20_Bicubic[5]
        ])

        Xm_im = Xm_im.transpose(2, 0, 1)
        if method == 'maxkeep':
            Xm_im, _ = normaliseData(Xm_im, method=method)
            d10_gt = d10_gt_norm
            d20_gt = d20_gt_norm
            d60_gt = d60_gt_norm

        scipy.io.savemat("../result/{}/{}_new.mat".format(data_name,
                                                          mat_name),
                         {'Xm_im': Xm_im, 'Yim': new_Yim, 'Bicubic': Bicubic})

        pic_new_size_60_w, pic_new_size_60_l = enforce_div8_new_size(d60[0])
        pic_new_size_20_w, pic_new_size_20_l = pic_new_size_60_w * 3, pic_new_size_60_l * 3
        pic_new_size_10_w, pic_new_size_10_l = pic_new_size_60_w * 6, pic_new_size_60_l * 6

        d10_crop = np.zeros((d10.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d10.shape[0]):
            d10_crop[i] = enforse_div8(d10[i], (pic_new_size_10_w, pic_new_size_10_l))

        d10gt_crop = np.zeros((d10_gt.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d10_gt.shape[0]):
            d10gt_crop[i] = enforse_div8(d10_gt[i], (pic_new_size_10_w, pic_new_size_10_l))

        d20_crop = np.zeros((d20_gt.shape[0], pic_new_size_20_w, pic_new_size_20_l))
        for i in range(d20_gt.shape[0]):
            d20_crop[i] = enforse_div8(d20[i], (pic_new_size_20_w, pic_new_size_20_l))

        d20gt_crop = np.zeros((d20_gt.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d20_gt.shape[0]):
            d20gt_crop[i] = enforse_div8(d20_gt[i], (pic_new_size_10_w, pic_new_size_10_l))

        d20_crop_bicubic = np.zeros((d20_gt.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d20_gt.shape[0]):
            d20_crop_bicubic[i] = enforse_div8(d20_Bicubic[i], (pic_new_size_10_w, pic_new_size_10_l))

        d60gt_crop = np.zeros((d60_gt.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d60_gt.shape[0]):
            d60gt_crop[i] = enforse_div8(d60_gt[i], (pic_new_size_10_w, pic_new_size_10_l))

        d60_crop_bicubic = np.zeros((d60_gt.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d60_gt.shape[0]):
            d60_crop_bicubic[i] = enforse_div8(d60_Bicubic[i], (pic_new_size_10_w, pic_new_size_10_l))

        d60_crop = np.zeros((d60_gt.shape[0], pic_new_size_60_w, pic_new_size_60_l))
        for i in range(d60_gt.shape[0]):
            d60_crop[i] = enforse_div8(d60[i], (pic_new_size_60_w, pic_new_size_60_l))

        return d10_crop, d10gt_crop, d20_crop, d20gt_crop, d20_crop_bicubic, d60_crop, d60gt_crop, d60_crop_bicubic

    Xm_im, Yim = read_data(mat_name, RealData)
    if RealData == False:
        d10_crop, d10gt_crop, d20_crop, d20gt_crop, d20_crop_bicubic, d60_crop, d60gt_crop, d60_crop_bicubic = _process_readedData(
            Xm_im, data_name, mat_name, Blur_10=Blur_10, Noise_10=Noise_10, method=method, RealData=RealData)
    else:
        d10_crop, d10gt_crop, d20_crop, d20gt_crop, d20_crop_bicubic, d60_crop, d60gt_crop, d60_crop_bicubic = _process_realdata(
            Yim)

    return d10_crop, d10gt_crop, d20_crop, d20gt_crop, d20_crop_bicubic, d60_crop, d60gt_crop, d60_crop_bicubic


def process_data_new(mat_name, input_type, result_mat_path, RealData):
    def read_data(mat_name, input_type, result_mat_path, RealData):
        fname = result_mat_path
        # fname = '../data/in_initial/Malmo_initial2.mat'
        data = scio.loadmat(fname)
        Yim = data['Yim']

        if RealData == True:  # if there is no GT in the real data, an all-zero tensor with the same GT shape is returned.
            Xm_im_shape = Yim[0][1].shape
            Xm_im = np.zeros(Xm_im_shape)

        else:
            Xm_im = data['Xm_im']
        return Xm_im, Yim

    def _process_readedData(Xm_im, Yim, RealData):
        def enforce_div8_new_size(img_orig_np, size=int(np.power(2, 4))):
            new_size = (img_orig_np.shape[0] - img_orig_np.shape[0] % size,
                        img_orig_np.shape[1] - img_orig_np.shape[1] % size)

            return new_size

        def enforse_div8(img_orig_np, new_size):
            bbox = [
                int((img_orig_np.shape[0] - new_size[0]) / 2),
                int((img_orig_np.shape[1] - new_size[1]) / 2),
                int((img_orig_np.shape[0] + new_size[0]) / 2),
                int((img_orig_np.shape[1] + new_size[1]) / 2),
            ]
            img_HR_np = img_orig_np[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            return img_HR_np

        d10_gt = np.zeros((4, Xm_im.shape[0], Xm_im.shape[1]))  # Yim  [1,2,3,7]
        d20_gt = np.zeros((6, Xm_im.shape[0], Xm_im.shape[1]))
        d60_gt = np.zeros((2, Xm_im.shape[0], Xm_im.shape[1]))
        if RealData:
            pass
        else:
            Xm_im_list = [1, 2, 3, 7]
            for i in range(len(Xm_im_list)):
                d10_gt[i, :, :] = Xm_im[:, :, Xm_im_list[i]]

            Xm_im_list = [4, 5, 6, 8, 10, 11]
            for i in range(len(Xm_im_list)):
                d20_gt[i, :, :] = Xm_im[:, :, Xm_im_list[i]]

            Xm_im_list = [0, 9]
            for i in range(len(Xm_im_list)):
                d60_gt[i, :, :] = Xm_im[:, :, Xm_im_list[i]]

        d10 = np.zeros((4, Xm_im.shape[0], Xm_im.shape[1]))
        d20 = np.zeros((6, int(Xm_im.shape[0] / 2), int(Xm_im.shape[1] / 2)))
        d60 = np.zeros((2, int(Xm_im.shape[0] / 6), int(Xm_im.shape[1] / 6)))

        Yim_list = [1, 2, 3, 7]
        for i in range(4):
            d10[i, :, :] = Yim[0][Yim_list[i]]

        Yim_list = [4, 5, 6, 8, 10, 11]
        for i in range(6):
            d20[i, :, :] = Yim[0][Yim_list[i]]

        Yim_list = [0, 9]
        for i in range(2):
            d60[i, :, :] = Yim[0][Yim_list[i]]

        pic_new_size_60_w, pic_new_size_60_l = enforce_div8_new_size(d60[0])
        pic_new_size_20_w, pic_new_size_20_l = pic_new_size_60_w * 3, pic_new_size_60_l * 3
        pic_new_size_10_w, pic_new_size_10_l = pic_new_size_60_w * 6, pic_new_size_60_l * 6

        d10_crop = np.zeros((d10.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d10.shape[0]):
            d10_crop[i] = enforse_div8(d10[i], (pic_new_size_10_w, pic_new_size_10_l))

        d10gt_crop = np.zeros((d10_gt.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d10_gt.shape[0]):
            d10gt_crop[i] = enforse_div8(d10_gt[i], (pic_new_size_10_w, pic_new_size_10_l))

        d20_crop = np.zeros((d20_gt.shape[0], pic_new_size_20_w, pic_new_size_20_l))
        for i in range(d20_gt.shape[0]):
            d20_crop[i] = enforse_div8(d20[i], (pic_new_size_20_w, pic_new_size_20_l))

        d20gt_crop = np.zeros((d20_gt.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d20_gt.shape[0]):
            d20gt_crop[i] = enforse_div8(d20_gt[i], (pic_new_size_10_w, pic_new_size_10_l))

        d60gt_crop = np.zeros((d60_gt.shape[0], pic_new_size_10_w, pic_new_size_10_l))
        for i in range(d60_gt.shape[0]):
            d60gt_crop[i] = enforse_div8(d60_gt[i], (pic_new_size_10_w, pic_new_size_10_l))

        d60_crop = np.zeros((d60_gt.shape[0], pic_new_size_60_w, pic_new_size_60_l))
        for i in range(d60_gt.shape[0]):
            d60_crop[i] = enforse_div8(d60[i], (pic_new_size_60_w, pic_new_size_60_l))

        d20_crop_bicubic = transform.resize(d20_crop, (d20_crop.shape[0], d10_crop.shape[1], d10_crop.shape[2]),
                                            order=3)
        d60_crop_bicubic = transform.resize(d60_crop, (d60_crop.shape[0], d10_crop.shape[1], d10_crop.shape[2]),
                                            order=3)

        return d10_crop, d10gt_crop, d20_crop, d20gt_crop, d20_crop_bicubic, d60_crop, d60gt_crop, d60_crop_bicubic

    Xm_im, Yim = read_data(mat_name, input_type, result_mat_path, RealData)
    return _process_readedData(Xm_im, Yim, RealData)
