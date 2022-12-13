# %%
from ast import arg
import os
import warnings
import time
import scipy.io as scio
import sys
import matplotlib.pyplot as plt
import scipy.io
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse
from tqdm import tqdm
import pandas as pd
from tensorboardX import SummaryWriter


warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
sys.path.append('../')
from utils.tools import *
from utils.RefeIqa import *
from utils.NoRefeIQA import *
from models.S2Downsampler import S2_Downsampler
from models import pytorch_ssim
from models.Perceptual_Loss import Perceptual_Loss


def compareRange(x):
    return x.max() - x.min()


def evalution_per_tail(data_, img_HR_np_):
    """
    Calculate index per spectrum
    """
    data = data_.copy()
    img_HR_np = img_HR_np_.copy()
    psnr_list = []
    ssim_list = []
    nrmse_list = []
    sre_list = []

    for i in range(data.shape[0]):
        psnr_list.append(compare_psnr(img_HR_np[i, :, :], data[i, :, :], data_range=compareRange(img_HR_np[i])))
        ssim_list.append(compare_ssim(img_HR_np[i, :, :], data[i, :, :], data_range=compareRange(img_HR_np[i])))
        nrmse_list.append(compare_nrmse(img_HR_np[i, :, :], data[i, :, :]))
        sre_list.append(SRE(img_HR_np[i, :, :], data[i, :, :]))
    return np.mean(psnr_list), np.mean(ssim_list), np.mean(nrmse_list), np.mean(sre_list)


def print_and_save_baseline(img_HR_np, d_bicubic, av_bic, Normmethod='maxkeep'):
    """
    print && save bicubic index as baseline
    """

    base_psnr, base_ssim, base_nrmse, base_sre = evalution_per_tail(
        unnormaliseData(d_bicubic, av_bic, method=Normmethod), img_HR_np)

    return base_psnr, base_ssim, base_nrmse, base_sre


def closure():
    global i, net_input, out_avg, av, NormMethod, LossWeight
    if sigma_random > 0:
        net_input_this = net_input.clone() + torch.from_numpy(
            get_noise(x=net_input.clone().detach().cpu().squeeze().numpy(),
                      input_depth=input_C,
                      spatial_size=(img_d10_dim1, img_d10_dim2),
                      noise_type=args.noise_type,
                      var=sigma_random)).type(dtype)[None, :].cuda()
    else:
        net_input_this = net_input.clone()
    if NetChoice == 'sep3d' or NetChoice == '3d' or NetChoice == 'resnet' or NetChoice == 'resnet3d_common':
        out = net1(net_input_this)
        out = torch.unsqueeze(out, 0)
        out = net_main(out)
        out = torch.squeeze(out, 0)
        out = net2(out)

    elif NetChoice == '2d' or NetChoice == 'resnet2d':
        out = net_main(net_input_this)

    # out = torch.clamp(out, min=0, max=1) 
    out_HR = out[:, 0:chunnel_bic, :, :]
    out_HR_10 = out[:, chunnel_bic::, :, :]
    if If60:
        out_HR_20 = out_HR[:, 0:6, :, :]
        out_HR_60 = out_HR[:, 6::, :, :]
        out_LR_20 = downsampler(out_HR_20)
        out_LR_60 = downsampler_60(out_HR_60)

    else:
        out_LR = downsampler(out_HR)

    if Blur_10 or Noise_10:
        out_HR_10 = downsampler_10(out_HR_10)

    #############################################################
    # Smoothing
    #############################################################
    if out_avg is None:
        out_avg = out_HR.detach()
    else:
        out_avg = out_avg * exp_weight + out_HR.detach() * (1 - exp_weight)
    #############################################################
    # Calculating loss
    #############################################################
    total_loss = 0
    per_loss = []
    name_loss = []
    if If60:
        if 'l1' in LossList:
            l1_10 = l1Norm(out_HR_10, img_10_var)
            l1_20 = l1Norm(out_LR_20, img_LR_var_20)
            l1_60 = l1Norm(out_LR_60, img_LR_var_60)
            l1 = l1_10 + l1_20 + l1_60
            total_loss += l1 * LossWeight['l1']
            per_loss.append(l1.detach().cpu().numpy().item())
            name_loss.append('l1')
        if 'l2' in LossList:
            l2_10 = l2Norm(out_HR_10, img_10_var)
            l2_20 = l2Norm(out_LR_20, img_LR_var_20)
            l2_60 = l2Norm(out_LR_60, img_LR_var_60)
            l2 = l2_10 + l2_20 + l2_60
            total_loss += l2 * LossWeight['l2']
            per_loss.append(l2.detach().cpu().numpy().item())
            name_loss.append('l2')
        if 'ssim' in LossList:
            ssim_10 = ssim_loss(out_HR_10, img_10_var)
            ssim_20 = ssim_loss(out_LR_20, img_LR_var_20)
            ssim_60 = ssim_loss(out_LR_60, img_LR_var_60)
            ssim = 1 - ((ssim_10 * 4 + ssim_20 * 6 + ssim_60 * 2) / 12)
            total_loss += ssim * LossWeight['ssim']
            per_loss.append(ssim.detach().cpu().numpy().item())
            name_loss.append('ssim')
        if 'vgg' in LossList:
            percep_10 = perceptual_loss(out_HR_10, img_10_var)
            percep_20 = perceptual_loss(out_LR_20, img_LR_var_20)
            percep_60 = perceptual_loss(out_LR_60, img_LR_var_60)
            percep = percep_10 + percep_20 + percep_60
            total_loss += percep * LossWeight['vgg']
            per_loss.append(percep.detach().cpu().numpy().item())
            name_loss.append('percep')
    else:
        if 'l1' in LossList:
            l1_10 = l1Norm(out_HR_10, img_10_var)
            l1_20 = l1Norm(out_LR, img_LR_var)
            l1 = l1_10 + l1_20
            total_loss += l1
            per_loss.append(l1.detach().cpu().numpy().item())
            name_loss.append('l1')
        if 'l2' in LossList:
            l2_10 = l2Norm(out_HR_10, img_10_var)
            l2_20 = l2Norm(out_LR, img_LR_var)
            l2 = l2_10 + l2_20
            total_loss += l2
            per_loss.append(l2.detach().cpu().numpy().item())
            name_loss.append('l2')
        if 'ssim' in LossList:
            ssim_10 = ssim_loss(out_HR_10, img_10_var)
            ssim_20 = ssim_loss(out_LR, img_LR_var)
            ssim = ssim_10 + ssim_20
            total_loss += ssim
            per_loss.append(ssim.detach().cpu().numpy().item())
            name_loss.append('ssim')
        if 'vgg' in LossList:
            percep_10 = perceptual_loss(out_HR_10, img_10_var)
            percep_20 = perceptual_loss(out_LR, img_LR_var)
            percep = percep_10 + percep_20
            total_loss += percep
            per_loss.append(percep.detach().cpu().numpy().item())
            name_loss.append('percep')

    if lambda_tv > 0:
        total_loss = total_loss + TV(out_HR) * lambda_tv
    total_loss.backward()
    #############################################################
    # Log
    #############################################################
    if If60 == True:
        out_LR_20_np = out_LR_20.detach().cpu().squeeze().numpy()
        out_LR_60_np = out_LR_60.detach().cpu().squeeze().numpy()
    else:
        out_LR_np = out_LR.detach().cpu().squeeze().numpy()
    out_HR_np = out_HR.detach().cpu().squeeze().numpy()
    out_avg_np = out_avg.cpu().squeeze().numpy()
    loss_here = total_loss.detach().cpu().numpy()

    if NormMethod != 'maxkeep':
        if If60 == True:
            out_LR_20_np = unnormaliseData(out_LR_20_np, av, method=NormMethod)
            out_LR_60_np = unnormaliseData(out_LR_60_np, av, method=NormMethod)
        else:
            out_LR_np = unnormaliseData(out_LR_np, av, method=NormMethod)
        out_HR_np = unnormaliseData(out_HR_np, av, method=NormMethod)
        out_avg_np = unnormaliseData(out_avg_np, av, method=NormMethod)

    if If60 == True:
        psnr_20_LR, ssim_20_LR, nrmse_20_LR, sre_20_LR = evalution_per_tail(
            out_LR_20_np, img_LR_np_20.astype(np.float32))
        psnr_60_LR, ssim_60_LR, nrmse_60_LR, sre_60_LR = evalution_per_tail(
            out_LR_60_np, img_LR_np_60.astype(np.float32))
        psnr_LR = (psnr_20_LR + psnr_60_LR) / 2
        ssim_LR = (ssim_20_LR + ssim_60_LR) / 2
        nrmse_LR = (nrmse_20_LR + nrmse_60_LR) / 2
        sre_LR = (sre_20_LR + sre_60_LR) / 2
    else:
        psnr_LR, ssim_LR, nrmse_LR, sre_LR = evalution_per_tail(
            out_LR_np, img_LR_np.astype(np.float32))

    if RealData == False:
        psnr_HR, ssim_HR, nrmse_HR, sre_HR = evalution_per_tail(
            out_HR_np, img_HR_np.astype(np.float32))
        psnr_avg, ssim_avg, nrmse_avg, sre_avg = evalution_per_tail(
            out_avg_np, img_HR_np.astype(np.float32))
    #############################################################
    # Send Loss to tensorboardX
    #############################################################

    summarywriter.add_scalar('loss', loss_here, i)
    if not RealData:
        summarywriter.add_scalar('ssim', ssim_avg, i)
        summarywriter.add_scalar('sre', sre_avg, i)
    summarywriter.add_images('out_HR_np', np.concatenate((
        np.expand_dims(my_regular(out_HR_np[band1, :, :], 'per'), 0),
        np.expand_dims(my_regular(out_avg_np[band1, :, :], 'per'), 0),
        np.expand_dims(my_regular(out_HR_np[band2, :, :], 'per'), 0),
        np.expand_dims(my_regular(out_avg_np[band2, :, :], 'per'), 0),
        np.expand_dims(my_regular(out_HR_np[band3, :, :], 'per'), 0),
        np.expand_dims(my_regular(out_avg_np[band3, :, :], 'per'), 0),
    ), axis=0), global_step=i,
                             dataformats='NCHW')
    #############################################################
    # Save History locally
    #############################################################
    if not RealData:
        sre_HR_list.append(sre_HR);
        sre_avg_list.append(sre_avg)
        sre_LR_list.append(sre_LR)
        psnr_HR_list.append(psnr_HR)
        psnr_avg_list.append(psnr_avg)
        psnr_LR_list.append(psnr_LR)
        ssim_HR_list.append(ssim_HR)
        ssim_avg_list.append(ssim_avg)
        ssim_LR_list.append(ssim_LR)
        nrmse_HR_list.append(nrmse_HR)
        nrmse_avg_list.append(nrmse_avg)
        nrmse_LR_list.append(nrmse_LR)
        loss_here_list.append(loss_here)
        per_loss_list.append(per_loss)
    else:
        sre_LR_list.append(sre_LR)
        psnr_LR_list.append(psnr_LR)
        ssim_LR_list.append(ssim_LR)
        nrmse_LR_list.append(nrmse_LR)
        loss_here_list.append(loss_here)
        per_loss_list.append(per_loss)

    if i % log_every == 0:
        if not RealData:
            print(
                f'\nNow is the {i + 1} epoch, loss = {loss_here}, respectively {per_loss}\nsre_HR:{sre_HR}\tsre_avg:{sre_avg}\tsre_LR:{sre_LR}\t\npsnr_HR:{psnr_HR}\tpsnr_avg:{psnr_avg}\tpsnr_LR:{psnr_LR}\t\nssim_HR:{ssim_HR}\tssim_avg:{ssim_avg}\tssim_LR:{ssim_LR}\t\nnrmse_HR:{nrmse_HR}\tnrmse_avg:{nrmse_avg}\tnrmse_LR:{nrmse_LR}\t\n\n')
        else:
            print(
                f'\nNow is the {i + 1} epoch, loss = {loss_here}, respectively {per_loss}\nsre_LR:{sre_LR}\npsnr_LR:{psnr_LR}\nssim_LR:{ssim_LR}\nnrmse_LR:{nrmse_LR}\n\n')
    if save_model is not None:
        if i % save_model == 0 and i != 0:
            if NetChoice == 'sep3d' or NetChoice == '3d' or NetChoice == 'resnet' or NetChoice == 'resnet3d_common':
                state = {
                    "net1": net1.state_dict(),
                    "net_main": net_main.state_dict(),
                    "net2": net2.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': i
                }
            elif NetChoice == '2d' or NetChoice == 'resnet2d':
                state = {
                    "net_main": net_main.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': i
                }
            torch.save(state, f'../result/{data_name}/checkpoint_{i + 1}.pth')
            if not RealData:
                temp_table = [loss_here_list,
                              sre_HR_list, sre_avg_list, sre_LR_list,
                              psnr_HR_list, psnr_avg_list, psnr_LR_list,
                              ssim_HR_list, ssim_avg_list, ssim_LR_list,
                              nrmse_HR_list, nrmse_avg_list, nrmse_LR_list]
                name_table = ['loss',
                              'SRE_HR', 'SRE_avg', 'SRE_LR',
                              'psnr_HR', 'psnr_avg', 'psnr_LR',
                              'ssim_HR', 'ssim_avg', 'ssim_LR',
                              'nrmse_HR', 'nrmse_avg', 'nrmse_LR']
            else:
                temp_table = [loss_here_list,
                              sre_LR_list,
                              psnr_LR_list,
                              ssim_LR_list,
                              nrmse_LR_list]
                name_table = ['loss',
                              'SRE_LR',
                              'psnr_LR',
                              'ssim_LR',
                              'nrmse_LR']
            temp_table = pd.DataFrame(temp_table).T
            temp_table.columns = name_table
            temp_table.index = list(range(1, temp_table.shape[0] + 1))
            temp_table.to_csv(f'../result/{data_name}/TrainLog_{i + 1}.csv')

            temp_table = per_loss_list.copy()
            temp_table = pd.DataFrame(temp_table)
            temp_table.columns = name_loss
            temp_table.index = list(range(1, temp_table.shape[0] + 1))
            temp_table.to_csv(f'../result/{data_name}/LossLog_{i + 1}.csv')

            scipy.io.savemat("../result/{}/result_sr_it{}.mat".format(data_name,
                                                                      i), {'pred': out_HR_np, 'avg': out_avg_np})

    if i % show_every == 0:
        draw_save_data(out_HR_np, 'np_iter_{}'.format(i), If60=If60, NormMethod=NormMethod)
        draw_save_data(out_avg_np, 'avg_iter_{}'.format(i), If60=If60, NormMethod=NormMethod)
    # if i in save_every:
    #     scipy.io.savemat("../result/{}/result_sr_it{}.mat".format(data_name,
    #                                                               i), {'pred': out_HR_np, 'avg': out_avg_np})

    return total_loss, psnr_LR


def closure_initial():
    global i, initial_input, img_d10_np, LossWeight
    if sigma_random > 0:
        net_input_this = initial_input.clone() + torch.from_numpy(
            get_noise(initial_input.clone().detach().cpu().squeeze().numpy(), input_C, (img_d10_dim1, img_d10_dim2),
                      noise_type=args.noise_type,
                      var=sigma_random)).type(dtype)[None, :].cuda()
    else:
        net_input_this = initial_input.clone()
    if NetChoice == 'sep3d' or NetChoice == '3d' or NetChoice == 'resnet' or NetChoice == 'resnet3d_common':
        out = net1(net_input_this)
        out = torch.unsqueeze(out, 0)
        out = net_main(out)
        out = torch.squeeze(out, 0)
        out = net2(out)
    elif NetChoice == '2d' or NetChoice == 'resnet2d':
        out = net_main(net_input_this)

    total_loss = 0
    if 'l1' in InitialLossList:
        l1 = l1Norm(out, initial_var)
        total_loss += l1 * LossWeight['l1']

    if 'l2' in InitialLossList:
        l2 = l2Norm(out, initial_var)
        total_loss += l2 * LossWeight['l2']

    if 'ssim' in InitialLossList:
        ssim = 1 - ssim_loss(out, initial_var)
        total_loss += ssim * LossWeight['ssim']

    if 'vgg' in InitialLossList:
        percep = perceptual_loss(out, initial_var)
        total_loss += percep * LossWeight['vgg']

    total_loss.backward()

    loss_here = total_loss.detach().cpu().numpy()
    out_np = out.detach().cpu().squeeze().numpy()

    out_np_1 = out_np[0, :, :]
    out_np_2 = out_np[1, :, :]
    out_np_3 = out_np[2, :, :]
    out_np_4 = out_np[3, :, :]
    out_np_5 = out_np[4, :, :]
    out_np_6 = out_np[5, :, :]
    if If60:
        out_np_7 = out_np[6, :, :]
        out_np_8 = out_np[7, :, :]
        out_np_9 = out_np[8, :, :]
        out_np_10 = out_np[9, :, :]
        out_np_11 = out_np[10, :, :]
        out_np_12 = out_np[11, :, :]

        temp_7 = initial_var_np[6, :, :]
        temp_8 = initial_var_np[7, :, :]
    else:

        out_np_9 = out_np[6, :, :]
        out_np_10 = out_np[7, :, :]
        out_np_11 = out_np[8, :, :]
        out_np_12 = out_np[9, :, :]

    temp_9 = img_d10_np[0, :, :]
    temp_10 = img_d10_np[1, :, :]
    temp_11 = img_d10_np[2, :, :]
    temp_12 = img_d10_np[3, :, :]

    temp_1 = initial_var_np[0, :, :]
    temp_2 = initial_var_np[1, :, :]
    temp_3 = initial_var_np[2, :, :]
    temp_4 = initial_var_np[3, :, :]
    temp_5 = initial_var_np[4, :, :]
    temp_6 = initial_var_np[5, :, :]

    error1 = np.abs(out_np_1 - temp_1)[np.newaxis, np.newaxis, :, :]
    error2 = np.abs(out_np_2 - temp_2)[np.newaxis, np.newaxis, :, :]
    error3 = np.abs(out_np_3 - temp_3)[np.newaxis, np.newaxis, :, :]
    error4 = np.abs(out_np_4 - temp_4)[np.newaxis, np.newaxis, :, :]
    error5 = np.abs(out_np_5 - temp_5)[np.newaxis, np.newaxis, :, :]
    error6 = np.abs(out_np_6 - temp_6)[np.newaxis, np.newaxis, :, :]
    if If60:
        error7 = np.abs(out_np_7 - temp_7)[np.newaxis, np.newaxis, :, :]
        error8 = np.abs(out_np_8 - temp_8)[np.newaxis, np.newaxis, :, :]
    error9 = np.abs(out_np_9 - temp_9)[np.newaxis, np.newaxis, :, :]
    error10 = np.abs(out_np_10 - temp_10)[np.newaxis, np.newaxis, :, :]
    error11 = np.abs(out_np_11 - temp_11)[np.newaxis, np.newaxis, :, :]
    error12 = np.abs(out_np_12 - temp_12)[np.newaxis, np.newaxis, :, :]

    errormap = np.concatenate(
        (
            1 - np.clip(error1, 0, 1),
            1 - np.clip(error2, 0, 1),
            1 - np.clip(error3, 0, 1),
            1 - np.clip(error4, 0, 1),
            1 - np.clip(error5, 0, 1),
            1 - np.clip(error6, 0, 1),
            1 - np.clip(error7, 0, 1),
            1 - np.clip(error8, 0, 1),
            1 - np.clip(error9, 0, 1),
            1 - np.clip(error10, 0, 1),
            1 - np.clip(error11, 0, 1),
            1 - np.clip(error12, 0, 1),
        ),
        axis=0
    ) if If60 else np.concatenate(
        (
            1 - np.clip(error1, 0, 1),
            1 - np.clip(error2, 0, 1),
            1 - np.clip(error3, 0, 1),
            1 - np.clip(error4, 0, 1),
            1 - np.clip(error5, 0, 1),
            1 - np.clip(error6, 0, 1),
            1 - np.clip(error9, 0, 1),
            1 - np.clip(error10, 0, 1),
            1 - np.clip(error11, 0, 1),
            1 - np.clip(error12, 0, 1),
        ),
        axis=0
    )

    summary_initial.add_scalar('PreLoss', loss_here, i)

    summary_initial.add_images(
        'errormap',
        errormap,
        global_step=i,
        dataformats='NCHW'
    )

    if i % log_every == 0:
        print(f'\nNow at {i + 1} epoch, loss is {loss_here}')

    return total_loss


def make_eva_table(data_, name, img_HR_np_, d_bicubic_):
    """
    Generate evaluation tables per spectral
    """

    def get_evalution_per_tail(data, img_HR_np):
        """
        Calculate the index per spectrum
        """
        psnr_list = []
        ssim_list = []
        nrmse_list = []

        rmse_list = []
        sre_list = []
        uiqa_list = []

        for i in range(data.shape[0]):
            psnr_list.append(compare_psnr(img_HR_np[i, :, :], data[i, :, :], data_range=compareRange(img_HR_np[i])))
            ssim_list.append(compare_ssim(img_HR_np[i, :, :], data[i, :, :], data_range=compareRange(img_HR_np[i])))
            nrmse_list.append(compare_nrmse(img_HR_np[i, :, :], data[i, :, :]))

            rmse_list.append(RMSE(img_HR_np[i, :, :], data[i, :, :]))
            sre_list.append(SRE(img_HR_np[i, :, :], data[i, :, :]))
            uiqa_list.append(UIQA(img_HR_np[i, :, :], data[i, :, :]))

        return np.array(psnr_list), np.array(ssim_list), np.array(nrmse_list), np.array(rmse_list), np.array(
            sre_list), np.array(uiqa_list)

    def get_blind_per_tail(data):
        """
        Calculate the index per spectrum
        """
        if data.max() > 1:
            data = my_regular(data, 'per')

        data = numpy2img(data)

        brenner_list = []
        smd2_list = []
        energy_list = []

        for i in range(data.shape[0]):
            brenner_list.append(Brenner(data[i, :, :]))
            smd2_list.append(SMD2(data[i, :, :]))
            energy_list.append(Energy(data[i, :, :]))

        return np.array(brenner_list), np.array(smd2_list), np.array(energy_list)

    def get_evalution(data, img_HR_np):
        """
        Algorithm that can not be processed per spectrum: SAM
        """
        sam = SAM(img_HR_np, data)
        return np.array(sam)

    data = data_.copy();
    img_HR_np = img_HR_np_.copy();
    d_bicubic = d_bicubic_.copy()

    psnr_list_DIP, ssim_list_DIP, nrmse_list_DIP, rmse_list_DIP, sre_list_DIP, uiqa_list_DIP = get_evalution_per_tail(
        data, img_HR_np)
    sam_DIP = get_evalution(data, img_HR_np)
    brenner_list_DIP, smd2_list_DIP, energy_list_DIP = get_blind_per_tail(data)

    psnr_list_base, ssim_list_base, nrmse_list_base, rmse_list_base, sre_list_base, uiqa_list_base = get_evalution_per_tail(
        d_bicubic, img_HR_np)
    sam_base = get_evalution(d_bicubic, img_HR_np)
    brenner_list_base, smd2_list_base, energy_list_base = get_blind_per_tail(
        d_bicubic)

    psnr_table = pd.DataFrame(
        data=[psnr_list_DIP, psnr_list_base], index=['DIP', 'Bicubic'])
    ssim_table = pd.DataFrame(
        data=[ssim_list_DIP, ssim_list_base], index=['DIP', 'Bicubic'])
    nrmse_table = pd.DataFrame(
        data=[nrmse_list_DIP, nrmse_list_base], index=['DIP', 'Bicubic'])
    rmse_table = pd.DataFrame(
        data=[rmse_list_DIP, rmse_list_base], index=['DIP', 'Bicubic'])
    sre_table = pd.DataFrame(
        data=[sre_list_DIP, sre_list_base], index=['DIP', 'Bicubic'])
    uiqa_table = pd.DataFrame(
        data=[uiqa_list_DIP, uiqa_list_base], index=['DIP', 'Bicubic'])

    sam_table = pd.DataFrame(
        data=[sam_DIP, sam_base], index=['DIP', 'Bicubic'])

    brenner_table = pd.DataFrame(
        data=[brenner_list_DIP, brenner_list_base], index=['DIP', 'Bicubic'])
    smd2_table = pd.DataFrame(
        data=[smd2_list_DIP, smd2_list_base], index=['DIP', 'Bicubic'])
    energy_table = pd.DataFrame(
        data=[energy_list_DIP, energy_list_base], index=['DIP', 'Bicubic'])

    psnr_table["Mean"] = psnr_table.mean(axis=1)
    ssim_table["Mean"] = ssim_table.mean(axis=1)
    nrmse_table["Mean"] = nrmse_table.mean(axis=1)
    rmse_table["Mean"] = rmse_table.mean(axis=1)
    sre_table["Mean"] = sre_table.mean(axis=1)
    uiqa_table["Mean"] = uiqa_table.mean(axis=1)
    brenner_table["Mean"] = brenner_table.mean(axis=1)
    smd2_table["Mean"] = smd2_table.mean(axis=1)
    energy_table["Mean"] = energy_table.mean(axis=1)

    with pd.ExcelWriter('../result/{}/{}_{}_eva.xlsx'.format(data_name, mat_name, name), engine='xlsxwriter') as writer:
        psnr_table.to_excel(writer, sheet_name='PSNR')
        ssim_table.to_excel(writer, sheet_name='SSIM')
        nrmse_table.to_excel(writer, sheet_name='NRMSE')
        rmse_table.to_excel(writer, sheet_name='RMSE')
        sre_table.to_excel(writer, sheet_name='SRE')
        uiqa_table.to_excel(writer, sheet_name='UIQA')

        sam_table.to_excel(writer, sheet_name='SAM')

        brenner_table.to_excel(writer, sheet_name='Brenner')
        smd2_table.to_excel(writer, sheet_name='SMD2')
        energy_table.to_excel(writer, sheet_name='Energy')


def draw_save_data(data_, name, size=30, av=None, If60=False, NormMethod='maxkeep'):
    data = data_.copy()
    if NormMethod == '01':
        data, _ = normaliseData(input_data=data, av=av, method='01')
    elif NormMethod == 'UnitPower':
        data, _ = normaliseData(input_data=data, av=None, method='01')
    elif NormMethod == 'maxkeep':
        pass

    long = size / (data.shape[1] / data.shape[2])
    width = size
    numbands = data.shape[0]

    if If60:
        num_line = 3
        num_colu = 3

    else:
        num_line = 2
        num_colu = 3

    figsize = (num_colu * long, num_line * width)
    plt.figure(figsize=figsize)
    for i in range(numbands):
        plt.subplot(num_line, num_colu, i + 1)
        plt.imshow(np.clip(data[i, :, :], 0, 1), cmap=color)
        plt.axis('off')

    plt.savefig('../result/{}/pertail/{}.png'.format(data_name,
                                                     name), bbox_inches='tight')
    plt.close()

    if If60:
        num_line = 1
        num_colu = 3

    else:
        num_line = 1
        num_colu = 2

    plt.figure(figsize=(num_colu * long, num_line * width))
    plt.subplot(num_line, num_colu, 1)
    plt.imshow(np.clip(data, 0, 1)[band1, :, :].transpose(1, 2, 0))
    plt.axis('off')
    plt.subplot(num_line, num_colu, 2)
    plt.imshow(np.clip(data, 0, 1)[band2, :, :].transpose(1, 2, 0))
    plt.axis('off')
    if If60 == True:
        plt.subplot(num_line, num_colu, 3)
        plt.imshow(np.clip(data, 0, 1)[band3, :, :].transpose(1, 2, 0))
        plt.axis('off')
    plt.savefig('../result/{}/compare/{}.png'.format(data_name,
                                                     name), bbox_inches='tight')
    plt.close()


# %%
#############################################################
# Settings
############################################################# 
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--id', type=str, default='')

    parser.add_argument('--mat_name', type=str, default='')
    parser.add_argument('--kernelSize', type=int, default=10)
    parser.add_argument('--SNR', type=int, default=40)
    parser.add_argument('--Loss', type=str, default='l1_ssim')
    parser.add_argument('--InitialLoss', type=str, default='l1_ssim')
    parser.add_argument('--LossWeight', type=str, default='1_1_1_1')  # l1_l2_ssim_vgg

    parser.add_argument('--NetChoice', type=str, default='resnet')
    parser.add_argument('--RealData', type=str, default='False')

    parser.add_argument('--If60', type=str, default='True')
    parser.add_argument('--Blur_10', type=str, default='False')
    parser.add_argument('--Noise_10', type=str, default='True')
    parser.add_argument('--feature_2d', type=int, default=256)
    parser.add_argument('--feature_3d', type=int, default=2)
    parser.add_argument('--LR', type=float, default=0.006)
    parser.add_argument('--num_iter', type=int, default=2000)
    parser.add_argument('--exp_weight', type=float, default=0.99)
    parser.add_argument('--lambda_tv', type=float, default=0)

    parser.add_argument('--sigma_random', type=str, default='1/30')
    parser.add_argument('--noise_type', type=str, default='u', help='u|n|s(signal)')

    parser.add_argument('--input_type', type=str, default='pre')

    parser.add_argument('--input_size', type=int, default=12)
    parser.add_argument('--result_mat_path', type=str, default='')
    parser.add_argument('--regular_strategy', type=str, default='no')
    parser.add_argument('--ifNorm', type=str, default='True')
    parser.add_argument('--NormMethod', type=str, default='maxkeep')
    parser.add_argument('--show_every', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--save_model', type=int, default=100)
    parser.add_argument('--resume', type=str, default='False')
    parser.add_argument('--deeperUnet', type=str, default='False')
    parser.add_argument('--initialNet', type=str, default='1000')
    parser.add_argument('--initialrate', type=float, default=0.02)
    parser.add_argument('--SimulateIter', type=int, default=20000)
    parser.add_argument('--SimulateRate', type=float, default=0.0006)
    parser.add_argument('--IfSimulate', type=str, default='True')
    parser.add_argument('--SimulateWeightPath', type=str, default='')
    args = parser.parse_args()
    id = args.id
    mat_name = args.mat_name
    If60 = True if args.If60 == 'True' else False
    deeper = True if args.deeperUnet == 'True' else False
    Blur_10 = True if args.Blur_10 == 'True' else False
    Noise_10 = True if args.Noise_10 == 'True' else False
    kernelSize = args.kernelSize
    SNR = args.SNR
    LossStr = args.Loss
    LossList = LossStr.split('_')
    InitialLossStr = args.InitialLoss
    InitialLossList = InitialLossStr.split('_')

    LossWeight = args.LossWeight
    LossWeightList = LossWeight.split('_')
    LossWeight = {}
    loss_name_list = ['l1', 'l2', 'ssim', 'vgg']
    for i in range(len(loss_name_list)):
        loss_name = loss_name_list[i]
        weight = LossWeightList[i]
        LossWeight[loss_name] = float(weight)

    NetChoice = args.NetChoice
    RealData = True if args.RealData == 'True' else False

    lambda_tv = args.lambda_tv

    sigma_random = args.sigma_random
    if '/' in sigma_random:
        temp = sigma_random.split('/')
        sigma_random = float(temp[0]) / float(temp[1])
    else:
        sigma_random = float(sigma_random)

    input_type = args.input_type
    input_size = args.input_size
    result_mat_path = args.result_mat_path
    regular_strategy = args.regular_strategy
    exp_weight = args.exp_weight
    LR = args.LR
    num_iter = args.num_iter
    ifNorm = True if args.ifNorm == 'True' else False
    NormMethod = args.NormMethod
    feature_2d = args.feature_2d
    feature_3d = args.feature_3d
    show_every = args.show_every
    log_every = args.log_every
    save_model = args.save_model
    resume = True if args.resume == 'True' else False
    initialNet = False if args.initialNet == 'False' else int(args.initialNet)
    initialrate = args.initialrate
    SimulateIter = args.SimulateIter
    SimulateRate = args.SimulateRate
    IfSimulate = True if args.IfSimulate == 'True' else False
    SimulateWeightPath = args.SimulateWeightPath

    TV = GTV2D_Loss()
    band1 = [0, 2, 1]
    band2 = [3, 4, 5]
    band3 = [6, 7, 7]
    color = plt.cm.gray

    save_every = list(range(num_iter - 100, num_iter + 1, 2))
    info = f'Id_{id}'
    data_name = '{}'.format(info)  # 如果文件夹名称与数据mat文件名称不一致， 此处输入文件夹名称
    try:
        os.makedirs('../result/{}/compare/'.format(data_name))
        os.makedirs('../result/{}/pertail/'.format(data_name))
        os.makedirs('../result/{}/combine_mat/'.format(data_name))
    except:
        pass
    #############################################################
    # Load Data
    #############################################################
    d10_crop, d10gt_crop, d20_crop, d20gt_crop, d20_crop_bicubic, d60_crop, d60gt_crop, d60_crop_bicubic = \
        process_data_new(mat_name, input_type, result_mat_path, RealData)

    img_d10_np = d10_crop
    img_LR_np = [d20_crop, d60_crop] if If60 == True else d20_crop
    if not RealData:
        img_HR_np = np.concatenate((d20gt_crop, d60gt_crop)) if If60 else d20gt_crop
    d_bicubic = np.concatenate((d20_crop_bicubic, d60_crop_bicubic)) if If60 else d20_crop_bicubic
    del d10_crop
    del d10gt_crop
    del d20_crop
    del d20gt_crop
    del d20_crop_bicubic
    del d60_crop
    del d60gt_crop
    del d60_crop_bicubic
    if NormMethod != 'maxkeep':
        img_d10_np, img_LR_np, d_bicubic, av, av_bic = groupNorm(
            img_d10_np, img_LR_np, d_bicubic, ifNorm=ifNorm, method=NormMethod, If60=If60)
        # img_d10_var,_ = normaliseData(img_d10_var, method=NormMethod)
    else:
        av = None
        av_bic = None

    if not RealData:
        base_psnr, base_ssim, base_nrmse, base_sre = print_and_save_baseline(
            img_HR_np, d_bicubic, av_bic)
    img_d10_dim1 = img_d10_np.shape[1]
    img_d10_dim2 = img_d10_np.shape[2]
    if not RealData:
        draw_save_data(img_HR_np, 'GT', If60=If60, NormMethod='maxkeep')

    draw_save_data(unnormaliseData(d_bicubic, av_bic, method=NormMethod), 'bicubic', av=av_bic, If60=If60,
                   NormMethod='maxkeep')

    if input_type != 'random':
        input_HR = get_input(input_type, d_bicubic, result_mat_path)
        input_HR = process_new_input(input_HR, If60, img_d10_np, NormMethod)
        if input_HR.shape[0] == 12:
            net_input = input_HR.copy()
        else:
            net_input = np.concatenate(
                (input_HR, img_d10_np), axis=0)
    else:
        net_input = get_input(input_type, d_bicubic,
                              result_mat_path, input_size)

    chunnel_bic = d_bicubic.shape[0]
    input_C = net_input.shape[0]
    out_C = 6 + 2 + 4 if If60 else 6 + 4
    net_input = torch.from_numpy(net_input).type(dtype)[None, :].cuda()
    #############################################################
    # If Parameters Init Required, Get Data for Toy Task
    #############################################################
    # %%
    if initialNet:
        def get_LR(IfSimulate, img_d10_np, img_LR_np, If60, SimulateWeightPath, RealData):
            def simulate_LR(img_d10_np, img_LR_np, If60, SimulateWeightPath, RealData):
                weights = None
                if SimulateWeightPath == '':
                    try:
                        weight_table = pd.read_csv(f'../result/{data_name}/Weight4Simulation.csv')
                    except:
                        try:
                            weight_table = pd.read_csv(f'../result/Weight4Simulation.csv')
                        except:
                            print('Weight File no Found, Start Calculating...')
                else:
                    try:
                        weight_table = pd.read_csv(SimulateWeightPath)
                    except:
                        print('Weight File no Found, Start Calculating...')

                try:
                    weight_table = weight_table.drop('Unnamed: 0', axis=1)
                    weights = weight_table.values
                except:
                    pass

                d10 = img_d10_np.copy().astype(np.float32)
                d10_1 = d10[0, :, :].copy()
                d10_2 = d10[1, :, :].copy()
                d10_3 = d10[2, :, :].copy()
                d10_4 = d10[3, :, :].copy()
                del d10
                if If60:
                    d20 = img_LR_np[0].copy().astype(np.float32)
                    d60 = img_LR_np[1].copy().astype(np.float32)
                    d60_1 = d60[0, :, :].copy()
                    d60_2 = d60[1, :, :].copy()
                    del d60
                else:
                    d20 = img_LR_np.copy()
                d20_1 = d20[0, :, :].copy()
                d20_2 = d20[1, :, :].copy()
                d20_3 = d20[2, :, :].copy()
                d20_4 = d20[3, :, :].copy()
                d20_5 = d20[4, :, :].copy()
                d20_6 = d20[5, :, :].copy()
                del d20
                if weights is None:
                    print('Start Fitting Fake Data for Toy Task.......')
                    print('Start Simulated Degradation Process......')
                    down20 = S2_Downsampler(scale=2, nr=img_d10_dim1, nc=img_d10_dim2, ifNoise=False,
                                            downMethod='angle', kernelSize=kernelSize, SNR=SNR,
                                            RealData=RealData).cuda()
                    d10_1_for_d20 = d10_1[np.newaxis, :, :]
                    d10_1_for_d20 = np.repeat(d10_1_for_d20, 6, axis=0)
                    d10_1_for_d20 = down20(torch.from_numpy(d10_1_for_d20)[
                                               None,].cuda()).squeeze().detach().cpu().numpy()
                    d10_1_for_d20_1 = d10_1_for_d20[0, :, :].copy()
                    d10_1_for_d20_2 = d10_1_for_d20[1, :, :].copy()
                    d10_1_for_d20_3 = d10_1_for_d20[2, :, :].copy()
                    d10_1_for_d20_4 = d10_1_for_d20[3, :, :].copy()
                    d10_1_for_d20_5 = d10_1_for_d20[4, :, :].copy()
                    d10_1_for_d20_6 = d10_1_for_d20[5, :, :].copy()
                    del d10_1_for_d20
                    d10_2_for_d20 = d10_2[np.newaxis, :, :]
                    d10_2_for_d20 = np.repeat(d10_2_for_d20, 6, axis=0)
                    d10_2_for_d20 = down20(torch.from_numpy(d10_2_for_d20)[
                                               None,].cuda()).squeeze().detach().cpu().numpy()
                    d10_2_for_d20_1 = d10_2_for_d20[0, :, :].copy()
                    d10_2_for_d20_2 = d10_2_for_d20[1, :, :].copy()
                    d10_2_for_d20_3 = d10_2_for_d20[2, :, :].copy()
                    d10_2_for_d20_4 = d10_2_for_d20[3, :, :].copy()
                    d10_2_for_d20_5 = d10_2_for_d20[4, :, :].copy()
                    d10_2_for_d20_6 = d10_2_for_d20[5, :, :].copy()
                    del d10_2_for_d20
                    d10_3_for_d20 = d10_3[np.newaxis, :, :]
                    d10_3_for_d20 = np.repeat(d10_3_for_d20, 6, axis=0)
                    d10_3_for_d20 = down20(torch.from_numpy(d10_3_for_d20)[
                                               None,].cuda()).squeeze().detach().cpu().numpy()
                    d10_3_for_d20_1 = d10_3_for_d20[0, :, :].copy()
                    d10_3_for_d20_2 = d10_3_for_d20[1, :, :].copy()
                    d10_3_for_d20_3 = d10_3_for_d20[2, :, :].copy()
                    d10_3_for_d20_4 = d10_3_for_d20[3, :, :].copy()
                    d10_3_for_d20_5 = d10_3_for_d20[4, :, :].copy()
                    d10_3_for_d20_6 = d10_3_for_d20[5, :, :].copy()
                    del d10_3_for_d20
                    d10_4_for_d20 = d10_4[np.newaxis, :, :]
                    d10_4_for_d20 = np.repeat(d10_4_for_d20, 6, axis=0)
                    d10_4_for_d20 = down20(torch.from_numpy(d10_4_for_d20)[
                                               None,].cuda()).squeeze().detach().cpu().numpy()
                    d10_4_for_d20_1 = d10_4_for_d20[0, :, :].copy()
                    d10_4_for_d20_2 = d10_4_for_d20[1, :, :].copy()
                    d10_4_for_d20_3 = d10_4_for_d20[2, :, :].copy()
                    d10_4_for_d20_4 = d10_4_for_d20[3, :, :].copy()
                    d10_4_for_d20_5 = d10_4_for_d20[4, :, :].copy()
                    d10_4_for_d20_6 = d10_4_for_d20[5, :, :].copy()
                    del d10_4_for_d20
                    if If60:
                        down60 = S2_Downsampler(scale=6, nr=img_d10_dim1, nc=img_d10_dim2, ifNoise=False,
                                                downMethod='angle', kernelSize=kernelSize, SNR=SNR,
                                                RealData=RealData).cuda()
                        d10_1_for_d60 = d10_1[np.newaxis, :, :]
                        d10_1_for_d60 = np.repeat(d10_1_for_d60, 2, axis=0)
                        d10_1_for_d60 = down60(torch.from_numpy(d10_1_for_d60)[
                                                   None,].cuda()).squeeze().detach().cpu().numpy()
                        d10_1_for_d60_1 = d10_1_for_d60[0, :, :].copy()
                        d10_1_for_d60_2 = d10_1_for_d60[1, :, :].copy()
                        del d10_1_for_d60
                        d10_2_for_d60 = d10_2[np.newaxis, :, :]
                        d10_2_for_d60 = np.repeat(d10_2_for_d60, 2, axis=0)
                        d10_2_for_d60 = down60(torch.from_numpy(d10_2_for_d60)[
                                                   None,].cuda()).squeeze().detach().cpu().numpy()
                        d10_2_for_d60_1 = d10_2_for_d60[0, :, :].copy()
                        d10_2_for_d60_2 = d10_2_for_d60[1, :, :].copy()
                        del d10_2_for_d60
                        d10_3_for_d60 = d10_3[np.newaxis, :, :]
                        d10_3_for_d60 = np.repeat(d10_3_for_d60, 2, axis=0)
                        d10_3_for_d60 = down60(torch.from_numpy(d10_3_for_d60)[
                                                   None,].cuda()).squeeze().detach().cpu().numpy()
                        d10_3_for_d60_1 = d10_3_for_d60[0, :, :].copy()
                        d10_3_for_d60_2 = d10_3_for_d60[1, :, :].copy()
                        del d10_3_for_d60
                        d10_4_for_d60 = d10_4[np.newaxis, :, :]
                        d10_4_for_d60 = np.repeat(d10_4_for_d60, 2, axis=0)
                        d10_4_for_d60 = down60(torch.from_numpy(d10_4_for_d60)[
                                                   None,].cuda()).squeeze().detach().cpu().numpy()
                        d10_4_for_d60_1 = d10_4_for_d60[0, :, :].copy()
                        d10_4_for_d60_2 = d10_4_for_d60[1, :, :].copy()
                        del d10_4_for_d60
                    print('Finish Simulated Degradation Process.......')
                    print('Start Fitting......')

                    class LinearRegression(nn.Module):
                        def __init__(self):
                            super(LinearRegression, self).__init__()
                            self.linear = nn.Linear(4, 4)

                        def forward(self, x):
                            x = self.linear(x)
                            return x

                    def get_weight(x1, x2, x3, x4, y, SimulateIter=SimulateIter, SimulateRate=SimulateRate):
                        model = LinearRegression().cuda()
                        criterion = nn.MSELoss().cuda()
                        optimizer = torch.optim.Adam(model.parameters(), lr=SimulateRate)
                        x1 = torch.from_numpy(x1).cuda()
                        x2 = torch.from_numpy(x2).cuda()
                        x3 = torch.from_numpy(x3).cuda()
                        x4 = torch.from_numpy(x4).cuda()
                        y = torch.from_numpy(y).cuda()
                        weights = np.ones((1, 4)).astype(np.float32)
                        weights = torch.from_numpy(weights).cuda()
                        best_loss = 1000
                        best_weight = None
                        for e in range(SimulateIter):
                            weights_out = model(weights)
                            weights_out = weights_out.squeeze(0)
                            out = weights_out[0] * x1 + weights_out[1] * x2 + \
                                  weights_out[2] * x3 + weights_out[3] * x4
                            loss = criterion(out, y)
                            log_loss = loss.detach().cpu().numpy().item()
                            weightresult = weights_out.detach().cpu().numpy().tolist()
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            if best_loss > log_loss:
                                best_loss = log_loss
                                best_weight = weightresult
                            if (e + 1) % 2000 == 0:
                                print('Epoch:{}, Loss:{:.5f}, bestLoss:{:.5f}.'.format(e + 1, log_loss, best_loss))
                        return best_weight

                    weight_for_d20_1 = get_weight(
                        d10_1_for_d20_1,
                        d10_2_for_d20_1,
                        d10_3_for_d20_1,
                        d10_4_for_d20_1,
                        d20_1)
                    weight_for_d20_2 = get_weight(
                        d10_1_for_d20_2,
                        d10_2_for_d20_2,
                        d10_3_for_d20_2,
                        d10_4_for_d20_2,
                        d20_2)
                    weight_for_d20_3 = get_weight(
                        d10_1_for_d20_3,
                        d10_2_for_d20_3,
                        d10_3_for_d20_3,
                        d10_4_for_d20_3,
                        d20_3)
                    weight_for_d20_4 = get_weight(
                        d10_1_for_d20_4,
                        d10_2_for_d20_4,
                        d10_3_for_d20_4,
                        d10_4_for_d20_4,
                        d20_4)
                    weight_for_d20_5 = get_weight(
                        d10_1_for_d20_5,
                        d10_2_for_d20_5,
                        d10_3_for_d20_5,
                        d10_4_for_d20_5,
                        d20_5)
                    weight_for_d20_6 = get_weight(
                        d10_1_for_d20_6,
                        d10_2_for_d20_6,
                        d10_3_for_d20_6,
                        d10_4_for_d20_6,
                        d20_6)
                    if If60:
                        weight_for_d60_1 = get_weight(
                            d10_1_for_d60_1,
                            d10_2_for_d60_1,
                            d10_3_for_d60_1,
                            d10_4_for_d60_1,
                            d60_1)
                        weight_for_d60_2 = get_weight(
                            d10_1_for_d60_2,
                            d10_2_for_d60_2,
                            d10_3_for_d60_2,
                            d10_4_for_d60_2,
                            d60_2)

                        weights = [
                            weight_for_d20_1, weight_for_d20_2,
                            weight_for_d20_3, weight_for_d20_4,
                            weight_for_d20_5, weight_for_d20_6,
                            weight_for_d60_1, weight_for_d60_2
                        ]
                        column_name = [
                            'weight_for_d20_1', 'weight_for_d20_2',
                            'weight_for_d20_3', 'weight_for_d20_4',
                            'weight_for_d20_5', 'weight_for_d20_6',
                            'weight_for_d60_1', 'weight_for_d60_2'
                        ]
                    else:
                        weights = [
                            weight_for_d20_1, weight_for_d20_2,
                            weight_for_d20_3, weight_for_d20_4,
                            weight_for_d20_5, weight_for_d20_6
                        ]
                        column_name = [
                            'weight_for_d20_1', 'weight_for_d20_2',
                            'weight_for_d20_3', 'weight_for_d20_4',
                            'weight_for_d20_5', 'weight_for_d20_6'
                        ]

                    weights = np.array(weights).astype(np.float32).T
                    weight_table = pd.DataFrame(weights, columns=column_name)
                    weight_table.to_csv(f'../result/{data_name}/Weight4Simulation.csv')
                    print('Weight Fitting Finished!')

                print('Start Generating Fake Simulated Data from Weight.......')
                simulate_d20_1 = weights[0, 0] * d10_1 + weights[1, 0] * d10_2 + \
                                 weights[2, 0] * d10_3 + weights[3, 0] * d10_4
                simulate_d20_2 = weights[0, 1] * d10_1 + weights[1, 1] * d10_2 + \
                                 weights[2, 1] * d10_3 + weights[3, 1] * d10_4
                simulate_d20_3 = weights[0, 2] * d10_1 + weights[1, 2] * d10_2 + \
                                 weights[2, 2] * d10_3 + weights[3, 2] * d10_4
                simulate_d20_4 = weights[0, 3] * d10_1 + weights[1, 3] * d10_2 + \
                                 weights[2, 3] * d10_3 + weights[3, 3] * d10_4
                simulate_d20_5 = weights[0, 4] * d10_1 + weights[1, 4] * d10_2 + \
                                 weights[2, 4] * d10_3 + weights[3, 4] * d10_4
                simulate_d20_6 = weights[0, 5] * d10_1 + weights[1, 5] * d10_2 + \
                                 weights[2, 5] * d10_3 + weights[3, 5] * d10_4
                simulate_d20_1 = simulate_d20_1[np.newaxis, :, :]
                simulate_d20_2 = simulate_d20_2[np.newaxis, :, :]
                simulate_d20_3 = simulate_d20_3[np.newaxis, :, :]
                simulate_d20_4 = simulate_d20_4[np.newaxis, :, :]
                simulate_d20_5 = simulate_d20_5[np.newaxis, :, :]
                simulate_d20_6 = simulate_d20_6[np.newaxis, :, :]

                if If60:
                    simulate_d60_1 = weights[0, 6] * d10_1 + weights[1, 6] * d10_2 + \
                                     weights[2, 6] * d10_3 + weights[3, 6] * d10_4
                    simulate_d60_2 = weights[0, 7] * d10_1 + weights[1, 7] * d10_2 + \
                                     weights[2, 7] * d10_3 + weights[3, 7] * d10_4
                    simulate_d60_1 = simulate_d60_1[np.newaxis, :, :]
                    simulate_d60_2 = simulate_d60_2[np.newaxis, :, :]

                    initial_input = np.concatenate(
                        (simulate_d20_1, simulate_d20_2, simulate_d20_3, simulate_d20_4, simulate_d20_5, simulate_d20_6,
                         simulate_d60_1, simulate_d60_2,
                         img_d10_np),
                        axis=0)
                else:
                    initial_input = np.concatenate(
                        (simulate_d20_1, simulate_d20_2, simulate_d20_3, simulate_d20_4, simulate_d20_5, simulate_d20_6,
                         img_d10_np),
                        axis=0)
                initial_var = initial_input.copy()
                return initial_input, initial_var

            def copy_LR(img_d10_np, If60):
                if If60 == True:
                    initial_input = np.concatenate([img_d10_np, img_d10_np, img_d10_np], axis=0)  # 6+2+4 = 4+4+4
                else:
                    initial_input = np.concatenate([img_d10_np, img_d10_np[0:2, :, :], img_d10_np],
                                                   axis=0)  # 6+4= 4+2+4
                initial_var = initial_input.copy()
                return initial_input, initial_var

            if IfSimulate:
                initial_input, initial_var = simulate_LR(img_d10_np, img_LR_np, If60, SimulateWeightPath, RealData)
            else:
                initial_input, initial_var = copy_LR(img_d10_np, If60)
            return initial_input, initial_var


        initial_input, initial_var = get_LR(IfSimulate, img_d10_np, img_LR_np, If60, SimulateWeightPath, RealData)
        initial_var_np = initial_var.copy()
        initial_input = torch.from_numpy(initial_input).type(dtype)[None, :].cuda()
        initial_var = torch.from_numpy(initial_var).type(dtype)[None, :].cuda()
        summary_initial = SummaryWriter('../result/{}/runs/exp_initial'.format(data_name))
        print("tensorboard --logdir={}\\result\\{}\\runs\\".format(os.path.abspath('../'), data_name))

    #############################################################
    # Data For Calculating Loss
    #############################################################
    if If60:
        img_LR_np_20 = img_LR_np[0]
        img_LR_np_60 = img_LR_np[1]
        img_LR_var_20 = torch.from_numpy(img_LR_np_20).type(dtype)
        img_LR_var_60 = torch.from_numpy(img_LR_np_60).type(dtype)
        img_LR_var_20 = img_LR_var_20[None, :].cuda()
        img_LR_var_60 = img_LR_var_60[None, :].cuda()
        if NormMethod != 'maxkeep':
            img_LR_np[0] = unnormaliseData(img_LR_np[0], av[0], method=NormMethod)
            img_LR_np[1] = unnormaliseData(img_LR_np[1], av[1], method=NormMethod)
    else:
        img_LR_var = img_LR_np
        img_LR_var = torch.from_numpy(img_LR_var).type(dtype)
        img_LR_var = img_LR_var[None, :].cuda()
        if NormMethod != 'maxkeep':
            img_LR_np = unnormaliseData(img_LR_np, av, method=NormMethod)

    img_10_var = img_d10_np
    img_10_var = torch.from_numpy(img_10_var).type(dtype)
    img_10_var = img_10_var[None, :].cuda()

    #############################################################
    # Network
    #############################################################

    if NetChoice == 'sep3d':
        try:
            from ..models.unet import *
        except:
            from models.unet import *
    elif NetChoice == '3d':
        try:
            from ..models.unet_standard_3d import *
        except:
            from models.unet_standard_3d import *
    elif NetChoice == 'resnet':
        try:
            from ..models.resnet3d import *
        except:
            from models.resnet3d import *
    elif NetChoice == '2d':
        try:
            from ..models.unet_2d import *
        except:
            from models.unet_2d import *

    elif NetChoice == 'resnet3d_common':
        try:
            from ..models.resnet3d_common import *
        except:
            from models.resnet3d_common import *

    elif NetChoice == 'resnet2d':
        try:
            from ..models.resnet2d import *
        except:
            from models.resnet2d import *

    if NetChoice == 'sep3d' or NetChoice == '3d' or NetChoice == 'resnet' or NetChoice == 'resnet3d_common':
        net1 = DoubleConv(input_C, feature_2d).cuda()
        if not deeper:
            net_main = resnet34(1, feature_3d, 1).cuda() if NetChoice in ['resnet', 'resnet3d_common'] else Unet_3D(1, feature_3d, 1).cuda()
        else:
            net_main = resnet34(1, feature_3d, 1).cuda() if NetChoice in ['resnet', 'resnet3d_common'] else Unet_3D_5(1, feature_3d, 1).cuda()
        net2 = DoubleConv(feature_2d, out_C).cuda()
        params = []
        params += [x for x in net1.parameters()]
        params += [x for x in net_main.parameters()]
        params += [x for x in net2.parameters()]
    elif NetChoice == '2d' or NetChoice == 'resnet2d':
        net_main = Unet(input_C, feature_2d, out_C).cuda() if NetChoice == '2d' else resnet34(input_C, feature_2d,
                                                                                              out_C).cuda()
        params = []
        params += [x for x in net_main.parameters()]

    #############################################################
    # Degradation
    #############################################################
    downsampler = S2_Downsampler(scale=2, nr=img_d10_dim1, nc=img_d10_dim2, ifNoise=False, downMethod='angle',
                                 kernelSize=kernelSize, SNR=SNR, RealData=RealData).cuda()
    if If60:
        downsampler_60 = S2_Downsampler(scale=6, nr=img_d10_dim1, nc=img_d10_dim2, ifNoise=False, downMethod='angle',
                                        kernelSize=kernelSize, SNR=SNR, RealData=RealData).cuda()

    if Blur_10 == True or Noise_10 == True:
        downsampler_10 = S2_Downsampler(scale=1, nr=img_d10_dim1, nc=img_d10_dim2, ifNoise=False, downMethod='angle',
                                        ifBlur_10=Blur_10, ifNoise_10=Noise_10, kernelSize=kernelSize, SNR=SNR,
                                        RealData=RealData).cuda()

    #############################################################
    # Loss Function
    #############################################################
    l2Norm = torch.nn.MSELoss().type(dtype)
    l1Norm = torch.nn.L1Loss().type(dtype)
    ssim_loss = pytorch_ssim.SSIM().type(dtype)
    perceptual_loss = Perceptual_Loss().cuda()

    out_avg = None

    #############################################################
    # If Parameters Init Required，Training and Setting Parameters
    #############################################################
    if initialNet:
        try:
            initial_weight = torch.load(f'../result/{data_name}/ParaInit.pth')
        except:
            initial_weight = None
        if initial_weight is None:
            optimizer_initial = torch.optim.Adam(params=params, lr=initialrate)
            from torch.optim.lr_scheduler import CosineAnnealingLR

            scheduler_initial = CosineAnnealingLR(optimizer_initial, T_max=10)
            i = 0
            print('Starting Training Initial Parameters......')
            pbar = tqdm(total=initialNet)
            for i in range(initialNet):
                optimizer_initial.zero_grad()
                total_loss = closure_initial()
                optimizer_initial.step()
                # scheduler_initial.step()
                pbar.update(1)
            pbar.close()
            if NetChoice == 'sep3d' or NetChoice == '3d' or NetChoice == 'resnet' or NetChoice == 'resnet3d_common':
                state = {
                    "net1": net1.state_dict(),
                    "net_main": net_main.state_dict(),
                    "net2": net2.state_dict(),
                }
            elif NetChoice == '2d' or NetChoice == 'resnet2d':
                state = {
                    "net_main": net_main.state_dict(),
                }
            torch.save(state, f'../result/{data_name}/ParaInit.pth')
            summary_initial.close()
            del initial_input, initial_var, net_main, scheduler_initial, optimizer_initial, state, params

        initial_weight = torch.load(f'../result/{data_name}/ParaInit.pth')

        if NetChoice == 'sep3d' or NetChoice == '3d' or NetChoice == 'resnet' or NetChoice == 'resnet3d_common':
            net1 = DoubleConv(input_C, feature_2d)
            if not deeper:
                net_main = resnet34(1, feature_3d, 1) if NetChoice == 'resnet' else Unet_3D(1, feature_3d,
                                                                                            1)
            else:
                net_main = resnet34(1, feature_3d, 1) if NetChoice == 'resnet' else Unet_3D_5(1, feature_3d, 1)
            net2 = DoubleConv(feature_2d, out_C)

            net1.load_state_dict(initial_weight['net1'])
            net_main.load_state_dict(initial_weight['net_main'])
            net2.load_state_dict(initial_weight['net2'])

            net1 = net1.cuda()
            net_main = net_main.cuda()
            net2 = net2.cuda()

            params = []
            params += [x for x in net1.parameters()]
            params += [x for x in net_main.parameters()]
            params += [x for x in net2.parameters()]

        elif NetChoice == '2d' or NetChoice == 'resnet2d':
            net_main = Unet(input_C, feature_2d, out_C) if NetChoice == '2d' else resnet34(input_C, feature_2d, out_C)
            net_main.load_state_dict(initial_weight['net_main'])
            net_main = net_main.cuda()
            params = []
            params += [x for x in net_main.parameters()]

    #############################################################
    # optimizer
    #############################################################
    optimizer = torch.optim.Adam(params=params, lr=LR)
    scheduler_1 = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=400, min_lr=0.005, cooldown=100)

    #############################################################
    # If Resume, Load state
    #############################################################
    if not resume:
        epoch_before = 0
    else:
        print('Start Resuming......')
        checkpoint = torch.load(resume)
        if NetChoice == 'sep3d' or NetChoice == '3d' or NetChoice == 'resnet' or NetChoice == 'resnet3d_common':
            net1.load_state_dict(checkpoint['net1'])
            net_main.load_state_dict(checkpoint['net_main'])
            net2.load_state_dict(checkpoint['net2'])
        elif NetChoice == '2d' or NetChoice == 'resnet2d':
            net_main.load_state_dict(checkpoint['net_main'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_before = checkpoint['epoch']
        except:
            pass
    #############################################################
    # Training
    #############################################################
    if not RealData:
        sre_HR_list = []
        sre_avg_list = []
        sre_LR_list = []
        psnr_HR_list = []
        psnr_avg_list = []
        psnr_LR_list = []
        ssim_HR_list = []
        ssim_avg_list = []
        ssim_LR_list = []
        nrmse_HR_list = []
        nrmse_avg_list = []
        nrmse_LR_list = []
        loss_here_list = []
        per_loss_list = []
    else:
        sre_LR_list = []
        psnr_LR_list = []
        ssim_LR_list = []
        nrmse_LR_list = []
        loss_here_list = []
        per_loss_list = []
    i = 0
    print('Starting optimization with ADAM')
    summarywriter = SummaryWriter('../result/{}/runs/exp_main'.format(data_name))
    print("tensorboard --logdir={}\\result\\{}\\runs\\".format(os.path.abspath('../'), data_name))

    time_start = time.time()
    pbar = tqdm(total=num_iter - epoch_before)
    for i in range(epoch_before, num_iter + 1):
        optimizer.zero_grad()
        total_loss, psnr_LR = closure()
        optimizer.step()
        scheduler_1.step(psnr_LR)
        summarywriter.add_text('LR', text_string='{}'.format(
            optimizer.state_dict()['param_groups'][0]['lr']), global_step=i)
        pbar.update(1)
    pbar.close()
    time_end = time.time()

    f = open(f'../result/{data_name}/timelog.txt', 'a+')
    print('totally cost', time_end - time_start, file=f)
    f.close()

    summarywriter.close()
