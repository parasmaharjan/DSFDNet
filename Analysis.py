import os, time, scipy.io
import numpy as np
from scipy import misc
import rawpy
import glob
import torch
import torch.nn as nn
import torch.optim as optim
#from Model import UNet
from SFNet import SFNet
import argparse
import WaveletDecomposition as wd
import skimage.measure as skm
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pytorch_ssim
import scipy.io as sio

input_dir = '../dataset/Sony/short/'
gt_dir = '../dataset/Sony/long/'
result_dir = './result_Sony/'
model_dir = './saved_model/'
ps = 512  # patch size for training
save_freq = 100
learning_rate = 1e-4
cluster = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Stage 1
parser1 = argparse.ArgumentParser(description='EDSR 1')
parser1.add_argument('--n_resblocks', type=int, default=32,
                    help='number of residual blocks')
parser1.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser1.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser1.add_argument('--scale', type=str, default=1,
                    help='super resolution scale')
parser1.add_argument('--n_colors', type=int, default=4,
                    help='number of input color channels to use')
parser1.add_argument('--o_colors', type=int, default=4,
                    help='number of output color channels to use')
args1 = parser1.parse_args()

# Stage 2
parser2 = argparse.ArgumentParser(description='EDSR 2')
parser2.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser2.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser2.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser2.add_argument('--scale', type=str, default=1,
                    help='super resolution scale')
parser2.add_argument('--n_colors', type=int, default=8,
                    help='number of input color channels to use')
parser2.add_argument('--o_colors', type=int, default=4,
                    help='number of output color channels to use')
args2 = parser2.parse_args()

# Stage 3
parser3 = argparse.ArgumentParser(description='EDSR 3')
parser3.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser3.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser3.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser3.add_argument('--scale', type=str, default=1,
                    help='super resolution scale')
parser3.add_argument('--n_colors', type=int, default=8,
                    help='number of input color channels to use')
parser3.add_argument('--o_colors', type=int, default=4,
                    help='number of output color channels to use')
args3 = parser3.parse_args()

# Stage 4
parser4 = argparse.ArgumentParser(description='EDSR 4')
parser4.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser4.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser4.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser4.add_argument('--scale', type=str, default=1,
                    help='super resolution scale')
parser4.add_argument('--n_colors', type=int, default=8,
                    help='number of input color channels to use')
parser4.add_argument('--o_colors', type=int, default=4,
                    help='number of output color channels to use')
args4 = parser4.parse_args()

# # Stage 5
# parser5 = argparse.ArgumentParser(description='EDSR 5')
# parser5.add_argument('--n_resblocks', type=int, default=4,
#                     help='number of residual blocks')
# parser5.add_argument('--n_feats', type=int, default=64,
#                     help='number of feature maps')
# parser5.add_argument('--res_scale', type=float, default=1,
#                     help='residual scaling')
# parser5.add_argument('--scale', type=str, default=1,
#                     help='super resolution scale')
# parser5.add_argument('--n_colors', type=int, default=4,
#                     help='number of input color channels to use')
# parser5.add_argument('--o_colors', type=int, default=4,
#                     help='number of output color channels to use')
# args5 = parser5.parse_args()

# get train and test IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level
    wr, wg1, wb, wg2 = np.array(raw.camera_whitebalance) / min(raw.camera_whitebalance)
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def unpack_raw(img):
    img_shape = img.shape
    H = img_shape[0]*2
    W = img_shape[1]*2

    out = np.zeros((H,W))
    out[0:H:2, 0:W:2] = img[:,:,0]
    out[0:H:2, 1:W:2] = img[:,:,1]
    out[1:H:2, 1:W:2] = img[:,:,2]
    out[1:H:2, 0:W:2] = img[:,:,3]
    return out

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


if cluster:
    # Raw data takes long time to load. Keep them in memory after loaded.
    gt4ch_images = [None] * 6000
    #gt3ch_images = [None] * 6000
    input_images = {}
    input_images['300'] = [None] * len(train_ids)
    input_images['250'] = [None] * len(train_ids)
    input_images['100'] = [None] * len(train_ids)

# g_loss = np.zeros((5000, 1))
# g_loss_LL = np.zeros((5000, 1))
# g_loss_LH = np.zeros((5000, 1))
# g_loss_HL = np.zeros((5000, 1))
# g_loss_HH = np.zeros((5000, 1))


allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

#LL
model_LL100 = SFNet(args1).to(device)
model_LL100.load_state_dict(torch.load(model_dir + 'ar100_model_LL_sony_e4000.pth'))
#model_LL100.load_state_dict(torch.load(model_dir + 'ar300_combine_all_model_LL_sony_e4000.pth'))
#opt_LL = optim.Adam(model_LL100.parameters(), lr=learning_rate)

model_LL250 = SFNet(args1).to(device)
model_LL250.load_state_dict(torch.load(model_dir + 'ar250_model_LL_sony_e4000.pth'))
#model_LL250.load_state_dict(torch.load(model_dir + 'ar300_combine_all_model_LL_sony_e4000.pth'))
#opt_LL = optim.Adam(model_LL250.parameters(), lr=learning_rate)

model_LL300 = SFNet(args1).to(device)
model_LL300.load_state_dict(torch.load(model_dir + 'ar300_combine_all_model_LL_sony_e4000.pth'))
#model_LL300.load_state_dict(torch.load(model_dir + 'ar300_model_LL_sony_e4000.pth'))
#opt_LL = optim.Adam(model_LL300.parameters(), lr=learning_rate)

#LH
model_LH = SFNet(args2).to(device)
#model_LH.load_state_dict(torch.load(model_dir + 'l1_ssim_LH/l1_ssim_model_LH_sony_e2000.pth'))
model_LH.load_state_dict(torch.load(model_dir + 'l1_ssim_with_added_ll_LH/l1_ssim_with_added_ll_model_LH_sony_e4000.pth'))
#opt_LH = optim.Adam(model_LH.parameters(), lr=learning_rate)

#HL
model_HL = SFNet(args3).to(device)
#model_HL.load_state_dict(torch.load(model_dir + 'l1_ssim_HL/l1_ssim_model_HL_sony_e1000.pth'))
model_HL.load_state_dict(torch.load(model_dir + 'l1_ssim_with_added_ll_HL/l1_ssim_with_added_ll_model_HL_sony_e4000.pth'))
#opt_HL = optim.Adam(model_HL.parameters(), lr=learning_rate)

#HH
model_HH = SFNet(args4).to(device)
#model_HH.load_state_dict(torch.load(model_dir + 'l1_ssim_HH/l1_ssim_model_HH_sony_e2000.pth'))
model_HH.load_state_dict(torch.load(model_dir + 'l1_ssim_with_added_ll_HH/l1_ssim_with_added_ll_model_HH_sony_e1200.pth'))
#opt_HH = optim.Adam(model_HH.parameters(), lr=learning_rate)

ssim_loss = pytorch_ssim.SSIM()

psnr = []
ssim = []

psnr_ll = []
ssim_ll = []

psnr_lh = []
ssim_lh = []

psnr_hl = []
ssim_hl = []

psnr_hh = []
ssim_hh = []
cnt = 0

with torch.no_grad():
    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]

            in_path = glob.glob(input_dir + '10003_00_0.1s.ARW')[0]

            _, in_fn = os.path.split(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
            gt_path = gt_files[0]

            #gt_path = glob.glob(gt_dir + '00230_00_10s.ARW')[0]

            _, gt_fn = os.path.split(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

            # if str(ratio)[0:3] != "300":
            #     continue

            # wr, wg, wb, _ = raw.daylight_whitebalance
            # input_full = input_full[:,:512, :512, :]

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # im = im[:1024,:1024]
            scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
            # scale_full = np.minimum(scale_full, 1.0)

            gt_raw = rawpy.imread(gt_path)
            max_val = gt_raw.raw_image_visible.max()
            gt4ch_images = np.expand_dims(pack_raw(gt_raw), axis=0)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # im = im[:1024, :1024]
            gt_full = np.float32(im / 65535.0)
            if 0:
                plt.figure()
                plt.imshow(gt_full)
                plt.title('Full ground truth image')

            row = 1900
            col = 3450
            patch = 256
            # plt.figure()
            # plt.title('Crop ground truth image')
            # plt.imshow(gt_full[row:row+patch,col:col+patch, :])

            input_full = np.minimum(input_full, 1.0)

            # Wavelet_decompose
            st = time.time()
            _, _, _, in_LL, in_LH, in_HL, in_HH = wd.decompose4ch(input_full)


            if 0:
                plt.figure()
                plt.suptitle('Wavelet decomposition of input noisy patch')
                plt.subplot(221)
                plt.imshow(in_LL[0, row//4: row//4 + 256//4, col//4: col//4 + 256//4, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(in_LH[0, row // 4: row // 4 + 256 // 4, col // 4: col // 4 + 256 // 4, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(223)
                plt.imshow(in_HL[0, row // 4: row // 4 + 256 // 4, col // 4: col // 4 + 256 // 4, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(224)
                plt.imshow(in_HH[0, row // 4: row // 4 + 256 // 4, col // 4: col // 4 + 256 // 4, 0], cmap='gray')
                plt.axis('off')

            #print('Time DWT: ', time.time() - st)
            _, _, _, gt4ch_LL, gt4ch_LH, gt4ch_HL, gt4ch_HH = wd.decompose4ch(gt4ch_images)

            if 0:
                plt.figure()
                plt.suptitle('Wavelet decomposition of ground truth patch')
                plt.subplot(221)
                plt.imshow(gt4ch_LL[0, row//4: row//4 + 256//4, col//4: col//4 + 256//4, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(gt4ch_LH[0, row // 4: row // 4 + 256 // 4, col // 4: col // 4 + 256 // 4, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(223)
                plt.imshow(gt4ch_HL[0, row // 4: row // 4 + 256 // 4, col // 4: col // 4 + 256 // 4, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(224)
                plt.imshow(gt4ch_HH[0, row // 4: row // 4 + 256 // 4, col // 4: col // 4 + 256 // 4, 0], cmap='gray')
                plt.axis('off')

                gt_out_img = wd.combine4ch(_, _, _, gt4ch_LL, gt4ch_LH, gt4ch_HL, gt4ch_HH)
                plt.figure()
                plt.suptitle('Reconstructed ground truth shown channel wise after inverse wavelet')
                plt.subplot(221)
                plt.imshow(gt_out_img[row//2: row//2 + 256//2, col//2: col//2 + 256//2, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(gt_out_img[row // 2: row // 2 + 256 // 2, col // 2: col // 2 + 256 // 2, 1], cmap='gray')
                plt.axis('off')
                plt.subplot(223)
                plt.imshow(gt_out_img[row // 2: row // 2 + 256 // 2, col // 2: col // 2 + 256 // 2, 2], cmap='gray')
                plt.axis('off')
                plt.subplot(224)
                plt.imshow(gt_out_img[row // 2: row // 2 + 256 // 2, col // 2: col // 2 + 256 // 2, 3], cmap='gray')
                plt.axis('off')

            in_LL = np.minimum(in_LL, 1.0)
            # #gt4ch_LL = np.minimum(gt4ch_LL, 1.0)
            # in_LH = np.minimum(in_LH, 1.0)
            # #gt4ch_LH = np.minimum(gt4ch_LH, 1.0)
            # in_HL = np.minimum(in_HL, 1.0)
            # #gt4ch_HL = np.minimum(gt4ch_HL, 1.0)
            # in_HH = np.minimum(in_HH, 1.0)
            # #gt4ch_HH = np.minimum(gt4ch_HH, 1.0)
            # #gt_patch = np.maximum(gt_patch, 0.0)

            # torch for input to model
            in_img_LL = torch.from_numpy(in_LL).permute(0, 3, 1, 2).to(device)
            #gt_img_LL = torch.from_numpy(gt4ch_LL).permute(0, 3, 1, 2).to(device)

            #in_img_LH = torch.from_numpy(in_LH).permute(0, 3, 1, 2).to(device)
            in_img_LH = torch.from_numpy(np.concatenate((in_LL, in_LH), axis=3)).permute(0, 3, 1, 2).to(device)
            #gt_img_LH = torch.from_numpy(gt4ch_LH).permute(0, 3, 1, 2).to(device)

            #in_img_HL = torch.from_numpy(in_HL).permute(0, 3, 1, 2).to(device)
            in_img_HL = torch.from_numpy(np.concatenate((in_LL, in_HL), axis=3)).permute(0, 3, 1, 2).to(device)
            #gt_img_HL = torch.from_numpy(gt4ch_HL).permute(0, 3, 1, 2).to(device)

            #in_img_HH = torch.from_numpy(in_HH).permute(0, 3, 1, 2).to(device)
            in_img_HH = torch.from_numpy(np.concatenate((in_LL, in_HH), axis=3)).permute(0, 3, 1, 2).to(device)
            #gt_img_HH = torch.from_numpy(gt4ch_HH).permute(0, 3, 1, 2).to(device)

            #gt_img = torch.from_numpy(gt_patch).permute(0, 3, 1, 2).to(device)

            if str(ratio)[0:3] == "100":
                model_LL100.eval()
                st = time.time()
                out_img_LL = model_LL100(in_img_LL)
                print("Time: ", time.time() - st)
            elif str(ratio)[0:3] == "250":
                model_LL250.eval()
                st = time.time()
                out_img_LL = model_LL250(in_img_LL)
                print("Time: ", time.time() - st)
            elif str(ratio)[0:3] == "300":
                model_LL300.eval()
                st = time.time()
                out_img_LL = model_LL300(in_img_LL)
                print("Time: ", time.time() - st)
            else:
                continue
            #psnr_ll.append(skm.compare_psnr(gt4ch_LL[0, :, :, :], out_img_LL[0,:,:,:]))
            #print('Time ll: ', time.time() - st)
            out_img_LL = out_img_LL.permute(0, 2, 3, 1).cpu().data.numpy()

            model_LH.eval()
            st = time.time()
            out_img_LH = model_LH(in_img_LH)
            #psnr_lh.append(skm.compare_psnr(gt4ch_LH[0, :, :, :], out_img_LH[0,:,:,:]))
            #print('Time lh: ', time.time() - st)
            out_img_LH = out_img_LH.permute(0, 2, 3, 1).cpu().data.numpy()

            model_HL.eval()
            st = time.time()
            out_img_HL = model_HL(in_img_HL)
            #psnr_hl.append(skm.compare_psnr(gt4ch_HL[0, :, :, :], out_img_HL[0,:,:,:]))
            #print('Time hl: ', time.time() - st)
            out_img_HL = out_img_HL.permute(0, 2, 3, 1).cpu().data.numpy()

            model_HH.eval()
            st = time.time()
            out_img_HH = model_HH(in_img_HH)
            #psnr_hh.append(skm.compare_psnr(gt4ch_HH[0, :, :, :], out_img_HH[0,:,:,:]))
            #print('Time hh: ', time.time() - st)
            out_img_HH = out_img_HH.permute(0, 2, 3, 1).cpu().data.numpy()

            st = time.time()
            out_img = wd.combine4ch(_, _, _, out_img_LL, out_img_LH, out_img_HL, out_img_HH)
            #out_img = wd.combine4ch(_, _, _, out_img_LL, gt4ch_LH, gt4ch_HL, gt4ch_HH)
            if 0:
                plt.figure()
                plt.suptitle('Reconstructed Wavelet decomposition of predicted image')
                plt.subplot(221)
                plt.imshow(out_img_LL[0, row//4: row//4 + 256//4, col//4: col//4 + 256//4, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(out_img_LH[0, row // 4: row // 4 + 256 // 4, col // 4: col // 4 + 256 // 4, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(223)
                plt.imshow(out_img_HL[0, row // 4: row // 4 + 256 // 4, col // 4: col // 4 + 256 // 4, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(224)
                plt.imshow(out_img_HH[0, row // 4: row // 4 + 256 // 4, col // 4: col // 4 + 256 // 4, 0], cmap='gray')
                plt.axis('off')

                # gt_out_img = wd.combine4ch(_, _, _, gt4ch_LL, gt4ch_LH, gt4ch_HL, gt4ch_HH)
                plt.figure()
                plt.suptitle('Reconstructed predicted image shown channel wise after inverse wavelet')
                plt.subplot(221)
                plt.imshow(out_img[row//2: row//2 + 256//2, col//2: col//2 + 256//2, 0], cmap='gray')
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(out_img[row // 2: row // 2 + 256 // 2, col // 2: col // 2 + 256 // 2, 1], cmap='gray')
                plt.axis('off')
                plt.subplot(223)
                plt.imshow(out_img[row // 2: row // 2 + 256 // 2, col // 2: col // 2 + 256 // 2, 2], cmap='gray')
                plt.axis('off')
                plt.subplot(224)
                plt.imshow(out_img[row // 2: row // 2 + 256 // 2, col // 2: col // 2 + 256 // 2, 3], cmap='gray')
                plt.axis('off')
            #print('Time IDWT: ', time.time() - st)




            #st = time.time()
            cnt +=1

            out = unpack_raw(out_img)
            out = np.maximum(out, 0.0) * 15871
            out = np.minimum(out, 15871)
            out = out.astype(np.uint16)
            out = out + 512
            out = np.minimum(out, 16383)
            out = np.maximum(out, 0)
            raw.raw_image_visible[:] = out

            # out_rgb = cv2.cvtColor(out, cv2.COLOR_BAYER_RG2BGR)
            im_ = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full_ = np.float32(im_ / 65535.0)
            # plt.figure()
            # plt.imshow(gt_full_)

            #out_img = np.expand_dims(np.minimum(np.maximum(out_img, 0), 1), axis=0)
            # print(out_img.shape, input_patch)
            #combine = np.concatenate((out_img, input_patch), axis=3)

            #misc.imsave()

            # scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            #     result_dir + '%05d_00_test_out_%d.jpg' % (test_id, ratio))
            # result = Image.fromarray((gt_full_ * 255).astype(np.uint8))
            # result.save(result_dir + '%05d_00_test_%d.png' % (test_id, ratio))
            #
            # sio.savemat('Raw_image/%05d_00.mat'%(test_id), {"gt_img": gt_full, "in_img": input_full,"out_img": gt_full_})
            psnr.append(skm.compare_psnr(gt_full, gt_full_))
            ssim.append(skm.compare_ssim(gt_full, gt_full_, multichannel=True))
            print(cnt, " psnr: ", psnr[-1], "SSIM: ", ssim[-1], "ratio:", ratio)
print('mean psnr:', np.mean(psnr))
# print('mean psnr ll:', np.mean(psnr_ll))
# print('mean psnr lh:', np.mean(psnr_lh))
# print('mean psnr hl:', np.mean(psnr_hl))
# print('mean psnr hh:', np.mean(psnr_hh))
print('mean ssim:', np.mean(ssim))
print('done...')

# kernel = np.ones((3,3,3),np.float32)/9
# ll = out_img_LL[:,300:300+25,90:90+25,:]
# lh = out_img_LH[:,300:300+25,90:90+25,:]
# lh = cv2.filter3D(lh[0,:,:,:],-1,kernel)
# hl = out_img_HL[:,300:300+25,90:90+25,:]
# hl = cv2.filter3D(hl[0,:,:,:],-1,kernel)
# hh = out_img_HH[:,300:300+25,90:90+25,:]
# hh = cv2.filter3D(hh[0,:,:,:],-1,kernel)
#
# o = wd.combine4ch(_, _, _, ll, lh*10, hl*10, hh*10)
# plt.figure()
# plt.imshow(o[:,:,0],cmap="gray")
#
# ll_ = gt4ch_LL[:,300:300+25,90:90+25,:]
# lh_ = gt4ch_LH[:,300:300+25,90:90+25,:]
# hl_ = gt4ch_HL[:,300:300+25,90:90+25,:]
# hh_ = gt4ch_HH[:,300:300+25,90:90+25,:]
# o_ = wd.combine4ch(_, _, _, ll_, lh_, hl_, hh_)
# plt.figure()
# plt.imshow(o_[:,:,0],cmap="gray")
#
# import scipy.io as sio
# gt4ch_LL = gt4ch_LL[0,:,:,:]
# sio.savemat('gt4ch_ll.mat', {'gt_ll':gt4ch_LL})
# gt4ch_LL = gt4ch_LH[0,:,:,:]
# sio.savemat('gt4ch_lh.mat', {'gt_lh':gt4ch_LH})
# gt4ch_HL = gt4ch_HL[0,:,:,:]
# sio.savemat('gt4ch_hl.mat', {'gt_hl':gt4ch_HL})
# gt4ch_HH = gt4ch_HH[0,:,:,:]
# sio.savemat('gt4ch_hh.mat', {'gt_hh':gt4ch_HH})
#
# out_img_LL = out_img_LL[0,:,:,:]
# sio.savemat('out_ll.mat', {'ll':out_img_LL})
# out_img_LH = out_img_LH[0,:,:,:]
# sio.savemat('out_lh.mat', {'lh':out_img_LH})
# out_img_HL = out_img_HL[0,:,:,:]
# sio.savemat('out_hl.mat', {'hl':out_img_HL})
# out_img_HH = out_img_HH[0,:,:,:]
# sio.savemat('out_hh.mat', {'hh':out_img_HH})
#
# #histogram analysis
# counts, bins = np.histogram(gt4ch_LH[:,300:300+25,90:90+25,:], 256)
# plt.figure()
# plt.plot(bins[0:-1], counts)
# plt.grid()
#
# counts, bins = np.histogram(out_img_LH[:,300:300+25,90:90+25,:], 256)
# plt.figure()
# plt.plot(bins[0:-1], counts)
# plt.grid()
#
# counts, bins = np.histogram(10*out_img_LH[:,300:300+25,90:90+25,:], 256)
# plt.figure()
# plt.plot(bins[0:-1], counts)
# plt.grid()
#
# # after bluring
# blur = cv2.blur(o[0,:,:,:],(5,5))
# counts, bins = np.histogram(10*blur, 256)
# plt.figure()
# plt.plot(bins[0:-1], counts)
# plt.grid()
#
# ll = out_img_LL[0,300:300+25,90:90+25,:]
# lh = out_img_LH[0,300:300+25,90:90+25,:]
# hl = out_img_HL[0,300:300+25,90:90+25,:]
# hh = out_img_HH[0,300:300+25,90:90+25,:]
# ll_ = gt4ch_LL[0,300:300+25,90:90+25,:]
# lh_ = gt4ch_LH[0,300:300+25,90:90+25,:]
# hl_ = gt4ch_HL[0,300:300+25,90:90+25,:]
# hh_ = gt4ch_HH[0,300:300+25,90:90+25,:]