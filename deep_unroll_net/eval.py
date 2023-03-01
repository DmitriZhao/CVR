import os
import torch
import random
import argparse
import numpy as np
import cv2
import sys
os.environ["KMP_BLOCKTIME"] = "0"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from package_core.generic_train_test import *
from forward_warp_package import *
from dataloader import *
from model_CVR import *
from package_core.metrics import *
from package_core.flow_utils import *
from lpips import lpips

import imageio

os.environ["KMP_BLOCKTIME"] = "0"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, ignoregamma=True)
    else:
        return imageio.imread(f)


def load_results(img_path):
    imgfiles = [os.path.join(img_path, f) for f in sorted(os.listdir(img_path)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    print(f"Loaded {len(imgfiles)} results")
    gt_files = [f for f in imgfiles if "gs" == f.split("_")[-2]]
    pred_files = [f for f in imgfiles if "pred" == f.split("_")[-2]]
    print(f"Loaded {len(gt_files)} results, {len(pred_files)} pred_files")
    return gt_files, pred_files


def load_imgs(imgfiles):
    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    print(f"Loaded {len(imgs)} images")
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    imgs = torch.Tensor(imgs)

    return imgs.cuda()


def test(img_path):
    sum_psnr = 0.
    sum_psnr_mask = 0.
    sum_ssim = 0.
    sum_lpips = 0.
    sum_time = 0.
    f_metric_all = None
    f_metric_avg = None
    n_frames = 0

    gt_files, pred_files = load_results(img_path)

    GT = load_imgs(gt_files)
    predict = load_imgs(pred_files)

    f_metric_all = open(os.path.join(img_path, 'metric_all.txt'), 'w')
    f_metric_avg = open(os.path.join(img_path, 'metric_avg.txt'), 'w')

    f_metric_all.write(
        '# frame_id, PSNR_pred, PSNR_pred_mask, SSIM_pred, LPIPS_pred\n')
    f_metric_avg.write(
        '# avg_PSNR_pred, avg_PSNR_pred_mask, avg_SSIM_pred, avg_LPIPS_pred\n')

    loss_fn_alex = lpips.LPIPS(net='alex')

    batch_size = GT.shape[0]

    for i in range(batch_size):

        # compute metrics
        predict_GS = predict[i].permute(2, 0, 1).unsqueeze(0)
        GT_GS = GT[i].permute(2, 0, 1).unsqueeze(0)

        psnr_pred = PSNR(predict_GS, GT_GS)
        psnr_pred_mask = PSNR(predict_GS, GT_GS)
        ssim_pred = SSIM(predict_GS, GT_GS)

        lpips_pred = 0.
        lpips_pred = loss_fn_alex(predict_GS, GT_GS)  # compute LPIPS

        sum_psnr += psnr_pred
        sum_psnr_mask += psnr_pred_mask
        sum_ssim += ssim_pred
        sum_lpips += lpips_pred
        n_frames += 1

        print('PSNR(%.6f dB) PSNR_mask(%.6f dB) SSIM(%.6f) LPIPS(%.6f)\n' %
              (psnr_pred, psnr_pred_mask, ssim_pred, lpips_pred))
        f_metric_all.write('%d %.2f %.2f %.2f %.4f\n' % (
            i, psnr_pred, psnr_pred_mask, ssim_pred, lpips_pred))

        psnr_avg = sum_psnr / n_frames
        psnr_avg_mask = sum_psnr_mask / n_frames
        ssim_avg = sum_ssim / n_frames
        lpips_avg = sum_lpips / n_frames

    print('PSNR_avg (%.6f dB) PSNR_avg_mask (%.6f dB) SSIM_avg (%.6f) LPIPS_avg (%.6f) ' % (
        psnr_avg, psnr_avg_mask, ssim_avg, lpips_avg))
    f_metric_avg.write('%.6f %.6f %.6f %.6f\n' %
                       (psnr_avg, psnr_avg_mask, ssim_avg, lpips_avg))

    f_metric_all.close()
    f_metric_avg.close()


if __name__ == "__main__":
    test(sys.argv[1])
