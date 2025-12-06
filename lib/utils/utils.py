# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn), Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim
from torchvision.utils import make_grid
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def save_checkpoint(states, predictions, is_best,
                    output_dir, filename='checkpoint.pth'):
    preds = predictions.cpu().data.numpy()
    torch.save(states, os.path.join(output_dir, filename))
    torch.save(preds, os.path.join(output_dir, 'current_pred.pth'))

    latest_path = os.path.join(output_dir, 'latest.pth')
    if os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.join(output_dir, filename), latest_path)

    if is_best and 'state_dict' in states.keys():
        torch.save(states['state_dict'].module, os.path.join(output_dir, 'model_best.pth'))

def visualize_heatmap(config, image, heatmap, save_path=None):
    """
    将模型输出的热图转换为可视化图像
    Args:
        config: 配置参数
        image: 输入图像（B,C,H,W）
        heatmap: 模型输出热图（B,K,H,W）
        save_path: 保存路径（None则不保存）
    Returns:
        vis_img: 可视化结果（B,H,W,3）
    """
    # 上采样到原图尺寸
    heatmap = nn.functional.interpolate(
        heatmap, size=image.shape[2:], mode='bilinear', align_corners=False
    )
    
    # 归一化到0-1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # 转换为颜色图
    num_joints = heatmap.shape[1]
    cmap = plt.get_cmap('jet')
    heatmap_colored = np.uint8(cmap(heatmap.squeeze().numpy())[:, :, :, :3] * 255)
    
    # 叠加在原图上（可选）
    vis_img = image.permute(0, 2, 3, 1).numpy()  # B,H,W,C
    alpha = 0.5
    for i in range(num_joints):
        vis_img[:, :, :, :] = cv2.addWeighted(
            vis_img[:, :, :, :], 
            1 - alpha, 
            heatmap_colored[i], 
            alpha, 
            0
        )
    
    # 转换为TensorBoard可显示格式
    grid_img = make_grid(torch.from_numpy(vis_img), nrow=num_joints)
    
    if save_path:
        cv2.imwrite(save_path, vis_img[0])  # 保存单张示例
    
    return grid_img