

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pprint
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')) 
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset, WFLW, UnlabeledWFLW
from lib.core import function
from lib.utils import utils
from torchvision.transforms import Compose, RandomApply, ColorJitter, GaussianBlur, ToTensor, Normalize


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.get_face_alignment_net(config)

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # loss
    criterion = torch.nn.MSELoss(reduction='mean').cuda()

    optimizer = utils.get_optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.TRAIN.LR_SCHEDULER.T_MAX,
    eta_min=config.TRAIN.COSINE_ETA_MIN
    )   
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'latest.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    dataset_type = get_dataset(config)
    sup_cfg = {**config, 'DATASET.TRAINSET': 'data/ct/landmarks_train.csv'}
    unsup_cfg = {**config, 'DATASET.TRAINSET': 'data/ct/landmarks_unsup.csv'}

    # 创建数据集实例
    # 创建监督数据集（带标签）
    sup_dataset = WFLW(sup_cfg, is_train=True)

    # 创建无监督数据集（不带标签）
    unsup_dataset = UnlabeledWFLW(
        img_dir="data/ct/images/uncropped",
        transform=Compose([...])
    )

    # 合并有监督和无监督数据集
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset([sup_dataset, unsup_dataset])

    # 使用自定义collate函数处理不同数据结构
    def custom_collate(batch):
        images = []
        targets = []
        metas = []
        for item in batch:
            if len(item) == 3:  # 有监督样本 (image, target, meta)
                images.append(item[0])
                targets.append(item[1])
                metas.append(item[2])
            elif len(item) == 1:  # 无监督样本 (image,)
                images.append(item[0])
                targets.append(torch.zeros(0))  # 填充虚拟目标
                metas.append({})  # 填充虚拟元数据
            else:
                raise ValueError("Invalid sample format")
        return torch.stack(images), torch.stack(targets), metas

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate  # 关键修改点
    )
    val_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        

        function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict)
        

        scheduler.step()
        # evaluate
        nme, predictions = function.validate(config, val_loader, model,
                                             criterion, epoch, writer_dict)

        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best:", is_best)
        utils.save_checkpoint(
            {"state_dict": model,
             "epoch": epoch + 1,
             "best_nme": best_nme,
             "optimizer": optimizer.state_dict(),
             }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
