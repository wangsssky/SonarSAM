#coding:utf-8

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

import torch.optim as opt
import os
import time
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import shutil

from model.model_proxy_SAM_box import SonarSAM, ModelWithLoss
from evaluate_box import evaluate
from utils.config import Config_SAM
from utils.logger import Logger
from utils.utils import rand_seed
from dataloader.data_loader import SAM_DebrisDataset, collate_fn_seq_box_seg_pair
from model.segment_anything.utils.transforms import ResizeLongestSide


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path (*.yaml)", required=True)
    parser.add_argument("--save_path", type=str, help="save path", default='')
    args = parser.parse_args()
    opt = Config_SAM(config_path=args.config)

    rand_seed(opt.RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log & model folder
    if args.save_path == '':
        opt.MODEL_DIR += '{}_{}'.format(opt.MODEL_NAME, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        opt.MODEL_DIR = args.save_path

    if not os.path.exists(opt.MODEL_DIR):
        os.mkdir(opt.MODEL_DIR)

    logger = Logger(opt.MODEL_NAME, path=opt.MODEL_DIR)

    if not os.path.exists(os.path.join(opt.MODEL_DIR, 'params.yaml')):
        shutil.copy(args.config, os.path.join(opt.MODEL_DIR, 'params.yaml'))

    # dataset
    train_dataset = SAM_DebrisDataset(root_path=opt.DATA_PATH, image_list=os.path.join(opt.IMAGE_LIST_PATH, 'train.txt'),
                                  input_size=opt.INPUT_SIZE, use_augment=True)
    val_dataset = SAM_DebrisDataset(root_path=opt.DATA_PATH, image_list=os.path.join(opt.IMAGE_LIST_PATH, 'val.txt'),
                                  input_size=opt.INPUT_SIZE, use_augment=False)

    train_loader = DataLoader(train_dataset, batch_size=opt.TRAIN_BATCHSIZE, shuffle=True, 
                              num_workers=opt.TRAIN_BATCHSIZE, collate_fn=collate_fn_seq_box_seg_pair)
    val_loader = DataLoader(val_dataset, batch_size=opt.VAL_BATCHSIZE, shuffle=False, 
                            num_workers=opt.VAL_BATCHSIZE*2, collate_fn=collate_fn_seq_box_seg_pair)    

    rand_seed(opt.RANDOM_SEED)

    # Training Config
    epochs = opt.EPOCH_NUM
    epoch_start = 0

    net = SonarSAM(model_name=opt.SAM_NAME, checkpoint=opt.SAM_CHECKPOINT, num_classes=opt.OUTPUT_CHN, 
                   is_finetune_image_encoder=opt.IS_FINETUNE_IMAGE_ENCODER,
                   use_adaptation=opt.USE_ADAPTATION, 
                   adaptation_type=opt.ADAPTATION_TYPE,
                   head_type=opt.HEAD_TYPE,
                   reduction=4, upsample_times=2, groups=4)
    net = ModelWithLoss(net)

    if opt.OPTIMIZER == 'ADAM':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=opt.LEARNING_RATE, weight_decay=opt.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(
            net.parameters(), lr=opt.LEARNING_RATE, momentum=opt.MOMENTUM,
            weight_decay=opt.WEIGHT_DECAY, nesterov=True)

    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * opt.WARM_LEN)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume
    if opt.RESUME_FROM > 0:
        ckpt = torch.load(
            os.path.join(opt.MODEL_DIR, '{}_{}.pth'.format(opt.MODEL_NAME, opt.RESUME_FROM)))
        net.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt.keys():
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt.keys():
            scheduler.load_state_dict(ckpt['scheduler'])
        epoch_start = opt.RESUME_FROM
    
    

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(device)

    best_score = 0
    # Training
    for epoch in range(epoch_start, epochs):
        net.train()
        print_str = '-------epoch {}/{}-------'.format(epoch+1, epochs)
        logger.write_and_print(print_str)
        start_t = time.time()

        for step, (image, box_mask_pairs) in enumerate(tqdm(train_loader)):
            image = image.to(device)

            # print('image shape', patch.shape)
            # print('GT shape', mask.shape)

            # prepare data
            optimizer.zero_grad()
            
            boxes_batch = []
            masks_batch = []
            box_mask_pairs = box_mask_pairs[0]
            for idx in range(len(box_mask_pairs)):
                boxes_item = box_mask_pairs[idx]['boxes']
                masks_item = box_mask_pairs[idx]['masks']
                boxes_xyxy = []
                masks = []
                for i in range(len(boxes_item)):
                    box = boxes_item[i]
                    box = box[:4]
                    jitter = np.random.randint(low=0, high=10, size=4)
                    box += jitter    
                    boxes_xyxy.append(box)            
                    masks.append(masks_item[i].cuda())
                boxes_xyxy = np.array(boxes_xyxy)   
                H, W = image.shape[-2], image.shape[-1]
                sam_trans = ResizeLongestSide(net.model.sam.image_encoder.img_size)
                boxes_trans = sam_trans.apply_boxes(boxes_xyxy, (H, W))
                boxes_trans = torch.as_tensor(boxes_trans, dtype=torch.float, device=device)
            
                boxes_batch.append(boxes_trans)
                masks_batch.append(masks)               
                
            loss, outputs = net.forward(image, masks_batch, boxes=boxes_batch)
            

            if torch.cuda.device_count() > 1:
                loss = loss.sum()
            
            if torch.isnan(loss):
                logger.write_and_print('***** Warning: loss is NaN *****')
                loss = torch.tensor(10000).to(device)

            # print loss
            print_str = '# total loss: {}\n'.format(loss.item())
            if opt.PRT_LOSS:
                logger.write_and_print(print_str)
            else:
                logger.write(print_str)

            loss.backward()
            optimizer.step()

            if epoch <= opt.WARM_LEN:
                warmup_scheduler.step()
                # print('lr', optimizer.param_groups[0]['lr'])
        end_t = time.time()
        duration = end_t-start_t
        logger.write_and_print('time: {}'.format(duration))

        # log learning_rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # save model
        if torch.cuda.device_count() > 1:
            ckpt = {
                'epoch': epoch + 1,                
                'state_dict': net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
        else:
            ckpt = {
                'epoch': epoch + 1,                
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
        torch.save(ckpt, os.path.join(opt.MODEL_DIR,
                                      '{}_{}.pth'.format(opt.MODEL_NAME, epoch+1)))

        # evaluate each epoch 
        if torch.cuda.device_count() > 1:
            metrics_dict = evaluate(net.module.model, val_loader, device, opt)
        else:       
            metrics_dict = evaluate(net.model, val_loader, device, opt)
        if metrics_dict['avg(exclude_bg)'] > best_score:
            best_score = metrics_dict['avg(exclude_bg)']
            torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_best.pth'.format(opt.MODEL_NAME)))
        print_str = ''
        for key in metrics_dict.keys():
            print_str += key + ':\t{:.2f}\n'.format(metrics_dict[key]*100)
        logger.write_and_print(print_str)

    # evaluate final
    test_dataset = SAM_DebrisDataset(root_path=opt.DATA_PATH, image_list=os.path.join(opt.IMAGE_LIST_PATH, 'test.txt'),
                                  input_size=opt.INPUT_SIZE, use_augment=False)
    test_loader = DataLoader(test_dataset, batch_size=opt.VAL_BATCHSIZE, shuffle=False, 
                             num_workers=opt.VAL_BATCHSIZE*2, collate_fn=collate_fn_seq_box_seg_pair)
    
    ckpt = torch.load(
        os.path.join(opt.MODEL_DIR, '{}_best.pth'.format(opt.MODEL_NAME)))
    
    net = SonarSAM(model_name=opt.SAM_NAME, checkpoint=opt.SAM_CHECKPOINT, num_classes=opt.OUTPUT_CHN, 
                   is_finetune_image_encoder=opt.IS_FINETUNE_IMAGE_ENCODER,
                   use_adaptation=opt.USE_ADAPTATION, adaptation_type=opt.ADAPTATION_TYPE,
                   head_type=opt.HEAD_TYPE,
                   reduction=4, upsample_times=2, groups=4)
    net = ModelWithLoss(net)
    net.load_state_dict(ckpt['state_dict'])
    
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)    
        
    net.to(device)
    rand_seed(opt.RANDOM_SEED)

    if torch.cuda.device_count() > 1:
        metrics_dict = evaluate(net.module.model, test_loader, device, opt)
    else:       
        metrics_dict = evaluate(net.model, test_loader, device, opt)

    logger.write_and_print("Dice on Test set:")
    for key in metrics_dict.keys():
        logger.write_and_print("{}:\t{:.2f}".format(key, metrics_dict[key]*100))


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
