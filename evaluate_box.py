import torch
from torch.utils.data import DataLoader

import numpy as np

import os
import argparse
from tqdm import tqdm

from utils.config import *
# from model.model_proxy import Proxy, ModelWithLoss
from model.model_proxy_SAM_box import SonarSAM, ModelWithLoss
from model.loss_functions import compute_dice_accuracy, compute_multilabel_dice_accuracy, compute_multilabel_IoU
from dataloader.data_loader import SAM_DebrisDataset, collate_fn_seq_box_seg_pair
from utils.utils import rand_seed
from model.segment_anything.utils.transforms import ResizeLongestSide


label_list = ["Background", "Bottle", "Can", "Chain", "Drink-carton", "Hook", 
              "Propeller", "Shampoo-bottle", "Standing-bottle", "Tire", "Valve", 
              "Wall"]

def evaluate(net, val_loader, device, opt):
    dice_ = [[], [], [], [], [], [], [], [], [], [], [], []]
    net.eval()
    with torch.no_grad():
        for val_step, (images, box_mask_pairs) in enumerate(tqdm(val_loader)):        
            images = images.to(device)
            
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
                    boxes_xyxy.append(box)            
                    masks.append(masks_item[i].cuda())
                boxes_xyxy = np.array(boxes_xyxy)   
                H, W = images.shape[-2], images.shape[-1]
                sam_trans = ResizeLongestSide(net.sam.image_encoder.img_size)
                boxes_trans = sam_trans.apply_boxes(boxes_xyxy, (H, W))
                boxes_trans = torch.as_tensor(boxes_trans, dtype=torch.float, device=device)
            
                boxes_batch.append(boxes_trans)
                masks_batch.append(masks)               
                
            predictions = net.forward(images, boxes_batch)
            start_x = int(opt.INPUT_SIZE / 3.0) // 2
            end_x = opt.INPUT_SIZE -1 -start_x
            # masks = masks[:, :, :, start_x:end_x].contiguous()
            # predictions = predictions[:, :, :, start_x:end_x].contiguous()

            # eval metric                                        
            for idx in range(len(box_mask_pairs)):
                boxes_item = box_mask_pairs[idx]['boxes']
                masks = masks_batch[idx]
                pred_masks = predictions[idx]
                for i in range(len(boxes_item)):
                    box = boxes_item[i]
                    label = box[-1]
                    # print(masks[i].shape, pred_masks[i].shape)
                    dice_iter = compute_dice_accuracy(masks[i][:, start_x:end_x].unsqueeze(0).contiguous(), 
                                                      (torch.sigmoid(pred_masks[i])>0.5)[:, start_x:end_x].unsqueeze(0).contiguous())
                    dice_[label].append(dice_iter.cpu().item())             

    # store in dict
    avg_list = []    
    metrics_dict = {}
    for i in range(len(label_list)):
        if len(dice_[i]) == 0:
            d = torch.tensor(0)
        else:
            d = torch.mean(torch.tensor(dice_[i]))
        metrics_dict[label_list[i]] = d
        avg_list.append(d)
    metrics_dict['avg'] = torch.mean(torch.tensor(avg_list))
    metrics_dict['avg(exclude_bg)'] = torch.mean(torch.tensor(avg_list[1:]))

    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path (*.yaml)", required=True)
    parser.add_argument("--save_path", type=str, help="save path", required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # opt = Config(config_path=args.config)
    opt = Config_SAM(config_path=args.config)

    # dataset
    test_dataset = SAM_DebrisDataset(root_path=opt.DATA_PATH, image_list=os.path.join(opt.IMAGE_LIST_PATH, 'test.txt'),
                                input_size=opt.INPUT_SIZE, use_augment=False)
    test_loader = DataLoader(test_dataset, batch_size=opt.VAL_BATCHSIZE, shuffle=False, collate_fn=collate_fn_seq_box_seg_pair)

    rand_seed(opt.RANDOM_SEED)

    net = SonarSAM(model_name=opt.SAM_NAME, checkpoint=opt.SAM_CHECKPOINT, num_classes=opt.OUTPUT_CHN, 
                   is_finetune_image_encoder=opt.IS_FINETUNE_IMAGE_ENCODER,
                   use_adaptation=opt.USE_ADAPTATION, adaptation_type=opt.ADAPTATION_TYPE,
                   head_type=opt.HEAD_TYPE,
                   reduction=4, upsample_times=2, groups=4)
    net = ModelWithLoss(net)

    ckpt = torch.load(os.path.join(args.save_path, '{}_best.pth'.format(opt.MODEL_NAME)))
    net.load_state_dict(ckpt['state_dict'])

    net.to(device)

    metrics_dict = evaluate(net.model, test_loader, device, opt)
    print("Dice on Test set:")
    for key in metrics_dict.keys():
        print("{}:\t{:.2f}".format(key, metrics_dict[key]*100))
