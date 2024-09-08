import argparse
import os
from glob import glob
import pandas as pd
import yaml
from utils import str2bool, write_csv
from collections import OrderedDict
from sklearn.model_selection import train_test_split

from trainer import trainer, validate
from dataset import CustomDataset
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from metrics import Dice, IOU, HD

from networks.RotCAtt_TransUNet_plusplus.RotCAtt_TransUNet_plusplus import RotCAtt_TransUNet_plusplus
from networks.RotCAtt_TransUNet_plusplus.config import get_config as rot_config



def parse_args():     
    
    # Training pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--pretrained', default=False,
                        help='pretrained or not (default: False)')
    parser.add_argument('--epochs', default=600, type=int, metavar='N',
                        help='number of epochs for training')
    parser.add_argument('--batch_size', default=6, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--num_workers', default=3, type=int)
    parser.add_argument('--val_mode', default=True, type=str2bool)
    
    # Network
    parser.add_argument('--network', default='RotCAtt_TransUNet_plusplus') 
    parser.add_argument('--input_channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='input patch size')
    parser.add_argument('--num_classes', default=12, type=int,
                        help='number of classes')
    parser.add_argument('--img_size', default=512, type=int,
                        help='input image img_size')
    
    # Dataset
    parser.add_argument('--dataset', default='VHSCDD', help='dataset name')
    parser.add_argument('--ext', default='.npy', help='file extension')
    parser.add_argument('--range', default=None, type=int, help='dataset size')
    
     # Criterion
    parser.add_argument('--loss', default='Dice Iou Cross entropy')
    
    # Optimizer
    parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'],
                        help='optimizer: ' + ' | '.join(['Adam', 'SGD']) 
                        + 'default (Adam)')
    parser.add_argument('--base_lr', '--learning_rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    
    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 
                                 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    return parser.parse_args()

def output_config(config):
    print('-' * 20)
    for key in config:
        print(f'{key}: {config[key]}')
    print('-' * 20)   
   

def loading_2D_data(config):
    image_paths = glob(f"data/{config.dataset}/images/*.npy")
    label_paths = glob(f"data/{config.dataset}/labels/*.npy")
    
    if config.range != None: 
        image_paths = image_paths[:config.range]
        label_paths = label_paths[:config.range]
    
    train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(image_paths, label_paths, test_size=0.2, random_state=41)
    train_ds = CustomDataset(config.num_classes, train_image_paths, train_label_paths, img_size=config.img_size)
    val_ds = CustomDataset(config.num_classes, val_image_paths, val_label_paths, img_size=config.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader

    
def load_pretrained_model(model_path):
    if os.path.exists(model_path):
        model = torch.load(model_path)
        return model
    else:
        print("No pretrained exists")
        exit()
        

def load_network(config):
    if config.network == 'RotCAtt_TransUNet_plusplus':
        model_config = rot_config()
        model_config.img_size = config.img_size
        model_config.num_classes = config.num_classes
        model = RotCAtt_TransUNet_plusplus(config=model_config).cuda()
        
    else:
        print("Add the custom network to the training pipeline please")
        exit(1)
        
    return model

 
def rlog(value):
    return round(value, 3)
        
def train(config):
    config_dict = vars(config)
    print(config.network)
    
    # Config name
    config.name = f"{config.dataset}_{config.network}_bs{config.batch_size}_ps{config.patch_size}_epo{config.epochs}_hw{config.img_size}"

    # Model
    print(f"=> Initialize model: {config.network}")
    if config.pretrained == False: 
        model = load_network(config)
        output_config(config_dict)
        print(f"=> Initialize output: {config.name}")
        model_path = f"outputs/{config.name}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            with open(f"{model_path}/config.yml", "w") as f:
                yaml.dump(config_dict, f)
                
    else: model = load_pretrained_model(f'outputs/{config.name}/model.pth')
    
    # Data loading
    if config.dataset == 'VHSCDD': config.dataset += f'_{config.img_size}'
    train_loader, val_loader = loading_2D_data(config)
    
    # logging
    log = OrderedDict([
        ('epoch', []),                                          # 0
        ('lr', []),                                             # 1
        
        ('Train loss', []),                                     # 2
        ('Train ce loss', []),                                  # 3
        ('Train dice score', []),                               # 4
        ('Train dice loss', []),                                # 5
        ('Train iou score', []),                                # 6
        ('Train iou loss', []),                                 # 7
        ('Train hausdorff', []),                                # 8
        
        ('Val loss', []),                                       # 8
        ('Val ce loss', []),                                    # 9
        ('Val dice score', []),                                 # 10
        ('Val dice loss', []),                                  # 11
        ('Val iou score', []),                                  # 12
        ('Val iou loss', []),                                   # 13
        ('Val hausdorff', []),                                  # 14
    ])
    
    if config.pretrained: 
        pre_log = pd.read_csv(f'outputs/{config.name}/epo_log.csv')
        print(pre_log)
        log = OrderedDict((key, []) for key in pre_log.keys())
        for column in pre_log.columns:
           log[column] = pre_log[column].tolist()

    # Optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.base_lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=config.base_lr, momentum=config.momentum,
        nesterov=config.nesterov, weight_decay=config.weight_decay)
        
    # Criterion
    ce = CrossEntropyLoss()
    dice = Dice(config.num_classes)
    iou = IOU(config.num_classes)
    hd = HD()
    
    # Training loop
    best_train_iou = 0
    best_train_dice_score = 0
    best_val_iou = 0
    best_val_dice_score = 0
    
    fieldnames = ['CE Loss', 'Dice Score', 'Dice Loss', 'IoU Score', 'IoU Loss', 'HausDorff Distance', 'Total Loss']
    iter_log_file = f'outputs/{config.name}/iter_log.csv'
    if not os.path.exists(iter_log_file): 
        write_csv(iter_log_file, fieldnames)
        
    for epoch in range(config.epochs):
        print(f"Epoch: {epoch+1}/{config.epochs}")
        train_log = trainer(config, train_loader, optimizer, model, ce, dice, iou, hd)
        if config.val_mode: val_log = validate(config, val_loader, model, ce, dice, iou, hd)
        
        print(f"Train loss: {rlog(train_log['loss'])} - Train ce loss: {rlog(train_log['ce_loss'])} - Train dice score: {rlog(train_log['dice_score'])} - Train dice loss: {rlog(train_log['dice_loss'])} - Train iou Score: {rlog(train_log['iou_score'])} - Train iou loss: {rlog(train_log['iou_loss'])} - Train hausdorff: {rlog(train_log['hausdorff'])}")
        if config.val_mode: print(f"Val loss: {rlog(val_log['loss'])} - Val ce loss: {rlog(val_log['ce_loss'])} - Val dice score: {rlog(val_log['dice_score'])} - Val dice loss: {rlog(val_log['dice_loss'])} - Val iou Score: {rlog(val_log['iou_score'])} - Val iou loss: {rlog(val_log['iou_loss'])} - Val hausdorff: {rlog(val_log['hausdorff'])}")
        
        log['epoch'].append(epoch)
        log['lr'].append(config.base_lr)
        
        log['Train loss'].append(train_log['loss'])
        log['Train ce loss'].append(train_log['ce_loss'])
        log['Train dice score'].append(train_log['dice_score'])
        log['Train dice loss'].append(train_log['dice_loss'])
        log['Train iou score'].append(train_log['iou_score'])
        log['Train iou loss'].append(train_log['iou_loss']) 
        log['Train hausdorff'].append(train_log['hausdorff']) 
        
        if config.val_mode:
            log['Val loss'].append(val_log['loss'])
            log['Val ce loss'].append(val_log['ce_loss'])
            log['Val dice score'].append(val_log['dice_score'])
            log['Val dice loss'].append(val_log['dice_loss'])
            log['Val iou score'].append(val_log['iou_score'])
            log['Val iou loss'].append(val_log['iou_loss'])
            log['Val hausdorff'].append(val_log['hausdorff'])
            
        else:
            log['Val loss'].append(None)
            log['Val ce loss'].append(None)
            log['Val dice score'].append(None)
            log['Val dice loss'].append(None)
            log['Val iou score'].append(None)
            log['Val iou loss'].append(None)
            log['Val hausdorff'].append(None)

        
        pd.DataFrame(log).to_csv(f'outputs/{config.name}/epo_log.csv', index=False)

        # Save best model
        if train_log['iou_score'] > best_train_iou and train_log['dice_score'] > best_train_dice_score and val_log['iou_score'] > best_val_iou and val_log['dice_score'] > best_val_dice_score:
                
            best_train_iou = train_log['iou_score']
            best_train_dice_score = train_log['dice_score']
            best_val_iou = val_log['iou_score']
            best_val_dice_score = val_log['dice_score']
            
            torch.save(model, f"outputs/{config.name}/model.pth")
            
        if (epoch+1) % 1 == 0:
            print(f'BEST TRAIN DICE: {best_train_dice_score} - BEST TRAIN IOU: {best_train_iou} - BEST VAL DICE SCORE: {best_val_dice_score} - BEST VAL IOU: {best_val_iou}')
            
if __name__ == '__main__':
    config = parse_args()
    train(config)