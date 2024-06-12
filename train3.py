# python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 20003 train_spup3.py

import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle
import dataset

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from networks.TransBTS.TransBTS_aux_GR import TransBTS
from networks.TransBTS.GradientReversal import *
import torch.distributed as dist
from networks import criterions

from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import nn

from dataset.build_dataset import *


local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='name of user', type=str)

parser.add_argument('--experiment', default='TransBTS', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBTS,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='path to training set', type=str)

parser.add_argument('--train_dir', default='Train', type=str)

parser.add_argument('--valid_dir', default='Valid', type=str)

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--train_file', default='train.txt', type=str)

parser.add_argument('--valid_file', default='valid.txt', type=str)

parser.add_argument('--dataset', default='breast2', type=str)

parser.add_argument('--model_name', default='TransBTS', type=str)

parser.add_argument('--input_C', default=1, type=int)

parser.add_argument('--input_H', default=256, type=int)

parser.add_argument('--input_W', default=256, type=int)

#parser.add_argument('--input_D', default=160, type=int)

parser.add_argument('--crop_H', default=256, type=int)

parser.add_argument('--crop_W', default=256, type=int)

#parser.add_argument('--crop_D', default=128, type=int)

#parser.add_argument('--output_D', default=155, type=int)

# Training Information
parser.add_argument('--lr', default=0.0002, type=float)

parser.add_argument('--weight_decay', default=1e-5, type=float)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)

#parser.add_argument('--num_class', default=4, type=int)

parser.add_argument('--seed', default=1000, type=int)

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='0,1,2,3', type=str)

#parser.add_argument('--num_workers', default=8, type=int)

parser.add_argument('--batch_size', default=8, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--end_epoch', default=120, type=int)

parser.add_argument('--save_freq', default=5, type=int)

parser.add_argument('--resume', default='', type=str)

parser.add_argument('--load', default=False, type=bool)

#parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()


def main_worker():
    #if args.local_rank == 0:
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log_aux3_1', args.experiment+args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)
    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    #torch.distributed.init_process_group('nccl')
    #torch.cuda.set_device(args.local_rank)
    device_num = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("DEVICE INFO:", device_num)

    dataset_name = 'breast2'
    data_loaders, data_sizes = build_breast_dataset(dataset_name=dataset_name, batch_size=args.batch_size)
    train_loader = data_loaders['train']

    #gradient_reverse = False
    _, model = TransBTS(dataset=dataset_name, _conv_repr=True, _pe_type="learned")
    

    #model.cuda(args.local_rank)
    #model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
    #                                            find_unused_parameters=True)
    model.to(device_num)
    # model.train()

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    #criterion = getattr(criterions, args.criterion)

    #if args.local_rank == 0:
    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint_aux3', args.experiment+args.date)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    resume = ''

    writer = SummaryWriter()

    if os.path.isfile(resume) and args.load:
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(args.resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)

    #train_set = BraTS(train_list, train_root, args.mode)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    #logging.info('Samples for train = {}'.format(len(train_set)))


    #num_gpu = (len(args.gpu)+1) // 2
    num_gpu = 1

    # train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size // num_gpu,
    #                           drop_last=True, num_workers=args.num_workers, pin_memory=True)
    

    start_time = time.time()

    torch.set_grad_enabled(True)
    mse_crit = nn.MSELoss()
    bce_crit = nn.BCELoss()

    for epoch in range(args.start_epoch, args.end_epoch):
        #train_sampler.set_epoch(epoch)  # shuffle
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch+1, args.end_epoch))
        start_epoch = time.time()
        total_loss = 0
        mse_loss = 0
        bce_loss = 0
        gr_loss = 0
        
        for i, data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)

            # # This approach output only 1 label 0 for no augmenbtation, 1 for augmentation
            x, label = data

            x = x.to(device_num)
            label = label.to(device_num)
            #target_labels = torch.cat((label.unsqueeze(1).float()), dim=1)
            target_labels = label.unsqueeze(1).float()

            # id data for MSE, MLP for both 
            model.set_reverse(False)
            x_id = x[label==0]
            mse_id = 0
            if x_id.shape[0] != 0: 
                output_id, z_id = model(x_id)
                mse_id = mse_crit(output_id, x_id)
                print(mse_id)
                # if mse_crit(output_id, x_id) > 3:
                #     exit()

            output, z_out = model(x)
            loss = 0.75*mse_id + 0.25*bce_crit(z_out, target_labels)
            loss.backward()
            # optimizer.step()

            # ood 
            x_ood = x[label==1]
            print(x_ood.shape)
            if x_ood.shape[0] != 0 and epoch > 20: 
                model.set_reverse(True)
                output_ood, z_out_ood = model(x_ood)
                loss_GR = 0.25*mse_crit(output_ood, x_ood) # 
                loss_GR.backward()
                gr_loss += loss_GR
            optimizer.step()
                
            
            total_loss += loss
            mse_loss += 0.75*mse_id
            bce_loss += 0.25*bce_crit(z_out, target_labels)
            
            
            
            logging.info('Epoch: {}_Iter:{}  loss: {:.5f} '
                        .format(epoch, i, loss))

            if i % 100 == 0:
                vutils.save_image(vutils.make_grid(x, nrow=4, normalize=True, scale_each=True), './log_aux3_1/real_samples'+str(epoch)+'.png')
                vutils.save_image(vutils.make_grid(output, nrow=4, normalize=True, scale_each=True), './log_aux3_1/fake_samples'+str(epoch)+'.png')
            #print("OOD check")
        
        loss_avg = total_loss/(len(train_loader))
        mse_avg = mse_loss/(len(train_loader))
        bce_avg = bce_loss/(len(train_loader))
        gr_avg = gr_loss/(len(train_loader)-20)
        end_epoch = time.time()
        if mse_avg > 2:
            exit()
        writer.add_scalar('lr:', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Total loss:', loss_avg, epoch)
        writer.add_scalar("MSE loss", mse_avg, epoch)
        writer.add_scalar("BCE loss", bce_avg, epoch)
        writer.add_scalar("GR MSE loss", gr_avg, epoch)

        writer.add_images("Input", x, epoch)
        writer.add_images("Reconstruct", output, epoch)
        # writer.add_images("Input OOD", x_ood, epoch)
        # writer.add_images("Reconstruct OOD", output_ood, epoch)
        #if args.local_rank == 0:
        if (epoch + 1) % int(args.save_freq) == 0 :
            file_name = os.path.join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

        epoch_time_minute = (end_epoch-start_epoch)/60
        remaining_time_hour = (args.end_epoch-epoch-1)*epoch_time_minute/60
        logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
        logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))
    

    #if args.local_rank == 0:
    writer.close()

    final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': args.end_epoch,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    },
        final_name)
    end_time = time.time()
    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))

    logging.info('----------------------------------The training process finished!-----------------------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)


def log_args(log_file):

    logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # # args FileHandler to save log file
    # fh = logging.FileHandler(log_file)
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)

    # # args StreamHandler to print log to console
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter)

    # # add the two Handler
    # logger.addHandler(ch)
    # logger.addHandler(fh)


if __name__ == '__main__':
    main_worker()
