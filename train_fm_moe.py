#import needed library
import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import net_builder, get_logger, count_parameters
from train_utils import TBLog, get_SGD, get_cosine_schedule_with_warmup
# from models.fixmatch.fixmatch import FixMatch
import models.moe as moe
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader

from config_utils import config_parser



def main(args):

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')
        
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    
    ngpus_per_node = torch.cuda.device_count() # number of gpus of each node
    
    main_worker(args.gpu, ngpus_per_node, args)
    

def main_worker(gpu, ngpus_per_node, args):

    global best_acc1
    args.gpu = gpu
    
    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    
    #SET save_path, tensorboard path and logger path
    save_path = os.path.join(args.save_dir, args.save_name)
   
    tb_log = TBLog(args.tb_dir, args.save_name)
    logger_level = "INFO"
    
    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")
    
    
    # mode;
    args.bn_momentum = 1.0 - args.ema_m
    _net_builder = net_builder(args.net, 
                               from_name = False,
                               net_conf={'depth': args.depth, 
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'bn_momentum': args.bn_momentum,
                                'dropRate': args.dropout})
    
    model = moe.FmMoeWrapper(_net_builder,
                             args.arch,
                             args.num_classes,
                             args.ema_m,
                             args.T,
                             args.p_cutoff,
                             args.ulb_loss_ratio,
                             args.ssl_ratio,
                             args.hard_label,
                             num_eval_iter=args.num_eval_iter,
                             tb_log=tb_log,
                             logger=logger)

    logger.info(f'Number of Trainable Params: {count_parameters(model.train_model)}')
        

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_SGD(model.train_model, 'SGD', args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_train_iter,
                                                num_warmup_steps=args.num_train_iter*0)
    ## set SGD and cosine lr on FixMatch 
    model.set_optimizer(optimizer, scheduler)
    
    
    # SET Devices for (Distributed) DataParallel
    if not torch.cuda.is_available():
        raise Exception('ONLY GPU TRAINING IS SUPPORTED')
 
    torch.cuda.set_device(args.gpu)
    model.train_model = model.train_model.cuda(args.gpu)
    model.eval_model = model.eval_model.cuda(args.gpu)
        

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {vars(args)}")
    
    cudnn.benchmark = True


    # Construct Dataset & DataLoader
    train_dset = SSL_Dataset(name=args.dataset, train=True, 
                             num_classes=args.num_classes, data_dir=args.data_dir)
    lb_dset, ulb_dset = train_dset.get_fmgssl_dataset(args.arch,num_labels=args.num_labels)
    
    _eval_dset = SSL_Dataset(name=args.dataset, train=False, 
                             num_classes=args.num_classes, data_dir=args.data_dir)
    eval_dset = _eval_dset.get_dset()
    
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}
    
    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
                                              args.batch_size,
                                              data_sampler = args.train_sampler,
                                              num_iters=args.num_train_iter,
                                              num_workers=args.num_workers, 
                                              distributed=False)
    
    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
                                               args.batch_size*args.uratio,
                                               data_sampler = args.train_sampler,
                                               num_iters=args.num_train_iter,
                                               num_workers=4*args.num_workers,
                                               distributed=False)
    
    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
                                          args.eval_batch_size, 
                                          num_workers=args.num_workers)
    
    ## set DataLoader on FixMatch
    model.set_data_loader(loader_dict)
    
    #If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path)
    
    # START TRAINING of FixMatch
    trainer = model.train
    for epoch in range(args.epoch):
        trainer(args, logger=logger)
        
    model.save_model('latest_model.pth', save_path)
        
    logging.warning(f"GPU training is FINISHED")
    
def path_correction(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Semi G-SSL Training')
    parser.add_argument('--path', '-p', type=str, help='config path')
    cli_parser = parser.parse_args()

    config = config_parser(path_correction('config/fmgssl_cifar10_4000_default.yaml'), cli_parser.path)
    args = config.get()

    # set save_name
    args.save_name += f'_{args.arch}_{args.dataset}_{args.num_labels}'
    main(args)
