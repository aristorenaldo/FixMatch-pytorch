import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import os
import contextlib
from train_utils import AverageMeter

from train_utils import ce_loss

from .moe import Moe1, Nomoe, Lorot

__all__ = ['FmMoeWrapper']

arch_dict ={
    'Moe1': Moe1,
    'Nomoe': Nomoe,
    'Lorot': Lorot,
}

import torch
import torch.nn.functional as F
from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value


def consistency_loss(logits_w, logits_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float() # greater than p_cutoff
        
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')

class FmMoeWrapper():
    """
    Model wraper for Fixmatch Moe
    containts model, loader, optimizer and update methods.

    Args:
        net_builder: backbone network class (see net_builder in utils.py)
        num_classes: # of label classes 
        ema_m: momentum of exponential moving average for eval_model
        T: Temperature scaling parameter for output sharpening (only when hard_label = False)
        p_cutoff: confidence cutoff parameters for loss masking
        lambda_u: ratio of unsupervised loss to supervised loss
        ssl_ratio: ratio self-supervised loss
        hard_label: If True, consistency regularization use a hard pseudo label.
        it: initial iteration count
        num_eval_iter: freqeuncy of iteration (after 500,000 iters)
        tb_log: tensorboard writer (see train_utils.py)
        logger: logger (see utils.py)
    """
    def __init__(self, net_builder, arch, num_classes, ema_m, T, p_cutoff, lambda_u, ssl_ratio, \
                 hard_label=True, it=0, num_eval_iter=1000, tb_log=None, logger=None) -> None:
        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        
        self.train_model = arch_dict[arch](net_builder, num_classes=num_classes) 
        self.eval_model = arch_dict[arch](net_builder, num_classes=num_classes)
        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T) #temperature params function
        self.p_fn = Get_Scalar(p_cutoff) #confidence cutoff function
        self.lambda_u = lambda_u
        self.ssl_ratio = ssl_ratio
        self.tb_log = tb_log
        self.use_hard_label = hard_label
        
        self.optimizer = None
        self.scheduler = None
        
        self.it = 0
        
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        
        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
            
        self.eval_model.eval()

    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        if hasattr(self.train_model, 'module'):
            for param_train, param_eval in zip(self.train_model.module.parameters(), self.eval_model.parameters()):
                param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        else:
            for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
                param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        
        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')    
            
    
    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __train_gssl(self, arch, input, ssl_label, return_sup=True):
        """
        training for gated self supervised learning
        """
        ssl_loss = 0
        ssl_dict ={}
        # choose arch
        if arch == "Moe1": # use rot flip sc
            sup_output, rot_output, flip_output, sc_output, gate_output = self.train_model(input)
            gate = torch.mean(F.softmax(gate_output, dim=1),dim=0)
            ssl_loss = (ce_loss(rot_output, ssl_label[0], reduction='mean') * gate[0].item()+
                        ce_loss(flip_output, ssl_label[1], reduction='mean') * gate[1].item()+
                        ce_loss(sc_output, ssl_label[2], reduction='mean') * gate[2]).item()
            ssl_dict['train/gate'] = {
                'rot': gate[0].detach(),
                'flip': gate[1].detach(),
                'sc': gate[2].detach(),
            }
            
        elif arch == "Nomoe": # use 2 task
            sup_output, rot_output, flip_output, sc_output = self.train_model(input)
            ssl_loss = (ce_loss(rot_output, ssl_label[0], reduction='mean') +
                        ce_loss(flip_output, ssl_label[1], reduction='mean') +
                        ce_loss(sc_output, ssl_label[2], reduction='mean') )
            
        elif arch == "Lorot":
            sup_output, rot_output = self.train_model(input)
            ssl_loss = ce_loss(rot_output, ssl_label, reduction='mean')
            
        else:
            self.logger.warning("training without gssl because arch error")

        if return_sup:
            return sup_output, ssl_loss, ssl_dict
        return ssl_loss, ssl_dict
    
    def train(self, args, logger=None):
        """
        train function for fixmatch moe
        """
        ngpus_per_node = torch.cuda.device_count()

        #lb: labeled, ulb: unlabeled
        self.train_model.train()
        
        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        start_batch.record()
        best_eval_acc, best_it = 0.0, 0
        
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        for ((x_lb, ssl_lb), y_lb), (x_ulb_w, x_ulb_s, _) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            # data to gpu
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]
            
            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)

            ssl_lb = torch.stack(ssl_lb).cuda(args.gpu) if isinstance(ssl_lb, (tuple, list)) else ssl_lb.cuda(args.gpu)
            inputs_ulb = torch.cat((x_ulb_w, x_ulb_s))
            # torch.autograd.set_detect_anomaly(True)
            with amp_cm():
                # supervised
                logits_lb, ssl_loss, ssl_dict = self.__train_gssl(args.arch, x_lb, ssl_lb, return_sup=True)
                logits_ulb = self.train_model(inputs_ulb, ssl_task=False)
                logits_x_ulb_w, logits_x_ulb_s = logits_ulb.chunk(2)
                
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)

                sup_loss = ce_loss(logits_lb, y_lb, reduction='mean')
                unsup_loss, mask = consistency_loss(logits_x_ulb_w, 
                                              logits_x_ulb_s, 
                                              'ce', T, p_cutoff,
                                               use_hard_labels=args.hard_label)
                total_loss = sup_loss + self.lambda_u * unsup_loss + self.ssl_ratio * ssl_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()
            self.train_model.zero_grad()
            
            with torch.no_grad():
                self._eval_model_update()
            
            end_run.record()
            torch.cuda.synchronize()

            #tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach()
            tb_dict['train/unsup_loss'] = unsup_loss.detach() 
            tb_dict['train/gssl_loss'] = ssl_loss 
            tb_dict['train/total_loss'] = total_loss.detach()
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach()  
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch)/1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run)/1000.

            tb_dict.update(ssl_dict)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)
                
                save_path = os.path.join(args.save_dir, args.save_name)
                
                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it
                
                self.print_fn(f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)
                
                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)
                
            self.it +=1
            del tb_dict
            start_batch.record()
            if self.it > 2**19:
                self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict
    
    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        use_ema = hasattr(self, 'eval_model')
        
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        for x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = eval_model(x, ssl_task=False)
            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)
            
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()
        
        if not use_ema:
            eval_model.train()
            
        return {'eval/loss': total_loss/total_num, 'eval/top-1-acc': total_acc/total_num}
    
    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)
        
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)
        
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        
        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key]) 
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")   
