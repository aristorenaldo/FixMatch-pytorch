import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import math
import random
import itertools
import time
import os


class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """
    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))
    
    def update(self, tb_dict, it, suffix=None):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            it: contains information of iteration (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        
        for key, value in tb_dict.items():
            if isinstance(value, dict):
                self.writer.add_scalars(suffix+key, value, it)
            else: 
                self.writer.add_scalar(suffix+key, value, it)         

            
class AverageMeter(object):
    """
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def get_SGD(net, name='SGD', lr=0.1, momentum=0.9, \
                  weight_decay=5e-4, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.
    If bn_wd_skip, the optimizer does not apply
    weight decay regularization on parameters in batch normalization.
    '''
    optim = getattr(torch.optim, name)
    
    decay = []
    no_decay = []
    for name, param in net.named_parameters():
        if ('bn' in name) and bn_wd_skip:
            no_decay.append(param)
        else:
            decay.append(param)
    
    per_param_args = [{'params': decay},
                      {'params': no_decay, 'weight_decay': 0.0}]
    
    optimizer = optim(per_param_args, lr=lr,
                    momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return optimizer
        
        
def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''
        
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    
    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    
    Args
        output: logits or probs (num of batch, num of classes)
        target: (num of batch, 1) or (num of batch, )
        topk: list of returned k
    
    refer: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    
    with torch.no_grad():
        maxk = max(topk) #get k in top-k
        batch_size = target.size(0) #get batch size of target

        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
        # return: value, index
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True) # pred: [num of batch, k]
        pred = pred.t() # pred: [k, num of batch]
        
        #[1, num of batch] -> [k, num_of_batch] : bool
        correct = pred.eq(target.view(1, -1).expand_as(pred)) 

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        #np.shape(res): [k, 1]
        return res 

    
def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return F.cross_entropy(logits, targets, reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss

def shuffle_channel(img: torch.Tensor, index_shuffle: int) -> torch.Tensor:
    """Mengacak urutan dimensi RGB sebagai bentuk transformasi

    Parameters
    ----------
    img : torch.Tensor
        Pixel image RGB

    index_shuffle : int
        Index pengacakan berdasarkan kombinasi RGB
    Returns
    -------
    torch.Tensor
        Shuffled result image
    """
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)

    list_to_permutations = list(itertools.permutations(range(3), 3))
    return img[list_to_permutations[index_shuffle], ...]

class SslTransform(object):
    '''
    Wrapper for SSL Transformation

    Parameter
    ---------
    arch : architecture of Gated-SSL to determine which transformations is used
        (Moe1, Lorot, Moe1Sc, Moe1Flip)
    
    Returns
    -------
    image : torch.Tensor
        Transformed image
    ssl_labels : int | tupple
        SSL label
    '''
    def __init__(self, arch) -> None:
        assert isinstance(arch,str)
        assert arch
        self.arch = arch

    def __transform_all(self, image):
        # print(image.shape)
        flip_label = random.randint(0, 1)
        sc_label = random.randint(0, 5)

        idx = random.randint(0, 3)
        idx2 = random.randint(0, 3)
        r2 = image.size(1)
        r = r2 // 2
        
        if idx == 0:
            w1 = 0
            w2 = r
            h1 = 0
            h2 = r
        elif idx == 1:
            w1 = 0
            w2 = r
            h1 = r
            h2 = r2
        elif idx == 2:
            w1 = r
            w2 = r2
            h1 = 0
            h2 = r
        elif idx == 3:
            w1 = r
            w2 = r2
            h1 = r
            h2 = r2
        if flip_label:
            image[:, w1:w2, h1:h2] = TF.hflip(image[:, w1:w2, h1:h2])
        # lorot
        image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
        # shuffle channel
        image[:, w1:w2, h1:h2] = shuffle_channel(image[:, w1:w2, h1:h2], sc_label)

        rot_label = idx * 4 + idx2
        ssl_label = (rot_label, flip_label, sc_label)

        return image, ssl_label
    
    def __transform_picker(self, image):
        idx = random.randint(0, 3) # select patch
        idx2 = random.randint(0, 3) # rotation
        r2 = image.size(1)
        r = r2 // 2
        
        if idx == 0:
            w1 = 0
            w2 = r
            h1 = 0
            h2 = r
        elif idx == 1:
            w1 = 0
            w2 = r
            h1 = r
            h2 = r2
        elif idx == 2:
            w1 = r
            w2 = r2
            h1 = 0
            h2 = r
        elif idx == 3:
            w1 = r
            w2 = r2
            h1 = r
            h2 = r2

        if self.arch == 'Moe1' or self.arch == 'Nomoe':
            flip_label = random.randint(0, 1)
            sc_label = random.randint(0, 5)
            if flip_label:
                image[:, w1:w2, h1:h2] = TF.hflip(image[:, w1:w2, h1:h2])
            # lorot
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            # shuffle channel
            image[:, w1:w2, h1:h2] = shuffle_channel(image[:, w1:w2, h1:h2], sc_label)

            rot_label = idx * 4 + idx2
            ssl_label = (rot_label, flip_label, sc_label)

            return image, ssl_label
        
        elif self.arch == 'Lorot':
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            rot_label = idx * 4 + idx2
            return image, rot_label
        
        elif self.arch == 'Moe1flip':
            flip_label = random.randint(0, 1)
            if flip_label:
                image[:, w1:w2, h1:h2] = TF.hflip(image[:, w1:w2, h1:h2])
            # lorot
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            rot_label = idx * 4 + idx2
            ssl_label = (rot_label, flip_label)
            return image, ssl_label
        
        elif self.arch == 'Moe1sc':
            sc_label = random.randint(0, 5)
             # lorot
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            # shuffle channel
            image[:, w1:w2, h1:h2] = shuffle_channel(image[:, w1:w2, h1:h2], sc_label)
            rot_label = idx * 4 + idx2
            ssl_label = (rot_label, sc_label)
            return image, ssl_label

        raise Exception('arch not implemented')
    
    def __transform_lorot(self, image):
        idx = random.randint(0, 3)
        idx2 = random.randint(0, 3)
        r2 = image.size(1)
        r = r2 // 2
        if idx == 0:
            w1 = 0
            w2 = r
            h1 = 0
            h2 = r
        elif idx == 1:
            w1 = 0
            w2 = r
            h1 = r
            h2 = r2
        elif idx == 2:
            w1 = r
            w2 = r2
            h1 = 0
            h2 = r
        elif idx == 3:
            w1 = r
            w2 = r2
            h1 = r
            h2 = r2
        image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
        rot_label = idx * 4 + idx2
        return image, rot_label
    
    def __call__(self, image: torch.Tensor):
        assert isinstance(image, torch.Tensor)
        return self.__transform_picker(image)