import argparse
from collections import deque
import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.multiprocessing import spawn
import torch.distributed as dist

# -------- gossip / models / data partition --------
from gossip import GossipDataParallel
from gossip import RingGraph, GridGraph, FullGraph, PetersenGraph, DyckGraph
from gossip import UniformMixing
from models import *  # resnet, vgg11, MobileNetV2, LeNet5, GUT, ...
from partition_data import *

try:
    from models.dpa import DPA
except Exception:
    DPA = None

try:
    from models.momentum_tracking import MomentumTracking
except Exception:
    MomentumTracking = None

try:
    from models.gradient_tracking import GradientTracking
except Exception:
    GradientTracking = None


parser = argparse.ArgumentParser(description='Decentralized training for image classification')

# ---------------------------
# Method control
# ---------------------------
parser.add_argument(
    '--method',
    default='gut',
    type=str,
    help='training method: gut, dpa, dpa-1g, mt, gt'
)
parser.add_argument(
    '--beta-1',
    dest='beta_1',
    default=0.9,
    type=float,
    help='DPA/DPA-1G beta_1 used in beta_1 * u_k + grad(z_k)'
)
parser.add_argument(
    '--beta-2',
    dest='beta_2',
    default=1.0,
    type=float,
    help='DPA/DPA-1G beta_2 used in u_{k+1} = (1-beta_2)u_k + beta_2*d_{k+1}'
)

# Learning-rate warmup and automatic epoch-based conversion to iteration counts.
parser.add_argument('--warmup-steps', default=0, type=int,
                    help='Warmup steps W for eta_k = eta * s_k')
parser.add_argument('--optimal-epoch', default=0.0, type=float,
                    help='If > 0 for DPA/DPA-1G, resolve a_steps, b_steps, warmup_steps from fractions * optimal_epoch * iters_per_epoch')
parser.add_argument('--a-epoch-frac', default=1.0, type=float,
                    help='For DPA/DPA-1G auto-resolution: a_steps = a_epoch_frac * optimal_epoch * iters_per_epoch')
parser.add_argument('--b-epoch-frac', default=0.1, type=float,
                    help='For DPA/DPA-1G auto-resolution: b_steps = b_epoch_frac * optimal_epoch * iters_per_epoch')
parser.add_argument('--warmup-epoch-frac', default=0.5, type=float,
                    help='For DPA/DPA-1G auto-resolution: warmup_steps = warmup_epoch_frac * optimal_epoch * iters_per_epoch')

# a-schedule for x-update
parser.add_argument('--a-start', default=0.3, type=float,
                    help='a_start for x-update mixing')
parser.add_argument('--a-end', default=0.1, type=float,
                    help='a_end for x-update mixing')
parser.add_argument('--a-steps', default=2000, type=int,
                    help='T_a iterations for a schedule; used directly unless auto-resolution is enabled')
parser.add_argument('--a-schedule', default='linear', type=str, choices=['linear', 'exp'],
                    help='Schedule type for a_{k+1}: linear or exp')
parser.add_argument('--qa', default=1.0, type=float,
                    help='Exponent q_a for exponential a schedule')

# b-schedule for z_k
parser.add_argument('--b-start', default=0.3, type=float,
                    help='b_start for z_k mixing')
parser.add_argument('--b-end', default=0.1, type=float,
                    help='b_end for z_k mixing')
parser.add_argument('--b-steps', default=2000, type=int,
                    help='T_b iterations for b schedule; used directly unless auto-resolution is enabled')
parser.add_argument('--b-schedule', default='linear', type=str, choices=['linear', 'exp'],
                    help='Schedule type for b_{k+1}: linear or exp')
parser.add_argument('--qb', default=1.0, type=float,
                    help='Exponent q_b for exponential b schedule')

# Periodic x-gossip control.
# DPA forces this to 0.
# DPA-1G requires this to be > 0.
parser.add_argument('--t-gossip-x', default=0, type=int,
                    help='T_x control: DPA forces T_x=0; DPA-1G requires T_x>0')

# Momentum Tracking params
parser.add_argument('--mt-beta', default=0.9, type=float,
                    help='Momentum Tracking beta')
parser.add_argument('--mt-wd', default=0.0, type=float,
                    help='Momentum Tracking weight decay (L2)')
parser.add_argument('--mt-paper-init', action='store_true',
                    help='Use paper initialization for MT (all-reduce grad init)')

# Gradient Tracking params
parser.add_argument('--gt-beta', default=0.9, type=float,
                    help='Unused in plain GT; kept only for CLI compatibility')
parser.add_argument('--gt-wd', default=0, type=float,
                    help='Gradient Tracking weight decay (L2)')

# model/data
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='resnet or vgg11 or mobilenet or lenet5')
parser.add_argument('-depth', '--depth', default=20, type=int,
                    help='depth of the resnet model')
parser.add_argument('--normtype', default='evonorm',
                    help='none or batchnorm or groupnorm or evonorm')
parser.add_argument('--data-dir', dest='data_dir',
                    help='The directory used to save datasets', default='../../data', type=str)
parser.add_argument('--dataset', dest='dataset', type=str,
                    help='available datasets: cifar10, cifar100, imagenette, fmnist', default='cifar10')
parser.add_argument('--skew', default=1.0, type=float,
                    help='parameter alpha that controls non-iidness')
parser.add_argument('--classes', default=10, type=int,
                    help='number of classes in the dataset')

# optimization
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    help='mini-batch size (global)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='base learning rate eta')
parser.add_argument('--gamma', default=1.0, type=float,
                    metavar='AR', help='averaging rate for gossip_averaging in GossipDataParallel')
parser.add_argument('--momentum', default=0.0, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--nesterov', action='store_true')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--scaling', default=0.9, type=float,
                    help='scaling factor for the bias correction term used by GUT tracker')

# distributed / topology
parser.add_argument('-world_size', '--world_size', default=5, type=int,
                    help='total number of nodes')
parser.add_argument('--graph', '-g', default='ring',
                    help='graph structure - [ring, torus, dyck, petersen, full]')
parser.add_argument('--neighbors', default=2, type=int,
                    help='number of neighbors per node')
parser.add_argument('-d', '--devices', default=4, type=int,
                    help='number of gpus/devices on the card')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--seed', default=1234, type=int,
                    help='set seed')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models', default='outputs', type=str)
parser.add_argument('--port', dest='port',
                    help='between 3000 to 65000', default='25500', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=5)

args = parser.parse_args()

# ---------------------------
# Method normalization and safety rules
# ---------------------------
args.method = args.method.lower()

if args.method == "dpa":
    if args.t_gossip_x != 0:
        print("DPA selected: overriding --t-gossip-x to 0.")
    args.t_gossip_x = 0

elif args.method in ("dpa-1g", "dpa_1g"):
    args.method = "dpa-1g"
    if args.t_gossip_x <= 0:
        raise ValueError("DPA-1G requires --t-gossip-x > 0, for example --t-gossip-x 5.")

elif args.method in ("gut", "mt", "gt"):
    pass

else:
    raise ValueError(
        f"Unknown method '{args.method}'. Supported methods are: gut, dpa, dpa-1g, mt, gt."
    )

if args.optimal_epoch < 0:
    raise ValueError("--optimal-epoch must be >= 0.")

if args.a_epoch_frac < 0 or args.b_epoch_frac < 0 or args.warmup_epoch_frac < 0:
    raise ValueError("Epoch fractions must be >= 0.")


def _build_method_tag():
    method_tag = args.method.lower()

    if method_tag in ("dpa", "dpa-1g"):
        auto_part = ""
        if args.optimal_epoch > 0:
            auto_part = (
                f"_oe{args.optimal_epoch:g}"
                f"_af{args.a_epoch_frac:g}"
                f"_bf{args.b_epoch_frac:g}"
                f"_wf{args.warmup_epoch_frac:g}"
            )

        method_tag = (
            f"{method_tag}"
            f"_b1{args.beta_1:g}"
            f"_b2{args.beta_2:g}"
            f"_as{args.a_schedule}_qa{args.qa:g}"
            f"_ast{args.a_start:g}_aen{args.a_end:g}_aT{args.a_steps}"
            f"_bs{args.b_schedule}_qb{args.qb:g}"
            f"_bst{args.b_start:g}_ben{args.b_end:g}_bT{args.b_steps}"
            f"_Tx{args.t_gossip_x}"
            f"{auto_part}"
        )

    elif method_tag == 'mt':
        method_tag = f"mt_b{args.mt_beta:g}"

    elif method_tag == 'gt':
        method_tag = f"gt_wd{args.gt_wd:g}"

    elif method_tag == 'gut':
        method_tag = f"gut_m{args.momentum:g}_s{args.scaling:g}"

    return method_tag


method_tag = _build_method_tag()

args.save_dir = os.path.join(
    args.save_dir,
    f"{method_tag}_{args.arch}"
    f"_{args.dataset}"
    f"_nodes_{args.world_size}"
    f"_{args.normtype}"
    f"_lr_{args.lr}"
    f"_seed_{args.seed}"
    f"_skew_{args.skew}"
    f"_{args.graph}"
)

excel_dir = os.path.join(args.save_dir, "excel_data")
os.makedirs(excel_dir, exist_ok=True)

torch.save(args, os.path.join(args.save_dir, "training_args.bin"))


def _resolve_auto_schedule_steps_from_loader(train_loader):
    """
    For DPA and DPA-1G only:
      if optimal_epoch > 0, resolve iteration-based schedule lengths from len(train_loader).

      a_steps      = round(a_epoch_frac      * optimal_epoch * iters_per_epoch)
      b_steps      = round(b_epoch_frac      * optimal_epoch * iters_per_epoch)
      warmup_steps = round(warmup_epoch_frac * optimal_epoch * iters_per_epoch)
    """
    if args.method not in ("dpa", "dpa-1g"):
        return None

    if float(args.optimal_epoch) <= 0.0:
        return None

    iters_per_epoch = int(len(train_loader))
    if iters_per_epoch <= 0:
        raise RuntimeError("len(train_loader) must be > 0 to auto-resolve schedule lengths.")

    resolved_a_steps = max(
        1,
        int(round(float(args.a_epoch_frac) * float(args.optimal_epoch) * float(iters_per_epoch)))
    )
    resolved_b_steps = max(
        1,
        int(round(float(args.b_epoch_frac) * float(args.optimal_epoch) * float(iters_per_epoch)))
    )
    resolved_warmup_steps = max(
        0,
        int(round(float(args.warmup_epoch_frac) * float(args.optimal_epoch) * float(iters_per_epoch)))
    )

    args.a_steps = resolved_a_steps
    args.b_steps = resolved_b_steps
    args.warmup_steps = resolved_warmup_steps
    args.iters_per_epoch = iters_per_epoch

    return {
        "iters_per_epoch": iters_per_epoch,
        "a_steps": resolved_a_steps,
        "b_steps": resolved_b_steps,
        "warmup_steps": resolved_warmup_steps,
        "optimal_epoch": float(args.optimal_epoch),
        "a_epoch_frac": float(args.a_epoch_frac),
        "b_epoch_frac": float(args.b_epoch_frac),
        "warmup_epoch_frac": float(args.warmup_epoch_frac),
    }


def partition_trainDataset(device):
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        dataset = datasets.CIFAR10(
            root=args.data_dir, train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True
        )
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(
            root=args.data_dir, train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True
        )
    elif args.dataset == 'fmnist':
        normalize = transforms.Normalize((0.5,), (0.5,))
        dataset = datasets.FashionMNIST(
            root=args.data_dir, train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            download=True
        )
    elif args.dataset == 'imagenette':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), data_transforms)
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    size = dist.get_world_size()
    bsz = int(args.batch_size / float(size))

    partition_sizes = [1.0 / size for _ in range(size)]
    partitioner = DataPartitioner(
        args.seed,
        dataset,
        partition_sizes,
        non_iid_alpha=args.skew,
        partition_type="non_iid_dirichlet"
    )
    partition, data_distribution = partitioner.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition,
        batch_size=bsz,
        shuffle=True,
        num_workers=2
    )
    return train_set, bsz, data_distribution


def test_Dataset():
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        dataset = datasets.CIFAR10(
            root=args.data_dir, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        dataset = datasets.CIFAR100(
            root=args.data_dir, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif args.dataset == 'fmnist':
        normalize = transforms.Normalize((0.5,), (0.5,))
        dataset = datasets.FashionMNIST(
            root=args.data_dir, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    elif args.dataset == 'imagenette':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), data_transforms)
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    val_bsz = 128
    val_set = torch.utils.data.DataLoader(dataset, batch_size=val_bsz, shuffle=False, num_workers=2)
    return val_set, val_bsz


def warmup_lr(step_k: int, base_lr: float, warmup_steps: int):
    k = int(step_k)
    W = int(warmup_steps)
    if W <= 0:
        return float(base_lr)
    if k < W:
        return float(base_lr) * float(k + 1) / float(W)
    return float(base_lr)


# ---------------------------
# training routines
# ---------------------------

def train_gut(train_loader, model, criterion, optimizer, epoch, lr, device, tracker):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_transferred = 0

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)

        _, amt_data_transfer, global_update, acc_y = model.transfer_params(
            epoch=epoch + (1e-3 * i),
            lr=optimizer.param_groups[0]['lr']
        )
        data_transferred += amt_data_transfer

        model.gossip_averaging()

        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()

        model.remove_gossip()

        tracker(global_update, acc_y, optimizer.param_groups[0]['lr'])
        tracker.modify_gradients()

        optimizer.step()
        optimizer.zero_grad()

        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Rank: {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      dist.get_rank(), epoch, i, len(train_loader),
                      batch_time=batch_time, loss=losses, top1=top1))

    return data_transferred, top1.avg, losses.avg, lr


def train_mt(train_loader, model, criterion, optimizer, epoch, lr, device, mt_obj):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_transferred = 0

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)

        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()

        dt = mt_obj.step_from_grads(lr=optimizer.param_groups[0]['lr'])
        data_transferred += dt

        optimizer.zero_grad()

        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Rank: {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      dist.get_rank(), epoch, i, len(train_loader),
                      batch_time=batch_time, loss=losses, top1=top1))

    return data_transferred, top1.avg, losses.avg, lr


def train_gt(train_loader, model, criterion, optimizer, epoch, lr, device, gt_obj):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_transferred = 0

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)

        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()

        dt = gt_obj.step_from_grads(lr=optimizer.param_groups[0]['lr'])
        data_transferred += dt

        optimizer.zero_grad()

        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Rank: {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      dist.get_rank(), epoch, i, len(train_loader),
                      batch_time=batch_time, loss=losses, top1=top1))

    return data_transferred, top1.avg, losses.avg, lr


def train_dpa(train_loader, model, criterion, optimizer, epoch, base_lr, device, dpa_obj):
    """
    DPA / DPA-1G training loop.

    DPA, args.method == 'dpa':
      T_x = 0
      Every iteration performs:
        y gossip and x gossip.

    DPA-1G, args.method == 'dpa-1g':
      T_x > 0
      Normal iterations:
        y gossip, x local update.
      Special iterations where (k+1) % T_x == 0:
        skip gradient, freeze y and u, gossip x.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_transferred = 0.0

    model.train()
    end = time.time()

    tx_mode = int(args.t_gossip_x)

    if args.method == "dpa" and tx_mode != 0:
        raise RuntimeError("Internal error: DPA must use T_x = 0.")

    if args.method == "dpa-1g" and tx_mode <= 0:
        raise RuntimeError("Internal error: DPA-1G must use T_x > 0.")

    method_name = "DPA" if args.method == "dpa" else "DPA-1G"

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_var = Variable(input).to(device)
        target_var = Variable(target).to(device)

        k = int(dpa_obj.t)
        eta_k = warmup_lr(step_k=k, base_lr=base_lr, warmup_steps=args.warmup_steps)
        optimizer.param_groups[0]['lr'] = eta_k

        a_k1, b_k1 = dpa_obj.begin_step(eta_k=eta_k)

        do_x_gossip_special = dpa_obj.is_x_gossip_step()
        always_mix_x = args.method == "dpa"

        loss = None
        output = None

        if not do_x_gossip_special:
            saved_x = dpa_obj.make_z_and_load(b_k1=b_k1)

            output = model(input_var)
            loss = criterion(output, target_var)
            loss.backward()

            dpa_obj.restore_x(saved_x)

            dt_y = dpa_obj.update_y_from_grads(lr=eta_k, do_mix=True)
            data_transferred += dt_y

            _a, dt_x = dpa_obj.update_x_from_y(a_k1=a_k1, do_mix=always_mix_x)
            data_transferred += dt_x

        else:
            _a, dt_x = dpa_obj.special_x_gossip_step(a_k1=a_k1)
            data_transferred += dt_x

        dpa_obj.zero_grads()

        if loss is not None and output is not None:
            output = output.float()
            loss_f = loss.float()

            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss_f.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if loss is None:
                loss_str = "n/a"
                top1_str = "n/a"
            else:
                loss_str = f"{losses.val:.4f} ({losses.avg:.4f})"
                top1_str = f"{top1.val:.3f} ({top1.avg:.3f})"

            x_is_mixed_this_iter = int(do_x_gossip_special or always_mix_x)
            y_is_mixed_this_iter = int(not do_x_gossip_special)

            print(
                'Rank: {0}\t'
                'Method: {1}\t'
                'Epoch: [{2}][{3}/{4}]\t'
                'Step k={5}\t'
                'LR {6:.5e}\t'
                'a_sched {7}\t'
                'qa {8:.4f}\t'
                'a_start {9:.6f}\t'
                'a_end {10:.6f}\t'
                'a_steps {11}\t'
                'a(k+1) {12:.6f}\t'
                'b_sched {13}\t'
                'qb {14:.4f}\t'
                'b_steps {15}\t'
                'b(k+1) {16:.6f}\t'
                'beta_1 {17:.4f}\t'
                'beta_2 {18:.4f}\t'
                'T_x {19}\t'
                'y_mixed {20}\t'
                'x_mixed {21}\t'
                'special {22}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {23}\t'
                'Prec@1 {24}'.format(
                    dist.get_rank(),
                    method_name,
                    epoch,
                    i,
                    len(train_loader),
                    k,
                    eta_k,
                    str(args.a_schedule),
                    float(args.qa),
                    float(args.a_start),
                    float(args.a_end),
                    int(args.a_steps),
                    float(a_k1),
                    str(args.b_schedule),
                    float(args.qb),
                    int(args.b_steps),
                    float(b_k1),
                    float(args.beta_1),
                    float(args.beta_2),
                    int(args.t_gossip_x),
                    y_is_mixed_this_iter,
                    x_is_mixed_this_iter,
                    int(do_x_gossip_special),
                    loss_str,
                    top1_str,
                    batch_time=batch_time,
                )
            )

    return data_transferred, top1.avg, losses.avg, optimizer.param_groups[0]['lr']


def gossip_avg(train_loader, model, criterion, optimizer, epoch, lr, device):
    data_transferred = 0
    n = 50
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input_var, target_var = Variable(input).to(device), Variable(target).to(device)
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.zero_grad()
        _, amt_data_transfer, _, _ = model.transfer_params(epoch=epoch + (1e-3 * i), lr=lr)
        model.gossip_averaging()
        data_transferred += amt_data_transfer
        if i == n:
            break
    return data_transferred


def _eval_on_x_enter(method_obj):
    if method_obj is None:
        return None, None

    if hasattr(method_obj, "load_x_and_save_y") and hasattr(method_obj, "restore_y"):
        saved = method_obj.load_x_and_save_y()
        return saved, lambda s: method_obj.restore_y(s)

    if hasattr(method_obj, "load_x_into_model") and hasattr(method_obj, "restore_from_saved"):
        saved = method_obj.load_x_into_model()
        return saved, lambda s: method_obj.restore_from_saved(s)

    return None, None


def validate(val_loader, model, criterion, batch_size, device,
             epoch=0, class_wise=False, list_of_classes=10,
             return_classwise=False, method_obj=None, eval_on_x=False):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    acc = [0 for _ in range(list_of_classes)]
    class_count = [0 for _ in range(list_of_classes)]

    saved = None
    exit_fn = None
    if eval_on_x and (method_obj is not None):
        saved, exit_fn = _eval_on_x_enter(method_obj)

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var, target_var = Variable(input).to(device), Variable(target).to(device)

            output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()

            prec1 = accuracy(output.data, target_var)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            if class_wise:
                _, preds = torch.max(output.data, 1)
                for c in range(list_of_classes):
                    acc[c] += ((preds == target_var) * (target_var == c)).sum().float()
                    class_count[c] += (target_var == c).sum()

            if i % args.print_freq == 0:
                print('Rank: {0}\t'
                      'Test: [{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          dist.get_rank(), i, len(val_loader),
                          loss=losses, top1=top1))

    if saved is not None and exit_fn is not None:
        exit_fn(saved)

    print('Rank:{0}, Prec@1 {top1.avg:.3f}'.format(dist.get_rank(), top1=top1))

    if class_wise:
        for c in range(list_of_classes):
            acc[c] = (acc[c].cpu().numpy() / class_count[c].cpu().numpy()) * 100
        print('Class-wise accuracy for rank {} is '.format(dist.get_rank()), acc)

    if return_classwise:
        return top1.avg, acc
    return top1.avg, losses.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', args.port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def _print_graph_debug(rank, graph):
    if rank != 0:
        return

    print("\n==== GRAPH DEBUG ====", flush=True)
    print(
        f"args.graph={args.graph}, "
        f"Graph class={graph.__class__.__name__}, "
        f"world_size={graph.world_size}, "
        f"peers_per_itr={graph.peers_per_itr}, "
        f"neighbors arg={args.neighbors}",
        flush=True
    )

    all_peers = []

    for r in range(graph.world_size):
        peers = [edge.dest // graph.nprocs_per_node for edge in graph.phone_book[r]]
        all_peers.append(peers)
        print(f"rank {r:02d}: {peers}", flush=True)

    print("\n---- graph checks ----", flush=True)
    ok = True

    for r, peers in enumerate(all_peers):
        if len(peers) != len(set(peers)):
            ok = False
            print(f"WARNING duplicate peers at rank {r}: {peers}", flush=True)

        if len(peers) < args.neighbors:
            ok = False
            print(
                f"WARNING rank {r} has only {len(peers)} peers, "
                f"but args.neighbors={args.neighbors}: {peers}",
                flush=True
            )

        for p in peers:
            if p < 0 or p >= graph.world_size:
                ok = False
                print(f"WARNING invalid peer: rank {r} -> {p}", flush=True)

            if r not in all_peers[p]:
                ok = False
                print(
                    f"WARNING asymmetric edge: {r} -> {p}, "
                    f"but {p} does not point back to {r}",
                    flush=True
                )

    print("\n---- group-index reverse-edge check ----", flush=True)

    for r, peers in enumerate(all_peers):
        for j, p in enumerate(peers):
            if j >= len(all_peers[p]):
                ok = False
                print(
                    f"WARNING index mismatch: rank {r} peer index {j} -> {p}, "
                    f"but rank {p} has only {len(all_peers[p])} peers",
                    flush=True
                )
                continue

            reverse_at_same_index = all_peers[p][j]
            if reverse_at_same_index != r:
                print(
                    f"NOTICE group-index not symmetric: "
                    f"rank {r} index {j} -> {p}, "
                    f"but rank {p} index {j} -> {reverse_at_same_index}",
                    flush=True
                )

    if ok:
        print("Basic graph checks passed.", flush=True)
    else:
        print("Basic graph checks found warnings.", flush=True)

    print("==== END GRAPH DEBUG ====\n", flush=True)


def run(rank, size):
    global args, best_prec1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device(f"cuda:{rank % args.devices}")
    best_prec1 = 0
    data_transferred = 0

    # build model
    if args.arch.lower() == 'resnet':
        model = resnet(num_classes=args.classes, depth=args.depth,
                       dataset=args.dataset, norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'vgg11':
        model = vgg11(num_classes=args.classes, dataset=args.dataset,
                      norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'mobilenet':
        model = MobileNetV2(num_classes=args.classes, dataset=args.dataset,
                            norm_type=args.normtype, groups=2)
    elif args.arch.lower() == 'lenet5':
        model = LeNet5()
    else:
        raise NotImplementedError(f"Unknown architecture: {args.arch}")

    if rank == 0:
        print(args)
        print('Printing model summary...')
        if args.dataset == "fmnist":
            print(summary(model, (1, 28, 28), batch_size=int(args.batch_size / size), device='cpu'))
        elif args.dataset == "imagenette":
            print(summary(model, (3, 224, 224), batch_size=int(args.batch_size / size), device='cpu'))
        else:
            print(summary(model, (3, 32, 32), batch_size=int(args.batch_size / size), device='cpu'))

    # graph
    if args.graph.lower() == 'ring':
        graph = RingGraph(rank, size, args.devices, peers_per_itr=args.neighbors)
    elif args.graph.lower() == 'torus':
        graph = GridGraph(rank, size, args.devices, peers_per_itr=args.neighbors)
    elif args.graph.lower() == 'petersen':
        graph = PetersenGraph(rank, size, args.devices, peers_per_itr=3)
    elif args.graph.lower() == 'dyck':
        graph = DyckGraph(rank, size, args.devices, peers_per_itr=3)
    elif args.graph.lower() == 'full':
        graph = FullGraph(rank, size, args.devices, peers_per_itr=args.world_size - 1)
    else:
        raise NotImplementedError(f"Unknown graph: {args.graph}")

    _print_graph_debug(rank, graph)
    dist.barrier()

    mixing = UniformMixing(graph, device)

    model = GossipDataParallel(
        model,
        device_ids=[rank % args.devices],
        rank=rank,
        world_size=size,
        graph=graph,
        mixing=mixing,
        comm_device=device,
        eta=args.gamma,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=0.0,
        lr=args.lr,
        neighbors=args.neighbors
    )
    model.to(device)

    train_loader, bsz_train, _ = partition_trainDataset(device=device)
    val_loader, bsz_val = test_Dataset()

    resolved_meta = _resolve_auto_schedule_steps_from_loader(train_loader)
    if resolved_meta is not None:
        if rank == 0:
            print("Auto-resolved DPA/DPA-1G schedule lengths from train_loader:")
            print(
                "  optimal_epoch={:.6f}, a_epoch_frac={:.6f}, b_epoch_frac={:.6f}, warmup_epoch_frac={:.6f}, "
                "iters_per_epoch={}, a_steps={}, b_steps={}, warmup_steps={}".format(
                    resolved_meta["optimal_epoch"],
                    resolved_meta["a_epoch_frac"],
                    resolved_meta["b_epoch_frac"],
                    resolved_meta["warmup_epoch_frac"],
                    resolved_meta["iters_per_epoch"],
                    resolved_meta["a_steps"],
                    resolved_meta["b_steps"],
                    resolved_meta["warmup_steps"],
                )
            )
            torch.save(resolved_meta, os.path.join(args.save_dir, "resolved_schedule_steps.pt"))
            torch.save(args, os.path.join(args.save_dir, "training_args.bin"))

    dist.barrier()

    optimizer = optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov
    )

    if rank == 0:
        print(optimizer)

    criterion = nn.CrossEntropyLoss().to(device)

    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestones)

    tracker = None
    dpa_obj = None
    mt_obj = None
    gt_obj = None

    m = args.method.lower()

    if m in ("dpa", "dpa-1g"):
        if DPA is None:
            raise RuntimeError("DPA requested but models/dpa.py could not be imported.")

        dpa_obj = DPA(
            model=model,
            device=device,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
            warmup_steps=args.warmup_steps,
            a_start=args.a_start,
            a_end=args.a_end,
            a_steps=args.a_steps,
            a_schedule=args.a_schedule,
            qa=args.qa,
            b_start=args.b_start,
            b_end=args.b_end,
            b_steps=args.b_steps,
            b_schedule=args.b_schedule,
            qb=args.qb,
            T_gossip_x=args.t_gossip_x,
        )

    elif m == 'mt':
        if MomentumTracking is None:
            raise RuntimeError("MT requested but models/momentum_tracking.py could not be imported.")
        mt_obj = MomentumTracking(
            model=model,
            device=device,
            beta=args.mt_beta,
            weight_decay=args.mt_wd,
            paper_init=args.mt_paper_init
        )

    elif m == 'gt':
        if GradientTracking is None:
            raise RuntimeError("GT requested but models/gradient_tracking.py could not be imported.")
        gt_obj = GradientTracking(
            model=model,
            device=device,
            beta=args.gt_beta,
            weight_decay=args.gt_wd
        )

    else:
        tracker = GUT(
            model, device, rank, args.lr, args.gamma,
            neighbors=args.neighbors,
            momentum=args.momentum,
            scaling=args.scaling
        )

    last_five_epoch_accuracies = deque(maxlen=5)

    for epoch in range(args.epochs):
        if m in ('gut', 'mt', 'gt'):
            curr_lr = optimizer.param_groups[0]['lr']
            print('current lr {:.5e}'.format(curr_lr))

        elif m in ("dpa", "dpa-1g"):
            method_name = "DPA" if m == "dpa" else "DPA-1G"

            if float(args.optimal_epoch) > 0:
                print(
                    'epoch {} ({}): base_lr={:.5e}, auto_steps=ON, optimal_epoch={}, '
                    'iters_per_epoch={}, a_epoch_frac={}, b_epoch_frac={}, warmup_epoch_frac={}, '
                    'warmup_steps={}, a_schedule={}, qa={}, a_start={}, a_end={}, a_steps={}, '
                    'b_schedule={}, qb={}, b_start={}, b_end={}, b_steps={}, '
                    'beta_1={}, beta_2={}, T_x={}'.format(
                        epoch,
                        method_name,
                        args.lr,
                        args.optimal_epoch,
                        getattr(args, "iters_per_epoch", -1),
                        args.a_epoch_frac,
                        args.b_epoch_frac,
                        args.warmup_epoch_frac,
                        args.warmup_steps,
                        args.a_schedule,
                        args.qa,
                        args.a_start,
                        args.a_end,
                        args.a_steps,
                        args.b_schedule,
                        args.qb,
                        args.b_start,
                        args.b_end,
                        args.b_steps,
                        args.beta_1,
                        args.beta_2,
                        args.t_gossip_x,
                    )
                )
            else:
                print(
                    'epoch {} ({}): base_lr={:.5e}, auto_steps=OFF, warmup_steps={}, '
                    'a_schedule={}, qa={}, a_start={}, a_end={}, a_steps={}, '
                    'b_schedule={}, qb={}, b_start={}, b_end={}, b_steps={}, '
                    'beta_1={}, beta_2={}, T_x={}'.format(
                        epoch,
                        method_name,
                        args.lr,
                        args.warmup_steps,
                        args.a_schedule,
                        args.qa,
                        args.a_start,
                        args.a_end,
                        args.a_steps,
                        args.b_schedule,
                        args.qb,
                        args.b_start,
                        args.b_end,
                        args.b_steps,
                        args.beta_1,
                        args.beta_2,
                        args.t_gossip_x,
                    )
                )

        model.block()

        if m in ("dpa", "dpa-1g"):
            dt, train_acc, train_loss, _lr = train_dpa(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                args.lr,
                device,
                dpa_obj,
            )
        elif m == 'mt':
            curr_lr = optimizer.param_groups[0]['lr']
            dt, train_acc, train_loss, _lr = train_mt(
                train_loader, model, criterion, optimizer, epoch,
                curr_lr, device, mt_obj
            )
        elif m == 'gt':
            curr_lr = optimizer.param_groups[0]['lr']
            dt, train_acc, train_loss, _lr = train_gt(
                train_loader, model, criterion, optimizer, epoch,
                curr_lr, device, gt_obj
            )
        else:
            curr_lr = optimizer.param_groups[0]['lr']
            dt, train_acc, train_loss, _lr = train_gut(
                train_loader, model, criterion, optimizer, epoch,
                curr_lr, device, tracker
            )

        data_transferred += dt

        if m not in ("dpa", "dpa-1g"):
            lr_scheduler.step()

        if m in ("dpa", "dpa-1g"):
            prec1, loss = validate(
                val_loader,
                model,
                criterion,
                bsz_val,
                device,
                epoch,
                class_wise=False,
                list_of_classes=args.classes,
                return_classwise=False,
                method_obj=dpa_obj,
                eval_on_x=True,
            )
        else:
            prec1, loss = validate(
                val_loader,
                model,
                criterion,
                bsz_val,
                device,
                epoch,
                class_wise=False,
                list_of_classes=args.classes,
                return_classwise=False,
            )

        last_five_epoch_accuracies.append(float(prec1))
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint(
            {'state_dict': model.state_dict(), 'best_prec1': best_prec1},
            filename=os.path.join(args.save_dir, f'model_{rank}.th')
        )

    # The last validation inside the training loop is the last-epoch result.
    last_epoch_prec1 = float(prec1)
    last_five_epoch_prec1 = (
        float(np.mean(last_five_epoch_accuracies))
        if len(last_five_epoch_accuracies) == 5
        else float("nan")
    )

    if m not in ("dpa", "dpa-1g"):
        dt_final = gossip_avg(
            train_loader, model, criterion, optimizer, epoch,
            optimizer.param_groups[0]['lr'], device
        )
        print('Post-training test accuracy (after additional averaging)')
        prec1_final, _ = validate(
            val_loader, model, criterion, bsz_val, device, epoch,
            class_wise=False, list_of_classes=args.classes,
            return_classwise=False
        )
        data_transferred_total_gb = (data_transferred + dt_final) / 1.0e9
        print(
            "Rank : ", rank,
            "Data transferred(in GB) during training: ", data_transferred / 1.0e9,
            "Data transferred(in GB) in final gossip averaging rounds: ", dt_final / 1.0e9, "\n"
        )
        torch.save(
            (last_epoch_prec1, prec1_final, data_transferred_total_gb, last_five_epoch_prec1),
            os.path.join(args.save_dir, "excel_data", f"rank_{rank}.sp")
        )
    else:
        print('Post-training test accuracy (DPA/DPA-1G repeated evaluation on x)')
        prec1_final, _ = validate(
            val_loader, model, criterion, bsz_val, device, epoch,
            class_wise=False, list_of_classes=args.classes,
            return_classwise=False,
            method_obj=dpa_obj,
            eval_on_x=True,
        )
        data_transferred_total_gb = data_transferred / 1.0e9
        print(
            "Rank : ", rank,
            "Data transferred(in GB) during training: ", data_transferred / 1.0e9, "\n"
        )
        torch.save(
            (last_epoch_prec1, prec1_final, data_transferred_total_gb, last_five_epoch_prec1),
            os.path.join(args.save_dir, "excel_data", f"rank_{rank}.sp")
        )

    print(
        f"Accuracy summary | rank={rank} | seed={args.seed} | "
        f"last_epoch={args.epochs} | last_epoch_accuracy={last_epoch_prec1:.3f}% | "
        f"last_five_epoch_accuracy={last_five_epoch_prec1:.3f}% | "
        f"post_training_accuracy={prec1_final:.3f}%",
        flush=True,
    )


if __name__ == '__main__':
    size = args.world_size
    spawn(init_process, args=(size, run), nprocs=size, join=True)

    resolved_path = os.path.join(args.save_dir, "resolved_schedule_steps.pt")
    if os.path.exists(resolved_path):
        resolved_meta = torch.load(resolved_path)
        args.a_steps = int(resolved_meta["a_steps"])
        args.b_steps = int(resolved_meta["b_steps"])
        args.warmup_steps = int(resolved_meta["warmup_steps"])
        args.iters_per_epoch = int(resolved_meta["iters_per_epoch"])

    excel_data = {
        'data': args.dataset,
        'arch': args.arch,
        "method": args.method,
        "beta_1": args.beta_1 if args.method in ("dpa", "dpa-1g") else None,
        "beta_2": args.beta_2 if args.method in ("dpa", "dpa-1g") else None,
        "momentum": args.momentum,
        "nesterov": args.nesterov,
        "learning rate": args.lr,
        "gamma": args.gamma,
        "graph": args.graph,
        "skew": args.skew,
        "norm": args.normtype,
        "epochs": args.epochs,
        "nodes": size,
        "avg test acc": [0.0 for _ in range(size)],
        "avg test acc last5": [0.0 for _ in range(size)],
        "avg test acc final": [0.0 for _ in range(size)],
        "data transferred": [0.0 for _ in range(size)],
        "seed": args.seed,
        "scaling": args.scaling,
        'depth': args.depth,
        "optimal_epoch": args.optimal_epoch,
        "a_epoch_frac": args.a_epoch_frac,
        "b_epoch_frac": args.b_epoch_frac,
        "warmup_epoch_frac": args.warmup_epoch_frac,
        "iters_per_epoch": getattr(args, "iters_per_epoch", None),
        "warmup_steps": args.warmup_steps,
        "a_start": args.a_start,
        "a_end": args.a_end,
        "a_steps": args.a_steps,
        "a_schedule": args.a_schedule,
        "qa": args.qa,
        "b_start": args.b_start,
        "b_end": args.b_end,
        "b_steps": args.b_steps,
        "b_schedule": args.b_schedule,
        "qb": args.qb,
        "t_gossip_x": args.t_gossip_x,
        "mt_beta": args.mt_beta,
        "mt_wd": args.mt_wd,
        "mt_paper_init": args.mt_paper_init,
        "gt_beta": args.gt_beta,
        "gt_wd": args.gt_wd,
    }

    for i in range(size):
        acc, acc_final, d_tfr, acc_last5 = torch.load(
            os.path.join(args.save_dir, "excel_data", f"rank_{i}.sp")
        )
        excel_data["avg test acc"][i] = acc
        excel_data["avg test acc last5"][i] = acc_last5
        excel_data["avg test acc final"][i] = acc_final
        excel_data["data transferred"][i] = d_tfr

    torch.save(excel_data, os.path.join(args.save_dir, "excel_data", "dict"))

    last_epoch_mean_accuracy = float(np.mean(excel_data["avg test acc"]))
    last_five_epoch_mean_accuracy = float(np.mean(excel_data["avg test acc last5"]))
    post_training_mean_accuracy = float(np.mean(excel_data["avg test acc final"]))

    print(
        f"\nTest accuracy summary | method={args.method} | seed={args.seed} | "
        f"mean across {size} agents",
        flush=True,
    )
    print(
        f"Last training epoch ({args.epochs}): "
        f"{last_epoch_mean_accuracy:.3f}%",
        flush=True,
    )
    if args.epochs >= 5:
        print(
            f"Last five training epochs ({args.epochs - 4}-{args.epochs}), L5E: "
            f"{last_five_epoch_mean_accuracy:.3f}%",
            flush=True,
        )
    else:
        print("L5E requires at least five training epochs (saved as NaN).", flush=True)
    if args.method not in ("dpa", "dpa-1g"):
        post_training_label = "after additional averaging"
    else:
        post_training_label = "repeated evaluation on x"
    print(
        f"Post-training evaluation ({post_training_label}): "
        f"{post_training_mean_accuracy:.3f}%",
        flush=True,
    )
