import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import curves
import data
import models
import utils

import numpy as np


parser = argparse.ArgumentParser(description='DNN curve training')
parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                    help='training directory (default: /tmp/curve/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init start point (default: None)')
parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                    help='fix start point (default: off)')
parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                    help='checkpoint to init end point (default: None)')
parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                    help='fix end point (default: off)')
parser.set_defaults(init_linear=True)
parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                    help='turns off linear initialization of intermediate points (default: on)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--threshold', type=float, default=0, # metavar='',
                    help='threshold (default: 0)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# parser.add_argument('--batch-norm',
#                     type = bool,
#                     default=True,
#                     action=argparse.BooleanOptionalAction,
#                     help='do we need batch norm or not')
# parser.add_argument('--residual',
#                     type = bool,
#                     default=True,
#                     action=argparse.BooleanOptionalAction,
#                     help='do we need residula connect or not')

parser.add_argument('--num_points', type=int, default=5, metavar='N',
                    help='number of points on the curve (default: 61)')
parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)




###############################################################################
### VIT (augmentation specific
###############################################################################

threshold = int(args.threshold)
if threshold == 100:
    
    # TODO: test vs. train for testing??
    # TODO: change so that we can use variable thresholds (e.g., < 100)
    
    from robustbench.data import load_cifar10c

    num_examples = len(loaders['train'])*args.batch_size
    num_cifar10c = int(num_examples * threshold)
    x, targets = load_cifar10c(n_examples=num_cifar10c, data_dir=args.data_path)
    print(x.size())
    
    x_c = [x[args.batch_size*i:args.batch_size*i + args.batch_size,:,:,:] for i in range(int(x.size()[0]/args.batch_size))]
    y_c = [targets[args.batch_size*i:args.batch_size*i + args.batch_size] for i in range(int(x.size()[0]/args.batch_size))]
    
    loaders['train'].train_data = x_c
    loaders['train'].train_labels = y_c
    
###############################################################################
    




architecture = getattr(models, args.model)

if args.curve is None:
    model = architecture.base(num_classes=num_classes, **architecture.kwargs)
else:
    curve = getattr(curves, args.curve)
    model = curves.CurveNet(
        num_classes,
        curve,
        architecture.curve,
        args.num_bends,
        args.fix_start,
        args.fix_end,
        architecture_kwargs=architecture.kwargs,
    )
    # model = torch.nn.DataParallel(model)

    base_model = None
    if args.resume is None:
        
        for (path, k) in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
            if path is not None:
                if base_model is None:
                    base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
                checkpoint = torch.load(path)
                # if 'model_state' not in checkpoint:
                #     checkpoint = dict(model_state=checkpoint)
                
                # TODO: hacky
                if list(checkpoint.keys())[0].startswith('module'):
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for state_dict_key, state_dict_value in checkpoint.items():
                        new_state_dict_key = state_dict_key.replace('module.','')
                        new_state_dict[new_state_dict_key] = state_dict_value
                    checkpoint = new_state_dict
                
                # # TODO: hacky
                # if any('.fn.' in _ for _ in checkpoint):
                #     from collections import OrderedDict
                #     new_state_dict = OrderedDict()
                #     for state_dict_key, state_dict_value in checkpoint.items():
                #         new_state_dict_key = state_dict_key.replace('.fn.','.')
                #         new_state_dict[new_state_dict_key] = state_dict_value
                #     checkpoint = new_state_dict
                
                # print(list(checkpoint.keys()))
                # print(base_model)
                
                print('Loading %s as point #%d' % (path, k))
                base_model.load_state_dict(checkpoint)
                model.import_base_parameters(base_model, k)
                
        if args.init_linear:
            print('Linear initialization.')
            model.init_linear()
            
# model = torch.nn.DataParallel(model)
model.cuda()


def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr



criterion = F.cross_entropy
regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)
# optimizer = torch.optim.SGD(
#     filter(lambda param: param.requires_grad, model.parameters()),
#     lr=args.lr,
#     momentum=args.momentum,
#     weight_decay=args.wd if args.curve is None else 0.0
# )
optimizer = torch.optim.Adam(base_model.parameters(), lr=args.lr)



start_epoch = 1
if args.resume is not None:
    # TODO: requires that optimizer_state was saved in checkpoint
    # TODO: so we can't use this with e.g., pre-trained ResNet20 
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    if 'model_state' not in checkpoint:
        checkpoint = dict(model_state=checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

utils.save_checkpoint(
    args.dir,
    start_epoch - 1,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)

has_bn = utils.check_bn(model)
test_res = {'loss': np.nan, 'accuracy': np.nan, 'nll': np.nan}
for epoch in range(start_epoch, args.epochs + 1):
    time_ep = time.time()

    # lr = args.lr
    # lr = learning_rate_schedule(args.lr, epoch, args.epochs)
    # utils.adjust_learning_rate(optimizer, lr)
    lr = optimizer.lr
    
    train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)
    # if args.curve is None or not has_bn:
    #    test_res = utils.test(loaders['test'], model, criterion, regularizer)

    if epoch % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep
    values = [epoch, lr, train_res['loss'], train_res['accuracy'], test_res['nll'],
              test_res['accuracy'], time_ep]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 1 or epoch == start_epoch:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

    
    
    
###############################################################################
### EVAL
###############################################################################

checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['model_state'])

# criterion = F.cross_entropy
# regularizer = curves.l2_regularizer(args.wd)

T = args.num_points
ts = np.linspace(0.0, 1.0, T)
tr_loss = np.zeros(T)
tr_nll = np.zeros(T)
tr_acc = np.zeros(T)
te_loss = np.zeros(T)
te_nll = np.zeros(T)
te_acc = np.zeros(T)
tr_err = np.zeros(T)
te_err = np.zeros(T)
dl = np.zeros(T)

previous_weights = None

columns = ['t', 'Train loss', 'Train nll', 'Train error (%)', 'Test nll', 'Test error (%)']


t = torch.FloatTensor([0.0]).cuda()
for i, t_value in enumerate(ts):
    t.data.fill_(t_value)
    weights = model.weights(t)
    if previous_weights is not None:
        dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
    previous_weights = weights.copy()

    # TODO: re run
    # utils.update_bn(loaders['train'], model, t=t)
    tr_res = utils.test(loaders['train'], model, criterion, regularizer, t=t)
    # te_res = utils.test(loaders['test'], model, criterion, regularizer, t=t)
    te_res = {'loss': np.nan, 'accuracy': np.nan, 'nll': np.nan}

    tr_loss[i] = tr_res['loss']
    tr_nll[i] = tr_res['nll']
    tr_acc[i] = tr_res['accuracy']
    tr_err[i] = 100.0 - tr_acc[i]
    te_loss[i] = te_res['loss']
    te_nll[i] = te_res['nll']
    te_acc[i] = te_res['accuracy']
    te_err[i] = 100.0 - te_acc[i]

    values = [t, tr_loss[i], tr_nll[i], tr_err[i], te_nll[i], te_err[i]]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


def stats(values, dl):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int


tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = stats(tr_nll, dl)
tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(tr_err, dl)

te_loss_min, te_loss_max, te_loss_avg, te_loss_int = stats(te_loss, dl)
te_nll_min, te_nll_max, te_nll_avg, te_nll_int = stats(te_nll, dl)
te_err_min, te_err_max, te_err_avg, te_err_int = stats(te_err, dl)

print('Length: %.2f' % np.sum(dl))
print(tabulate.tabulate([
        ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
        ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
        ['test nll', te_nll[0], te_nll[-1], te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
        ['test error (%)', te_err[0], te_err[-1], te_err_min, te_err_max, te_err_avg, te_err_int],
    ], [
        '', 'start', 'end', 'min', 'max', 'avg', 'int'
    ], tablefmt='simple', floatfmt='10.4f'))


### define metric
tr_err_argmax = np.argmax(np.abs(tr_err - (tr_err[0] + tr_err[-1])/2))
mc_metric = (tr_err[0] + tr_err[-1])/2 - tr_err[tr_err_argmax]
print(f"Mode Connectivity: {mc_metric}")


### TODO: update this to include the DIR basename too
### save things
save_as_npz = os.path.join(args.dir, os.path.basename(args.ckpt).replace('.pt','_curve.npz'))

np.savez(
    # os.path.join(args.dir, 'curve.npz'),
    save_as_npz,
    ts=ts,
    dl=dl,
    tr_loss=tr_loss,
    tr_loss_min=tr_loss_min,
    tr_loss_max=tr_loss_max,
    tr_loss_avg=tr_loss_avg,
    tr_loss_int=tr_loss_int,
    tr_nll=tr_nll,
    tr_nll_min=tr_nll_min,
    tr_nll_max=tr_nll_max,
    tr_nll_avg=tr_nll_avg,
    tr_nll_int=tr_nll_int,
    tr_acc=tr_acc,
    tr_err=tr_err,
    tr_err_min=tr_err_min,
    tr_err_max=tr_err_max,
    tr_err_avg=tr_err_avg,
    tr_err_int=tr_err_int,
    te_loss=te_loss,
    te_loss_min=te_loss_min,
    te_loss_max=te_loss_max,
    te_loss_avg=te_loss_avg,
    te_loss_int=te_loss_int,
    te_nll=te_nll,
    te_nll_min=te_nll_min,
    te_nll_max=te_nll_max,
    te_nll_avg=te_nll_avg,
    te_nll_int=te_nll_int,
    te_acc=te_acc,
    te_err=te_err,
    te_err_min=te_err_min,
    te_err_max=te_err_max,
    te_err_avg=te_err_avg,
    te_err_int=te_err_int,
    mc_metric=mc_metric,
)

print(f"[+] {save_as_npz}")