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

### PINN specific
parser.add_argument('--system', type=str, default='convection', help='System to study.')
parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--N_f', type=int, default=100, help='Number of collocation points to sample.')
parser.add_argument('--optimizer_name', type=str, default='LBFGS', help='Optimizer of choice.')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
parser.add_argument('--L', type=float, default=1.0, help='Multiplier on loss f.')

parser.add_argument('--xgrid', type=int, default=256, help='Number of points in the xgrid.')
parser.add_argument('--nt', type=int, default=100, help='Number of points in the tgrid.')
parser.add_argument('--nu', type=float, default=1.0, help='nu value that scales the d^2u/dx^2 term. 0 if only doing advection.')
parser.add_argument('--rho', type=float, default=1.0, help='reaction coefficient for u*(1-u) term.')
parser.add_argument('--beta', type=float, default=1.0, help='beta value that scales the du/dx term. 0 if only doing diffusion.')
parser.add_argument('--u0_str', default='sin(x)', help='str argument for initial condition if no forcing term.')
parser.add_argument('--source', default=0, type=float, help="If there's a source term, define it here. For now, just constant force terms.")

parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions/layers of the NN, minus the first layer.')
parser.add_argument('--net', type=str, default='DNN', help='The net architecture that is to be used.')
parser.add_argument('--activation', default='tanh', help='Activation to use in the network.')
parser.add_argument('--loss_style', default='mean', help='Loss for the network (MSE, vs. summing).')


args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)




###############################################################################
### PINN specific
###############################################################################

nu = args.nu
beta = args.beta
rho = args.rho

if args.system == 'diffusion': # just diffusion
    beta = 0.0
    rho = 0.0
elif args.system == 'convection':
    nu = 0.0
    rho = 0.0
elif args.system == 'rd': # reaction-diffusion
    beta = 0.0
elif args.system == 'reaction':
    nu = 0.0
    beta = 0.0

print('nu', nu, 'beta', beta, 'rho', rho)

# parse the layers list here
orig_layers = args.layers
layers = [int(item) for item in args.layers.split(',')]


############################
# Process data
############################

x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
t = np.linspace(0, 1, args.nt).reshape(-1, 1)
X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

# remove initial and boundaty data from X_star
t_noinitial = t[1:]
# remove boundary at x=0
x_noboundary = x[1:]
X_noboundary, T_noinitial = np.meshgrid(x_noboundary, t_noinitial)
X_star_noinitial_noboundary = np.hstack((X_noboundary.flatten()[:, None], T_noinitial.flatten()[:, None]))

# sample collocation points only from the interior (where the PDE is enforced)
X_f_train = sample_random(X_star_noinitial_noboundary, args.N_f)

if 'convection' in args.system or 'diffusion' in args.system:
    u_vals = convection_diffusion(args.u0_str, nu, beta, args.source, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
elif 'rd' in args.system:
    u_vals = reaction_diffusion_discrete_solution(args.u0_str, nu, rho, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
elif 'reaction' in args.system:
    u_vals = reaction_solution(args.u0_str, rho, args.xgrid, args.nt)
    G = np.full(X_f_train.shape[0], float(args.source))
else:
    print("WARNING: System is not specified.")

u_star = u_vals.reshape(-1, 1) # Exact solution reshaped into (n, 1)
Exact = u_star.reshape(len(t), len(x)) # Exact on the (x,t) grid

xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) # initial condition, from x = [-end, +end] and t=0
uu1 = Exact[0:1,:].T # u(x, t) at t=0
bc_lb = np.hstack((X[:,0:1], T[:,0:1])) # boundary condition at x = 0, and t = [0, 1]
uu2 = Exact[:,0:1] # u(-end, t)

# generate the other BC, now at x=2pi
t = np.linspace(0, 1, args.nt).reshape(-1, 1)
x_bc_ub = np.array([2*np.pi]*t.shape[0]).reshape(-1, 1)
bc_ub = np.hstack((x_bc_ub, t))

u_train = uu1 # just the initial condition
X_u_train = xx1 # (x,t) for initial condition

layers.insert(0, X_u_train.shape[-1])

############################
# setup the model
############################

set_seed(args.seed) # for weight initialization

architecture = getattr(models, args.model)

# manually update kwargs here
architecture.kwargs = dict(
    system=args.system,
    X_u_train=X_u_train, 
    u_train=u_train, 
    X_f_train=X_f_train,
    bc_lb=bc_lb, 
    bc_ub=bc_ub, 
    layers=layers, 
    G=G, 
    nu=nu, 
    beta=beta, 
    rho=rho,
    optimizer_name=args.optimizer_name,
    lr=args.lr,
    net=args.net, 
    L=args.L, 
    activation=args.activation, 
    loss_style=args.loss_style,
)

###############################################################################




# loaders, num_classes = data.loaders(
#     args.dataset,
#     args.data_path,
#     args.batch_size,
#     args.num_workers,
#     args.transform,
#     args.use_test
# )

#  architecture = getattr(models, args.model)

if args.curve is None:
    # model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    model = architecture.base(**architecture.kwargs)
else:
    curve = getattr(curves, args.curve)
    model = curves.CurveNet(
        num_classes, # TODO: we don't need this anymore ...
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
                    # base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
                    base_model = architecture.base(**architecture.kwargs)
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
                    
                print('Loading %s as point #%d' % (path, k))
                # base_model = torch.nn.DataParallel(base_model)
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


# criterion = F.cross_entropy
# regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)
# optimizer = torch.optim.SGD(
#     filter(lambda param: param.requires_grad, model.parameters()),
#     lr=args.lr,
#     momentum=args.momentum,
#     weight_decay=args.wd if args.curve is None else 0.0
# )


# start_epoch = 1
# if args.resume is not None:
#     # TODO: requires that optimizer_state was saved in checkpoint
#     # TODO: so we can't use this with e.g., pre-trained ResNet20 
#     print('Resume training from %s' % args.resume)
#     checkpoint = torch.load(args.resume)
#     if 'model_state' not in checkpoint:
#         checkpoint = dict(model_state=checkpoint)
#     start_epoch = checkpoint['epoch'] + 1
#     model.load_state_dict(checkpoint['model_state'])
#     optimizer.load_state_dict(checkpoint['optimizer_state'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

utils.save_checkpoint(
    args.dir,
    start_epoch - 1,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)





# has_bn = utils.check_bn(model)
test_res = {'loss': None, 'accuracy': None, 'nll': None}
for epoch in range(start_epoch, args.epochs + 1):
    time_ep = time.time()

    # lr = learning_rate_schedule(args.lr, epoch, args.epochs)
    # utils.adjust_learning_rate(optimizer, lr)

    # train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)
    # if args.curve is None or not has_bn:
    # test_res = utils.test(loaders['test'], model, criterion, regularizer)

    ###############################################################################
    ### PINN specific
    ###############################################################################

    # TODO: model refers to CurveNet
    #       but we just want to train the DNN?
    model.train()
    loss = model.loss_pinn()
    u_pred = model.predict(X_star)
    
    
    
    error_u_relative = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
    error_u_abs = np.mean(np.abs(u_star - u_pred))
    error_u_linf = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)

    print('Error u rel: %e' % (error_u_relative))
    print('Error u abs: %e' % (error_u_abs))
    print('Error u linf: %e' % (error_u_linf))
    
    train_res = dict(
        loss=loss,
        error_u_rel=error_u_relative,
        error_u_abs=error_u_abs,
        error_u_linf=error_u_linf,
    )


    ###############################################################################

    
    if epoch % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep
    
    values = [epoch, lr,
              train_res['loss'], 
              train_res['error_u_rel'],
              train_res['error_u_abs'],
              train_res['error_u_linf'],
              # test_res['nll'],
              # test_res['accuracy'], 
              time_ep]

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
