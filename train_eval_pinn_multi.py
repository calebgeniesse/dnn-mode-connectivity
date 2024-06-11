import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F
import copy

import curves
from curve_net_pinn import CurveNetPINN
import data
import models
import utils
import numpy as np
import pandas as pd


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

parser.add_argument('--model', type=str, default='PINNDNN', metavar='MODEL', required=True,
                    help='model name (default: None)')

parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                    help='curve type to use (default: None)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
# parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
#                    help='checkpoint to init start point (default: None)')
parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                    help='fix start point (default: off)')
# parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
#                     help='checkpoint to init end point (default: None)')
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
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#                     help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')


### MULTI PAIR VERSION
parser.add_argument('--init_file', type=str, default=None, metavar='CKPT',
                    help='CSV file listing pairs to compute MC between')
parser.add_argument('--input_folder', type=str, default="", help='where to find inputs')
parser.add_argument('--output_folder', type=str, default="", help='where to store outputs')
parser.add_argument('--eval_epoch', type=int, default=0, help='which epoch to use for MC')




# parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

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

parser.add_argument('--visualize', default=False, action="store_true", help='Visualize the solution.')



parser.add_argument('--num_points', type=int, default=5, metavar='N',
                    help='number of points on the curve (default: 61)')

# TODO: update this file based on input pairs
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




###############################################################################
### PINN specific
###############################################################################

from net_pbc import *
from systems_pbc import *
from pinn_utils import *
# from .visualize import *


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










########################################################
# Load & loop over pairs of models
########################################################

df_pairs = pd.read_csv(args.init_file)

for pair_idx in df_pairs.index.values:
    

    # get endpoints
    init_start = df_pairs.loc[pair_idx, 'init_start']
    init_end = df_pairs.loc[pair_idx, 'init_end']
    
    # configure ckpt
    # 
    # checkpoints/PINN_convection_beta_1_seed_0_seed_123/checkpoint-50.pt
    # 
    # TODO: sort and check for existing checkpoint ?
    # 
    # NOTE: just combine full names so we include lr info
    ckpt_folder = f"{init_start}_{init_end}".replace(".pt", "")
    ckpt = f"{ckpt_folder}/checkpoint-{args.eval_epoch}.pt"
    
    # TODO: configure full path based on another argument
    # if not os.path.exists(init_start):
    init_start = os.path.join(args.input_folder, init_start)
    init_end = os.path.join(args.input_folder, init_end)
    
    # TODO: configure full path based on another argument
    # if not os.path.exists(os.path.dirname(ckpt)):
    ckpt = os.path.join(args.output_folder, ckpt)
    dir_ = os.path.dirname(ckpt)
    
    # update args
    args.init_start = init_start
    args.init_end = init_end
    args.ckpt = ckpt
    args.dir = dir_
    
    # make target directories
    os.makedirs(args.dir, exist_ok=True)

    # log some stuff
    print(f"")
    print(f"Training Mode Connectivity for pair_idx={pair_idx} ...")
    print(f"    init_start = {os.path.relpath(args.init_start)}")
    print(f"    init_end = {os.path.relpath(args.init_end)}")
    print(f"    ckpt = {os.path.relpath(args.ckpt)}")
    print(f"")



    # check for existing checkponit
    # if os.path.exists(args.ckpt):
    #     print(f"Found existing ckpt!!!")
    #     print("")
    
    
    # check for existign result file 
    save_as_npz = os.path.join(args.dir, os.path.basename(args.ckpt).replace('.pt','_curve.npz'))
    if os.path.exists(args.ckpt):
        print(f"Found existing result file for ckpt (remove to recompute)")
        with np.load(save_as_npz) as results:
            mc_metric = results["mc_metric"]
        print(f"Mode Connectivity: {mc_metric}")
        print(f"[+] {save_as_npz}") 

        continue











    ############################
    # Process data
    ############################

    x = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
    t = np.linspace(0, 1, args.nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t) # all the X grid points T times, all the T grid points X times
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) # all the x,t "test" data

    # Training Error = (pred(X) vs. t) 
    # Test Error = (pred(X_star) vs. u_star)

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

    # parse the layers list here
    orig_layers = args.layers
    layers = [int(item) for item in args.layers.split(',')]
    layers.insert(0, X_u_train.shape[-1])

    ############################
    # setup the model
    ############################

    set_seed(args.seed) # for weight initialization

    architecture = getattr(models, args.model)

    # manually update kwargs here
    # architecture.kwargs = dict(
    #     system=args.system,
    #     X_u_train=X_u_train, 
    #     u_train=u_train, 
    #     X_f_train=X_f_train,
    #     bc_lb=bc_lb, 
    #     bc_ub=bc_ub, 
    #     layers=layers, 
    #     G=G, 
    #     nu=nu, 
    #     beta=beta, 
    #     rho=rho,
    #     optimizer_name=args.optimizer_name,
    #     lr=args.lr,
    #     net=args.net, 
    #     L=args.L, 
    #     activation=args.activation, 
    #     loss_style=args.loss_style,
    # )
    architecture.kwargs = dict(
            layers=layers, # [50,50,50,50,1], 
            activation='tanh',
            use_batch_norm=False, 
            use_instance_norm=False,
        )

    ###############################################################################

    PINN_MODEL = getattr(models, 'PINN')
    pinn_model = PINN_MODEL(
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
        model = CurveNetPINN(
            # num_classes, # TODO: we don't need this anymore ...
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

                    checkpoint_pinn = torch.load(path)
                    checkpoint = checkpoint_pinn.dnn.state_dict()
                    # if 'model_state' not in checkpoint:
                    #     checkpoint = dict(model_state=checkpoint)

                    # TODO: hacky
                    # if list(checkpoint.keys())[0].startswith('module'):
                    #     from collections import OrderedDict
                    #     new_state_dict = OrderedDict()
                    #     for state_dict_key, state_dict_value in checkpoint.items():
                    #         new_state_dict_key = state_dict_key.replace('module.','')
                    #         new_state_dict[new_state_dict_key] = state_dict_value
                    #     checkpoint = new_state_dict

                    print('Loading %s as point #%d' % (path, k))
                    # base_model = torch.nn.DataParallel(base_model)
                    base_model.load_state_dict(checkpoint)
                    model.import_base_parameters(base_model, k)
            if args.init_linear:
                print('Linear initialization.')
                model.init_linear()

    # model = torch.nn.DataParallel(model)
    model.cuda()
    pinn_model.cuda()
    pinn_model.dnn = copy.deepcopy(model) 

    # reset optimizer based on the curve model params
    from models.choose_optimizer_pbc import choose_optimizer
    if pinn_model.optimizer_name == "LBFGS":
        pinn_model.optimizer = choose_optimizer(
            args.optimizer_name, 
            filter(lambda param: param.requires_grad, pinn_model.dnn.parameters()),
            lr=args.lr,
            max_iter=20,
            max_eval=1.25 * 20,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1e-7,
        )

    elif pinn_model.optimizer_name == "SGD":
        pinn_model.optimizer = choose_optimizer(
            args.optimizer_name, 
            filter(lambda param: param.requires_grad, pinn_model.dnn.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd if args.curve is None else 0.0
        )
    else:
        pinn_model.optimizer = choose_optimizer(
            args.optimizer_name, 
            filter(lambda param: param.requires_grad, pinn_model.dnn.parameters()),
            lr=args.lr,
            weight_decay=args.wd if args.curve is None else 0.0
        )

    if torch.is_grad_enabled():
        pinn_model.optimizer.zero_grad(set_to_none=False)


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

    optimizer = pinn_model.optimizer


    start_epoch = 1
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

    # columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']
    columns = ['ep', 'lr', 'tr_loss', 
               'tr_error_u_rel', 'tr_error_u_abs', 'tr_error_u_linf',
               'te_error_u_rel', 'te_error_u_abs', 'te_error_u_linf',
               'time']

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
        # coeff_ts = [1.0, 0.5, 0]
        # for j,coeff_t in enumerate(coeff_ts):



        # pinn_models[j].dnn = copy.deepcopy(model.net)


        # pinn_model = PINN(dnn=DNN)
        # CurveNetPINN(architecture=DNNCurve)
        # def get_weights_copy(m):
        #     weights_path = 'weights_temp.pt'
        #     torch.save(m.state_dict(), weights_path)
        #     return mode.load_state_dict(torch.load(weights_path))

        # pinn_model.dnn.load_state_dict(copy.deepcopy(model.state_dict()))
        pinn_model.train()
        # model.load_state_dict(copy.deepcopy(pinn_model.dnn.state_dict())) 

        # TODO: we were running self.optimizer.zero_grad() everytime we run pinn model
        # TODO: now we do it during initialization, and after resetting the optimizer
        loss = pinn_model.loss_pinn()



        # Training Error = (pred(X) vs. t) 
        x_ = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
        t_ = np.linspace(0, 1, args.nt).reshape(-1, 1)
        X_, T_ = np.meshgrid(x_, t_) # all the X grid points T times, all the T grid points X times
        X_star_ = np.hstack((X_.flatten()[:, None], T_.flatten()[:, None])) # all the x,t "test" data

        u_pred = pinn_model.predict(X_star_)

        # print(u_pred)
        # print(u_star)
        # print(u_pred.max(), u_pred.min())
        # print(u_star.max(), u_star.min())

        tr_error_u_relative = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
        tr_error_u_abs = np.mean(np.abs(u_star - u_pred))
        tr_error_u_linf = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)


        # Test Error = (pred(X_star) vs. u_star)
        u_pred = pinn_model.predict(X_star)
        te_error_u_relative = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
        te_error_u_abs = np.mean(np.abs(u_star - u_pred))
        te_error_u_linf = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)


        # print('Error u rel: %e' % (error_u_relative))
        # print('Error u abs: %e' % (error_u_abs))
        # print('Error u linf: %e' % (error_u_linf))

        train_res = dict(
            loss=loss,
            tr_error_u_rel=tr_error_u_relative,
            tr_error_u_abs=tr_error_u_abs,
            tr_error_u_linf=tr_error_u_linf,
            te_error_u_rel=te_error_u_relative,
            te_error_u_abs=te_error_u_abs,
            te_error_u_linf=te_error_u_linf,
        )


        ###############################################################################

        # TODO: can we save the whole model? is state dict enough??

        if epoch % args.save_freq == 0:
            utils.save_checkpoint(
                args.dir,
                epoch,
                model_state=pinn_model.dnn.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        time_ep = time.time() - time_ep

        values = [epoch, args.lr,
                  train_res['loss'], 
                  train_res['tr_error_u_rel'],
                  train_res['tr_error_u_abs'],
                  train_res['tr_error_u_linf'],
                  train_res['te_error_u_rel'],
                  train_res['te_error_u_abs'],
                  train_res['te_error_u_linf'],              
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




        if args.visualize:

            from visualize_pinn import *
            path = f"heatmap_results/{args.system}_epoch_{epoch}"
            if not os.path.exists(path):
                os.makedirs(path)


            x_ = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
            t_ = np.linspace(0, 1, args.nt).reshape(-1, 1)
            X_, T_ = np.meshgrid(x_, t_) # all the X grid points T times, all the T grid points X times
            X_star_ = np.hstack((X_.flatten()[:, None], T_.flatten()[:, None])) # all the x,t "test" data

            # Training Error = (pred(X) vs. t) 
            u_pred = pinn_model.predict(X_star_)
            # print(u_pred.shape)
            # print(len(t_))
            # print(len(x_))

            u_pred = u_pred.reshape(len(t_), len(x_))

            exact_u(Exact, x_, t_, nu, beta, rho, orig_layers,
                    args.N_f, args.L, args.source, args.u0_str, args.system, 
                    path=path)
            u_diff(Exact, u_pred, x_, t_, nu, beta, rho, args.seed, orig_layers, 
                   args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, 
                   path=path)
            u_predict(u_vals, u_pred, x_, t_, nu, beta, rho, args.seed, orig_layers, 
                      args.N_f, args.L, args.source, args.lr, args.u0_str, args.system,
                      path=path)

            plt.close('all')

    if args.epochs % args.save_freq != 0:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            model_state=pinn_model.dnn.state_dict(),
            optimizer_state=optimizer.state_dict()
        )





    ###############################################################################
    ### EVAL
    ###############################################################################


    checkpoint = torch.load(args.ckpt)
    pinn_model.dnn.load_state_dict(checkpoint['model_state'])
    # pinn_model.dnn = copy.deepcopy(model)
    # pinn_model.load_state_dict(copy.deepcopy(model.state_dict()))

    # criterion = F.cross_entropy
    # regularizer = curves.l2_regularizer(args.wd)

    T = args.num_points
    ts = np.linspace(0.0, 1.0, T)
    tr_loss = np.zeros(T)
    tr_error_u_rel =  np.zeros(T)
    tr_error_u_abs =  np.zeros(T)
    tr_error_u_linf = np.zeros(T)
    te_error_u_rel =  np.zeros(T)
    te_error_u_abs =  np.zeros(T)
    te_error_u_linf = np.zeros(T)

    dl = np.zeros(T)

    previous_weights = None

    columns = ['t', 'Train loss', 
               'Train error (rel)', 'Train error (abs)', 'Train error (linf)',
               'Test error (rel)', 'Test error (abs)', 'Test error (linf)',
              ]

    t = torch.FloatTensor([0.0]).cuda()
    for i, t_value in enumerate(ts):
        t.data.fill_(t_value)

        # pinn_model.dnn = model

        weights = model.weights(t)
        if previous_weights is not None:
            dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
        previous_weights = weights.copy()

        ### PINN STUFF
        # pinn_model.dnn = copy.deepcopy(model)
        # pinn_model.dnn.eval() #.to(device)

        # pinn_model.dnn.load_state_dict(copy.deepcopy(model.state_dict()))

        # pinn_model.dnn = copy.deepcopy(model)        
        tr_loss[i] = pinn_model.loss_pinn(coeff_t=t)




        # utils.update_bn(loaders['train'], model, t=t)
        # r_res = utils.test(loaders['train'], model, criterion, regularizer, t=t)
        # te_res = utils.test(loaders['test'], model, criterion, regularizer, t=t)


        x_ = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
        t_ = np.linspace(0, 1, args.nt).reshape(-1, 1)
        X_, T_ = np.meshgrid(x_, t_) # all the X grid points T times, all the T grid points X times
        X_star_ = np.hstack((X_.flatten()[:, None], T_.flatten()[:, None])) # all the x,t "test" data

        # Training Error = (pred(X) vs. t) 
        # u_pred = pinn_model.predict(X_, coeff_t=t)
        # tr_error_u_rel[i] = np.linalg.norm(t_-u_pred, 2)/np.linalg.norm(t_, 2)
        # tr_error_u_abs[i] = np.mean(np.abs(t_ - u_pred))
        # tr_error_u_linf[i] = np.linalg.norm(t_ - u_pred, np.inf)/np.linalg.norm(t_, np.inf)

        u_pred = pinn_model.predict(X_star_, coeff_t=t)
        tr_error_u_rel[i] = np.linalg.norm(u_star - u_pred, 2)/np.linalg.norm(u_star, 2)
        tr_error_u_abs[i] = np.mean(np.abs(u_star - u_pred))
        tr_error_u_linf[i] = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)



        # Test Error = (pred(X_star) vs. u_star)
        u_pred = pinn_model.predict(X_star, coeff_t=t)
        te_error_u_rel[i] = np.linalg.norm(u_star-u_pred, 2)/np.linalg.norm(u_star, 2)
        te_error_u_abs[i] = np.mean(np.abs(u_star - u_pred))
        te_error_u_linf[i] = np.linalg.norm(u_star - u_pred, np.inf)/np.linalg.norm(u_star, np.inf)



        values = [t, tr_loss[i],
                  tr_error_u_rel[i], tr_error_u_abs[i], tr_error_u_linf[i],
                  te_error_u_rel[i], te_error_u_abs[i], te_error_u_linf[i]
                 ]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)



        if args.visualize:
            from visualize_pinn import *
            path = f"heatmap_results/{args.system}_t_{t_value}"
            if not os.path.exists(path):
                os.makedirs(path)


            x_ = np.linspace(0, 2*np.pi, args.xgrid, endpoint=False).reshape(-1, 1) # not inclusive
            t_ = np.linspace(0, 1, args.nt).reshape(-1, 1)
            X_, T_ = np.meshgrid(x_, t_) # all the X grid points T times, all the T grid points X times
            X_star_ = np.hstack((X_.flatten()[:, None], T_.flatten()[:, None])) # all the x,t "test" data

            # Training Error = (pred(X) vs. t) 
            u_pred = pinn_model.predict(X_star_, coeff_t=t)
            # print(u_pred.shape)
            # print(len(t_))
            # print(len(x_))

            u_pred = u_pred.reshape(len(t_), len(x_))

            exact_u(Exact, x_, t_, nu, beta, rho, orig_layers,
                    args.N_f, args.L, args.source, args.u0_str, args.system, 
                    path=path)
            u_diff(Exact, u_pred, x_, t_, nu, beta, rho, args.seed, orig_layers, 
                   args.N_f, args.L, args.source, args.lr, args.u0_str, args.system, 
                   path=path)
            u_predict(u_vals, u_pred, x_, t_, nu, beta, rho, args.seed, orig_layers, 
                      args.N_f, args.L, args.source, args.lr, args.u0_str, args.system,
                      path=path)

            plt.close('all')



    def stats(values, dl):
        min = np.min(values)
        max = np.max(values)
        avg = np.mean(values)
        int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
        return min, max, avg, int


    tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
    tr_error_u_rel_min, tr_error_u_rel_max, tr_error_u_rel_avg, tr_error_u_rel_int = stats(tr_error_u_rel, dl)
    tr_error_u_abs_min, tr_error_u_abs_max, tr_error_u_abs_avg, tr_error_u_abs_int = stats(tr_error_u_abs, dl)
    tr_error_u_linf_min, tr_error_u_linf_max, tr_error_u_linf_avg, tr_error_u_linf_int = stats(tr_error_u_linf, dl)

    te_error_u_rel_min, te_error_u_rel_max, te_error_u_rel_avg, te_error_u_rel_int = stats(te_error_u_rel, dl)
    te_error_u_abs_min, te_error_u_abs_max, te_error_u_abs_avg, te_error_u_abs_int = stats(te_error_u_abs, dl)
    te_error_u_linf_min, te_error_u_linf_max, te_error_u_linf_avg, te_error_u_linf_int = stats(te_error_u_linf, dl)

    print('Length: %.2f' % np.sum(dl))
    print(tabulate.tabulate([
            ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
    ['train error (rel)', te_error_u_rel[0], te_error_u_rel[-1], te_error_u_rel_min, te_error_u_rel_max, te_error_u_rel_avg, te_error_u_rel_int],
    ['train error (abs)', te_error_u_abs[0], te_error_u_abs[-1], te_error_u_abs_min, te_error_u_abs_max, te_error_u_abs_avg, te_error_u_abs_int],
    ['train error (linf)', te_error_u_linf[0], te_error_u_linf[-1], te_error_u_linf_min, te_error_u_linf_max, te_error_u_linf_avg, te_error_u_linf_int],

    ], [
            '', 'start', 'end', 'min', 'max', 'avg', 'int'
        ], tablefmt='simple', floatfmt='10.4f'))


    ### define metric
    # tr_err = tr_error_u_abs
    tr_err = tr_error_u_rel
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
        tr_error_u_rel=tr_error_u_abs,
        tr_error_u_rel_min=tr_error_u_rel_min,
        tr_error_u_rel_max=tr_error_u_rel_max,
        tr_error_u_rel_avg=tr_error_u_rel_avg,
        tr_error_u_rel_int=tr_error_u_rel_int,
        tr_error_u_abs=tr_error_u_abs,
        tr_error_u_abs_min=tr_error_u_abs_min,
        tr_error_u_abs_max=tr_error_u_abs_max,
        tr_error_u_abs_avg=tr_error_u_abs_avg,
        tr_error_u_abs_int=tr_error_u_abs_int,
        # TODO: save the rest
        mc_metric=mc_metric,
    )

    print(f"[+] {save_as_npz}")