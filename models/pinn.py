import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
from copy import deepcopy

import curves

from .choose_optimizer_pbc import *


__all__ = [
    'PINNDNN',
    'PINN',
]




# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

    
    

################################################################################
### helpers
################################################################################  

class SequentialCurve(nn.Sequential):
    """ TODO: move to curves.Sequential at some point """
    def forward(self, x, coeff_t):
        for module in self._modules.values():
            try:
                x = module(x, coeff_t)
            except TypeError:
                x = module(x)
        return x    
    
    
      
################################################################################
### DNN, DNNCurve
################################################################################  
    
# the deep neural network
class DNNBase(torch.nn.Module):
    def __init__(self, layers, activation, use_batch_norm=False, use_instance_norm=False):
        super(DNNBase, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'gelu':
            self.activation = torch.nn.GELU
        elif activation == 'sin':
            self.activation = Sine
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )

            if self.use_batch_norm:
                layer_list.append(('batchnorm_%d' % i, torch.nn.BatchNorm1d(num_features=layers[i+1])))
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, torch.nn.InstanceNorm1d(num_features=layers[i+1])))

            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out
    
    
    
# the deep neural network
class DNNCurve(torch.nn.Module):
    def __init__(self, layers, activation, fix_points, use_batch_norm=False, use_instance_norm=False):
        super(DNNCurve, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        if activation == 'identity':
            self.activation = torch.nn.Identity
        elif activation == 'tanh':
            self.activation = torch.nn.Tanh
        elif activation == 'relu':
            self.activation = torch.nn.ReLU
        elif activation == 'gelu':
            self.activation = torch.nn.GELU
        elif activation == 'sin':
            self.activation = Sine
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, curves.Linear(layers[i], layers[i+1], fix_points))
            )

            if self.use_batch_norm:
                # layer_list.append(('batchnorm_%d' % i, curves.BatchNorm1d(num_features=layers[i+1])))
                layer_list.append(('batchnorm_%d' % i, curves._BatchNorm(num_features=layers[i+1], fix_points=fix_points)))
            
            # TODO: this may cause problems (no curve version)
            if self.use_instance_norm:
                layer_list.append(('instancenorm_%d' % i, nn.InstanceNorm1d(num_features=layers[i+1])))

            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), curves.Linear(layers[-2], layers[-1], fix_points))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = SequentialCurve(layerDict)

    def forward(self, x, coeff_t):
        # print(coeff_t)
        out = self.layers(x, coeff_t)
        return out    
    
    
    
    
 
################################################################################
### PINNBase, PINNCurve
################################################################################     

# class PhysicsInformedNN_pbc():
class PINN(torch.nn.Module):
    
    """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    def __init__(self, system, X_u_train, u_train, X_f_train, bc_lb, bc_ub, layers, G, nu, beta, rho, optimizer_name, lr,
        net, L=1, activation='tanh', loss_style='mean'):

        super(PINN, self).__init__()
        
        self.system = system

        self.x_u = torch.tensor(X_u_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_u = torch.tensor(X_u_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).float().to(device)
        self.net = net

        if self.net == 'DNN':
            self.dnn = DNNBase(layers, activation).to(device)
        else: # "pretrained" can be included in model path
            # the dnn is within the PINNs class
            self.dnn = torch.load(net).dnn

        self.u = torch.tensor(u_train, requires_grad=True).float().to(device)
        self.layers = layers
        self.nu = nu
        self.beta = beta
        self.rho = rho

        self.G = torch.tensor(G, requires_grad=True).float().to(device)
        self.G = self.G.reshape(-1, 1)

        self.L = L

        self.lr = lr
        self.optimizer_name = optimizer_name

        self.optimizer = choose_optimizer(self.optimizer_name, self.dnn.parameters(), self.lr)

        self.loss_style = loss_style

        self.iter = 0
        
        if torch.is_grad_enabled():
            self.optimizer.zero_grad(set_to_none=False)

    def net_u(self, x, t, coeff_t=None):
        """The standard DNN that takes (x,t) --> u."""
        if coeff_t is not None:
            u = self.dnn(torch.cat([x, t], dim=1), coeff_t)
        else:
            u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t, coeff_t=None):
        """ Autograd for calculating the residual for different systems."""
    
        u = self.net_u(x, t, coeff_t=coeff_t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]

        if 'convection' in self.system or 'diffusion' in self.system:
            f = u_t - self.nu*u_xx + self.beta*u_x - self.G
        elif 'rd' in self.system:
            f = u_t - self.nu*u_xx - self.rho*u + self.rho*u**2
        elif 'reaction' in self.system:
            f = u_t - self.rho*u + self.rho*u**2
        return f

    def net_b_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
        """For taking BC derivatives."""

        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]

        return u_lb_x, u_ub_x

    def loss_pinn(self, coeff_t=None, verbose=True):
        """ Loss function. """
        
        if torch.is_grad_enabled():
            self.optimizer.zero_grad(set_to_none=False)
            
        u_pred = self.net_u(self.x_u, self.t_u, coeff_t=coeff_t)
        u_pred_lb = self.net_u(self.x_bc_lb, self.t_bc_lb, coeff_t=coeff_t)
        u_pred_ub = self.net_u(self.x_bc_ub, self.t_bc_ub, coeff_t=coeff_t)
        if self.nu != 0:
            u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
        f_pred = self.net_f(self.x_f, self.t_f, coeff_t=coeff_t)

        if self.loss_style == 'mean':
            loss_u = torch.mean((self.u - u_pred) ** 2)
            loss_b = torch.mean((u_pred_lb - u_pred_ub) ** 2)
            if self.nu != 0:
                loss_b += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.mean(f_pred ** 2)
        elif self.loss_style == 'sum':
            loss_u = torch.sum((self.u - u_pred) ** 2)
            loss_b = torch.sum((u_pred_lb - u_pred_ub) ** 2)
            if self.nu != 0:
                loss_b += torch.sum((u_pred_lb_x - u_pred_ub_x) ** 2)
            loss_f = torch.sum(f_pred ** 2)

        loss = loss_u + loss_b + self.L*loss_f

            
        if loss.requires_grad:
            loss.backward()

        grad_norm = np.nan
        for (name,p) in self.dnn.named_parameters():
            # skip gradient of fixed ends
            if p.requires_grad is False:
                # print(f"({name}).requires_grad is False)")
                continue
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5

        if verbose:
            if self.iter % 100 == 0:
                print(
                    'epoch %d, gradient: %.5e, loss: %.5e, loss_u: %.5e, loss_b: %.5e, loss_f: %.5e' % (self.iter, grad_norm, loss.item(), loss_u.item(), loss_b.item(), loss_f.item())
                )
            self.iter += 1

        return loss

    def train(self):
        self.dnn.train()
        self.optimizer.step(self.loss_pinn)

    def predict(self, X, coeff_t=None):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t, coeff_t=coeff_t)
        u = u.detach().cpu().numpy()

        return u

    # def forward(self, X, coeff_t):
    #     x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
    #     # t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

    #     self.dnn.eval()
    #     return self.dnn(x, coeff_t)


    # # class PINNCurve(PINNBase):

#     """PINNs (convection/diffusion/reaction) for periodic boundary conditions."""
    
#     ### TODO: I don't think this will work because no "forward method"
#     ###       ... not sure where coeff_t will be passed?
#     ###       ... we may need to adapt a new CurveNet architecture
    
#     def __init__(self, fix_points, *args, **kwargs):
#     super(PINNCurve, self).__init__(*args, **kwargs):
#         self.dnn = DNNCurve(layers, activation, fix_points).to(device)
        
    
    
    
################################################################################
### models
################################################################################     

class PINNDNN:
    """ Use BN and Residuals by default """
    base = DNNBase
    curve = DNNCurve
    kwargs = dict(
        layers=[50,50,50,50,1], 
        activation='tanh',
        use_batch_norm=False, 
        use_instance_norm=False,
    )
   
    

# class PINN_convection_beta_1:
#     base = PINNBase
#     curve = PINNCurve
#     kwargs = {
#         'system': 'convection',
#         'beta': 1.0, 
#     }

    

# class PINN_convection_beta_50:
#     base = PINNBase
#     curve = PINNCurve
#     kwargs = {
#         'system': 'convection',
#         'beta': 50.0, 
#     }
    