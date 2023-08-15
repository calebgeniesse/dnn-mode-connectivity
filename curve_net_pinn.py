import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
# from torch.nn.modules.utils import _pair
# from scipy.special import binom

from curves import CurveModule




class CurveNetPINN(Module):
    
    """ TODO: only difference now is to drop num_classes,
              but this could be done using num_classes=None?
    """

    def __init__(self, curve, architecture, num_bends, fix_start=True, fix_end=True,
                 architecture_kwargs={}):
        super(CurveNetPINN, self).__init__()

        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]

        self.curve = curve
        self.architecture = architecture

        self.l2 = 0.0
        self.coeff_layer = self.curve(self.num_bends)
        self.net = self.architecture(fix_points=self.fix_points, **architecture_kwargs)
        self.curve_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, CurveModule):
                self.curve_modules.append(module) 


    def import_base_parameters(self, base_model, index):
        # print([_ for _,__ in list(self.net.named_parameters())[index::self.num_bends]])
        # print([_ for _,__ in list(base_model.named_parameters())])
        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        # named_parameters = list(self.net.named_parameters())[index::self.num_bends]
        # base_named_parameters = list(base_model.named_parameters())
        for i,(parameter, base_parameter) in enumerate(zip(parameters, base_parameters)):
            parameter.data.copy_(base_parameter.data)


    def import_base_buffers(self, base_model):
        for buffer, base_buffer in zip(self.net._all_buffers(), base_model._all_buffers()):
            buffer.data.copy_(base_buffer.data)

    def export_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            base_parameter.data.copy_(parameter.data)

    def init_linear(self):
        parameters = list(self.net.parameters())
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i:i+self.num_bends]
            for j in range(1, self.num_bends - 1):
                alpha = j * 1.0 / (self.num_bends - 1)
                weights[j].data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

    def weights(self, t):
        coeffs_t = self.coeff_layer(t)
        weights = []
        for module in self.curve_modules:
            weights.extend([w for w in module.compute_weights_t(coeffs_t) if w is not None])
        return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])

    def _compute_l2(self):
        self.l2 = sum(module.l2 for module in self.curve_modules)

    def forward(self, input, t=None):
        if t is None:
            t = input.data.new(1).uniform_()
        coeffs_t = self.coeff_layer(t)
        output = self.net(input, coeffs_t)
        self._compute_l2()
        return output


