from __future__ import absolute_import
'''Resnet for cifar dataset. 
Ported form 
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei 
'''
import torch.nn as nn
import math
from copy import deepcopy

import curves


__all__ = [
    'resnet', 
    'resnet20',
    'resnet20_batch_norm_True_residual_True',
    'resnet20_batch_norm_True_residual_False',    
    'resnet20_batch_norm_False_residual_True',
]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv3x3curve(in_planes, out_planes, fix_points, stride=1):
    return curves.Conv2d(in_planes, 
                         out_planes, 
                         kernel_size=3, 
                         fix_points=fix_points, 
                         stride=stride,
                         padding=1, 
                         bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 residual_not,
                 batch_norm_not,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm_not:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm_not:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual_not:
            out += residual
        out = self.relu(out)

        return out

    
class BasicBlockCurve(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,
                 residual_not,
                 batch_norm_not,
                 fix_points, 
                 stride=1, 
                 downsample=None):
        super(BasicBlockCurve, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3curve(inplanes, planes, stride=stride, fix_points=fix_points)
        self.conv2 = conv3x3curve(planes, planes, fix_points=fix_points)
        if self.batch_norm_not:
            self.bn1 = curves.BatchNorm2d(planes, fix_points=fix_points)
            self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
            # self.bn3 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, coeffs_t):
        
        # TODO: adapted PreResNet (double check order of operations?)
        
        residual = x

        out = self.conv1(x, coeffs_t)
        if self.batch_norm_not:
            out = self.bn1(out, coeffs_t)
        out = self.relu(out)

        out = self.conv2(out, coeffs_t)
        if self.batch_norm_not:
            out = self.bn2(out, coeffs_t)

        if self.residual_not and (self.downsample is not None):
            residual = self.downsample(x, coeffs_t)
            # residual = self.bn3(residual, coeffs_t)
            
        if self.residual_not:
            out += residual
            
        out = self.relu(out)
        
    
        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 residual_not,
                 batch_norm_not,
                 stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batch_norm_not:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm_not:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.batch_norm_not:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            
        if self.residual_not:
            out += residual

        out = self.relu(out)

        return out

    
class BottleneckCurve(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, 
                 residual_not,
                 batch_norm_not,
                 fix_points, stride=1, 
                 downsample=None):
        super(BottleneckCurve, self).__init__()
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.conv1 = curves.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                                   fix_points=fix_points)
        self.conv2 = curves.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False, fix_points=fix_points)
        self.conv3 = curves.Conv2d(planes, planes * 4, kernel_size=1, bias=False,
                                   fix_points=fix_points)
        if self.batch_norm_not:
            self.bn1 = curves.BatchNorm2d(inplanes, fix_points=fix_points)
            self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
            self.bn3 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, coeffs_t):
        residual = x

        # TODO: adapted PreResNet (double check order of operations?)

        out = self.conv1(x, coeffs_t)
        if self.batch_norm_not:
            out = self.bn1(out, coeffs_t)
        out = self.relu(out)

        out = self.conv2(out, coeffs_t)
        if self.batch_norm_not:
            out = self.bn2(out, coeffs_t)
        out = self.relu(out)

        out = self.conv3(out, coeffs_t)
        if self.batch_norm_not:
            out = self.bn3(out, coeffs_t)

        if self.downsample is not None:
            residual = self.downsample(x, coeffs_t)

        if self.residual_not:
            out += residual

        out = self.relu(out)

        return out

    
    
    
    
    
ALPHA_ = 1


class ResNetBase(nn.Module):

    def __init__(self,
                 depth,
                 residual_not=True,
                 batch_norm_not=True,
                 base_channel=16,
                 num_classes=10):
        super(ResNetBase, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        # block = Bottleneck if depth >=44 else BasicBlock
        block = BasicBlock

        self.base_channel = int(base_channel)
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.inplanes = self.base_channel * ALPHA_
        self.conv1 = nn.Conv2d(3,
                               self.base_channel * ALPHA_,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        if self.batch_norm_not:
            self.bn1 = nn.BatchNorm2d(self.base_channel * ALPHA_)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.base_channel * ALPHA_, n,
                                       self.residual_not, self.batch_norm_not)
        self.layer2 = self._make_layer(block,
                                       self.base_channel * 2 * ALPHA_,
                                       n,
                                       self.residual_not,
                                       self.batch_norm_not,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       self.base_channel * 4 * ALPHA_,
                                       n,
                                       self.residual_not,
                                       self.batch_norm_not,
                                       stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.base_channel * 4 * ALPHA_ * block.expansion,
                            num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    residual_not,
                    batch_norm_not,
                    stride=1):
        downsample = None
        if (stride != 1 or
                self.inplanes != planes * block.expansion) and (residual_not):
            if batch_norm_not:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,
                              planes * block.expansion,
                              kernel_size=1,
                              stride=stride,
                              bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,
                              planes * block.expansion,
                              kernel_size=1,
                              stride=stride,
                              bias=False),)

        layers = nn.ModuleList()
        layers.append(
            block(self.inplanes, planes, residual_not, batch_norm_not, stride,
                  downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, residual_not, batch_norm_not))

        # return nn.Sequential(*layers)
        return layers

    def forward(self, x):
        output_list = []
        x = self.conv1(x)
        if self.batch_norm_not:
            x = self.bn1(x)
        x = self.relu(x)  # 32x32
        output_list.append(x.view(x.size(0), -1))

        for layer in self.layer1:
            x = layer(x)  # 32x32
            output_list.append(x.view(x.size(0), -1))
        for layer in self.layer2:
            x = layer(x)  # 16x16
            output_list.append(x.view(x.size(0), -1))
        for layer in self.layer3:
            x = layer(x)  # 8x8
            output_list.append(x.view(x.size(0), -1))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output_list.append(x.view(x.size(0), -1))

        # return output_list, x
        return x

    
class ModuleListCurve(nn.ModuleList):
    def forward(self, x, coeff_t):
        for module in self.modules():
            x = module(x, coeff_t)
        return x

            
class ResNetCurve(nn.Module):

    def __init__(self, 
                 num_classes, 
                 fix_points, 
                 depth=20,
                 residual_not=True,
                 batch_norm_not=True,
                 base_channel=16):
        super(ResNetCurve, self).__init__()
        if depth >= 44:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = BottleneckCurve
        else:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlockCurve

        self.base_channel = int(base_channel)
        self.residual_not = residual_not
        self.batch_norm_not = batch_norm_not
        self.inplanes = 16
        self.conv1 = curves.Conv2d(3, 16, kernel_size=3, padding=1,
                                   bias=False, fix_points=fix_points)
        if self.batch_norm_not:
            self.bn = curves.BatchNorm2d(self.base_channel * block.expansion, fix_points=fix_points)
        # self.layer1 = self._make_layer(block, 16, n, fix_points=fix_points)
        # self.layer2 = self._make_layer(block, 32, n, stride=2, fix_points=fix_points)
        # self.layer3 = self._make_layer(block, 64, n, stride=2, fix_points=fix_points)
        self.layer1 = self._make_layer(block, 
                                       self.base_channel * ALPHA_, 
                                       n,
                                       residual_not=self.residual_not,
                                       batch_norm_not=self.batch_norm_not,
                                       fix_points=fix_points)
        self.layer2 = self._make_layer(block,
                                       self.base_channel * 2 * ALPHA_,
                                       n,
                                       residual_not=self.residual_not,
                                       batch_norm_not=self.batch_norm_not,
                                       stride=2,
                                       fix_points=fix_points)
        self.layer3 = self._make_layer(block,
                                       self.base_channel * 4 * ALPHA_,
                                       n,
                                       residual_not=self.residual_not,
                                       batch_norm_not=self.batch_norm_not,
                                       stride=2,
                                       fix_points=fix_points)
        
        
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = curves.Linear(64 * block.expansion, num_classes, fix_points=fix_points)

        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, curves.BatchNorm2d):
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.fill_(1)
                    getattr(m, 'bias_%d' % i).data.zero_()

        # def _make_layer_preresnet(self, 
        #                 block, 
        #                 planes, 
        #                 blocks, 
        #                 fix_points, 
        #                 stride=1):
        #     downsample = None
        #     if stride != 1 or self.inplanes != planes * block.expansion:
        #         downsample = curves.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
        #                                    stride=stride, bias=False, fix_points=fix_points)

        #     layers = list()
        #     layers.append(block(self.inplanes, planes, fix_points=fix_points, stride=stride,
        #                         downsample=downsample))
        #     self.inplanes = planes * block.expansion
        #     for i in range(1, blocks):
        #         layers.append(block(self.inplanes, planes, fix_points=fix_points))

        #     return nn.ModuleList(layers)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    residual_not=True,
                    batch_norm_not=True,
                    fix_points=False, 
                    stride=1):
        downsample = None
        if (stride != 1 or
                self.inplanes != planes * block.expansion) and (residual_not):
            if batch_norm_not:
                downsample = nn.Sequential(
                    # nn.Conv2d(self.inplanes,
                    #           planes * block.expansion,
                    #           kernel_size=1,
                    #           stride=stride,
                    #           bias=False),
                    curves.Conv2d(self.inplanes, 
                                  planes * block.expansion, 
                                  kernel_size=1,
                                  stride=stride, 
                                  bias=False, 
                                  fix_points=fix_points),
                    # TODO: do we need a curves.BatchNorm2d???
                    # nn.BatchNorm2d(planes * block.expansion),
                    # curves.BatchNorm2d(planes * block.expansion),
                    curves.BatchNorm2d(planes * block.expansion, fix_points=fix_points),
                    )
            else:
                downsample = nn.Sequential(
                    curves.Conv2d(self.inplanes, 
                                  planes * block.expansion, 
                                  kernel_size=1,
                                  stride=stride, 
                                  bias=False, 
                                  fix_points=fix_points),
                )
            
        
                            
        
        layers = nn.ModuleList()
        layers.append(
            block(self.inplanes, planes, 
                  residual_not=residual_not, 
                  batch_norm_not=batch_norm_not, 
                  fix_points=fix_points, stride=stride, 
                  downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, 
                      residual_not=residual_not, 
                      batch_norm_not=batch_norm_not, 
                      fix_points=fix_points))

        # return nn.Sequential(*layers)
        return layers

    def forward2(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        if self.batch_norm_not:
            x = self.bn(x, coeffs_t)
        x = self.relu(x)

        for block in self.layer1:  # 32x32
            x = block(x, coeffs_t)
        for block in self.layer2:  # 16x16
            x = block(x, coeffs_t)
        for block in self.layer3:  # 8x8
            x = block(x, coeffs_t)
       
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, coeffs_t)

        return x    
    
    def forward(self, x, coeffs_t):
        output_list = []
        x = self.conv1(x, coeffs_t)
        if self.batch_norm_not:
            x = self.bn(x, coeffs_t)
        x = self.relu(x)  # 32x32
        output_list.append(x.view(x.size(0), -1))

        for layer in self.layer1:
            x = layer(x, coeffs_t)  # 32x32
            output_list.append(x.view(x.size(0), -1))
        for layer in self.layer2:
            x = layer(x, coeffs_t)  # 16x16
            output_list.append(x.view(x.size(0), -1))
        for layer in self.layer3:
            x = layer(x, coeffs_t)  # 8x8
            output_list.append(x.view(x.size(0), -1))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, coeffs_t)
        output_list.append(x.view(x.size(0), -1))

        # return output_list, x
        return x



def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNetBase(**kwargs)



class resnet20:
    """ Use BN and Residuals by default """
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {
        'depth': 20,
        'batch_norm_not': True,
        'residual_not' : True,
    }
    
    
class resnet20_batch_norm_True_residual_True:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {
        'depth': 20,
        'batch_norm_not': True,
        'residual_not' : True,
    }
    
class resnet20_batch_norm_True_residual_False:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {
        'depth': 20,
        'batch_norm_not': True,
        'residual_not' : False,
    }
    
class resnet20_batch_norm_False_residual_True:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {
        'depth': 20,
        'batch_norm_not': False,
        'residual_not' : True,
    }

    
