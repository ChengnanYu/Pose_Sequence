import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
###############################################################################
# Functions
###############################################################################


def weight_init_googlenet(key, module, weights=None):
    if weights is None:
        init.constant_(module.bias.data, 0.0)
        if key == "XYZ":
            init.normal_(module.weight.data, 0.0, 0.5)
        else:
            init.normal_(module.weight.data, 0.0, 0.01)
    else:
        # print(key, weights[(key+"_1").encode()].shape, module.bias.size())
        module.bias.data[...] = torch.from_numpy(weights[(key+"_1").encode()])
        module.weight.data[...] = torch.from_numpy(weights[(key+"_0").encode()])
    return module

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, sx, sq, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[],
             init_from=None, isTest=False):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'posenet':
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids)
    elif which_model_netG == 'poselstm':
        netG = PoseLSTM(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids)
    elif which_model_netG == 'posenetnobeta':
        netG = PoseNetNoBeta(input_nc, sx=sx, sq=sq, weights=init_from, isTest=isTest, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG

##############################################################################
# Classes
##############################################################################

# defines the spatial LSTM after googlenet
'''
class SpLSTM(nn.Module):
    def __init__(self, weights=None):
        super(SpLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)

    def forward(self, input):
        #input shape [batch size, 2048]
        input1 = input.view(input.size(0),32,64)
        input2 = input1.transpose(1,2)
        #inverse the input tensor at axis=1
        inv_idx = torch.arange(input.size(1)-1, -1, -1, dtype=torch.long, device=input.device)
        inv_input = input.index_select(1,inv_idx)
        input3 = inv_input.view(input.size(0),32,64)
        input4 = input3.transpose(1,2)
        outlstm1, _ = self.lstm1(input1) #shape (batch_size, seq_length, hidden_size)
        outlstm2, _ = self.lstm2(input2)
        outlstm3, _ = self.lstm1(input3)
        outlstm4, _ = self.lstm2(input4)
        out1 = outlstm1[:, -1, :] # take the state of the last time step
        out2 = outlstm2[:, -1, :]
        out3 = outlstm3[:, -1, :]
        out4 = outlstm4[:, -1, :]
        output = torch.cat((out1, out2, out3, out4), 1) #shape [batch size, 128]
        return output
'''

class SpLSTM(nn.Module):
    def __init__(self, seq_len1, seq_len2, hidden_size, weights=None):
        super(SpLSTM, self).__init__()
        self.seq_len1 = seq_len1
        self.seq_len2 = seq_len2
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size=self.seq_len1, hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=self.seq_len2, hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, input):
        #input shape [batch size, 2048]
        input1 = input.view(input.size(0), self.seq_len2, self.seq_len1)
        input2 = input1.transpose(1,2)
        # initialize hidden states and cell states of shape (num_layers * num_directions, batch, hidden_size)
        hidden1 = (torch.randn(2, input.size(0), self.hidden_size).to(input.device),torch.randn(2, input.size(0), self.hidden_size).to(input.device))
        hidden2 = (torch.randn(2, input.size(0), self.hidden_size).to(input.device),torch.randn(2, input.size(0), self.hidden_size).to(input.device))
        outlstm1, hidden1 = self.lstm1(input1, hidden1) #shape (batch_size, seq_length, hidden_size*2)
        outlstm2, hidden2 = self.lstm2(input2, hidden2)
        out1 = hidden1[0][0]
        out2 = hidden1[0][1]
        out3 = hidden2[0][0]
        out4 = hidden2[0][1]
        output = torch.cat((out1, out2, out3, out4), 1) #shape [batch size, 4*hiddensize]
        return output

# defines the regression heads for googlenet
class RegressionHead1(nn.Module):
    def __init__(self, lossID, weights=None):
        super(RegressionHead1, self).__init__()
        nc = {"loss1": 512, "loss2": 528}
        self.projection = nn.Sequential(*[nn.AvgPool2d(kernel_size=5, stride=3),
                                              weight_init_googlenet(lossID+"/conv", nn.Conv2d(nc[lossID], 128, kernel_size=1), weights),
                                              nn.ReLU(inplace=True)])
        self.cls_splstm = SpLSTM(seq_len1=32, seq_len2=32, hidden_size=256)
        self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet(lossID+"/fc", nn.Linear(2048, 1024), weights),
                                               nn.ReLU(inplace=True), self.cls_splstm, nn.Dropout(0.7)])
        self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(1024, 3))
        self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(1024, 4))

    def forward(self, input):
        output = self.projection(input)
        output = self.cls_fc_pose(output.view(output.size(0), -1))
        output_xy = self.cls_fc_xy(output)
        output_wpqr = self.cls_fc_wpqr(output)
        output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
        return [output_xy, output_wpqr]

class RegressionHead2(nn.Module):
    def __init__(self, lossID, weights=None):
        super(RegressionHead2, self).__init__()
        self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
        self.cls_splstm = SpLSTM(seq_len1=64, seq_len2=32, hidden_size=256)
        self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet("pose", nn.Linear(1024, 2048)),
                                               nn.ReLU(inplace=True), self.cls_splstm, nn.Dropout(0.5)])
        self.cls_fc_xy = nn.Linear(1024, 3)
        self.cls_fc_wpqr = nn.Linear(1024, 4)

    def forward(self, input):
        output = self.projection(input)
        output = self.cls_fc_pose(output.view(output.size(0), -1))
        output_xy = self.cls_fc_xy(output)
        output_wpqr = self.cls_fc_wpqr(output)
        output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
        return [output_xy, output_wpqr]


# defines the regression heads for googlenet
class RegressionHead(nn.Module):
    def __init__(self, lossID, weights=None):
        super(RegressionHead, self).__init__()
        if lossID != "loss3":
            nc = {"loss1": 512, "loss2": 528}
            self.projection = nn.Sequential(*[nn.AvgPool2d(kernel_size=5, stride=3),
                                               weight_init_googlenet(lossID+"/conv", nn.Conv2d(nc[lossID], 128, kernel_size=1), weights),
                                               nn.ReLU(inplace=True)])
            self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet(lossID+"/fc", nn.Linear(2048, 1024), weights),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.7)])
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(1024, 3))
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(1024, 4))
        else:
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet("pose", nn.Linear(1024, 2048)),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5)])
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(2048, 3))
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(2048, 4))
 
    def forward(self, input):
        output = self.projection(input)
        output = self.cls_fc_pose(output.view(output.size(0), -1))
        output_xy = self.cls_fc_xy(output)
        output_wpqr = self.cls_fc_wpqr(output)
        output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
        return [output_xy, output_wpqr]

# define inception block for GoogleNet
class InceptionBlock(nn.Module):
    def __init__(self, incp, input_nc, x1_nc, x3_reduce_nc, x3_nc, x5_reduce_nc,
                 x5_nc, proj_nc, weights=None, gpu_ids=[]):
        super(InceptionBlock, self).__init__()
        self.gpu_ids = gpu_ids
        # first
        self.branch_x1 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/1x1", nn.Conv2d(input_nc, x1_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True)])

        self.branch_x3 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/3x3_reduce", nn.Conv2d(input_nc, x3_reduce_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("inception_"+incp+"/3x3", nn.Conv2d(x3_reduce_nc, x3_nc, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True)])

        self.branch_x5 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/5x5_reduce", nn.Conv2d(input_nc, x5_reduce_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("inception_"+incp+"/5x5", nn.Conv2d(x5_reduce_nc, x5_nc, kernel_size=5, padding=2), weights),
            nn.ReLU(inplace=True)])

        self.branch_proj = nn.Sequential(*[
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            weight_init_googlenet("inception_"+incp+"/pool_proj", nn.Conv2d(input_nc, proj_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True)])

        if incp in ["3b", "4e"]:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = None

    def forward(self, input):
        outputs = [self.branch_x1(input), self.branch_x3(input),
                   self.branch_x5(input), self.branch_proj(input)]
        # print([[o.size()] for o in outputs])
        output = torch.cat(outputs, 1)
        if self.pool is not None:
            return self.pool(output)
        return output

class PoseNet(nn.Module):
    def __init__(self, input_nc, weights=None, isTest=False, gpu_ids=[]):
        super(PoseNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        self.before_inception = nn.Sequential(*[
            weight_init_googlenet("conv1/7x7_s2", nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(5),
            weight_init_googlenet("conv2/3x3_reduce", nn.Conv2d(64, 64, kernel_size=1)),
            nn.ReLU(inplace=True),
            weight_init_googlenet("conv2/3x3", nn.Conv2d(64, 192, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ])

        self.inception_3a = InceptionBlock("3a", 192, 64, 96, 128, 16, 32, 32, weights, gpu_ids)
        self.inception_3b = InceptionBlock("3b", 256, 128, 128, 192, 32, 96, 64, weights, gpu_ids)
        self.inception_4a = InceptionBlock("4a", 480, 192, 96, 208, 16, 48, 64, weights, gpu_ids)
        self.inception_4b = InceptionBlock("4b", 512, 160, 112, 224, 24, 64, 64, weights, gpu_ids)
        self.inception_4c = InceptionBlock("4c", 512, 128, 128, 256, 24, 64, 64, weights, gpu_ids)
        self.inception_4d = InceptionBlock("4d", 512, 112, 144, 288, 32, 64, 64, weights, gpu_ids)
        self.inception_4e = InceptionBlock("4e", 528, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5a = InceptionBlock("5a", 832, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5b = InceptionBlock("5b", 832, 384, 192, 384, 48, 128, 128, weights, gpu_ids)

        self.cls1_fc = RegressionHead(lossID="loss1", weights=weights)
        self.cls2_fc = RegressionHead(lossID="loss2", weights=weights)
        self.cls3_fc = RegressionHead(lossID="loss3", weights=weights)

        self.model = nn.Sequential(*[self.inception_3a, self.inception_3b,
                                   self.inception_4a, self.inception_4b,
                                   self.inception_4c, self.inception_4d,
                                   self.inception_4e, self.inception_5a,
                                   self.inception_5b, self.cls1_fc,
                                   self.cls2_fc, self.cls3_fc
                                   ])
        if self.isTest:
            self.model.eval() # ensure Dropout is deactivated during test

    def forward(self, input):

        output_bf = self.before_inception(input)
        output_3a = self.inception_3a(output_bf)
        output_3b = self.inception_3b(output_3a)
        output_4a = self.inception_4a(output_3b)
        output_4b = self.inception_4b(output_4a)
        output_4c = self.inception_4c(output_4b)
        output_4d = self.inception_4d(output_4c)
        output_4e = self.inception_4e(output_4d)
        output_5a = self.inception_5a(output_4e)
        output_5b = self.inception_5b(output_5a)

        if not self.isTest:
            return self.cls1_fc(output_4a) + self.cls2_fc(output_4d) +  self.cls3_fc(output_5b)
        return self.cls3_fc(output_5b)

class PoseLSTM(nn.Module):
    def __init__(self, input_nc, weights=None, isTest=False,  gpu_ids=[]):
        super(PoseLSTM, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        self.before_inception = nn.Sequential(*[
            weight_init_googlenet("conv1/7x7_s2", nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(5),
            weight_init_googlenet("conv2/3x3_reduce", nn.Conv2d(64, 64, kernel_size=1)),
            nn.ReLU(inplace=True),
            weight_init_googlenet("conv2/3x3", nn.Conv2d(64, 192, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ])

        self.inception_3a = InceptionBlock("3a", 192, 64, 96, 128, 16, 32, 32, weights, gpu_ids)
        self.inception_3b = InceptionBlock("3b", 256, 128, 128, 192, 32, 96, 64, weights, gpu_ids)
        self.inception_4a = InceptionBlock("4a", 480, 192, 96, 208, 16, 48, 64, weights, gpu_ids)
        self.inception_4b = InceptionBlock("4b", 512, 160, 112, 224, 24, 64, 64, weights, gpu_ids)
        self.inception_4c = InceptionBlock("4c", 512, 128, 128, 256, 24, 64, 64, weights, gpu_ids)
        self.inception_4d = InceptionBlock("4d", 512, 112, 144, 288, 32, 64, 64, weights, gpu_ids)
        self.inception_4e = InceptionBlock("4e", 528, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5a = InceptionBlock("5a", 832, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5b = InceptionBlock("5b", 832, 384, 192, 384, 48, 128, 128, weights, gpu_ids)

        self.cls1_fc = RegressionHead1(lossID="loss1", weights=weights)
        self.cls2_fc = RegressionHead1(lossID="loss2", weights=weights)
        self.cls3_fc = RegressionHead2(lossID="loss3", weights=weights)

        self.model = nn.Sequential(*[self.inception_3a, self.inception_3b,
                                   self.inception_4a, self.inception_4b,
                                   self.inception_4c, self.inception_4d,
                                   self.inception_4e, self.inception_5a,
                                   self.inception_5b, self.cls1_fc, self.cls2_fc, self.cls3_fc
                                   ])
        if self.isTest:
            self.model.eval() # ensure Dropout is deactivated during test

    def forward(self, input):

        output_bf = self.before_inception(input)
        output_3a = self.inception_3a(output_bf)
        output_3b = self.inception_3b(output_3a)
        output_4a = self.inception_4a(output_3b)
        output_4b = self.inception_4b(output_4a)
        output_4c = self.inception_4c(output_4b)
        output_4d = self.inception_4d(output_4c)
        output_4e = self.inception_4e(output_4d)
        output_5a = self.inception_5a(output_4e)
        output_5b = self.inception_5b(output_5a)

        if not self.isTest:
            return self.cls1_fc(output_4a) + self.cls2_fc(output_4d) +  self.cls3_fc(output_5b)
        return self.cls3_fc(output_5b)

class PoseNetNoBeta(nn.Module):
    def __init__(self, input_nc, sx, sq, weights=None, isTest=False, gpu_ids=[]):
        super(PoseNetNoBeta, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        # initilize learning beta
        self.l1_sx = nn.Parameter(torch.tensor(sx),requires_grad=True)
        self.l1_sq = nn.Parameter(torch.tensor(sq),requires_grad=True)
        self.l2_sx = nn.Parameter(torch.tensor(sx),requires_grad=True)
        self.l2_sq = nn.Parameter(torch.tensor(sq),requires_grad=True)
        self.l3_sx = nn.Parameter(torch.tensor(sx),requires_grad=True)
        self.l3_sq = nn.Parameter(torch.tensor(sq),requires_grad=True)
        self.learning_beta = [(self.l1_sx, self.l1_sq), (self.l2_sx, self.l2_sq), (self.l3_sx, self.l3_sq)]
        self.before_inception = nn.Sequential(*[
            weight_init_googlenet("conv1/7x7_s2", nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(5),
            weight_init_googlenet("conv2/3x3_reduce", nn.Conv2d(64, 64, kernel_size=1)),
            nn.ReLU(inplace=True),
            weight_init_googlenet("conv2/3x3", nn.Conv2d(64, 192, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ])

        self.inception_3a = InceptionBlock("3a", 192, 64, 96, 128, 16, 32, 32, weights, gpu_ids)
        self.inception_3b = InceptionBlock("3b", 256, 128, 128, 192, 32, 96, 64, weights, gpu_ids)
        self.inception_4a = InceptionBlock("4a", 480, 192, 96, 208, 16, 48, 64, weights, gpu_ids)
        self.inception_4b = InceptionBlock("4b", 512, 160, 112, 224, 24, 64, 64, weights, gpu_ids)
        self.inception_4c = InceptionBlock("4c", 512, 128, 128, 256, 24, 64, 64, weights, gpu_ids)
        self.inception_4d = InceptionBlock("4d", 512, 112, 144, 288, 32, 64, 64, weights, gpu_ids)
        self.inception_4e = InceptionBlock("4e", 528, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5a = InceptionBlock("5a", 832, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5b = InceptionBlock("5b", 832, 384, 192, 384, 48, 128, 128, weights, gpu_ids)

        self.cls1_fc = RegressionHead(lossID="loss1", weights=weights)
        self.cls2_fc = RegressionHead(lossID="loss2", weights=weights)
        self.cls3_fc = RegressionHead(lossID="loss3", weights=weights)

        self.model = nn.Sequential(*[self.inception_3a, self.inception_3b,
                                   self.inception_4a, self.inception_4b,
                                   self.inception_4c, self.inception_4d,
                                   self.inception_4e, self.inception_5a,
                                   self.inception_5b, self.cls1_fc,
                                   self.cls2_fc, self.cls3_fc
                                   ])
        if self.isTest:
            self.model.eval() # ensure Dropout is deactivated during test

    def forward(self, input):

        output_bf = self.before_inception(input)
        output_3a = self.inception_3a(output_bf)
        output_3b = self.inception_3b(output_3a)
        output_4a = self.inception_4a(output_3b)
        output_4b = self.inception_4b(output_4a)
        output_4c = self.inception_4c(output_4b)
        output_4d = self.inception_4d(output_4c)
        output_4e = self.inception_4e(output_4d)
        output_5a = self.inception_5a(output_4e)
        output_5b = self.inception_5b(output_5a)

        if not self.isTest:
            return self.cls1_fc(output_4a) + self.cls2_fc(output_4d) +  self.cls3_fc(output_5b)
        return self.cls3_fc(output_5b)

