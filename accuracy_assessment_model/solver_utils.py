import torch
import os
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch.autograd import Variable

class Criterion(object):
    def __init__(self, args):
        # self.setupNormalCrit(args)
        self.lambda_L1 = args.lambda_L1
        self.lambda_perceptual = args.lambda_perceptual
        self.lambda_mse = args.lambda_mse
        
        self.mes_loss = torch.nn.MSELoss()
        self.smooth_l1_loss = torch.nn.SmoothL1Loss()
        if args.cuda:
            self.mes_loss = self.mes_loss.cuda()
            self.smooth_l1_loss = self.smooth_l1_loss.cuda()
            
        self.perceptual_layers = args.perceptual_layers
        self.gpu_ids = args.gpu_ids
        
        vgg = models.vgg19(pretrained=True).features
        self.vgg_submodel = nn.Sequential()
        for i,layer in enumerate(list(vgg)):
            self.vgg_submodel.add_module(str(i),layer)
            if i == self.perceptual_layers:
                break
        self.vgg_submodel = self.vgg_submodel.cuda()

    # def setupNormalCrit(self, args):
        
        

        
    #     print('=> Using {} for criterion normal'.format(args.normal_loss))
    #     self.normal_loss = args.normal_loss
    #     self.normal_w    = args.normal_w
    #     if args.normal_loss == 'mse':
    #         self.n_crit = torch.nn.MSELoss()
    #         self.n_crit_2 = torch.nn.SmoothL1Loss()
    #     elif args.normal_loss == 'cos':
    #         self.n_crit = torch.nn.CosineEmbeddingLoss()
    #     else:
    #         raise Exception("=> Unknown Criterion '{}'".format(args.normal_loss))
    #     if args.cuda:
    #         self.n_crit = self.n_crit.cuda()
    #         self.n_crit_2 = self.n_crit_2.cuda()

    def forward(self, output, target, rel_img, target_2):
        
        loss_l1 =self.smooth_l1_loss(rel_img, target_2)*self.lambda_L1
        
        
        # perceptual L1
        mean = torch.FloatTensor(3)
        mean[0] = 0.485
        mean[1] = 0.456
        mean[2] = 0.406
        mean = Variable(mean)
        mean = mean.resize(1, 3, 1, 1).cuda()        

        std = torch.FloatTensor(3)
        std[0] = 0.229
        std[1] = 0.224
        std[2] = 0.225
        std = Variable(std)
        std = std.resize(1, 3, 1, 1).cuda()
        
        fake_p2_norm = (rel_img + 1)/2 # [-1, 1] => [0, 1]
        fake_p2_norm = (fake_p2_norm - mean)/std

        input_p2_norm = (target_2 + 1)/2 # [-1, 1] => [0, 1]
        input_p2_norm = (input_p2_norm - mean)/std
        
        
        fake_p2_norm = self.vgg_submodel(fake_p2_norm)
        input_p2_norm = self.vgg_submodel(input_p2_norm)
        input_p2_norm_no_grad = input_p2_norm.detach()
        
        loss_perceptual = self.smooth_l1_loss(fake_p2_norm, input_p2_norm_no_grad) * self.lambda_perceptual
        loss_relighting = loss_l1 + loss_perceptual
       
        loss_angular_map = self.mes_loss(output,target)*self.lambda_mse
        self.total_loss = loss_relighting+loss_angular_map
        
        # if self.normal_loss == 'cos':
        #     num = target.nelement() // target.shape[1]
        #     if not hasattr(self, 'flag') or num != self.flag.nelement():
        #         self.flag = torch.autograd.Variable(target.data.new().resize_(num).fill_(1))

        #     self.out_reshape = output.permute(0, 2, 3, 1).contiguous().view(-1, 1)
        #     self.gt_reshape  = target.permute(0, 2, 3, 1).contiguous().view(-1, 1)
        #     self.loss        = self.n_crit(self.out_reshape, self.gt_reshape, self.flag)
        # elif self.normal_loss == 'mse':
        #     self.loss = self.n_crit(output, target) + self.n_crit_2(rel_img, target_2)
        out_loss = {'total_loss': self.total_loss.item()}
        
        
        return out_loss

    def backward(self):
        self.total_loss.backward()
        

def getOptimizer(args, params):
    print('=> Using %s solver for optimization' % (args.solver))
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(params, args.init_lr, betas=(args.beta_1, args.beta_2))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(params, args.init_lr, momentum=args.momentum)
    else:
        raise Exception("=> Unknown Optimizer %s" % (args.solver))
    return optimizer

def getLrScheduler(args, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=args.milestones, gamma=args.lr_decay, last_epoch=args.start_epoch-2)
    return scheduler

def loadRecords(path, model, optimizer):
    records = None
    if os.path.isfile(path):
        records = torch.load(path[:-8] + '_rec' + path[-8:])
        optimizer.load_state_dict(records['optimizer'])
        start_epoch = records['epoch'] + 1
        records = records['records']
        print("=> loaded Records")
    else:
        raise Exception("=> no checkpoint found at '{}'".format(path))
    return records, start_epoch

def configOptimizer(args, model):
    records = None
    optimizer = getOptimizer(args, model.parameters())
    if args.resume:
        print("=> Resume loading checkpoint '{}'".format(args.resume))
        records, start_epoch = loadRecords(args.resume, model, optimizer)
        args.start_epoch = start_epoch
    scheduler = getLrScheduler(args, optimizer)
    return optimizer, scheduler, records
