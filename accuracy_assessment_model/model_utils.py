import os
import torch
import torch.nn as nn

def getInput(args, data):

    normal_map = data['normal_map']
    angular_map12 = data['angular_map12'] 

    pre_image = data["pre_image"]
    next_light = data["next_light"]
    
    
    return normal_map, angular_map12,pre_image,next_light

def parseData(args, sample, timer=None, split='train'):
        
    normal_map,angular_map12,angular_map2,mask = sample['normal_map'],sample['angular_map12'],sample['angular_map2'],sample['mask']
    
    pre_image = sample["pre_image"]
    next_image = sample["next_image"]
    next_light = sample["next_light"].expand_as(pre_image)

    if timer: timer.updateTime('ToCPU')
    if args.cuda:
        normal_map = normal_map.cuda(); angular_map12 = angular_map12.cuda() ;angular_map2 = angular_map2.cuda() ;mask = mask.cuda();
        pre_image = pre_image.cuda()
        next_image = next_image.cuda()
        next_light = next_light.cuda()
              

    normal_map_var  = torch.autograd.Variable(normal_map)
    angular_map12_var = torch.autograd.Variable(angular_map12)
    angular_map2_var = torch.autograd.Variable(angular_map2)
    mask_var   = torch.autograd.Variable(mask, requires_grad=False);
    
    pre_image_var = torch.autograd.Variable(pre_image)
    next_image_var = torch.autograd.Variable(next_image)
    next_light_var = torch.autograd.Variable(next_light)

    

    if timer: timer.updateTime('ToGPU')
    data = { 'normal_map': normal_map_var, 'angular_map12': angular_map12_var, \
        'angular_map2': angular_map2_var ,"mask":mask_var, \
        'pre_image':pre_image_var, 'next_image':next_image_var, 'next_light':next_light_var}

    return data 

def getInputChanel(args):
    print('[Network Input] Color image as input')
    c_in = 3
    if args.in_light:
        print('[Network Input] Adding Light direction as input')
        c_in += 3
    print('[Network Input] Input channel: {}'.format(c_in))
    return c_in

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def loadCheckpoint(path, model, cuda=True):
    if cuda:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

def saveCheckpoint(save_path, epoch=-1, model=None, optimizer=None, records=None, args=None):
    state   = {'state_dict': model.state_dict(), 'model': args.model}
    records = {'epoch': epoch, 'optimizer':optimizer.state_dict(), 'records': records, 
            'args': args}
    torch.save(state, os.path.join(save_path, 'checkp_%d.pth.tar' % (epoch)))
    torch.save(records, os.path.join(save_path, 'checkp_%d_rec.pth.tar' % (epoch)))

def conv(batchNorm, cin, cout, k=3, stride=1, pad=-1):
    pad = (k - 1) // 2 if pad < 0 else pad
 
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.1, inplace=True)
                )
    else:
        return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
                )

def deconv(cin, cout):
    return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
            )
