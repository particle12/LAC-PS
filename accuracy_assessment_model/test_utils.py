import os
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import model_utils
from utils import eval_utils, time_utils 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
test = False

def get_itervals(args, split):
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    return disp_intv, save_intv

def test(args, split, loader, model, criterion,log, epoch, recorder):
    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    disp_intv, save_intv = get_itervals(args, split)
    losses= []
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData(args, sample, timer, split)
            normal_map, angular_map,pre_image,next_light = model_utils.getInput(args, data)

            # low_bound,up_bound = model(img,err2); timer.updateTime('Forward')
            output ,rel_img= model(normal_map, angular_map,pre_image,next_light); timer.updateTime('Forward')
      
        
            # mse_loss = torch.nn.MSELoss()
            # loss = mse_loss(output,data['angular_map2']);
            
            out_loss = criterion.forward(output,data['angular_map2'],rel_img,data["next_image"])
            # out_loss = {'loss': loss.item()}
            recorder.updateIter(split, out_loss.keys(),out_loss.values())
            
            losses.append(list(out_loss.values())[0])
            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                        'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)


    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)
    return sum(losses)/len(losses)

