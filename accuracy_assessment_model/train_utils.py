import model_utils
from utils  import time_utils 

def train(args, loader, model, criterion, optimizer, log, epoch, recorder):
    model.train()
    print('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);
    losses = []
    for i, sample in enumerate(loader):
        
        data  = model_utils.parseData(args, sample, timer, 'train')
        normal_map,angular_map,pre_image,next_light = model_utils.getInput(args, data)

        output ,rel_img= model(normal_map, angular_map,pre_image,next_light); timer.updateTime('Forward')
        

        optimizer.zero_grad()
        loss = criterion.forward(output,data['angular_map2'],rel_img,data["next_image"]); timer.updateTime('Crit');
        criterion.backward(); timer.updateTime('Backward')

        recorder.updateIter('train', loss.keys(), loss.values())

        optimizer.step(); timer.updateTime('Solver')

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            log.printItersSummary(opt)
            
        losses.append(list(loss.values())[0])
        
    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)
    
    return sum(losses)/len(losses)
