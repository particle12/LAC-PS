import torch
import os
import matplotlib.pyplot as plt

from network import Errormap_FCN
import Custom_Dataloader
import Custom_Model
import train_utils
import test_utils
import model_utils
import solver_utils
from utils    import logger, recorders
from options  import train_opts

args = train_opts.TrainOpts().parse()
log  = logger.Logger(args)

def main(args):


    train_loader,val_loader=Custom_Dataloader.Custom_Dataloader(args)
    model =Custom_Model.buildModel(args)
    optimizer, scheduler, records = solver_utils.configOptimizer(args, model)
    criterion = solver_utils.Criterion(args)
    recorder  = recorders.Records(args.log_dir, records)
    train_losses = []
    val_losses = []

    
    for epoch in range(args.start_epoch, args.epochs+1):
        scheduler.step()
        recorder.insertRecord('train', 'lr', epoch, scheduler.get_lr()[0])

        train_loss = train_utils.train(args, train_loader, model, criterion, optimizer, log, epoch, recorder)
        train_losses.append(train_loss)
        
        if epoch % args.save_intv == 0: 
            model_utils.saveCheckpoint(args.cp_dir, epoch, model,optimizer, recorder.records, args)

        if epoch % args.val_intv == 0:
            val_loss = test_utils.test(args, 'val', val_loader, model,criterion, log, epoch, recorder)  
            val_losses.append(val_loss)  

    plt.plot(train_losses, 'b-.',label='Train Loss')
    plt.plot(val_losses, 'r--',label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
        
if __name__ == '__main__':

    torch.manual_seed(args.seed)
    main(args)