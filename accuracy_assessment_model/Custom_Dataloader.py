# from Custom_Dataset import Custom_Dataset
import torch.utils.data
from torch.utils.data.dataset import random_split

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

def Custom_Dataloader(args):
    
    train_ratio = 0.95
    val_ratio = 0.05


    
    if args.dataset == 'PS_Synth_Dataset':
        from Custom_Dataset import Custom_Dataset
        dataset = Custom_Dataset(args, args.data_dir)

    else:
        raise Exception('Unknown dataset: %s' % (args.dataset))
    
    if args.concat_data:
        print('****** Using cocnat data ******')
        print("=> fetching img pairs in %s" % (args.data_dir2))
        dataset2 = Custom_Dataset(args, args.data_dir2)

        dataset  = torch.utils.data.ConcatDataset([dataset,dataset2])
   
    
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoaderX(train_set, batch_size=args.train_batch,
                        num_workers=args.workers, pin_memory=args.cuda, shuffle=True)
    
    val_loader = DataLoaderX(val_set, batch_size=args.val_batch,
                        num_workers=args.workers, pin_memory=args.cuda,shuffle=False) 
    
    return train_loader,val_loader

def Test_Dataloader(args):
    from Custom_Dataset import Custom_Dataset
    test_set  = Custom_Dataset(args)

    test_loader = DataLoaderX(test_set, batch_size=args.test_batch,
                        num_workers=args.workers, pin_memory=args.cuda, shuffle=False)
    return test_loader
