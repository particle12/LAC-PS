import Custom_Dataloader
import Custom_Model
from utils    import logger, recorders
from options  import run_model_opts
import torch
import test_utils

args = run_model_opts.RunModelOpts().parse()
log  = logger.Logger(args)

def main(args):
    test_loader = Custom_Dataloader.Test_Dataloader(args)
    model    = Custom_Model.buildModel(args)
    recorder = recorders.Records(args.log_dir)
    test_utils.test(args, 'test', test_loader, model, log, 1, recorder)

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
