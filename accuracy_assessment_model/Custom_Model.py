import network
import model_utils

def buildModel(args):
    print('Creating Model %s' % (args.model))
    
    model=network.Errormap_FCN(batchnorm=args.batchnorm)
   
    if args.cuda: 
        model = model.cuda()

    if args.retrain: 
        print("=> using pre-trained model %s" % (args.retrain))
        model_utils.loadCheckpoint(args.retrain, model, cuda=args.cuda)

    if args.resume:
        print("=> Resume loading checkpoint %s" % (args.resume))
        model_utils.loadCheckpoint(args.resume, model, cuda=args.cuda)
    print(model)
    print("=> Model Parameters: %d" % (model_utils.get_n_params(model)))
    return model