import torch
import argparse
import network
from network import *
from model import set_random, load_data, train, test
import sys
from datetime import datetime

#ag import os
#ag os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_args():
    parser = argparse.ArgumentParser(description='Args for training networks')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-resnet_version', type=int, default=1, help='ResNet version')
    parser.add_argument('-resnet_size', type=int, default=18, help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    parser.add_argument('-num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('-num_epochs', type=int, default=20, help='num epochs')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-drop', type=float, default=0.3, help='dropout rate')
    parser.add_argument(
    '-mode', type=str, default='d',
    choices=['a', 'b', 'c', 'd'],
    help="Experiment mode: a (no BN), b (BN), c (BN + dropout), d (BN + dropout + tuning)"
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = get_args()
    # Set flags based on mode
    args.use_bn = args.mode in ['b', 'c', 'd']
    args.use_dropout = args.mode in ['c', 'd']
    args.use_finetune = args.mode == 'd'

    # Fine-tuning hyperparameters
    if args.use_finetune:
        args.num_epochs = 150
        args.batch = 32
        args.lr = 0.01
        args.drop = 0.3

    resnet_type = 'basic' if args.resnet_version == 1 else 'bottleneck'

    # Define filename based on mode and architecture
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"log_mode-{args.mode}_{resnet_type}_{timestamp}.txt"

    sys.stdout = open(log_filename, 'w')
    set_random(args.seed)
    net = ResNet(args)
    # print(f"The net is {net}.")
    if torch.cuda.is_available():
        net.cuda()
        print("the cuda is available.")
    print(torch.version.cuda)  
    print(torch.cuda.is_available()) 
    print(torch.cuda.get_device_name(0))
    # Step 3: Print experiment metadata
    print("========== Experiment Info ==========")
    print(f"Mode (part of question): {args.mode}")
    print(f"ResNet version: {args.resnet_version} ({resnet_type})")
    print(f"ResNet size (n): {args.resnet_size}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Dropout rate: {args.drop}")
    print(f"Epochs: {args.num_epochs}")
    print("=====================================\n")

    # print(f"the args are:", args)
    # print("ResNet class loaded from:", network.__file__)
    
    # print("before the train and test function.")
    trainloader, testloader = load_data(args.batch)
    # print(f"In the load data. The trainloader and testloader are {trainloader} and {testloader}.")
    # print("Checking net:")
    # print(net)  
    # print("Number of parameters:", sum(p.numel() for p in net.parameters()))
    # print("Trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))
    # print("ResNet type:", type(net))
    # print("ResNet defined in module:", type(net).__module__)
    train(args, net, trainloader)
    test(net, testloader)
