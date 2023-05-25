import torch.nn as nn
import torch.nn.init as init
import torch 
import torchvision 
import torchvision.transforms as transforms 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
from pathlib import Path
import torchnet as tnt
from torchvision.datasets import ImageFolder
import argparse
import json
from data.fruit_dataset import get_loaders
from models.ConvNet import ConvNet
from utils.utils import show_tensorboard_losses

def weight_init(m):
    '''
    Initializes a model's parameters.
    Credits to: https://gist.github.com/jeasinema

    Usage:
        model = Model()
        model.apply(weight_init)
    '''

    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        try:
            init.normal_(m.bias.data)
        except AttributeError:
            pass
        
def train_epoch(model, criterion, device, optimizer, train_loader, writer, total_iterations, batch_size, epoch):
    """
    Train a single epoch
    Arguments:
      model: model from ConvNet class
      criterion: criterion to be used for calculating losses
      device: device to be used gpu 0,1,2,3 or cpu
      optimiser: optimiser used for gradient descent
      train_loader: loader from DataLoader class
      writer: Tensorboard summary writer
      total_iterations: No of iterations passed
      batch_size: batch_size used
      epoch: No of epochs passed
    Returns:
      total_iterations: No of iterations passed
    """
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    y_true = []
    y_pred = []
    for i, data in enumerate(train_loader, 0):
        total_iterations = total_iterations + batch_size
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        pred = outputs.detach()
        y_p = pred.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
        acc_meter.add(pred, labels)
        loss_meter.add(loss.item())
        if total_iterations % 10 == 0:
            print("Epoch: ", epoch, "iterations: ", i*batch_size, "training loss", loss_meter.value()[0])
            show_tensorboard_losses(writer, loss_meter.value()[0], total_iterations, mode = "train")
    return total_iterations    

def val(model, criterion, device, val_loader, writer, total_iterations):
    """
    Performs validation
    Arguments:
      model: model from ConvNet class
      criterion: criterion to be used for calculating losses
      device: device to be used gpu 0,1,2,3 or cpu
      train_loader: loader from DataLoader class
      writer: Tensorboard summary writer
      total_iterations: No of iterations passed
    """
    y_true = []
    y_pred = []
    test_results = dict()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
    loss_meter = tnt.meter.AverageValueMeter()
    for i, data in enumerate(val_loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward + backward + optimize
        with torch.no_grad():
            prediction = model(inputs)           
            loss = criterion(prediction, labels)
            
        acc_meter.add(prediction, labels)
        loss_meter.add(loss.item())
        y_p = prediction.argmax(dim=1).cpu().numpy()
        y_pred.extend(list(y_p))
    metrics = {'{}_accuracy'.format("val"): acc_meter.value()[0],
               '{}_loss'.format("val"): loss_meter.value()[0]}
    print("Validating model")
    print("Validation loss", loss_meter.value()[0])
    show_tensorboard_losses(writer, loss_meter.value()[0], total_iterations, mode ="val")
    
                 
    
def test(net, test_loader):
    """
    Testing the model
    Arguments:
        net: model from ConvNet class
        test_loader: loader from DataLoader class
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main(config):
    """
    Main function
    Arguments:
        config: config file of type dict    
    """
    np.random.seed(config['rdm_seed'])
    torch.manual_seed(config['rdm_seed'])
    classes = config["classes"]
    device = torch.device(config['device'])
    train_loader, val_loader, test_loader = get_loaders(config)
    net = ConvNet(classes)
    net = net.to(device)
    net.apply(weight_init) ## Weight initialisation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'])
    runs_dir = Path("./runs") / config["name"]
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
    os.makedirs(runs_dir,  exist_ok = True)     
    writer= SummaryWriter(runs_dir)
    total_iterations = 0

    ### Model training
 
    for epoch in range(config['epochs']):  # loop over the dataset multiple times  
        net.train()
        total_iterations = train_epoch(net, criterion, config['device'], optimizer, train_loader, writer, total_iterations, config['batch_size'], epoch)
        net.eval()
        val(net, criterion, config['device'], val_loader, writer, total_iterations)
    PATH = Path('./checkpoints')/ config["name"] / "fruit_net.pth"
    torch.save(net.state_dict(), PATH)
     
    net = ConvNet()
    net.load_state_dict(torch.load(PATH))
    test(net, test_loader)    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument('--name', default='exp', type=str,
                        help='Experiment name')
    parser.add_argument('--dataset_folder', default='./dataset/big', type=str,
                        help='Path to the folder where the results are saved.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')

    # Training parameters
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default= 4, type=int, help='Batch size') ## 128
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for Adam optimizer')

    # Architecture Hyperparameters

    ## Classifier
    parser.add_argument('--classes', default=2, type=int, help='Number of classes')

    config = parser.parse_args()
    config = vars(config)
    args_file_path = Path('./checkpoints')/ config["name"] /"args.json"
    args_file_path.parent.mkdir(exist_ok=True, parents=True)
    print("Writing arguments to", str(args_file_path))
    with open(args_file_path,'w') as file:
        file.write(json.dumps(config))
    main(config)