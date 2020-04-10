import datetime
import torch
from torch import nn, optim
from torchvision import models

from collections import OrderedDict

def build_model(input_size, output_size, hidden_layer, drop_p=0.5, arch='vgg16'):
    
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    
    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)
     
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
     
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # The new untrained feed-forward work
    # Use OrderedDict for build the model
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_layer, bias=True)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=drop_p)),
        ('fc2', nn.Linear(hidden_layer, output_size, bias=True)),
        ('output', nn.LogSoftmax(dim=1))  
    ]))
                           
    model.classifier = classifier 
    
    return model

def save_checkpoint(model, optimizer, traindataset, arch, save_directory, epochs=10,):
    
    date = datetime.datetime.utcnow().strftime("_%Y-%m-%d_%H:%M:%S")
    model.class_to_idx = traindataset.class_to_idx
    
    checkpoint = {'arch' : arch,
              'classifier' : model.classifier,
              'state_dict' : model.state_dict(),
              'epochs' : epochs,
              'optimizer' : optimizer.state_dict(),
              'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, save_directory + '/classifier_' + arch + date + '.pth')
        
def load_checkpoint(filepath, arch='vgg16'):
    
     
    # https://stackoverflow.com/questions/55759311/runtimeerror-cuda-runtime-error-35-cuda-driver-version-is-insufficient-for
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(filepath, map_location=map_location)
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    
    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)
     
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model
 
    
        
    