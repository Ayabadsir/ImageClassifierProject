import torch
from torchvision import datasets, transforms

import argparse

import numpy as np
from PIL import Image

def get_train_input_args():
    '''
    Retrieves and parses the command line arguments provided by the user for the training
    '''
    
 
    parser = argparse.ArgumentParser()
  
    parser.add_argument('data_dir', type=str, default='data', help='image directory') 
    parser.add_argument('--cat_names_file', type=str, default='cat_to_name.json',
                    help='Path to json file mapping categories to names')
    
    parser.add_argument('--check_dir', type=str, default='checkpoints',
                    help='directory to store the checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN Model Architecture to use')
    parser.add_argument('--input_size', type=int, default=25088, help='Network input size')
    parser.add_argument('--hidden_layer', type=int , default=1024, help='One layer size of the Network')
    parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
    parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs')
    parser.add_argument('--print_every', type=int, default=60,
                    help='print frequency')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
    parser.add_argument('--dropout_p', type=float, default=0.5,
                    help='probability of dropping weights')
    parser.add_argument('--gpu', action='store_true', default=True,
                    help='run the network on the GPU')
    
    return parser.parse_args()

def get_predict_input_args():
    '''
    Retrieves and parses the  command line arguments provided by the user 
    '''
  
    parser = argparse.ArgumentParser()
  
    parser.add_argument('image_path', type=str, default=None, help='Path to input image')
    parser.add_argument('checkpoint', type=str, default=None,
                    help='Load checkpoint for prediction')
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN Model Architecture to use')
    parser.add_argument('--cat_names_file', type=str, default='cat_to_name.json',
                    help='Path to json file mapping categories to names')
    parser.add_argument('--top_k', type=int, default=5,
                    help='Predict the top K character probabilities')
    parser.add_argument('--gpu', action='store_true', default=True,
                    help='Run the network on a GPU')
    
    return parser.parse_args()


def load_data(train_dir, valid_dir, test_dir, batch_size=32):
    ''' Loading of the different batches of data after having prepared them
    '''
    
    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    'valid' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])     
    ]),
    'test' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) 
    ])
}
    #Load the datasets with ImageFolder
    image_datasets = {
    'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test']),
     }

    #Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=32),
}
    return image_datasets, dataloaders

def load_label(file_path):
    ''' Map class indexes to classe names
    '''
    
    import json
    
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Resize
    pil_image = Image.open(image)
    pil_image = pil_image.resize((256, 256))

    # Crop out the center 
    # Cf. https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    coord_1 = (256 - 224)/2
    coord_2 = (256 + 224)/2
    
    pil_image = pil_image.crop((coord_1, coord_1, coord_2, coord_2))
    
    # Convert array values
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    return np_image.transpose(2,0,1)

