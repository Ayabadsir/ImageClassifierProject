#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

import time

import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models

import utils
import classifier

args = utils.get_train_input_args()
print(args)

# Implement a function for the validation pass
def validation(model, dataloader, criterion, cuda=True):
    
    ''' This function is the validation pass of for the training
    
        Arguments
        ---------
    
        dataloaders : dataloader dictionary
        model : Training model used
        epochs : The number of epochs for training
    '''
    valid_loss = 0
    accuracy = 0
    
    if cuda and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    
    for images, labels in dataloader:
        
        # Put image and label 
        images, labels = Variable(images), Variable(labels)

        if cuda:
            images, labels = images.cuda(), labels.cuda()
        
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return valid_loss, accuracy
    

# Implement train function
def train(dataloaders, model, epochs=10, cuda=True):
    
    ''' This function serve to update the feed-forward network's weights
    
        Arguments
        ---------
    
        dataloaders : Dataloader dictionary
        model : Training model used
        epochs : The number of epochs for training
    '''
    print_every = args.print_every
    steps = 0
    running_loss = 0

    if cuda and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
        
    model.train()
    for e in range (epochs):
        
        for images, labels in dataloaders['train']:
        
            # Put image and label 
            images, labels = Variable(images), Variable(labels)

            if cuda:
                images, labels = images.cuda(), labels.cuda()
           
            steps += 1
            
            # Clear the gradients
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Put network in eval mode
                model.eval()
                
                # Turn off gradients for validation, saves 
                with torch.no_grad():
                    validloader = dataloaders['valid']
                    valid_loss, accuracy = validation(model, validloader, criterion)
                    # Add validation on the test set
                    
                print("Epoch: {}/{}..".format(e+1, epochs),
                     "Training Loss: {:.4f}..".format(running_loss/print_every),
                     "Validation Loss: {:.4f}".format(valid_loss/len(validloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))
                
                running_loss = 0
                model.train()
                
        # Validation on the test set  
        testloader = dataloaders['test']
        with torch.no_grad():
            valid_loss, accuracy = validation(model, testloader, criterion, args.gpu)

        print("Test Accuracy: {:.4f}".format(accuracy/len(testloader)))
                

# Main function
if __name__== "__main__":
    
    # Define the differents sets directories
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Load_data
    image_datasets, dataloaders = utils.load_data(train_dir,
                                                  valid_dir, test_dir, args.batch_size)
    
    # Label mapping
    file_path = args.cat_names_file
    cat_to_name = utils.load_label(file_path)
    
    # Call build model
    model = classifier.build_model(args.input_size, 
            len(cat_to_name), args.hidden_layer, args.dropout_p, args.arch)
    
    # Define criterion and the optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    # Train now the model
    print('Train started...')
    # For measure total program runtime 
    start = time.time()
    
    train(dataloaders, model, epochs=args.epochs, cuda = args.gpu)
    
    # End time
    end = time.time()
    tot_time = end - start
    
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)))
    
    classifier.save_checkpoint(model, optimizer, image_datasets['train'], args.arch, args.check_dir)
    
    

