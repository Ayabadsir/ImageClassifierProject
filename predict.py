#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

import torch

import numpy as np

import utils
from torch.autograd import Variable
from classifier import load_checkpoint
from utils import load_label, process_image

args = utils.get_predict_input_args()
print(args)

def predict (image_path, model, top_k=5, cuda=True):
    ''' This function allows the classification and the probability of an image
    
        Arguments
        ---------
    
        image_path : Path to image for the prediction
        model : Training model used
        topk = The k most probable classes  
        cuda : Set to True to run the model on GPU
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Cf https://howieko.com/projects/classifying_flowers_pytorch/
    
    model.eval()
    
    if cuda and torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
      
    # Process image
    image = process_image(image_path)
    image = torch.from_numpy(np.array([image])).float()
    
    # Put image and label 
    image = Variable(image)

    if cuda:
        image = image.cuda()
            
    # The output
    with torch.no_grad():
        output = model.forward(image)
    
    # Get the probality
    ps = torch.exp(output)
    
    top_ps, top_labels = ps.topk(top_k)
    
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key
        
    # Convert to numpy array
    np_top_labels = top_labels[0].cpu().numpy()
    
    top_labs = []
    for label in np_top_labels:
        top_labs.append(int(idx_to_class[label]))
        
    return top_ps, top_labs

if __name__ == '__main__':

      
    image_path = args.image_path
    model = load_checkpoint(args.checkpoint, args.arch)
  
    # Label mapping
    file_path = args.cat_names_file
    cat_to_name = utils.load_label(file_path)
    
    probs, classes = predict(image_path, model, top_k=args.top_k, cuda=args.gpu)

    np_probs = probs[0].cpu().numpy()
    labels = []
    
    for cl in classes:
        labels.append(cat_to_name[str(cl)])

    for label, prob in zip(labels, np_probs):
       print(f"{label}: {prob:.4f}")
        
    