# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 01:33:08 2023

@author: clayt
"""


import tensorflow as tf
import numpy as np
import copy


def train_step(model, loss_fn, x, y, config, optimizer, rho = 0.05):
    y_encoded = tf.one_hot(y, depth = config['num_classes'], axis = 1) 
    
    with tf.GradientTape() as tape:
        predictions = model.call(x, training = True)
        loss = loss_fn(y_encoded, predictions)
    
    e_ws = []
    
    trainable_params = model.trainable_variables
    gradients = tape.gradient(loss, trainable_params)
    grad_norm = _grad_norm(gradients)
    scale = rho / (grad_norm + 1e-12)

    for (grad, param) in zip(gradients, trainable_params):
        e_w = grad * scale
        param.assign_add(e_w)
        e_ws.append(e_w)
         
    with tf.GradientTape() as tape:
        predictions = model.call(x, training = True)
        loss = loss_fn(y_encoded, predictions)
    
    sam_gradients = tape.gradient(loss, trainable_params)
    for (param, e_w) in zip(trainable_params, e_ws):
        param.assign_sub(e_w)
    
    optimizer.apply_gradients(zip(sam_gradients, trainable_params))

    return predictions, loss, gradients

def train_step_cc(model, loss_fn, x, y, config, optimizer, rho = 0.05):

    
    y_flat = y.reshape(-1)
    seen_classes = list(set(y_flat))
    
    
    sam_gradients = []
    total_loss = 0
    total_predictions = np.zeros((len(y_flat), config['num_classes']))
    
    for idx, class_id in enumerate(seen_classes):
        class_indices = list(np.where(y_flat == class_id)[0])
        class_count = len(class_indices)
        
        class_x = x[class_indices, ...]
        class_y = y[class_indices, ...]
        
        class_y_encoded = tf.one_hot(class_y, depth = config['num_classes'], axis = 1) 
        
        with tf.GradientTape() as tape:
            predictions = model.call(class_x, training = True)
            total_predictions[class_indices, ...] = predictions
            loss = loss_fn(class_y_encoded, predictions) 
            total_loss = loss*class_count
    
        e_ws = []
    
        trainable_params = model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = _grad_norm(gradients)
        scale = rho / (grad_norm + 1e-12)
        
        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)
         
    
        with tf.GradientTape() as tape:
            predictions = model.call(class_x, training = True)
            loss = loss_fn(class_y_encoded, predictions)
    
        class_sam_gradients = tape.gradient(loss, trainable_params)
        
        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)
            
        if not idx:
            sam_gradients = copy.deepcopy(class_sam_gradients)
            for i in range(0, len(sam_gradients)):
                sam_gradients[i] *= class_count
        else:
            for i in range(0, len(sam_gradients)):
                sam_gradients[i] += class_sam_gradients[i]*class_count
        
        
    
    for i in range(0, len(sam_gradients)):
        sam_gradients[i] = sam_gradients[i]/len(y_flat)
        
    total_loss = total_loss/len(y_flat)
    optimizer.apply_gradients(zip(sam_gradients, trainable_params))

    return total_predictions, total_loss, sam_gradients


def _grad_norm(gradients):
    norm = tf.norm(
        tf.stack([
            tf.norm(grad) for grad in gradients if grad is not None
        ])
    )
    return norm
 