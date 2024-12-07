# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 01:31:22 2023

@author: clayt
"""

import tensorflow as tf
import time
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import numpy as np

def evaluate(dataset, dataset_mode, model, config):
    # Run test
    predictions_agg = []
    labels_agg = []
    
    if dataset_mode == 'test':
        num_batch = dataset.num_test_batches
    elif dataset_mode == 'val':
        num_batch = dataset.num_val_batches
    else:
        num_batch = dataset.num_train_batches
    
    start = time.time()
    seen_classes = []
    
    for batch_index in range(num_batch):
        x, y = dataset.get_batch(dataset_mode, batch_index)
        
        if model.model_name == 'retnet':
            predictions = model.call(x, force_recurrent = True)
        else:
            predictions = model.call(x)
        
        predictions = tf.argmax(predictions, axis = 1)
        predictions_agg += list(predictions.numpy())
        labels_agg += list(y[:, 0].astype(int))
    
    end = time.time()
    elapsed_time = end - start
    seen_classes = list(set(labels_agg))
    
    f1 = f1_score(labels_agg, predictions_agg, average= 'macro', labels = seen_classes, zero_division = 0)
    f1_weighted = f1_score(labels_agg, predictions_agg, average= 'weighted', labels = seen_classes, zero_division = 0)
    #f1_cw = f1_score(labels_agg, predictions_agg, average= None, labels = seen_classes, zero_division = 0)
    
    accuracy = accuracy_score(labels_agg, predictions_agg)
    cm = confusion_matrix(labels_agg, predictions_agg)

    if dataset_mode == 'test':
        print(f"Test. F1 Macro: {f1:.4f}. F1 Weighted: {f1_weighted:.4f}. Elapsed time: {elapsed_time:.4f}")
        #print(f"Test. F1 Class-wise: {f1_cw}")
    elif dataset_mode == 'val':
        print(f"Validation. F1 Macro: {f1:.4f}. F1 Weighted: {f1_weighted:.4f}. Elapsed time: {elapsed_time:.4f}")
        
    return f1, f1_weighted, accuracy, cm


def evaluate_mse(dataset, dataset_mode, model, config):
    # Run test
    
    if dataset_mode == 'test':
        num_batch = dataset.num_test_batches
    elif dataset_mode == 'val':
        num_batch = dataset.num_val_batches
    else:
        num_batch = dataset.num_train_batches
    
    start = time.time()
    mse = []
    for batch_index in range(num_batch):
        x, y = dataset.get_batch(dataset_mode, batch_index)
        
        predictions = model.call(x)
        mse_ = np.mean(np.square(predictions - y))
        mse.append(mse_)
    
    end = time.time()
    elapsed_time = end - start


    mse = np.mean(mse)

    if dataset_mode == 'test':
        print(f"Test. MSE: {mse:.4f}. Elapsed time: {elapsed_time:.4f}")
    elif dataset_mode == 'val':
        print(f"Validation. MSE: {mse:.4f}. Elapsed time: {elapsed_time:.4f}")
        
    return mse