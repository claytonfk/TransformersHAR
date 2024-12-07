# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 01:23:52 2023

@author: clayt
"""
import numpy as np
import tensorflow as tf
import os
import time

from datasets.dataset_config import get_dataset_config
from datasets.dataset_loader import Dataset
from models.get_model import get_model, get_model_from_path
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score
from evaluate import evaluate
from optimization.sam import train_step, train_step_cc
#import matplotlib.pyplot as plt
from analysis import get_num_params, loss_landscape, hessian_analysis, plot_hessian, evaluate_with_adversarial, convert_to_tflite, evaluate_from_tflite

dataset_name   = 'PAMAP2_H3D_REAL'
model_name     = 'doublevi_t'  
# Options: residual_bi_lstm, ttn, vi_t, te, cross_vi_t, inno_har, tasked, ret_net_decoder
batch_size     = 64
train          = True
load_best      = False
num_epochs     = 35
learning_rate  = 0.00125
seed           = 0
sam_option     = 1 # 0 to not use SAM, 1 to use SAM, 2 to use CCSAM
plot_ll        = False # Plot the loss landscape
plot_hde       = False # Plot the Hessian density spectrum
adv_analysis   = False # Perform adversarial attack analysis
quant_analysis = False # Perform quantization analysis

config = get_dataset_config(dataset_name)
print("Loading dataset...")
dataset = Dataset(config, batch_size)
print("Dataset loaded.")
print("Creating model...")


if not load_best:
    model = get_model(model_name, config)
else:
    load_path =  f"NOVEL_models/{dataset_name}/{model_name}"
    model, load_path = get_model_from_path(model_name, config, load_path)

print("Model created.")
loss_fn     = CategoricalCrossentropy()
optimizer   = Adam(learning_rate=learning_rate)
best_val_f1 = 0

# Setting the seed
np.random.seed(seed)
tf.random.set_seed(seed)
directory = f"temporary/{dataset_name}/{model.name}"

if sam_option == 1:
    save_name = f"lr_{learning_rate}_bs_{batch_size}_{model.info}_SAM.ckpt"
elif sam_option == 2:
    save_name = f"lr_{learning_rate}_bs_{batch_size}_{model.info}_CCSAM.ckpt"
else:
    save_name = f"lr_{learning_rate}_bs_{batch_size}_{model.info}.ckpt"
    
save_path = os.path.join(directory, save_name)

if not os.path.exists(directory):
    os.makedirs(directory)

if train and num_epochs > 0:
    for epoch in range(num_epochs):
        predictions_agg = []
        labels_agg      = []
        avg_loss = 0
        start = time.time()
        for batch_index in range(dataset.num_train_batches):
            x, y = dataset.get_batch('train', batch_index)
            batch_size = x.shape[0]
            y_encoded = tf.one_hot(y, depth = config['num_classes'], axis = 1) 
    
            if sam_option == 1:
                predictions, loss, gradients = train_step(model, loss_fn, x, y, config, optimizer)
                save_name = save_name.replace(".ckpt", "_SAM.ckpt")
            elif sam_option == 2:
                predictions, loss, gradients = train_step_cc(model, loss_fn, x, y, config, optimizer)
                save_name = save_name.replace(".ckpt", "_CCSAM.ckpt")
            else:
                with tf.GradientTape() as tape:
                    predictions = model.call(x, training = True)
                    loss        = loss_fn(y_encoded, predictions)
                    
                
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
            avg_loss        += loss.numpy()*batch_size
            predictions      = tf.argmax(predictions, axis = 1)
            predictions_agg += list(predictions.numpy())
            labels_agg      += list(y[:, 0].astype(int))
            
        end = time.time()
        elapsed_time = end - start
        f1 = f1_score(labels_agg, predictions_agg, average= 'macro', labels = [i for i in range(0, config['num_classes'])], zero_division = 0)
        f1_weighted = f1_score(labels_agg, predictions_agg, average= 'weighted', labels = [i for i in range(0, config['num_classes'])], zero_division = 0)
        avg_loss = avg_loss/dataset.num_train_windows
        
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, F1 Macro: {f1:.4f}. F1 Weighted: {f1_weighted:.4f}. Elapsed time: {elapsed_time:.4f}.")
        
        f1, f1_weighted, accuracy, cm = evaluate(dataset, 'val', model, config)
        np.random.shuffle(dataset.train_windows) 
        
        if f1_weighted >= best_val_f1:
            best_val_f1 = f1_weighted
            test_f1, test_f1_weighted, test_acc, test_cm = evaluate(dataset, 'test', model, config)
            model.save_weights(save_path)
    
    if f1_weighted != best_val_f1:
        model.load_weights(save_path)
else:
    x, y = dataset.get_batch('train', 0)
    _ = model.call(x)
    if load_best:
        model.load_weights(load_path.replace(".index", ""))

txt_file                                     = save_path.replace('.ckpt', '.txt')
f1, f1_weighted, accuracy, cm                = evaluate(dataset, 'val', model, config)
test_f1, test_f1_weighted, test_acc, test_cm = evaluate(dataset, 'test', model, config)
num_params                                   = get_num_params(model)

f = open(txt_file, 'w')
f.write(f'Validation mean F1 {f1}, weighted F1 {f1_weighted} \n')
f.write(f'Test mean F1 {test_f1}, weighted F1 {test_f1_weighted} \n')
f.write(f'Number of params: {num_params.numpy()} \n')
f.write(model.info + '\n')
    

if plot_ll:
    loss_landscape(model, dataset, loss_fn, config, save_path, grid_length = 30, extension = 3, save = True)

if plot_hde:
    if model_name in ['inno_har', 'residual_bi_lstm']:
        # Redefine NN with unrolling LSTM or GRU layers
        if load_best:
            model, save_path = get_model_from_path(model_name, config, os.path.dirname(save_path), unroll = True)
            save_path = save_path.replace(".index", "")
        else:
            model = get_model(model_name, config, unroll = True)
        model.load_weights(save_path)
    elif model_name == 'hart':
        if load_best:
            model, save_path = get_model_from_path(model_name, config, os.path.dirname(save_path), pre_embedding = True)
            save_path = save_path.replace(".index", "")
        else:
            model = get_model(model_name, config, pre_embedding = True)
            
        model.load_weights(save_path)
            
    V, T, density, grids = hessian_analysis(model, config, loss_fn, dataset, order=50)
    plot_hessian(grids, density, label = None, savepath = save_path.replace('.ckpt', '_HESSIAN.png'))
    np.savez(save_path.replace(".ckpt", "_HESSIAN.npz"), V=V, T=T, density=density, grids=grids)
        
if adv_analysis:
    eps = [0.01, 0.05]
    
    for eps_ in eps:
        adv_f1, adv_f1_weighted, _, _ = evaluate_with_adversarial(dataset, model, config, loss_fn, eps = eps_)
        f1_degradation = (test_f1 - adv_f1)*100
        weighted_f1_degradation = (test_f1_weighted - adv_f1_weighted)*100
        
        info = f'AA with eps {eps_} on test set. Mean F1 {adv_f1} (-{f1_degradation}%). Weighted F1 {adv_f1_weighted} (-{weighted_f1_degradation}%)'
        print(info)
        f.write(info + "\n")

if quant_analysis:
    
    fp16_modelpath = save_path.replace(".ckpt", "_FP16.tflite")
    int8_modelpath = save_path.replace(".ckpt", "_INT8.tflite")
    
    convert_to_tflite(model, fp16_modelpath, dataset, int8 = False, full_bs = True, pt_batches = int(256/batch_size))
    convert_to_tflite(model, int8_modelpath, dataset, int8 = True, full_bs = True, pt_batches = int(256/batch_size))
    
    q_f1, q_f1_weighted, _, _ = evaluate_from_tflite(dataset, fp16_modelpath, config)
    f1_degradation = (test_f1 - q_f1)*100
    weighted_f1_degradation = (test_f1_weighted - q_f1_weighted)*100
    info = f'FP16 Quantization on test set. Mean F1 {q_f1} (-{f1_degradation}%). Weighted F1 {q_f1_weighted} (-{weighted_f1_degradation}%)'
    print(info)
    f.write(info + "\n")
    
    q_f1, q_f1_weighted, _, _ = evaluate_from_tflite(dataset, int8_modelpath, config)
    f1_degradation = (test_f1 - q_f1)*100
    weighted_f1_degradation = (test_f1_weighted - q_f1_weighted)*100
    info = f'INT8 Quantization on test set. Mean F1 {q_f1} (-{f1_degradation}%). Weighted F1 {q_f1_weighted} (-{weighted_f1_degradation}%)'
    print(info)
    f.write(info + "\n")

f.close()