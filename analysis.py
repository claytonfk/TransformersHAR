# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 01:38:56 2023

@author: clayt
"""

import landscapeviz
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from hessian_est.lanczos_algorithm import approximate_hessian
from hessian_est.density import tridiag_to_density
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
#from models.TASKED_architecture_conversion.TASKEDArchitecture_t import TASKEDArchitecture

def loss_landscape(model, dataset, loss_fn, config, save_path, grid_length = 80, extension = 10, save = True):
    plt.rcParams['figure.figsize'] = [50, 50]
    plt.rcParams.update({'font.size': 30})

    landscapeviz.build_mesh(model, dataset, loss_fn, config['num_classes'], grid_length=grid_length, verbose=0, extension = extension)
    landscapeviz.plot_3d(key="loss", savepath = save_path.replace('.ckpt', '_LL.png'), save=save)
    landscapeviz.plot_contour(key="loss", savepath = save_path.replace('.ckpt', '_C.png'), save=save)
    
def get_num_params(model):
    num_params = 0
    # Iterate through the trainable variables and count parameters
    for variable in model.trainable_variables:
        num_params += tf.reduce_prod(variable.shape)
        
    return num_params

def hessian_analysis(model, config, loss_fn, dataset, order = 90):
    x, y = dataset.get_batch('train', 0)
    _ = model(x)
    
    data = [dataset.get_batch('train', i) for i in range(0, dataset.num_train_batches)]
    def _loss_fn(model, inputs):
        x, y = inputs
        y_encoded = tf.one_hot(y, depth = config['num_classes'], axis = 1)
        
        predictions = model.call(x, training = True)
        loss = loss_fn(y_encoded, predictions)
        return loss
    
    
    V, T = approximate_hessian(model, _loss_fn, data, order=order, random_seed=1)
    density, grids = tridiag_to_density([T.numpy()], grid_len=10000, sigma_squared=1e-3)
    
    return V, T, density, grids

def plot_hessian(grids, density, label=None, savepath='hessian.png'):
    plt.clf()
    plt.cla()
    plt.rcParams['figure.figsize'] = [50, 50]
    plt.rcParams.update({'font.size': 60})
    plt.semilogy(grids, density, label=label)
    plt.ylim(1e-10, 1e2)
    plt.ylabel("Density")
    plt.xlabel("Eigenvalue")
    plt.rcParams.update({'font.size': 60})
    plt.legend()
    plt.savefig(savepath)
    

def evaluate_with_adversarial(dataset, model, config, loss_fn, eps = 0.05):
    # Run test
    predictions_agg = []
    labels_agg = []
    
    num_batch = dataset.num_test_batches
    seen_classes = []
    
    for batch_index in range(num_batch):
        x, y = dataset.get_batch('test', batch_index)
        y_encoded = tf.one_hot(y, depth = config['num_classes'], axis = 1) 
        
        x = tf.convert_to_tensor(x, dtype = tf.float32)

        with tf.GradientTape() as tape:
          tape.watch(x)
          
          if model.model_name == 'tasked':
              predictions = model.call(x, training = True)
          else:
              predictions = model.call(x, training = True)

          loss = loss_fn(y_encoded, predictions)
          
        gradient = tape.gradient(loss, x)
        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)

        x = x + eps*signed_grad
        
        if model.model_name == 'retnet':
            predictions = model.call(x, force_recurrent = True)
        elif model.model_name == 'tasked':
            predictions = model.call(x)
        else:
            predictions = model.call(x)
            
        
        predictions = tf.argmax(predictions, axis = 1)
        predictions_agg += list(predictions.numpy())
        labels_agg += list(y[:, 0].astype(int))
        
    seen_classes = list(set(labels_agg))
    
    f1 = f1_score(labels_agg, predictions_agg, average= 'macro', labels = seen_classes, zero_division = 0)
    f1_weighted = f1_score(labels_agg, predictions_agg, average= 'weighted', labels = seen_classes, zero_division = 0)

    accuracy = accuracy_score(labels_agg, predictions_agg)
    cm = confusion_matrix(labels_agg, predictions_agg)
        
    
    return f1, f1_weighted, accuracy, cm

def convert_to_tflite(model, savepath, dataset, int8 = True, full_bs = True, pt_batches = 5):
    def representative_dataset():        
        num_batch = min(pt_batches, dataset.num_train_batches)

        for batch_index in range(num_batch):
            x, _ = dataset.get_batch('train', batch_index)
            
            if full_bs:
                yield [x.astype(np.float32)]
                
            else:
                for e in range(0, x.shape[0]):
                    x_ = x[e, ...]
                    x_ = x_[None, ...]

                    yield [x_.astype(np.float32)]
    
    #converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if full_bs:
        batch_size = dataset.batch_size_train
    else:
        batch_size = 1
        

    func = tf.function(model).get_concrete_function(tf.TensorSpec((batch_size, dataset.config['num_timesteps'], dataset.config['num_channels']), tf.float32))
    converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    if int8:
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops += [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()

    # Save the converted model to a .tflite file
    with open(savepath, 'wb') as f:
        f.write(tflite_model)
        
def evaluate_from_tflite(dataset, model_path, config, full_bs = True):
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run test
    predictions_agg = []
    labels_agg = []
    
    num_batch = dataset.num_test_batches
    seen_classes = []
    
    for batch_index in range(num_batch):
        x, y = dataset.get_batch('test', batch_index)
                
        if full_bs:
            bs = x.shape[0] 
            if x.shape[0] != dataset.batch_size_train:
                diff = dataset.batch_size_train - x.shape[0]
                diff_shape = list(x.shape)
                diff_shape[0] = diff
                
                x = np.concatenate([x, np.zeros(diff_shape)], axis = 0)            
            
            interpreter.set_tensor(input_index, x.astype(np.float32))
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_index)
            

            predictions = predictions[:bs, ...]

            predictions = tf.argmax(predictions, axis = 1)
            predictions_agg += list(predictions.numpy())
            labels_agg += list(y[:, 0].astype(int))
        else:
            for sample in range(0, x.shape[0]):
                input_x = x[sample, ...]
                input_x = np.expand_dims(input_x, axis = 0)
    
                input_x = input_x.astype(np.float32)
                
                interpreter.set_tensor(input_index, input_x)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_index)
        
                predictions = tf.argmax(predictions, axis = 1)
                
                predictions_agg += list(predictions.numpy())
                labels_agg.append(y[sample, 0].astype(int))
        
    seen_classes = list(set(labels_agg))
    
    f1 = f1_score(labels_agg, predictions_agg, average= 'macro', labels = seen_classes, zero_division = 0)
    f1_weighted = f1_score(labels_agg, predictions_agg, average= 'weighted', labels = seen_classes, zero_division = 0)
    
    accuracy = accuracy_score(labels_agg, predictions_agg)
    cm = confusion_matrix(labels_agg, predictions_agg)
    
    return f1, f1_weighted, accuracy, cm