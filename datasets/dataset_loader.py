# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 01:14:52 2023

@author: clayt
"""

import pandas as pd
import numpy as np
import math, os

class Dataset:
    def __init__(self, config, batch_size_train, batch_size_eval = -1):
        self.config = config
        self.batch_size_train = batch_size_train
        if batch_size_eval == -1:
            self.batch_size_eval = batch_size_train
        
        train_set = self.read_set('training_files')
        val_set = self.read_set('validation_files')
        test_set = self.read_set('test_files')
        
        mean, std = self.get_mean_std(train_set)
        self.mean = mean
        self.std = std
        train_set = self.normalize_set(train_set, mean, std)
        test_set = self.normalize_set(test_set, mean, std)
        val_set = self.normalize_set(val_set, mean, std)
        
        self.train_windows = self.create_sliding_windows(train_set)
        self.test_windows = self.create_sliding_windows(test_set, overlap = 0)
        self.val_windows = self.create_sliding_windows(val_set)
        
        self.num_train_windows = self.train_windows.shape[0]
        self.num_train_batches = math.ceil(self.num_train_windows/self.batch_size_train)
        
        self.num_test_windows = self.test_windows.shape[0]
        self.num_test_batches = math.ceil(self.num_test_windows/self.batch_size_eval)  
        
        self.num_val_windows = self.val_windows.shape[0]
        self.num_val_batches = math.ceil(self.num_val_windows/self.batch_size_eval)
        np.random.shuffle(self.train_windows)

    def read_file(self, filepath):
        index_col = 0 if self.config['skip_first_col'] else False
        header    = 0 if self.config['skip_header'] else None
        
        filepath  = os.path.join(self.config['dataset_path'], filepath)

        data = pd.read_csv(filepath, index_col = index_col, header = header, sep = self.config['column_sep'], keep_default_na = True).values
        
        filename = os.path.basename(filepath)
        sid = self.config['fcn_subjectid'](filename)
        datax = data[:, self.config['input_columns']]
        datay = data[:, self.config['class_index']]
        datay = datay[:, np.newaxis]
        dataz = np.ones((data.shape[0], 1))*sid
        
        
        data = np.concatenate([datax, dataz, datay], axis = 1)
        # Note that from now on the last column represents the label

        # Deal with the NaN values in the data
        if self.config['keep_nan']:
            _, num_cols = data.shape
            num_cols = num_cols - 2
            for col in range(0, num_cols):
                col_data = data[:, col]
                series = pd.Series(col_data)
                interpolated_series = series.interpolate(method='linear', limit_direction="both",  limit_area="inside")
                interpolated_data = interpolated_series.to_numpy()
                data[:, col] = interpolated_data
            data = np.nan_to_num(data, copy=False)
        else:
            data = data[~np.isnan(data).any(axis=1)]

        # Reduce sampling frequency by decimation
        if self.config['frequency_reducer'] != 1:
            data = data[::self.config['frequency_reducer'], :]  
            
        # Exclude classes
        for exclude_class_id in self.config['exclude_classes']:
            class_row_idx = data[:, -1] == exclude_class_id
            class_row_idx = [i for i, x in enumerate(class_row_idx) if x]
            data = np.delete(data, class_row_idx, axis = 0)

        # Remap classes
        if self.config['remap_classes'] != -1 and len(self.config['remap_classes']) > 0:
            for old_class_id, new_class_id in self.config['remap_classes'].items():
                class_row_idx = data[:, -1] ==  old_class_id
                class_row_idx = [i for i, x in enumerate(class_row_idx) if x]
                data[class_row_idx, -1] = new_class_id    
                
        return data
    
    def create_sliding_windows(self, data, overlap = -1):
        num_samples = data.shape[0]
        if overlap == -1:
            overlap = self.config['overlap_steps'] 

        step_size = self.config['num_timesteps'] -  overlap
        num_windows = (num_samples - self.config['num_timesteps']) // step_size + 1
        num_channels = data.shape[-1]

        sliding_windows = np.zeros((num_windows, self.config['num_timesteps'], num_channels))

        for i in range(0, num_windows):
            sliding_windows[i, ...] = data[i*step_size: i*step_size + self.config['num_timesteps']]
            
        return sliding_windows
  
    
    def rand_rotation_matrix(self, deflection=1.0, randnums=None):
        """
        Creates a random rotation matrix.
        
        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
        """
        # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        
        if randnums is None:
            randnums = np.random.uniform(size=(3,))
            
        theta, phi, z = randnums
        
        theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0*np.pi  # For direction of pole deflection.
        z = z * 2.0*deflection  # For magnitude of pole deflection.
        
        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.
        
        r = np.sqrt(z)
        Vx, Vy, Vz = V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
            )
        
        st = np.sin(theta)
        ct = np.cos(theta)
        
        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
        
        # Construct the rotation matrix  ( V Transpose(V) - I ) R.
        
        M = (np.outer(V, V) - np.eye(3)).dot(R)
        return M
    
    def read_set(self, set_name):
        files = self.config[set_name]
        
        set_data  = []
        
        for file in files:
            data = self.read_file(file)
            set_data.append(data)

        set_data = np.concatenate(set_data, axis=0)
        
        classes = set_data[:, -1].astype(int)
        classes = np.bincount(classes)
        print(classes)
        print(classes/np.sum(classes))

        return set_data
    
    def split_data(self, batch, return_sid = False):
        x = batch[..., :-2] 
        y_ = batch[..., -1].astype(int)
        z_ = batch[..., -2].astype(int)
        
        
        num_windows = y_.shape[0]
        y = np.zeros((num_windows, 1))
        z = np.zeros((num_windows, 1))
    
        for window in range(0, y_.shape[0]):
            y[window, 0] = np.argmax(np.bincount(y_[window, :]), axis=0)
            z[window, 0] = np.argmax(np.bincount(z_[window, :]), axis=0)
        
        if return_sid:
            return x, y.astype(int), z.astype(int)
        
        return x, y.astype(int)
        
        
    def get_mean_std(self, data):
        num_channels = data.shape[1] - 2
        norm_indices = [i for i in range(0, num_channels)]
        
        mean = np.mean(data[:, norm_indices], axis = 0)
        std = np.std(data[:, norm_indices], axis = 0)
        
        return mean, std

    def normalize_set(self, set_data, mean, std):
        num_channels = set_data.shape[1] - 2
        norm_indices = [i for i in range(0, num_channels)]
        
        set_data[:, norm_indices] = set_data[:, norm_indices] - mean
        set_data[:, norm_indices] = set_data[:, norm_indices]/std
        
        return set_data

    
    def get_batch(self, set_name, batch_index, return_sid = False):
        
        
        
        if set_name == 'training' or set_name == 'train':
            assert batch_index < self.num_train_windows
            windows = self.train_windows[batch_index*self.batch_size_train:(batch_index+1)*self.batch_size_train, ...]
        elif set_name == "validation" or set_name == 'val':
            assert batch_index < self.num_val_windows
            windows = self.val_windows[batch_index*self.batch_size_eval:(batch_index+1)*self.batch_size_eval, ...]
        elif set_name == 'test':
            assert batch_index < self.num_test_windows
            windows = self.test_windows[batch_index*self.batch_size_eval:(batch_index+1)*self.batch_size_eval, ...]
        else:
            return None
            
            
        x, y = self.split_data(windows, return_sid = return_sid)
            
        return x, y
        

