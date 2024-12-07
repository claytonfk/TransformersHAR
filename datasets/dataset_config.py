# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 01:14:52 2023

@author: clayt
"""
import glob, os
import random

def get_dataset_config(dataset_name):
    config = {}
    if dataset_name == 'PAMAP2-TASKED':
        # Dataset protocol according to the paper TASKED for HAR
        config['dataset_name']   = 'PAMAP2'
        config['dataset_path']   = "./datasets/pamap2/"
        config['training_files'] = ['subject101.dat',
                                    'subject102.dat',
                                    'subject103.dat',
                                    'subject104.dat',
                                    'subject107.dat',
                                    'subject108.dat']
                    
        config['validation_files'] = ['subject105.dat']
        
        config['test_files']       = ['subject106.dat']
        
        config['class_index']          = 1
        config['column_sep']           = " "
        config['keep_nan']             = True
        config['skip_header']          = False
        config['skip_first_col']       = False
        config['input_columns']        = [4,5,6,7,8,9,10,11,12,13,14,15,21,22,23,24,25,26,27,28,29,30,31,32,38,39,40,41,42,43,44,45,46,47,48,49] # According to the paper TASKED
        config['exclude_classes']      = [0]
        config['remap_classes']        = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 9:7, 10: 8, 11: 9, 12: 10, 13: 11, 16: 12, 
                                          17: 13, 18: 14, 19: 15, 20: 16, 24: 17} 
        config['num_classes']          = 18 
        config['frequency_reducer']    = 1
        config['num_timesteps']        = 200
        config['overlap_steps']        = 150
        
        config['fcn_subjectid']        = lambda fname: int(fname.replace(".dat", "").replace("subject10", "")) - 1
        
        
        
    elif dataset_name == 'PAMAP2-TTN':
           # Dataset protocol according to the paper Two-Stream Transformers for HAR
            config['dataset_name']   = 'PAMAP2'
            config['dataset_path']   = "./datasets/pamap2/"
            config['training_files'] = ['subject101.dat', 
                                        'subject102.dat',
                                        'subject103.dat',
                                        'subject104.dat',
                                        'subject107.dat',
                                        'subject108.dat',
                                        'subject109.dat' ]
                        
            config['validation_files'] = ['subject105.dat']
            
            config['test_files']       = ['subject106.dat']
            
            config['class_index']          = 1
            config['column_sep']           = " "
            config['keep_nan']             = True
            config['skip_header']          = False
            config['skip_first_col']       = False
            config['input_columns']        = [4, 5, 6, 10, 11, 12, 21, 22, 23, 27, 28, 29, 38, 39, 40, 44, 45, 46 ] # According to TTN           
            config['exclude_classes']      = [0, 8, 11, 14, 15, 18, 19, 20, 21, 22, 23]
            config['remap_classes']        = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 12: 7, 13: 8, 16: 9, 17: 10, 24: 11}
            config['frequency_reducer']    = 3
            config['num_classes']          = 12
            config['num_timesteps']        = 32
            config['overlap_steps']        = 16
            
            config['fcn_subjectid']        = lambda fname: int(fname.replace(".dat", "").replace("subject10", "")) - 1 
            
            
    elif dataset_name == 'PAMAP2-GRUINC':
           # Dataset protocol according to the paper GRUINC
        config['dataset_name']   = 'PAMAP2'
        config['dataset_path']   = "./datasets/pamap2/"
        config['training_files'] = ['subject101.dat', 'subject101_2.dat',
                                    'subject102.dat',
                                    'subject103.dat',
                                    'subject104.dat',
                                    'subject107.dat',
                                    'subject108.dat', 'subject108_2.dat',
                                    'subject109.dat', 'subject109_2.dat' ]
                    
                    
        config['validation_files'] = ['subject106.dat', 'subject106_2.dat']
        
        config['test_files']       = ['subject105.dat', 'subject105_2.dat']
        
        config['class_index']          = 1
        config['column_sep']           = " "
        config['keep_nan']             = True
        config['skip_header']          = False
        config['skip_first_col']       = False
        config['input_columns']        = [i for i in range(2, 54)]
        config['exclude_classes']      = [0]
        config['remap_classes']        = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 9:7, 10: 8, 11: 9, 12: 10, 13: 11, 16: 12, 
                                          17: 13, 18: 14, 19: 15, 20: 16, 24: 17} 
        config['num_classes']          = 18 
        config['frequency_reducer']    = 1
        config['num_timesteps']        = 256
        config['overlap_steps']        = 128
        
        config['fcn_subjectid']        = lambda fname: int(fname.replace(".dat", "").replace("subject10", "")) - 1
        
        
        
    elif dataset_name == 'PAMAP2':
         config['dataset_name']   = 'PAMAP2'
         config['dataset_path']   = "./datasets/pamap2/"
         config['training_files'] = ['subject101.dat', 
                                     'subject102.dat',
                                     'subject103.dat',
                                     'subject104.dat',
                                     'subject107.dat',
                                     'subject108.dat',
                                     'subject109.dat' ]
                     
         config['validation_files'] = ['subject105.dat']
         
         config['test_files']       = ['subject106.dat']
         
         config['class_index']          = 1
         config['column_sep']           = " "
         config['keep_nan']             = True
         config['skip_header']          = False
         config['skip_first_col']       = False
         config['input_columns']        = [i for i in range(2, 54)]          
         config['exclude_classes']      = [0, 8, 11, 14, 15, 18, 19, 20, 21, 22, 23]
         config['remap_classes']        = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 12: 7, 13: 8, 16: 9, 17: 10, 24: 11}
         config['frequency_reducer']    = 3
         config['num_classes']          = 12
         config['num_timesteps']        = 32
         config['overlap_steps']        = 16
         
         config['fcn_subjectid']        = lambda fname: int(fname.replace(".dat", "").replace("subject10", "")) - 1 
         
    elif dataset_name == 'PAMAP2_H3D_REAL':
         config['dataset_name']   = 'PAMAP2'
         config['dataset_path']   = "./datasets/pamap2/"
         config['training_files'] = ['subject101.dat', 
                                     'subject102.dat',
                                     'subject103.dat',
                                     'subject104.dat',
                                     'subject107.dat',
                                     'subject108.dat',
                                     'subject109.dat' ]
                     
         config['validation_files'] = ['subject105.dat']
         
         config['test_files']       = ['subject106.dat']
         
         config['class_index']          = 1
         config['column_sep']           = " "
         config['keep_nan']             = True
         config['skip_header']          = False
         config['skip_first_col']       = False
         config['input_columns']        = [i for i in range(7, 13)] + [i for i in range(24, 30)] + [i for i in range(42, 48)] 
         config['exclude_classes']      = [0, 8, 11, 14, 15, 18, 19, 20, 21, 22, 23]
         config['remap_classes']        = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 12: 7, 13: 8, 16: 9, 17: 10, 24: 11}
         config['frequency_reducer']    = 5
         config['num_classes']          = 12
         config['num_timesteps']        = 64
         config['overlap_steps']        = 32
         
         config['fcn_subjectid']        = lambda fname: int(fname.replace(".dat", "").replace("subject10", "")) - 1 
    elif dataset_name == 'PAMAP2_H3D':
         config['dataset_name']   = 'PAMAP2_H3D'
         config['dataset_path']   = "./datasets/pamap2_h3d/sensor_data/"
         
         files = glob.glob(os.path.join(config['dataset_path'], "*.npy"))
         files = [os.path.basename(f) for f in files]
         random.shuffle(files)
         
         num_files = 5000
         
         files = files[:num_files]
         
         num_files_train = int(num_files*0.9)
         num_files_val   = int(num_files*0.05)
         
         
         config['training_files']   = files[:num_files_train]
         config['validation_files'] = files[num_files_train:num_files_val+num_files_train]
         config['test_files']       = files[num_files_val+num_files_train:]
         
         config['frequency_reducer']    = 1
         config['num_timesteps']        = 64
         config['overlap_steps']        = 32
         config['num_classes']          = 512
         config['input_columns']        = [0,1,2,3,4,5,9,10,11,12,13,14,18,19,20,21,22,23]
            
    elif dataset_name == "OpportunityHL":
        config['dataset_name']         = 'OpportunityHL'
        config['dataset_path']         = "./datasets/opp/"
        config['training_files']       = ["S1-ADL1.dat", "S1-ADL3.dat", "S1-ADL4.dat", "S1-ADL5.dat", 
                                          "S2-ADL1.dat", "S2-ADL2.dat", "S2-ADL3.dat", 
                                          "S3-ADL1.dat", "S3-ADL2.dat", "S3-ADL3.dat", "S3-Drill.dat",
                                          "S4-ADL1.dat",  "S4-ADL2.dat", "S4-ADL3.dat", "S4-ADL4.dat", "S4-ADL5.dat", 
                                          "S1-Drill.dat",  "S2-Drill.dat","S3-Drill.dat", "S4-Drill.dat"]
        config['validation_files']     = ["S1-ADL2.dat" ]
        config['test_files']           = ["S2-ADL4.dat", "S2-ADL5.dat", "S3-ADL4.dat", "S3-ADL5.dat"]
        config['class_index']          = 244
        config['input_columns']        = [i for i in range(1, 241)] # Body-worn and environmental sensors
        config['exclude_classes']      = []
        '''101   -   HL_Activity   -   Relaxing
           102   -   HL_Activity   -   Coffee time
           103   -   HL_Activity   -   Early morning
           104   -   HL_Activity   -   Cleanup
           105   -   HL_Activity   -   Sandwich time'''       
        config['remap_classes']        = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}
        config['keep_nan']             = True
        config['skip_header']          = False
        config['skip_first_col']       = False
        config['column_sep']           = " "
        config['frequency_reducer']    = 1
        config['num_classes']          = 5
        config['num_timesteps']        = 256
        config['overlap_steps']        = 128         
        config['fcn_subjectid']        = lambda fname: int(fname.split('-')[0].replace("S", "")) - 1
    elif dataset_name == "OpportunityML":
        config['dataset_name']         = 'OpportunityML'
        config['dataset_path']         = "./datasets/opp/"
        config['training_files']       = ["S1-ADL1.dat", "S1-ADL3.dat", "S1-ADL4.dat", "S1-ADL5.dat", 
                                          "S2-ADL1.dat", "S2-ADL2.dat", "S2-ADL3.dat", 
                                          "S3-ADL1.dat", "S3-ADL2.dat", "S3-ADL3.dat", "S3-Drill.dat",
                                          "S4-ADL1.dat",  "S4-ADL2.dat", "S4-ADL3.dat", "S4-ADL4.dat", "S4-ADL5.dat", 
                                          "S1-Drill.dat",  "S2-Drill.dat","S3-Drill.dat", "S4-Drill.dat"]
        config['validation_files']     = ["S1-ADL2.dat" ]
        config['test_files']           = ["S2-ADL4.dat", "S2-ADL5.dat", "S3-ADL4.dat", "S3-ADL5.dat"]
        config['class_index']          = 249
        config['input_columns']        = [i for i in range(1, 134)] # Only body-worn sensors
        config['exclude_classes']      = []
        '''406516   -   ML_Both_Arms   -   Open Door 1
        406517   -   ML_Both_Arms   -   Open Door 2
        404516   -   ML_Both_Arms   -   Close Door 1
        404517   -   ML_Both_Arms   -   Close Door 2
        406520   -   ML_Both_Arms   -   Open Fridge
        404520   -   ML_Both_Arms   -   Close Fridge
        406505   -   ML_Both_Arms   -   Open Dishwasher
        404505   -   ML_Both_Arms   -   Close Dishwasher
        406519   -   ML_Both_Arms   -   Open Drawer 1
        404519   -   ML_Both_Arms   -   Close Drawer 1
        406511   -   ML_Both_Arms   -   Open Drawer 2
        404511   -   ML_Both_Arms   -   Close Drawer 2
        406508   -   ML_Both_Arms   -   Open Drawer 3
        404508   -   ML_Both_Arms   -   Close Drawer 3
        408512   -   ML_Both_Arms   -   Clean Table
        407521   -   ML_Both_Arms   -   Drink from Cup
        405506   -   ML_Both_Arms   -   Toggle Switch'''
        config['remap_classes']        = {0:0, 406516: 1, 406517: 2, 404516: 3, 404517: 4, 406520: 5, 404520: 6, 406505: 7, 404505: 8, 406519: 9, 404519: 10, 
                                          406511: 11, 404511: 12, 406508: 13, 404508: 14, 408512: 15, 407521: 16, 405506: 17}
        config['keep_nan']             = True
        config['skip_header']          = False
        config['skip_first_col']       = False
        config['column_sep']           = " "
        config['frequency_reducer']    = 1
        config['num_classes']          = 18
        config['num_timesteps']        = 32
        config['overlap_steps']        = 16
        config['fcn_subjectid']        = lambda fname: int(fname.split('-')[0].replace("S", "")) - 1
    elif dataset_name == "OpportunityML-GRUINC":
        config['dataset_name']         = 'OpportunityML'
        config['dataset_path']         = "./datasets/opp/"
        config['training_files']       = ["S1-ADL1.dat", "S1-ADL3.dat", "S1-ADL4.dat",  
                                          "S2-ADL1.dat", "S2-ADL2.dat", "S2-ADL3.dat", 
                                          "S3-ADL2.dat", "S3-ADL3.dat", "S3-Drill.dat",
                                          "S4-ADL1.dat",  "S4-ADL2.dat", "S4-ADL3.dat",  "S4-ADL5.dat", 
                                          "S1-Drill.dat",  "S3-Drill.dat", "S4-Drill.dat", "S2-ADL4.dat",  "S2-ADL5.dat",  "S3-ADL5.dat"]
        config['validation_files']     = ["S1-ADL2.dat" ]
        config['test_files']           = ["S2-Drill.dat", "S1-ADL5.dat",  "S3-ADL1.dat", "S3-ADL4.dat", "S4-ADL4.dat"]
        config['class_index']          = 249
        config['input_columns']        = [i for i in range(1, 134)] # Only body-worn sensors
        config['exclude_classes']      = []
        '''406516   -   ML_Both_Arms   -   Open Door 1
        406517   -   ML_Both_Arms   -   Open Door 2
        404516   -   ML_Both_Arms   -   Close Door 1
        404517   -   ML_Both_Arms   -   Close Door 2
        406520   -   ML_Both_Arms   -   Open Fridge
        404520   -   ML_Both_Arms   -   Close Fridge
        406505   -   ML_Both_Arms   -   Open Dishwasher
        404505   -   ML_Both_Arms   -   Close Dishwasher
        406519   -   ML_Both_Arms   -   Open Drawer 1
        404519   -   ML_Both_Arms   -   Close Drawer 1
        406511   -   ML_Both_Arms   -   Open Drawer 2
        404511   -   ML_Both_Arms   -   Close Drawer 2
        406508   -   ML_Both_Arms   -   Open Drawer 3
        404508   -   ML_Both_Arms   -   Close Drawer 3
        408512   -   ML_Both_Arms   -   Clean Table
        407521   -   ML_Both_Arms   -   Drink from Cup
        405506   -   ML_Both_Arms   -   Toggle Switch'''
        config['remap_classes']        = {0:0, 406516: 1, 406517: 2, 404516: 3, 404517: 4, 406520: 5, 404520: 6, 406505: 7, 404505: 8, 406519: 9, 404519: 10, 
                                          406511: 11, 404511: 12, 406508: 13, 404508: 14, 408512: 15, 407521: 16, 405506: 17}
        config['keep_nan']             = True
        config['skip_header']          = False
        config['skip_first_col']       = False
        config['column_sep']           = " "
        config['frequency_reducer']    = 1
        config['num_classes']          = 18
        config['num_timesteps']        = 90
        config['overlap_steps']        = 20  
        config['fcn_subjectid']        = lambda fname: int(fname.split('-')[0].replace("S", "")) - 1
    elif dataset_name == "OpportunityLoco":
        config['dataset_name']         = 'OpportunityLoco'
        config['dataset_path']         = "./datasets/opp/"
        config['training_files']       = ["S1-ADL1.dat", "S1-ADL3.dat", "S1-ADL4.dat", "S1-ADL5.dat", 
                                          "S2-ADL1.dat", "S2-ADL2.dat", "S2-ADL3.dat", 
                                          "S3-ADL1.dat", "S3-ADL2.dat", "S3-ADL3.dat", "S3-Drill.dat",
                                          "S4-ADL1.dat",  "S4-ADL2.dat", "S4-ADL3.dat", "S4-ADL4.dat", "S4-ADL5.dat", 
                                          "S1-Drill.dat",  "S2-Drill.dat","S3-Drill.dat", "S4-Drill.dat"]
        config['validation_files']     = ["S1-ADL2.dat" ]
        config['test_files']           = ["S2-ADL4.dat", "S2-ADL5.dat", "S3-ADL4.dat", "S3-ADL5.dat"]
        config['class_index']          = 243
        config['input_columns']        = [i for i in range(1, 134)] # Only body-worn sensors
        config['exclude_classes']      = []
        '''1   -   Locomotion   -   Stand
            2   -   Locomotion   -   Walk
            4   -   Locomotion   -   Sit
            5   -   Locomotion   -   Lie'''
        config['remap_classes']        = {1: 0, 2: 1, 4: 2, 5: 3}
        config['keep_nan']             = True
        config['skip_header']          = False
        config['skip_first_col']       = False
        config['column_sep']           = " "
        config['frequency_reducer']    = 1
        config['num_classes']          = 4
        config['num_timesteps']        = 32
        config['overlap_steps']        = 16
        config['fcn_subjectid']        = lambda fname: int(fname.split('-')[0].replace("S", "")) - 1
    elif dataset_name == "Skoda":
        config['dataset_name']         = 'Skoda'
        config['dataset_path']         = "./datasets/skoda/"
        config['training_files']       = ["skoda_train.csv"]
        config['validation_files']     = ["skoda_val.csv"]
        config['test_files']           = ["skoda_test.csv"]
        config['class_index']          = 0
        config['input_columns']        = [i for i in range(1, 61)] 
        config['exclude_classes']      = []
        '''1   -   Locomotion   -   Stand
            2   -   Locomotion   -   Walk
            4   -   Locomotion   -   Sit
            5   -   Locomotion   -   Lie'''
        config['remap_classes']        = {}
        config['keep_nan']             = True
        config['skip_header']          = False
        config['skip_first_col']       = False
        config['column_sep']           = ","
        config['frequency_reducer']    = 3
        config['num_classes']          = 10
        config['num_timesteps']        = 32
        config['overlap_steps']        = 16
        config['fcn_subjectid']        = lambda fname: 0
    elif dataset_name == "USC-HAD":
        config['dataset_name']         = 'USC-HAD'
        config['dataset_path']         = "./datasets/USC-HAD/"
        config['training_files']       = ["Subject1.csv", "Subject2.csv", "Subject3.csv", "Subject4.csv", "Subject5.csv",
                                          "Subject6.csv", "Subject7.csv", "Subject8.csv", "Subject9.csv", "Subject10.csv"]
        config['validation_files']     = ["Subject11.csv", "Subject12.csv"]
        config['test_files']           = ["Subject13.csv", "Subject14.csv"]
        config['class_index']          = 6
        config['input_columns']        = [i for i in range(0, 6)] 
        config['exclude_classes']      = []
        config['remap_classes']        = {}
        config['keep_nan']             = True
        config['skip_header']          = False
        config['skip_first_col']       = False
        config['column_sep']           = ","
        config['frequency_reducer']    = 3
        config['num_classes']          = 12
        config['num_timesteps']        = 32
        config['overlap_steps']        = 16
        config['fcn_subjectid']        = lambda fname: int(fname.replace("Subject", "").replace(".csv", "")) - 1
        
    
    config['num_channels']         = len(config['input_columns'])   
    return config
