# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 01:15:29 2023

@author: clayt
"""

from .resbilstm import ResidualBiLSTM
from .innohar import InnoHAR
from .vit import ViT, DoubleViT
from .crossvit import CrossViT
from .te import TE
from .ttn import TTN
from .retnet import RetNetDecoder
from .tasked.TASKEDArchitecture_t import TASKED
import os, glob
from .hart import HART

def get_model(model_name, config, unroll = False, pre_embedding = False):
    if model_name == 'residual_bi_lstm':
        model = ResidualBiLSTM(config['num_classes'], 
                               conv_filters = 32, 
                               lstm_units = 64,
                               unroll = unroll)
    elif model_name == 'inno_har':
        model = InnoHAR(config['num_classes'], 
                        conv_num_filters = 128, 
                        gru_units = 64,
                        dropout_rate = 0.5,
                        unroll = unroll)
    elif model_name == 'vi_t':
        model = ViT(config['num_timesteps'],
                    config['num_channels'], 
                    config['num_classes'],
                    patch_size = [8, 8],
                    embedding_dim = 64,
                    num_heads = 4,
                    transformer_layers = 2,
                    mlp_head_units = 64,
                    softmax = True)
        
    elif model_name == 'doublevi_t':
        model = DoubleViT(config['num_timesteps'],
                    config['num_channels'], 
                    12,
                    patch_size = [8, 8],
                    embedding_dim = 64,
                    num_heads = 4,
                    transformer_layers = 2,
                    mlp_head_units = 64)
            
    elif model_name == 'cross_vi_t':
        model = CrossViT(config['num_timesteps'], 
                          config['num_channels'], 
                          config['num_classes'], 
                          patch_size_small = [8, 8], 
                          patch_size_large = [16, 16],
                          small_dim=32,
                          large_dim=128,
                          heads=4,
                          multi_scale_enc_depth=1,
                          dropout=0.5, 
                          emb_dropout=0.5,
                          pool='cls')
        
    elif model_name == 'ttn':
        model = TTN(config['num_classes'], 
                    config['num_channels'], 
                    config['num_timesteps'], 
                    model_dim = 128,
                    num_heads_spatial = 4, 
                    num_heads_temporal = 4,
                    num_layers_spatial = 2, 
                    num_layers_temporal = 2, 
                    dropout_rate = 0.5)
        
    elif model_name == 'te':
        model = TE(config['num_classes'], 
                    config['num_channels'], 
                    config['num_timesteps'],
                    num_heads_spatial = 4, 
                    num_layers_spatial = 2, 
                    model_dim = 64,
                    num_filters = 64,
                    dropout_rate = 0.5)
    elif model_name == 'ret_net_decoder':
        model = RetNetDecoder(config['num_classes'], 
                              config['num_channels'], 
                              config['num_timesteps'],
                              dropout = 0.25, 
                              ffn_dim = 128,
                              num_decoder_layers = 2,  
                              num_retention_heads = 4,
                              conv_filters = 128, 
                              conv_kernel = 2)
    elif model_name == 'tasked':
        model = TASKED( 2, # Cs, channels per sensor
                       config['num_timesteps'],  # W, Window length
                       config['num_channels'] // 3,  # S, number of sensors (total channels divided by channels per sensor)
                       config['num_classes'], # na, class count
                       len(config['training_files']))  # N, number of subjects. NOTE: this assumes each subject is their individual file, like in PAMAP. Subject count is not available in configs by default

    elif model_name == 'hart':    
        model = HART(config['num_timesteps'], 
                     config['num_channels'], 
                     config['num_classes'], 
                     projection_dim = 192, 
                     patchSize = 16,
                     time_stride = 16,
                     num_heads = 3, 
                     filterAttentionHead = 4, 
                     convKernels = [3, 7, 15, 31, 31, 31], 
                     mlp_head_units = 1024,
                     dropout_rate = 0.3, 
                     useTokens = False, 
                     pre_embedding = pre_embedding)
    else:
        raise ValueError(f"Unknown model type {model_name}")

    return model



def get_model_from_path(model_name, config, filepath, unroll = False, pre_embedding = False):
    
    filepath = glob.glob(os.path.join(filepath, "*.index"))
    filepath = filepath[0]
    filename = os.path.basename(filepath)

    filename = filename.replace(".ckpt", "").replace(".index", "")
    filename = filename.split("_")
    if model_name == 'residual_bi_lstm':

        conv_filters = int(filename[5])
        lstm_units =  int(filename[7])
        model = ResidualBiLSTM(config['num_classes'], 
                               conv_filters = conv_filters, 
                               lstm_units = lstm_units,
                               unroll = unroll)
    elif model_name == 'inno_har':
        conv_num_filters = int(filename[5])
        gru_units =  int(filename[7])
        dropout_rate = float(filename[9])
        
        model = InnoHAR(config['num_classes'], 
                        conv_num_filters = conv_num_filters, 
                        gru_units = gru_units,
                        dropout_rate = dropout_rate,
                        unroll = unroll)
    elif model_name == 'vi_t':
        
        patch_size = filename[5].split("by")
        patch_size = [int(e) for e in patch_size]
        embedding_dim = int(filename[7])
        num_heads = int(filename[9])
        transformer_layers =  int(filename[11])
        mlp_head_units = int(filename[13])
        
        
        model = ViT(config['num_timesteps'],
                    config['num_channels'], 
                    config['num_classes'],
                    patch_size = patch_size,
                    embedding_dim = embedding_dim,
                    num_heads = num_heads,
                    transformer_layers = transformer_layers,
                    mlp_head_units = mlp_head_units)
        
    elif model_name == 'cross_vi_t':
        
        patch_size_small = filename[5].split("by")
        patch_size_small = [int(e) for e in patch_size_small]
        patch_size_large = filename[7].split("by")
        patch_size_large = [int(e) for e in patch_size_large]
        small_dim=int(filename[9])
        large_dim=int(filename[11])
        heads=int(filename[13])
        multi_scale_enc_depth=int(filename[15])
        dropout=float(filename[17])
        emb_dropout=float(filename[17])
        
        model = CrossViT(config['num_timesteps'], 
                          config['num_channels'], 
                          config['num_classes'], 
                          patch_size_small = patch_size_small, 
                          patch_size_large = patch_size_large,
                          small_dim=small_dim,
                          large_dim=large_dim,
                          heads=heads,
                          multi_scale_enc_depth=multi_scale_enc_depth,
                          dropout=dropout, 
                          emb_dropout=emb_dropout,
                          pool='cls')
        
    elif model_name == 'ttn':
        
        model_dim = int(filename[6])
        num_heads_spatial = int(filename[8])
        num_heads_temporal = int(filename[10])
        num_layers_spatial = int(filename[12])
        num_layers_temporal = int(filename[14])
        dropout_rate = float(filename[16])
        
        model = TTN(config['num_classes'], 
                    config['num_channels'], 
                    config['num_timesteps'], 
                    model_dim = model_dim,
                    num_heads_spatial = num_heads_spatial, 
                    num_heads_temporal = num_heads_temporal,
                    num_layers_spatial = num_layers_spatial, 
                    num_layers_temporal = num_layers_temporal, 
                    dropout_rate = dropout_rate)
        
    elif model_name == 'te':
        num_heads_spatial = int(filename[5])
        num_layers_spatial = int(filename[7])
        model_dim = int(filename[9])
        num_filters = int(filename[11])
        dropout_rate = float(filename[13])
                
        
        model = TE(config['num_classes'], 
                    config['num_channels'], 
                    config['num_timesteps'],
                    num_heads_spatial = num_heads_spatial, 
                    num_layers_spatial = num_layers_spatial, 
                    model_dim = model_dim,
                    num_filters = num_filters,
                    dropout_rate = dropout_rate)
    elif model_name == 'ret_net_decoder':

        ffn_dim = int(filename[5])
        num_decoder_layers = int(filename[7])
        num_retention_heads = int(filename[9])
        conv_filters = int(filename[11])
        conv_kernel = int(filename[13])
        dropout = float(filename[15])
        
        model = RetNetDecoder(config['num_classes'], 
                              config['num_channels'], 
                              config['num_timesteps'],
                              dropout = dropout, 
                              ffn_dim = ffn_dim,
                              num_decoder_layers = num_decoder_layers,  
                              num_retention_heads = num_retention_heads,
                              conv_filters = conv_filters, 
                              conv_kernel = conv_kernel)
    elif model_name == 'tasked':  
        Cs = int(filename[5])
        S = int(filename[9])

        model = TASKED(Cs, 
                    config['num_timesteps'], 
                    S,  
                    config['num_classes'],
                    len(config['training_files']))
    elif model_name == 'hart':    
        projection_dim = int(filename[5])
        patchSize = int(filename[7])
        time_stride = int(filename[9])
        num_heads = int(filename[11])
        filterAttentionHead = int(filename[13])
        mlp_head_units = int(filename[15])
        dropout = float(filename[17])
        
        model = HART(config['num_timesteps'], 
                     config['num_channels'], 
                     config['num_classes'], 
                     projection_dim = projection_dim, 
                     patchSize = patchSize, 
                     time_stride = time_stride,
                     num_heads = num_heads, 
                     filterAttentionHead = filterAttentionHead, 
                     convKernels = [3, 7, 15, 31, 31, 31], 
                     mlp_head_units = mlp_head_units, 
                     dropout_rate = dropout, 
                     useTokens = False, 
                     pre_embedding = pre_embedding)
        
    return model, filepath