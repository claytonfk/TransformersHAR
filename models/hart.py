# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 03:05:04 2024

@author: clayt
"""


import tensorflow as tf

from tensorflow.keras import layers
import numpy as np

class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x,training=None):
        if(training):
            input_shape = tf.shape(x)
            batch_size = input_shape[0]
            rank = x.shape.rank
            shape = (batch_size,) + (1,) * (rank - 1)
            random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
            path_mask = tf.floor(random_tensor)
            output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
            return output
        else:
            return x 

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'drop_prob': self.drop_prob,})
        return config


class GatedLinearUnit(layers.Layer):
    def __init__(self,units,**kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.units = units
        self.linear = layers.Dense(units * 2)
        self.sigmoid = tf.keras.activations.sigmoid
    def call(self, inputs):
        linearProjection = self.linear(inputs)
        softMaxProjection = self.sigmoid(linearProjection[:,:,self.units:])
        return tf.multiply(linearProjection[:,:,:self.units],softMaxProjection)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim,  **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, patch):
        encoded = patch + self.position_embedding(self.positions)
        return encoded
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,})
        return config
    
class PatchEncoderV2(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoderV2, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)
        self.embeddings = tf.constant(self.position_embedding(tf.stop_gradient(self.positions)))
        self.position_embedding = -1

    def call(self, patch):
        encoded = patch + self.embeddings
        return encoded
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,})
        return config

class ClassToken(layers.Layer):
    def __init__(self, hidden_size,**kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.cls_init = tf.random.normal
        self.hidden_size = hidden_size
        self.cls = tf.Variable(
            name="cls",
            initial_value=self.cls_init(shape=(1, 1, self.hidden_size), seed=0, dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_size': self.hidden_size,})
        return config

class Prompts(layers.Layer):
    def __init__(self, projectionDims,promptCount = 1,**kwargs):
        super(Prompts, self).__init__(**kwargs)
        self.cls_init = tf.random.normal
        self.projectionDims = projectionDims
        self.promptCount = promptCount
        self.prompts = [tf.Variable(
            name="prompt"+str(_),
            initial_value=self.cls_init(shape=(1, 1, self.projectionDims), seed=0, dtype="float32"),
            trainable=True,
        )  for _ in range(promptCount)]

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        prompt_broadcasted = tf.concat([tf.cast(tf.broadcast_to(promptInits, [batch_size, 1, self.projectionDims]),dtype=inputs.dtype,)for promptInits in self.prompts],1)
        return tf.concat([inputs,prompt_broadcasted], 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionDims': self.projectionDims,
            'promptCount': self.promptCount,})
        return config
    
class SensorWiseMHA(layers.Layer):
    def __init__(self, projectionQuarter, num_heads,startIndex,stopIndex,dropout_rate = 0.0,dropPathRate = 0.0, **kwargs):
        super(SensorWiseMHA, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.MHA = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projectionQuarter, dropout = dropout_rate )
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.dropPathRate = dropPathRate
        self.DropPath = DropPath(dropPathRate)
    def call(self, inputData, training=None, return_attention_scores = False):
        extractedInput = inputData[:,:,self.startIndex:self.stopIndex]
        if(return_attention_scores):
            MHA_Outputs, attentionScores = self.MHA(extractedInput,extractedInput,return_attention_scores = True )
            return MHA_Outputs , attentionScores
        else:
            MHA_Outputs = self.MHA(extractedInput,extractedInput)
            MHA_Outputs = self.DropPath(MHA_Outputs)
            return MHA_Outputs
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'num_heads': self.num_heads,
            'startIndex': self.startIndex,
            'dropout_rate': self.dropout_rate,
            'stopIndex': self.stopIndex,
            'dropPathRate': self.dropPathRate,})
        return config
    
def softDepthConv(inputs):
    kernel = inputs[0]
    inputData = inputs[1]
    convOutputs = tf.nn.conv1d(
    inputData,
    kernel,
    stride = 1,
    padding = 'SAME',
    data_format='NCW',)
    return convOutputs




class liteFormer(layers.Layer):
    def __init__(self,startIndex,stopIndex, projectionSize, kernelSize = 16, attentionHead = 3, use_bias=False, dropPathRate = 0.0,dropout_rate = 0,**kwargs):
        super(liteFormer, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.kernelSize = kernelSize
        self.softmax = tf.nn.softmax
        self.projectionSize = projectionSize
        self.attentionHead = attentionHead 
        self.DropPathLayer = DropPath(dropPathRate)
        self.projectionHalf = projectionSize // 2
    def build(self,inputShape):
        self.depthwise_kernel = [self.add_weight(
            shape=(self.kernelSize,1,1),
            initializer="glorot_uniform",
            trainable=True,
            name="convWeights"+str(_),
            dtype="float32") for _ in range(self.attentionHead)]
        if self.use_bias:
            self.convBias = self.add_weight(
                shape=(self.attentionHead,), 
                initializer="glorot_uniform", 
                trainable=True,  
                name="biasWeights",
                dtype="float32"
            )
        
    def call(self, inputs,training=None):
        formattedInputs = inputs[:,:,self.startIndex:self.stopIndex]
        inputShape = tf.shape(formattedInputs)
        reshapedInputs = tf.reshape(formattedInputs,(-1,self.attentionHead,inputShape[1]))
        if(training):
            for convIndex in range(self.attentionHead):
                self.depthwise_kernel[convIndex].assign(self.softmax(self.depthwise_kernel[convIndex], axis=0))
        convOutputs = tf.convert_to_tensor([tf.nn.conv1d(
            reshapedInputs[:,convIndex:convIndex+1,:],
            self.depthwise_kernel[convIndex],
            stride = 1,
            padding = 'SAME',
            data_format='NCW',) for convIndex in range(self.attentionHead) ])
        convOutputsDropPath = self.DropPathLayer(convOutputs)
        localAttention = tf.reshape(convOutputsDropPath,(-1,inputShape[1],self.projectionSize))
        return localAttention
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_bias': self.use_bias,
            'kernelSize': self.kernelSize,
            'startIndex': self.startIndex,
            'stopIndex': self.stopIndex,
            'projectionSize': self.projectionSize,
            'attentionHead': self.attentionHead,})
        return config          

class mixAccGyro(layers.Layer):
    def __init__(self,projectionQuarter,projectionHalf,projection_dim,**kwargs):
        super(mixAccGyro, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.projectionHalf = projectionHalf
        self.projection_dim = projection_dim
        self.projectionThreeFourth = self.projectionHalf+self.projectionQuarter
        self.mixedAccGyroIndex = tf.reshape(tf.transpose(tf.stack(
            [np.arange(projectionQuarter,projectionHalf), np.arange(projectionHalf,projectionHalf + projectionQuarter)])),[-1])
        self.newArrangement = tf.concat((np.arange(0,projectionQuarter),self.mixedAccGyroIndex,np.arange(self.projectionThreeFourth,projection_dim)),axis = 0)
    def call(self, inputs):
        return tf.gather(inputs,self.newArrangement,axis= 2)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'projectionHalf': self.projectionHalf,
            'projection_dim': self.projection_dim,
        })
        return config

class MLP2(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, dropout=0.3):
        super(MLP2, self).__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(hidden_dim[0]),
            layers.Activation(tf.nn.swish),
            layers.Dropout(dropout),
            layers.Dense(hidden_dim[1])
        ])

    def call(self, x, training = False):
        return self.net(x, training = training)
    
    
class MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, dropout=0.3):
        super(MLP, self).__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(hidden_dim),
            layers.Activation(tf.nn.swish),
            layers.Dropout(dropout)
        ])

    def call(self, x, training = False):
        return self.net(x, training = training)

class SensorPatchesTimeDistributed(layers.Layer):
    def __init__(self, projection_dim,filterCount,patchCount,frameSize = 128, channelsCount = 6,**kwargs):
        super(SensorPatchesTimeDistributed, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.frameSize = frameSize
        self.channelsCount = channelsCount
        self.patchCount = patchCount
        self.filterCount = filterCount
        self.reshapeInputs = layers.Reshape((patchCount, frameSize // patchCount, channelsCount))
        self.kernelSize = (projection_dim//2 + filterCount) // filterCount
        self.accProjection = layers.TimeDistributed(layers.Conv1D(filters = filterCount,kernel_size = self.kernelSize,strides = 1, data_format = "channels_last"))
        self.gyroProjection = layers.TimeDistributed(layers.Conv1D(filters = filterCount,kernel_size = self.kernelSize,strides = 1, data_format = "channels_last"))
        self.flattenTime = layers.TimeDistributed(layers.Flatten())
        assert (projection_dim//2 + filterCount) / filterCount % self.kernelSize == 0
        print("Kernel Size is "+str((projection_dim//2 + filterCount) / filterCount))
#         assert 
    def call(self, inputData):
        inputData = self.reshapeInputs(inputData)
        accProjections = self.flattenTime(self.accProjection(inputData[:,:,:,:3]))
        gyroProjections = self.flattenTime(self.gyroProjection(inputData[:,:,:,3:]))
        Projections = tf.concat((accProjections,gyroProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projection_dim': self.projection_dim,
            'filterCount': self.filterCount,
            'patchCount': self.patchCount,
            'frameSize': self.frameSize,
            'channelsCount': self.channelsCount,})
        return config
    
class SensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize, timeStep, num_channels,  **kwargs):
        super(SensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim/2),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim/2),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.num_channels = num_channels
    def call(self, inputData):
        #accProjections = self.accProjection(inputData[:,:,:3])
        #gyroProjections = self.gyroProjection(inputData[:,:,3:])
        num_ch = int(self.num_channels / 2)
        accProjections = self.accProjection(inputData[:, :, :num_ch])
        gyroProjections = self.gyroProjection(inputData[:, :, num_ch:])
        Projections = tf.concat((accProjections,gyroProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config

def extract_intermediate_model_from_base_model(base_model, intermediate_layer=4):
    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model


class HART(tf.keras.Model):
    def __init__(self, num_timesteps, num_channels, num_classes, projection_dim = 128, patchSize = 16, time_stride = 16, num_heads = 2, filterAttentionHead = 4, 
             convKernels = [3, 7, 15, 31, 31, 31],  mlp_head_units = 256, dropout_rate = 0.3, useTokens = False, pre_embedding = False): #, 31, 31],
                 
        super(HART, self).__init__()
        projectionHalf = projection_dim//2
        projectionQuarter = projection_dim//4
        dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
        transformer_units = [projection_dim * 2, projection_dim, ]  

        self.model_name = "hart"
        
        self.sensor_patches = SensorPatches(projection_dim, patchSize, time_stride, num_channels)
        self.useTokens = useTokens
        if pre_embedding:
            self.patch_encoder = PatchEncoderV2(num_timesteps // patchSize, projection_dim)
        else:
            self.patch_encoder = PatchEncoder(num_timesteps // patchSize, projection_dim)
        if useTokens:
            self.class_token = ClassToken(projection_dim)
        
        self.norm = []
        self.branches = []
        self.branches_acc = []
        self.branches_gyro = []
        self.x3_norm = []
        self.drop_path = []
        self.mlp2 = []
        
        self.convKernels = convKernels
        for layerIndex, kernelLength in enumerate(convKernels):
            self.norm.append(layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex)))
            self.branches.append(liteFormer(
                              startIndex = projectionQuarter,
                              stopIndex = projectionQuarter + projectionHalf,
                              projectionSize = projectionHalf,
                              attentionHead =  filterAttentionHead, 
                              kernelSize = kernelLength,
                              dropPathRate = dropPathRate[layerIndex],
                              dropout_rate = dropout_rate,
                              name = "liteFormer_"+str(layerIndex)))
            self.branches_acc.append(SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex)))
            self.branches_gyro.append(SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf, projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex)))
            self.x3_norm.append(layers.LayerNormalization(epsilon=1e-6))
            self.drop_path.append(DropPath(dropPathRate[layerIndex]))
            self.mlp2.append(MLP2(transformer_units, dropout_rate))
            
        if useTokens:
            self.post_representation = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")
        else:
            self.post_representation = layers.GlobalAveragePooling1D()
        
        self.mlp = MLP(mlp_head_units, dropout_rate)
        self.representation_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dense = layers.Dense(num_classes)            
        self.info = f"pd_{projection_dim}_ps_{patchSize}_ts_{time_stride}_nh_{num_heads}_fah_{filterAttentionHead}_mlp_{mlp_head_units}_do_{dropout_rate}"
        
    def call(self, inputs, training = False):
        # Forward pass
    
        patches = self.sensor_patches(inputs)
        if self.useTokens:
            patches = self.class_token(patches)
        encoded_patches = self.patch_encoder(patches)
        x1 = encoded_patches

        #print(patches.shape[1])
        
        for i in range(len(self.convKernels)):
            x1 = self.norm[i](x1)
            xb = self.branches[i](x1, training = training)
            xa = self.branches_acc[i](x1, training = training)
            xg = self.branches_gyro[i](x1, training = training)
            
            concat_att = tf.concat((xa,xb,xg),axis= 2 )
            x2 = concat_att + encoded_patches
            x3 = self.x3_norm[i](x2)
            x3 = self.mlp2[i](x3, training = training)
            x3 = self.drop_path[i](x3, training = training)
            x1 = x3 + x2
        
        x = self.representation_norm(x1)
        x = self.post_representation(x)
        x = self.mlp(x, training = training)
        x = self.dense(x)
        x = tf.nn.softmax(x)
        
        return x




# def HART(num_timesteps, num_channels, num_classes, projection_dim = 192, patchSize = 16, num_heads = 3, filterAttentionHead = 4, 
#          convKernels = [3, 7, 15, 31, 31, 31], mlp_head_units = [1024], dropout_rate = 0.3, useTokens = False):
#     projectionHalf = projection_dim//2
#     projectionQuarter = projection_dim//4
#     dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
#     transformer_units = [projection_dim * 2, projection_dim, ]  
#     inputs = layers.Input(shape=(num_timesteps, num_channels))


#     patches = SensorPatches(projection_dim, patchSize, num_timesteps, num_channels)(inputs)

#     if useTokens:
#         patches = ClassToken(projection_dim)(patches)
        
#     patchCount = patches.shape[1] 
#     encoded_patches = PatchEncoder(patchCount, projection_dim)(patches)
#     # Create multiple layers of the Transformer block.
#     for layerIndex, kernelLength in enumerate(convKernels):        
#         x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
#         branch1 = liteFormer(
#                           startIndex = projectionQuarter,
#                           stopIndex = projectionQuarter + projectionHalf,
#                           projectionSize = projectionHalf,
#                           attentionHead =  filterAttentionHead, 
#                           kernelSize = kernelLength,
#                           dropPathRate = dropPathRate[layerIndex],
#                           dropout_rate = dropout_rate,
#                           name = "liteFormer_"+str(layerIndex))(x1)

                          
#         branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)

#         branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf, projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x1)

#         concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
#         x2 = layers.Add()([concatAttention, encoded_patches])
#         x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
#         x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
#         x3 = DropPath(dropPathRate[layerIndex])(x3)
#         encoded_patches = layers.Add()([x3, x2])
        
#     representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
#     if useTokens:
#         representation = layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(representation)
#     else:
#         representation = layers.GlobalAveragePooling1D()(representation)
        
#     features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)
#     logits = layers.Dense(num_classes)(features)
#     probs = tf.nn.softmax(logits)
#     model = tf.keras.Model(inputs=inputs, outputs=probs)
    
#     model.model_name = 'hart'
#     model.info = 'no_info'
    
#     return model
# # ------------------------------specific module for MobileHART------------------------------




# ACTIVITY_LABEL = ['Walking', 'Upstair','Downstair', 'Sitting', 'Standing', 'Lying']
# activityCount = len(ACTIVITY_LABEL)
# segment_size = 64
# num_input_channels = 100 #Total number of sensors
# input_shape = (segment_size,num_input_channels)
# print(input_shape)
# model_classifier = HART(input_shape,activityCount)
# print(model_classifier)
# learningRate = 3e-4
# optimizer = tf.keras.optimizers.Adam(learningRate)
# model_classifier.compile(
#     optimizer=optimizer,
#     loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
#     metrics=["accuracy"],
# )
# model_classifier.summary()

