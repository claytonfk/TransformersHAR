# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 19:34:16 2023

@author: clayt
"""
import tensorflow as tf
from tensorflow.keras import layers

class PatchExtractor(layers.Layer):
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size

    def segment_input(self, input_tensor, patch_size, stride):
        """
        Segment a 4D input tensor in the second and third dimensions.
    
        Parameters:
        - input_tensor: 4D input tensor (batch, height, width, channels)
        - patch_size: Tuple specifying the size of the patches (patch_height, patch_width)
        - stride: Tuple specifying the stride of the sliding window (stride_height, stride_width)
    
        Returns:
        - segmented_patches: Segmented patches tensor
        """
    
        # Ensure the input tensor has the correct rank
        assert input_tensor.shape.ndims == 4, "Input tensor must be 4D"
    
        # Get input tensor dimensions
        batch_size, height, width, channels = input_tensor.shape
    
        # Calculate the number of patches in each dimension
        num_patches_height = (height - patch_size[0]) // stride[0] + 1
        num_patches_width = (width - patch_size[1]) // stride[1] + 1
    
        # Initialize an empty list to store patches
        patches = []
    
        # Iterate over patches in the second dimension
        for j in range(num_patches_width):
           w_start = j * stride[1]
           w_end = w_start + patch_size[1]

            # Iterate over patches in the third dimension
           for i in range(num_patches_height):
                h_start = i * stride[0]
                h_end = h_start + patch_size[0]
        
                # Slice the input tensor to get the current patch
                patch = input_tensor[:, h_start:h_end, w_start:w_end, :]
    
                # Append the patch to the list
                patches.append(patch)
    
        # Concatenate the patches along a new dimension (axis=1)
        segmented_patches = tf.concat(patches, axis=-1)
    
        return segmented_patches

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        
        # patches = self.segment_input(images, [self.patch_size[0], self.patch_size[1]], [self.patch_size[0], self.patch_size[1]])

        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches_spatial, num_patches_temporal, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches_spatial = num_patches_temporal
        self.num_patches_temporal = num_patches_spatial
        self.projection_dim = projection_dim
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = self.add_weight(shape=(1, self.num_patches_spatial*self.num_patches_temporal + 1, self.projection_dim), name="embedding") #layers.Embedding(input_dim=self.num_patches_spatial*self.num_patches_temporal + 1, output_dim=projection_dim)

    def call(self, patch):
        batch = tf.shape(patch)[0]
        n = tf.shape(patch)[1]
        class_token = tf.tile(self.class_token, multiples = [batch, 1])
        class_token = tf.reshape(class_token, (batch, 1, self.projection_dim))
        # calculate patches embeddings
        patches_embed = self.projection(patch)
        patches_embed = tf.concat([patches_embed, class_token], 1)

        #positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
        
        # positions = tf.tile(tf.range(start=0, limit=self.num_patches_spatial , delta=1), multiples = [self.num_patches_temporal])

        # positions = tf.concat([positions, [self.num_patches_spatial]], axis = 0)
        
        # positions_embed = self.position_embedding(positions)
        # add both embeddings

        encoded = patches_embed + self.position_embedding[:, :(n + 1)]
        return encoded

class MLP(layers.Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.dense1 = layers.Dense(hidden_features, activation=tf.nn.gelu)
        self.dense2 = layers.Dense(out_features)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training):
        x = self.dense1(x)
        x = self.dropout(x, training = training)
        x = self.dense2(x)
        y = self.dropout(x, training = training)
        return y
    
class TransformerBlock(layers.Layer):
    def __init__(self, projection_dim, num_heads, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.dropout = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)

    def call(self, x, training):
        x1 = self.norm1(x) # encoded_patches
        attention_output = self.attn(x1, x1)
        attention_output = self.dropout(attention_output, training = training)
        x2 = layers.Add()([attention_output, x])
        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        y = layers.Add()([x3, x2])
        return y
    

class ViT(tf.keras.Model):
    def __init__(
        self,
        num_timesteps,
        num_channels,
        num_classes,
        patch_size,
        embedding_dim,
        num_heads,
        transformer_layers,
        mlp_head_units,
        softmax = True
    ):
        super(ViT, self).__init__()
        self.model_name = 'vit'

        # Calculate the number of patches
        
        self.num_patches = (num_timesteps // patch_size[0]) * (num_channels // patch_size[1])
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.patch_extrator = PatchExtractor(self.patch_size)
        self.patch_encoder = PatchEncoder((num_channels // patch_size[1]), (num_timesteps // patch_size[0]), self.embedding_dim)
        self.softmax = softmax


        # Transformer encoder layers
        self.transformer_layers = []
        for _ in range(transformer_layers):
            self.transformer_layers.append(TransformerBlock(embedding_dim, num_heads))

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        # Global average pooling
        self.pooling = layers.GlobalAveragePooling1D()

        # MLP head
       # self.awc = AttentionWithContext(embedding_dim)
        self.mlp_output  = MLP(mlp_head_units, num_classes)
        
        patch_size_str = str(patch_size).replace(", ", "by").replace("[", "").replace("]", "")
        
        
        self.info = f"ps_{patch_size_str}_ed_{embedding_dim}_nh_{num_heads}_tl_{transformer_layers}_hu_{mlp_head_units}"

    
    def call(self, inputs, training = False):
        inputs = tf.expand_dims(inputs, axis = 3)
        x = self.patch_extrator(inputs) 
        #print(x.shape)
        x = self.patch_encoder(x)
        #print(x.shape)
        for layer in self.transformer_layers:
            x = layer(x, training)
            
        x = self.norm(x)
        #print(x.shape)
        #print(x.shape)
        x = tf.reshape(x, (-1, x.shape[1]*x.shape[2]))
        #x = x[:, -1, :]

        #x = self.pooling(x)
        #print(x.shape)
        #x = self.awc(x)
        x = self.mlp_output(x)
        if self.softmax:
            x = tf.nn.softmax(x)
        return x
    
class DoubleViT(tf.keras.Model):
    def __init__(
        self,
        num_timesteps,
        num_channels,
        num_classes,
        patch_size,
        embedding_dim,
        num_heads,
        transformer_layers,
        mlp_head_units
    ):
        super(DoubleViT, self).__init__()
        self.model_name = 'doublevit'

        # Calculate the number of patches
        
        self.stream1 = ViT(num_timesteps, num_channels, 512, patch_size, embedding_dim, num_heads, transformer_layers, mlp_head_units, softmax=False)
        self.stream2 = ViT(num_timesteps, num_channels, num_classes, patch_size, embedding_dim, num_heads, transformer_layers, mlp_head_units, softmax=False)
        
      
        self.mlp_output  = MLP(mlp_head_units + 512, num_classes)
        self.do = layers.Dropout(0.3)
        
        patch_size_str = str(patch_size).replace(", ", "by").replace("[", "").replace("]", "")
        self.info = f"ps_{patch_size_str}_ed_{embedding_dim}_nh_{num_heads}_tl_{transformer_layers}_hu_{mlp_head_units}"

    def call_stream1(self, inputs, training = False):   
        return self.stream1.call(inputs, training=training)


    def call_stream2(self, inputs, training = False):
        s2 = self.stream2.call(inputs, training=training)
        s2 = self.do(s2, training=training)
        return s2
    
    def call(self, inputs, training = False):
        s1 = self.call_stream1(inputs, training=training)
        s2 = self.call_stream2(inputs, training=training)
        s2 = self.do(s2, training=training)
        
        s = tf.concat([s1, s2], axis = -1)
        s = tf.nn.relu(s)
        
        s = self.mlp_output(s)
        s = tf.nn.softmax(s)
        
        return s
    
    def call_returnall(self, inputs, training = False):
        s1 = self.call_stream1(inputs, training=training)
        s2 = self.call_stream2(inputs, training=training)
        s2 = self.do(s2, training=training)
        
        s = tf.concat([s1, s2], axis = -1)
        s = tf.nn.relu(s)
        
        s = self.mlp_output(s)
        s = tf.nn.softmax(s)
        
        return s1, s2, s
        
        
        
    