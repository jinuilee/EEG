#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Activation, GlobalAveragePooling1D, Multiply, Permute
from tensorflow.keras.models import Model
import tensorflow as tf

input_shape=(input_length, n_channel, 1)
input_layer = tf.keras.Input(shape=input_shape)

sliced_layer = input_layer[:, :, 1:3, :]

# the model needs to be conveyed with the specified number of layers
n_layers = 4 

output_list = []

# Temporal wise convolution
conv1 = Conv2D(filters=16, kernel_size=(5, 1), strides=(3, 1), padding='SAME')(sliced_layer)
conv1 = Activation('relu')(conv1)

# Channel wise convolution
conv2 = Conv2D(filters=16, kernel_size=(1, 2), padding='SAME')(sliced_layer)
conv2 = Activation('relu')(conv2)
conv2 = MaxPooling2D(pool_size=(5, 1), strides=(3, 1), padding='SAME')(conv2)
concat = Concatenate()([conv1, conv2])

# Self-attention
query = Conv2D(filters=32, kernel_size=(1, 1), padding='SAME')(concat)
query = tf.reshape(query, (-1, query.shape[1], query.shape[2] * query.shape[3]))
key = Conv2D(filters=32, kernel_size=(1, 1), padding='SAME')(concat)
key = tf.reshape(key, (-1, key.shape[1], key.shape[2] * key.shape[3]))
value = Conv2D(filters=32, kernel_size=(1, 1), padding='SAME')(concat)
value = tf.reshape(value, (-1, value.shape[1], value.shape[2] * value.shape[3]))

attention_weights = tf.matmul(query, key, transpose_b=True)
attention_weights = Activation('softmax')(attention_weights)
output_attended = tf.matmul(attention_weights, value)

# Residual connection
output_attended = tf.reshape(output_attended, tf.shape(concat))
output = tf.add(concat, output_attended)
output = Activation('relu')(output)

flatten = Flatten()(output)
output_list.append(flatten)
print("firsit output list: ", output_list)

iterations = n_layers
for i in range(iterations-1):
    tensor_split1 = output[:, :, :, :16]
    tensor_split2 = output[:, :, :, 16:]

    conv1 = Conv2D(filters=16, kernel_size=(5, 1), strides=(3, 1), padding='SAME')(tensor_split1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(filters=16, kernel_size=(1, 2), padding='SAME')(tensor_split2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(5, 1), strides=(3, 1), padding='SAME')(conv2)
    concat = Concatenate()([conv1, conv2])

    # Self-attention
    query = Conv2D(filters=32, kernel_size=(1, 1), padding='SAME')(concat)
    query = tf.reshape(query, (-1, query.shape[1], query.shape[2] * query.shape[3]))
    key = Conv2D(filters=32, kernel_size=(1, 1), padding='SAME')(concat)
    key = tf.reshape(key, (-1, key.shape[1], key.shape[2] * key.shape[3]))
    value = Conv2D(filters=32, kernel_size=(1, 1), padding='SAME')(concat)
    value = tf.reshape(value, (-1, value.shape[1], value.shape[2] * value.shape[3]))
    
    attention_weights = tf.matmul(query, key, transpose_b=True)
    attention_weights = Activation('softmax')(attention_weights)
    output_attended = tf.matmul(attention_weights, value)

    # Residual connection
    output_attended = tf.reshape(output_attended, tf.shape(concat))
    output = tf.add(concat, output_attended)
    output = Activation('relu')(output)
    
    flatten = Flatten()(output)
    output_list.append(flatten)
    
output_concat = Concatenate()(output_list)
dense = Dense(units=64, activation='relu')(output_concat)
output = Dense(units=1, activation='sigmoid')(dense)
model = Model(inputs=input_layer, outputs=output)

model.summary()

