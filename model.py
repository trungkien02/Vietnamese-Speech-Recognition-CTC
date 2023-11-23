import tensorflow as tf
from keras import layers
from keras.models import Model
from mltu.tensorflow.model_utils import activation_layer

def train_model_LSTM(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input")

    input = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)

    x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")

    x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")
    
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(512, return_sequences=True))(x)

    x = layers.Dense(1024)(x)
    x = activation_layer(x, activation="leaky_relu")
    x = layers.Dropout(dropout)(x)

    output = layers.Dense(output_dim + 1, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model

def train_model_GRU(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    
    inputs = layers.Input(shape=input_dim, name="input")

    input = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)

    x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")

    x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")
    
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)


    for i in range(1, 6):
        recurrent = layers.GRU(
            units=512,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < 5:
            x = layers.Dropout(rate=0.5)(x)

    x = layers.Dense(1024)(x)
    x = activation_layer(x, activation="leaky_relu")
    x = layers.Dropout(dropout)(x)

    output = layers.Dense(output_dim + 1, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model