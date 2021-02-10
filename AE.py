import keras
from keras import layers
import numpy as np

def train_AE(input_data, k_features):
    """ builds AE k_features """
    input_dim = len(input_data[0])
    print (input_dim, input_data.shape)
    encoded_input = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(k_features, activation='relu')(encoded_input)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    ae = keras.Model(encoded_input, decoded)
    #--- extract RE-----
    reconstruction_loss = keras.losses.binary_crossentropy(encoded_input, decoded)
    ae.add_loss(reconstruction_loss)
    ae.compile(optimizer='adam')
    ae.compile(optimizer='adam', loss='binary_crossentropy')

    ae.fit(input_data, input_data,
                epochs=110,
                batch_size=16,
                shuffle=True) #validation_split=0.2)

    encoder = keras.Model(encoded_input, encoded)
    return encoder.predict(input_data), reconstruction_loss
