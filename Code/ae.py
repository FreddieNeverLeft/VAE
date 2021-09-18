import keras
from keras import layers, models
from keras import backend as K
from keras.layers import Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class autoencoder_util:

    def __init__(self):
        self.name = 'autoencoder'
        return

    def flatten(self, x):
        x = x.astype('float32') / 255
        x = x.reshape((len(x), np.prod(x.shape[1:])))
        return x

    def build_ae(self, training_shape):
        hidden_dims = [64, 16]
        latent_dims = 8

        input_data = keras.Input(shape=(training_shape,))
        e1 = layers.Dense(hidden_dims[0], activation='relu')(input_data)
        e2 = layers.Dense(hidden_dims[1], activation='relu')(e1)
        latent = layers.Dense(latent_dims, activation='relu')(e2)
        d1 = layers.Dense(hidden_dims[1], activation='relu')(latent)
        d2 = layers.Dense(hidden_dims[0], activation='relu')(d1)
        output_data = Dense(training_shape, activation='sigmoid')(d2)

        ae = keras.Model(input_data, output_data)

        return ae

    #### Start Here ####    

    def build_vae(self, training_shape):
        # latent_layer = layers.Lambda(self.sampling)[latent_mean, latent_log_cov]

        # vae.add_loss(self.add_vae_loss(latent_mean, latent_log_cov, input_data, output_data))

        return

    def sampling(self, params):
        return

    def add_vae_loss(self, mean, log_cov, input_data, output_data):
        return
