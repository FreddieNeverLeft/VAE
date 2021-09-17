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
        self.latent_dims=8
        return


    def flatten(self, x):
        x = x.astype('float32')/255
        x = x.reshape((len(x), np.prod(x.shape[1:])))
        return x
    
    
    
    def build_vae(self, training_shape):
        hidden_dims = [128,64]
        self.latent_dims = 10
        
        input_data = keras.Input(shape=(training_shape,))
        e1 = layers.Dense(hidden_dims[0],activation='relu')(input_data)
        e2 = layers.Dense(hidden_dims[1],activation='relu')(e1)
        
        latent_mean = layers.Dense(self.latent_dims, activation='relu')(e2)
        latent_log_cov = layers.Dense(self.latent_dims, activation='relu')(e2)
        
        latent = layers.Lambda(self.sampling)([latent_mean, latent_log_cov])
    
        d1 = layers.Dense(hidden_dims[1], activation = 'relu')(latent)
        d2 = layers.Dense(hidden_dims[0], activation = 'relu')(d1)
        output_data = Dense(training_shape, activation = 'sigmoid')(d2)
        
        vae = keras.Model(input_data, output_data)
        loss = self.get_vae_loss(latent_mean, latent_log_cov, input_data, output_data)
        vae.add_loss(loss)
        
        return vae
    
    def sampling(self, params):
        mean, log_cov = params
        sd_normal = K.random_normal(shape=(K.shape(mean)[0], self.latent_dims), mean = 0, stddev=0.1)
        return mean + K.exp(log_cov)*sd_normal
    
    
    def get_vae_loss(self, mean, log_cov, input_data, output_data):
        ae_loss = 28*28*keras.losses.binary_crossentropy(input_data, output_data)
        kl_loss = -0.5*K.sum(1 + log_cov - K.square(mean) - K.exp(log_cov), axis=-1)
        return K.mean(ae_loss + kl_loss)

from keras.datasets import mnist
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

ae_util = autoencoder_util()

## Importing/Transforming mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = ae_util.flatten(x_train)
x_test = ae_util.flatten(x_test)
training_shape = x_train.shape[1]

## Build simple autoencoder model
vae = ae_util.build_vae(training_shape)
print(vae.summary())


## Compile and Fit model
vae.compile(optimizer='adam')
vae.fit(x_train, x_train, epochs=20, batch_size=256)

## Make predictions
pred = vae.predict(x_test)


## Plot predictions
sample_num = random.sample(range(0, len(x_test)), 10)
pred_out = pred[sample_num]
test_out = x_test[sample_num]

fig1, ax1 = plt.subplots(10,1)
fig2, ax2 = plt.subplots(10,1)

fig1.tight_layout()
fig2.tight_layout()


for x in range(0,10):
    test_img = np.reshape(test_out[x], (28,28))
    pred_img = np.reshape(pred_out[x], (28,28))
    ax1[x].imshow(test_img)
    ax2[x].imshow(pred_img)


fig1.set_figheight(30)
fig1.set_figwidth(30)
fig2.set_figheight(30)
fig2.set_figwidth(30)

fig1.savefig('vae_test.png', dpi=500)  
fig2.savefig('vae_pred.png', dpi=500)  
plt.show()

