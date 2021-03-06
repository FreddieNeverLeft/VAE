from ae import autoencoder_util
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
ae = ae_util.build_ae(training_shape)
print(ae.summary())

## Compile and Fit model
ae.compile(optimizer='adam', loss='binary_crossentropy')
ae.fit(x_train, x_train, epochs=10, batch_size=256)

# vae.compile(optimizer='adam')
##

## Make predictions
pred = ae.predict(x_test)

## Plot predictions
sample_num = random.sample(range(0, len(x_test)), 10)
pred_out = pred[sample_num]
test_out = x_test[sample_num]

fig1, ax1 = plt.subplots(10, 1)
fig2, ax2 = plt.subplots(10, 1)

fig1.tight_layout()
fig2.tight_layout()

for x in range(0, 10):
    test_img = np.reshape(test_out[x], (28, 28))
    pred_img = np.reshape(pred_out[x], (28, 28))
    ax1[x].imshow(test_img)
    ax2[x].imshow(pred_img)

fig1.set_figheight(30)
fig1.set_figwidth(30)
fig2.set_figheight(30)
fig2.set_figwidth(30)

fig1.savefig('ae_test.png', dpi=500)
fig2.savefig('ae_pred.png', dpi=500)
plt.show()
