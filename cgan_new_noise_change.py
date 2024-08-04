import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import os
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from keras.models import Model
from keras.layers import Input, Reshape, multiply, Embedding, merge, Concatenate
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from keras import initializers
import keras
np.random.seed(10)
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import time
import keras.backend as K
from scipy.special import kl_div
from scipy.spatial import distance

def data_load(all_classes, frac=1):
    
    S = pd.read_csv('Data/S.csv', header=None)
    V = pd.read_csv('Data/V.csv', header=None)
    F = pd.read_csv('Data/F.csv', header=None)
    
    if all_classes == True:
        N = pd.read_csv('Data/N.csv', header=None)
        Main_X = pd.concat((N,S,V,F))
        label_dict = {0: 'N', 1: 'S', 2: 'V', 3: 'F'}
        test_data = pd.read_csv('Data/test4.csv', header=None)
    else:
        Main_X = pd.concat((S,V,F))
        label_dict = {1: 'S', 2: 'V', 3: 'F'}
        test_data = pd.read_csv('Data/test3.csv', header=None)
    
    Main_X = Main_X.sample(frac=frac).reset_index(drop=True)
    Main_X = Main_X.values
    return Main_X, label_dict, test_data.values

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def generator(G_in, input_classes=5, noise_dim=100, data_dim=186, optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy'):

    label = Input(shape=(1,))
    x = Embedding(input_classes, 50)(label)
    x = Dense(int(G_in.shape[1]))(x)
    x = Reshape((noise_dim,1))(x)
    x = Concatenate()([G_in, x])
    
    x = UpSampling1D()(x)
    x = Conv1D(filters=32*16, kernel_size=10, strides=1, padding='valid', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling1D()(x)
    x = Conv1D(filters=32*8, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling1D()(x)
    x = Conv1D(filters=32*4, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling1D()(x)
    x = Conv1D(filters=32*2, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling1D()(x)
    x = Conv1D(filters=32, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling1D()(x)
    x = Conv1D(filters=1, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(data_dim)(x)
    out = Activation('sigmoid')(x)

    model = Model(inputs=[G_in, label], outputs=out)
#     model.compile(loss=loss, optimizer=optimizer)
    
    return model, out


def discriminator(D_in, out_layer='sigmoid', data_dim=186, out_dim=1, input_classes=5, optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy'):

    label = Input(shape=(1,))
    x = Embedding(input_classes, 50)(label)
    x = Dense(int(D_in.shape[1]))(x)
    x = Reshape((data_dim,1))(x)
    x = Concatenate()([D_in, x])
    
    x = Conv1D(filters=32, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv1D(filters=32*2, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv1D(filters=32*4, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv1D(filters=32*8, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv1D(filters=32*16, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Flatten()(x)
    x = Dense(out_dim)(x)
    if out_layer == 'linear':
        out = Activation('linear')(x)
    elif out_layer == 'sigmoid':
        out = Activation('sigmoid')(x)
    
    model = Model(inputs=[D_in, label], outputs=out)
    model.compile(loss=loss, optimizer=optimizer)
    return model, out


def set_trainability(model, trainable=False, optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy'):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
    model.compile(loss=loss, optimizer=optimizer)
    return model
    
def create_gan(GAN_in, G, D, input_classes=5, optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy'):
    D = set_trainability(D, False, optimizer=optimizer)
    gen_noise, gen_label = G.input
    gen_output = G.output
    gen_output = Reshape((int(gen_output.shape[1]),1))(gen_output)
    GAN_out = D([gen_output, gen_label])
    GAN = Model([gen_noise, gen_label], GAN_out)
    GAN.compile(loss=loss, optimizer=optimizer)
    return GAN, GAN_out

def reshape(X):
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1)
        return X
    else:
        if X.shape[-1] == 1:
            return X
        else:
            X = X.reshape(X.shape[0], X.shape[1], 1)
            return X

def save_model(model, data_dir, type='G', epoch=1):
    json_name = data_dir+str(epoch)+'_'+type+'.json'
    h5name = data_dir+str(epoch)+'_'+type+'.h5'
    # serialize model to JSON
    model_json = model.to_json()
    with open(json_name, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5name)
    # print("Saved model to disk")
    del model

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=4):
    # generate points in the latent space
    # X_fake = np.random.uniform(0, 1.0, size=[n_samples, latent_dim])
    X_fake = np.random.normal(0,1.0,(n_samples,latent_dim))
    # generate labels
    labels_fake = np.random.randint(0, n_classes, n_samples)
    return [reshape(X_fake), reshape(labels_fake)]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, n_classes=4):
    # generate points in latent space
    X_fake, labels_fake = generate_latent_points(latent_dim, n_samples, n_classes)
    # predict outputs
    X_fake = generator.predict([X_fake, labels_fake])
    # create class labels
    y_fake = np.ones((n_samples, 1))*0.1
    return [reshape(X_fake), reshape(labels_fake)], y_fake

def generator1(Main_X, n_samples=128):
#     Main_X = Main_X.values
    num_rows = Main_X.shape[0]
    # print (num_rows)
    counter = 0
    while True:
        new_counter = counter
        counter = counter + n_samples
        if counter >= num_rows:
            yield [reshape(Main_X[new_counter:num_rows-new_counter,:-1]), reshape(Main_X[new_counter:num_rows-new_counter,-1])], np.ones((num_rows-new_counter, 1))*0.9
        else:
            yield [reshape(Main_X[new_counter:counter,:-1]), reshape(Main_X[new_counter:counter,-1])], np.ones((n_samples, 1))*0.9

def generator2(X_fake, labels_fake, y_fake):
    while True:
        yield [X_fake, labels_fake], y_fake
def generator3(noise_dim, batch_size):
    while True:
        [z_input, labels_input] = generate_latent_points(noise_dim, batch_size)
        # create inverted labels for the fake samples
        y_gan = np.ones((batch_size, 1))*0.9
        yield [z_input, labels_input], y_gan


def load_model(data_dir, epoch=1, type='G', loss='binary_crossentropy', optim=Adam(0.0002, 0.5)):
    json_name = data_dir+str(epoch)+'_'+type+'.json'
    h5name = data_dir+str(epoch)+'_'+type+'.h5'
    from keras.models import model_from_json
    # load json and create model
    json_file = open(json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5name)
    # print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss=loss, optimizer=optim)
    return loaded_model



# def generate_real_samples(Main_X, n_samples):
#     # split into images and labels
#     X = Main_X.values[:,:-1]
#     labels = Main_X.values[:,-1]
#     # choose random instances
#     ix = np.random.randint(0, X.shape[0], n_samples)
#     # select images and labels
#     X, labels = X[ix], labels[ix]
#     # generate class labels
#     y_real = np.ones((n_samples, 1))*0.9
#     return [reshape(X), reshape(labels)], reshape(y_real)