import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import os
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
from sklearn.preprocessing import MinMaxScaler
import time
import keras.backend as K
from scipy.special import kl_div
from scipy.spatial import distance
from cgan_new_noise_change import *
from evaluation_metrics import *
import sys

all_classes = True
frac = 1.0
loss = 'mse'
# loss = wasserstein_loss
# loss = 'binary_crossentropy'
nb_batches = 400
checkpoint=5


# In[3]:


main_dir = 'Final_Results_Noise_Change_0_1/'
if not os.path.exists(main_dir):
    os.mkdir(main_dir)

# In[4]:


data_dim = 260
noise_dim = 100
optim = Adam(0.0002, 0.5)
batch_size=128
verbose = 0

Main_X, label_dict, test_data = data_load(all_classes,frac=frac)
n_classes=len(label_dict)
half_batch = int(batch_size // 2) 
metric_to_calculate = ['FID', 'MMD', 'DTW', 'ED', 'PC', 'KLD', 'RMSE', 'TWED']
# RE and KLD are same. removed RE


# In[ ]:


G_in = Input(shape=[noise_dim,1])
G, G_out = generator(G_in, input_classes=len(label_dict), noise_dim=100, data_dim=260, optimizer=optim, loss=loss)
D_in = Input(shape=[data_dim,1])
# D, D_out = discriminator(D_in, out_layer='linear', data_dim=260, out_dim=1, input_classes=len(label_dict), optimizer=optim, loss=loss)
D, D_out = discriminator(D_in, out_layer='sigmoid', data_dim=260, out_dim=1, input_classes=len(label_dict), optimizer=optim, loss=loss)
# D, D_out = discriminator(D_in, z1=16, z2=outputs, data_dim=260, out_dim=1, input_classes=len(label_dict), optimizer=optim, loss=loss)
GAN_in = Input([noise_dim,1])
GAN, GAN_out = create_gan(GAN_in, G, D, input_classes=len(label_dict), optimizer=optim, loss=loss)
# print (G.summary(), D.summary(), GAN.summary())


# In[ ]:

if loss == wasserstein_loss:
    data_dir = main_dir+ 'Four_Classes_' + str(all_classes) + '_Loss_WS/'
else:
    data_dir = main_dir+ 'Four_Classes_' + str(all_classes) + '_Loss_' + str(loss) + '/'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
else:
    sys.exit()

f = open(data_dir+'Stats.csv', 'a')
f.write('Epoch, d_loss1, d_loss2, g_loss, ')
for i in range(len(label_dict)):
    for mtc in metric_to_calculate:
        f.write(str(mtc)+str(i)+',')
f.write('Time,\n')
f.close()


# In[ ]:


# train_samples=np.ceil(Main_X.shape[0] // batch_size)
for i in range(nb_batches):

    start = time.time()
    
    D = set_trainability(D, True, optimizer=optim, loss=loss)
    train_generator = generator1(Main_X, n_samples=half_batch)
    # [X_real, labels_real], y_real = generate_real_samples(Main_X, n_samples=half_batch)
    d_loss1 = D.fit(train_generator, steps_per_epoch=1, epochs=1, verbose=verbose)
    d_loss1 = float(d_loss1.history['loss'][0])

    [X_fake, labels_fake], y_fake = generate_fake_samples(G, noise_dim, half_batch, n_classes)
    train_generator = generator2(X_fake, labels_fake, y_fake)
    d_loss2 = D.fit(train_generator, steps_per_epoch=1, epochs=1, verbose=verbose)
    d_loss2 = float(d_loss2.history['loss'][0])

    D = set_trainability(D, False, optimizer=optim, loss=loss)
    train_generator = generator3(noise_dim, batch_size)
    g_loss = GAN.fit(train_generator, steps_per_epoch=1, epochs=1, verbose=verbose)
    g_loss = float(g_loss.history['loss'][0])
    end = time.time()
    print('> %d/%d, d1=%.3f, d2=%.3f g=%.3f t=%.3f' %(i+1, nb_batches, d_loss1, d_loss2, g_loss, end-start))
    
    if i%checkpoint == 0:
        
        f = open(data_dir+'Stats.csv', 'a')
        f.write(str(i+1)+','+str(d_loss1)+','+str(d_loss2)+','+str(g_loss)+',')
        save_model(model=G, data_dir=data_dir, type='G', epoch=i)
        save_model(model=D, data_dir=data_dir, type='D', epoch=i)
        save_model(model=GAN, data_dir=data_dir, type='GAN', epoch=i)
        
        for k,metric in enumerate(label_dict.keys()):    
            temp_x = test_data[200*(k):200*(k+1),:-1]
            [z_input, labels_input] = generate_class_specific_latent_input(200, n_classes=n_classes, noise_dim=noise_dim, category=float(metric))
            z_input = G.predict([z_input, labels_input], verbose=verbose)

            for j in range(2):
                plt.plot(z_input[j])
            plt.savefig(data_dir+str(i)+'_Label_'+str(metric)+'.png')
            plt.close()
            plt.clf()
            
            results = evaluate(temp_x,z_input,metric_to_calculate)
            for r in results:
                f.write(str(r)+',')
            
        f.write(str(end-start)+'\n')
        f.close()


# In[ ]:




