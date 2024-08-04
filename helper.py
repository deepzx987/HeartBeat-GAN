import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, TimeDistributed, Bidirectional, LeakyReLU, BatchNormalization, UpSampling1D
from keras.layers import Dense, Activation, Flatten, Input, GRU, MaxPooling1D, Dropout, SimpleRNN, Reshape
from keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D, concatenate, AveragePooling1D, Conv1D
from keras.models import Model
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Lambda
from keras import backend as K
import tensorflow as tf
from pycm import *
import itertools
from sklearn.metrics import classification_report, confusion_matrix
import os
from keras.layers import RepeatVector, Permute, multiply, GlobalAveragePooling1D, Attention, Embedding
from sklearn.preprocessing import MinMaxScaler
from keras.layers.core import Lambda
from keras.models import model_from_json

def load_model(filename):
    # load json and create model
    json_file = open(str(filename)+'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(str(filename)+"/check.h5")
    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(), 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='Accuracy'), 
                           tf.keras.metrics.Recall(name='Recall'),  
                           tf.keras.metrics.Precision(name='Precision'), 
                           tf.keras.metrics.AUC(num_thresholds=200, summation_method="interpolation", 
                                                name="AUC", dtype=None, curve="ROC", thresholds=None, 
                                                multi_label=True, label_weights=None)])
    return model

def write_history(filename, my_dict):
    with open(str(filename)+'/History.csv', 'w') as f:
        for key in my_dict.keys():
            f.write("%s,%s\n"%(key,my_dict[key]))

def save_the_model(filename, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(str(filename)+"/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(str(filename)+"/check.h5")
    print("Saved model to disk")
    # model.save(str(filename)+"/full_model.h5")

def plot_confusion_matrix(cm, classes, filename, title='Confusion matrix', cmap=plt.cm.Greys):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        new_str = "{:d}".format(cm[i, j]) + '\n' + "{:.3f}".format(cm1[i,j]) 
        plt.text(j, i, new_str ,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(str(filename)+ '/' + title + '.pdf',dpi=250)
    plt.close()


def prediction(X_test, y_test, filename, title):
    model = load_model(filename)
    print ('Model Loaded')
    y_pred=model.predict(X_test,verbose=0)
    
    y_pred_arg = np.argmax(y_pred, axis=1)
    y_test_arg = np.argmax(y_test, axis=1)
    
    # y_pred=model.predict(X_test,verbose=1)
    cm = ConfusionMatrix(actual_vector=y_test_arg, predict_vector=y_pred_arg)
    totalt = cm.__dict__

    TP = totalt['TP']
    FP = totalt['FP']
    TN = totalt['TN']
    FN = totalt['FN']

    PPV = totalt['PPV']
    ACC = totalt['ACC']
    SEN = totalt['TPR']
    SPE = totalt['TNR']
    F1S = totalt['F1']
    AUC = totalt['AUC']

    f = open(str(filename)+'/Results.csv', "w")
    f.write('TP,FP,TN,FN,Precision,Accuracy,Sensitivity,Specificity,F1Score,AUC\n')
    for i in range(3):
        f.write(str(TP[i])+','+str(FP[i])+','+str(TN[i])+','+str(FN[i])+','+str(PPV[i])+','+
                str(ACC[i])+','+str(SEN[i])+','+str(SPE[i])+','+str(F1S[i])+','+str(AUC[i])+'\n')
    f.close()
    
    cnf_matrix = confusion_matrix(y_test_arg, y_pred_arg)
    classes = ['Normal','SVEB','VEB']
    # plt.figure(figsize=(8,5))
    plot_confusion_matrix(cnf_matrix, classes, filename, title)

def parallel_NN_LSTM(WINDOW_SIZE,INPUT_FEAT,OUTPUT_CLASS):
    input1 = Input(shape=(WINDOW_SIZE,INPUT_FEAT), name='input')

    x = Conv1D(filters=48, kernel_size=19, padding='same', strides=4, kernel_initializer='he_normal')(input1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=64, kernel_size=15, padding='same', strides=3, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=80, kernel_size=11, padding='same', strides=2, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=96, kernel_size=9, padding='same', strides=2, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=112, kernel_size=7, padding='same', strides=2, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2)(x)
    x = LSTM(128, return_sequences=False)(x)

    x1 = Conv1D(filters=48, kernel_size=9, padding='same', strides=4, kernel_initializer='he_normal')(input1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=64, kernel_size=7, padding='same', strides=3, kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=80, kernel_size=5, padding='same', strides=2, kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=96, kernel_size=3, padding='same', strides=2, kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=112, kernel_size=3, padding='same', strides=2, kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(pool_size=2, strides=2)(x1)
    x1 = LSTM(128, return_sequences=False)(x1)

    xx = concatenate([x,x1])
    xx = Dense(100)(xx)
    xx = Dense(100)(xx)
    del x,x1
    # concatenate more features here
    outs = Dense(OUTPUT_CLASS, activation='softmax')(xx)
    model = Model(inputs=input1, outputs=outs)
    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='Accuracy'), tf.keras.metrics.Recall(name='Recall'), 
        tf.keras.metrics.Precision(name='Precision'), 
        tf.keras.metrics.AUC(num_thresholds=200, summation_method="interpolation", name="AUC", dtype=None, curve="ROC", 
            thresholds=None, multi_label=True, label_weights=None)])
    return model

def Parallel_GAP(WINDOW_SIZE, INPUT_FEAT, OUTPUT_CLASS):
    # WINDOW_SIZE=5000
    # INPUT_FEAT=12
    # OUTPUT_CLASS=27
    input1 = Input(shape=(WINDOW_SIZE,INPUT_FEAT), name='input')

    x = Conv1D(filters=48, kernel_size=19, padding='same', strides=4, kernel_initializer='he_normal')(input1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=64, kernel_size=15, padding='same', strides=3, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=80, kernel_size=11, padding='same', strides=2, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=96, kernel_size=9, padding='same', strides=2, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters=112, kernel_size=7, padding='same', strides=2, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)

    x1 = Conv1D(filters=48, kernel_size=9, padding='same', strides=4, kernel_initializer='he_normal')(input1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=64, kernel_size=7, padding='same', strides=3, kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=80, kernel_size=5, padding='same', strides=2, kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=96, kernel_size=3, padding='same', strides=2, kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv1D(filters=112, kernel_size=3, padding='same', strides=2, kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = GlobalAveragePooling1D()(x1)

    xx = concatenate([x,x1])
    xx = Dense(100)(xx)
    xx = Dense(100)(xx)
    del x,x1
    # # concatenate more features here
    outs = Dense(OUTPUT_CLASS, activation='softmax')(xx)
    model = Model(inputs=input1, outputs=outs)
    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.BinaryAccuracy(name='Accuracy'), tf.keras.metrics.Recall(name='Recall'), 
        tf.keras.metrics.Precision(name='Precision'), 
        tf.keras.metrics.AUC(num_thresholds=200, summation_method="interpolation", name="AUC", dtype=None, curve="ROC", 
            thresholds=None, multi_label=True, label_weights=None)])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# def CNN_model(WINDOW_SIZE,INPUT_FEAT,OUTPUT_CLASS,LAYER):
#     input1 = Input(shape=(WINDOW_SIZE,INPUT_FEAT), name='input')
#     x = Conv1D(filters=64, kernel_size=16, padding='same', strides=1, kernel_initializer='he_normal',activation='relu')(input1)
#     for layer in range(LAYER-1):
#         x = Conv1D(filters=64, kernel_size=16, padding='same', strides=1, kernel_initializer='he_normal',activation='relu')(x)
#     x = Flatten()(x)
#     x = Dense(3)(x)
#     out = Dense(OUTPUT_CLASS, activation='softmax')(x)
#     model = Model(inputs=input1, outputs=out)
#     # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def ResNet_model(WINDOW_SIZE,INPUT_FEAT,OUTPUT_CLASS,LAYER):
#     # Add CNN layers left branch (higher frequencies)
#     # Parameters from paper
#     k = 1    # increment every 4th residual block
#     p = True # pool toggle every other residual block (end with 2^8)
#     convfilt = 64
#     convstr = 1
#     ksize = 16
#     poolsize = 2
#     poolstr  = 2
#     drop = 0.5
    
#     # Modelling with Functional API
#     #input1 = Input(shape=(None,1), name='input')
#     input1 = Input(shape=(WINDOW_SIZE,INPUT_FEAT), name='input')
    
#     ## First convolutional block (conv,BN, relu)
#     x = Conv1D(filters=convfilt,
#                kernel_size=ksize,
#                padding='same',
#                strides=convstr,
#                kernel_initializer='he_normal')(input1)                
#     x = BatchNormalization()(x)        
#     x = Activation('relu')(x)  
    
#     ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
#     # Left branch (convolutions)
#     x1 =  Conv1D(filters=convfilt,
#                kernel_size=ksize,
#                padding='same',
#                strides=convstr,
#                kernel_initializer='he_normal')(x)      
#     x1 = BatchNormalization()(x1)    
#     x1 = Activation('relu')(x1)
#     x1 = Dropout(drop)(x1)
#     x1 =  Conv1D(filters=convfilt,
#                kernel_size=ksize,
#                padding='same',
#                strides=convstr,
#                kernel_initializer='he_normal')(x1)
#     x1 = MaxPooling1D(pool_size=poolsize,
#                       strides=poolstr)(x1)
#     # Right branch, shortcut branch pooling
#     x2 = MaxPooling1D(pool_size=poolsize,
#                       strides=poolstr)(x)
#     # Merge both branches
#     x = keras.layers.add([x1, x2])
#     del x1,x2
    
#     ## Main loop
#     p = not p 
#     for l in range(LAYER):
        
#         if (l%4 == 0) and (l>0): # increment k on every fourth residual block
#             k += 1
#              # increase depth by 1x1 Convolution case dimension shall change
#             xshort = Conv1D(filters=convfilt*k,kernel_size=1)(x)
#         else:
#             xshort = x        
#         # Left branch (convolutions)
#         # notice the ordering of the operations has changed        
#         x1 = BatchNormalization()(x)
#         x1 = Activation('relu')(x1)
#         x1 = Dropout(drop)(x1)
#         x1 =  Conv1D(filters=convfilt*k,
#                kernel_size=ksize,
#                padding='same',
#                strides=convstr,
#                kernel_initializer='he_normal')(x1)        
#         x1 = BatchNormalization()(x1)
#         x1 = Activation('relu')(x1)
#         x1 = Dropout(drop)(x1)
#         x1 =  Conv1D(filters=convfilt*k,
#                kernel_size=ksize,
#                padding='same',
#                strides=convstr,
#                kernel_initializer='he_normal')(x1)        
#         if p:
#             x1 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(x1)                

#         # Right branch: shortcut connection
#         if p:
#             x2 = MaxPooling1D(pool_size=poolsize,strides=poolstr)(xshort)
#         else:
#             x2 = xshort  # pool or identity            
#         # Merging branches
#         x = keras.layers.add([x1, x2])
#         # change parameters
#         p = not p # toggle pooling

    
#     # Final bit    
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x) 
#     x = Flatten()(x)
#     #x = Dense(1000)(x)
#     x = Dense(3)(x)
#     out = Dense(OUTPUT_CLASS, activation='softmax')(x)
#     model = Model(inputs=input1, outputs=out)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
