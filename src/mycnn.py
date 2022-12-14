# -*- coding: utf-8 -*-

#Dependencies


import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Nadam, Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

from keras.layers import Dropout, Flatten, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.models import Model

import random
import tensorflow as tf
import cv2
import os
import time
import my_functions as my_func
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from architectures import architectures
from paths import directory_path_benignos,directory_path_malignos,file_path

"""#Loading the dataset

"""

directory_files = os.listdir(directory_path_malignos)
array_of_images_malignos = [plt.imread( os.path.join(directory_path_malignos,file),format='.png' ) for file in directory_files]

directory_files = os.listdir(directory_path_benignos)
array_of_images_benignos = [plt.imread( os.path.join(directory_path_benignos,file),format='.png' ) for file in directory_files]

print(f'Number of cropped maligno images: {len(array_of_images_malignos)}')
print(f'Number of cropped benigno images: {len(array_of_images_benignos)}')
x =np.array([*array_of_images_malignos,*array_of_images_benignos])
y =np.array([*[1 for _ in array_of_images_malignos],*[0 for _ in array_of_images_benignos]])

"""###Simulations"""

[qtd_simulations,tolerance,show_summary,use_augmentation] = [1,15,False,True]

results = {key:{'accuracy':[],'recall':[],'precision':[],'time_used_to_train':[],'epochs':[],'all_epochs':[],'graph_loss':[],'graph_val_loss':[],'graph_accuracy':[],'graph_val_accuracy':[],'predicted_values':[],'actual_values':[]} for key in architectures.keys()}

skf = StratifiedKFold(n_splits=5,shuffle=True)

for key in architectures:
  
  for train_index, test_index in skf.split(x, y):

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    if use_augmentation:
        
      new_x_train, new_y_train = my_func.DataAumentation(x_train,y_train)
      x_train = np.array([*new_x_train,*x_train])
      y_train = np.array([*new_y_train,*y_train])

      temp = list(zip(x_train, y_train))  # Shuffling the x_train and y_train together.
      random.shuffle(temp)
      res1, res2 = zip(*temp)
      x_train, y_train = np.array(list(res1)), np.array(list(res2))
  
    print(f'\n\nPerforming the {key}:\n\n')

    for index in range(qtd_simulations):

      print(f'loop: {index}:')
      model = keras.Sequential()

      for layer_number in architectures[key]['layers']:

        if architectures[key]['layers'][layer_number]['type'] == 'conv2':
          model.add(keras.layers.Conv2D( **architectures[key]['layers'][layer_number]['parameters'] ))
          continue

        if architectures[key]['layers'][layer_number]['type'] == 'maxPooling':
          model.add(keras.layers.MaxPooling2D(**architectures[key]['layers'][layer_number]['parameters']))
          continue
        
        if architectures[key]['layers'][layer_number]['type'] == 'AveragePooling':
          model.add(keras.layers.AveragePooling2D(**architectures[key]['layers'][layer_number]['parameters']))
          continue

        if architectures[key]['layers'][layer_number]['type'] == 'BatchNormalization':
          model.add(keras.layers.BatchNormalization())
          continue

        if architectures[key]['layers'][layer_number]['type'] == 'flatten':
          model.add(keras.layers.Flatten())
          continue

        if architectures[key]['layers'][layer_number]['type'] == 'dense':
          model.add(Dense(**architectures[key]['layers'][layer_number]['parameters']))
          continue
        
        print('Something went wrong')
      
      #model.summary()
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01,patience=tolerance,restore_best_weights=False) 
      start = time.time()
      model.compile(**architectures[key]['compile'])
      history = model.fit(x_train,y_train,**architectures[key]['fit'],verbose=1,callbacks=[callback])
      end = time.time()
      
      total_epochs_used = len(history.history['loss'])

      results[key]['graph_loss'].append(history.history['loss'])
      results[key]['graph_val_loss'].append(history.history['val_loss'])
      results[key]['graph_accuracy'].append(history.history['accuracy'])
      results[key]['graph_val_accuracy'].append(history.history['val_accuracy'])
      time_used = end-start
      
      prediction_result = model.predict(x_test)
      prediction_result_tranformed = [0 if number[0]>number[1] else 1 for number in prediction_result]

      accuracy = accuracy_score(y_test,prediction_result_tranformed)
      recall = recall_score(y_test,prediction_result_tranformed)
      precision = precision_score(y_test,prediction_result_tranformed)
      results[key]['all_epochs'].append(total_epochs_used)
      results[key]['epochs'].append(total_epochs_used-tolerance)
      results[key]['accuracy'].append(accuracy)
      results[key]['recall'].append(recall)
      results[key]['precision'].append(precision)
      results[key]['actual_values'].append(y_test.tolist())
      results[key]['predicted_values'].append(prediction_result_tranformed)
      results[key]['time_used_to_train'].append(time_used)
      
      print(results[key])

  with open(file_path+'\\results\\'+str(key)+'.txt', 'w') as convert_file:
     convert_file.write(json.dumps(results[key]))
    

results.keys()

agrouped_result = dict()

for simulation in results.keys():

  agrouped_result[simulation] = dict()

  for metric in ['accuracy','recall','precision','time_used_to_train']:
    
    agrouped_result[simulation][metric] = dict()
    agrouped_result[simulation][metric]['mean'] = np.mean(results[simulation][metric])
    agrouped_result[simulation][metric]['std'] = np.std(results[simulation][metric])

  print(pd.DataFrame(agrouped_result[simulation]))

with open(file_path+'\\results\\groupedResults.txt', 'w') as convert_file:
     convert_file.write(json.dumps(agrouped_result))
     
