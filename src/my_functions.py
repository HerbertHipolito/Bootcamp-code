import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def display_img(image,title='Image',x_label=None,y_label=None,cmap_type='gray',show_axis=False,colorBar=False,F_size=(8,6)):
  plt.figure(figsize=F_size)
  plt.imshow(image,cmap=cmap_type)
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  if colorBar: plt.colorbar()
  if not show_axis: plt.axis('off')
  plt.show()

def show_graphs(curve1,curve2,title,xlabel,ylabel,x_dot=None,y_dot=None):

  plt.plot(curve1)
  plt.plot(curve2)
  plt.plot(x_dot,y_dot,'xr')
  plt.title(f'Model {title}')
  plt.ylabel(xlabel)
  plt.xlabel(ylabel)
  plt.legend(['train', 'validation'])
  plt.show()

#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
def DataAumentation(x_train,y_train,batch_size=2,loop_number=30):
  
  x_train_tf = tf.expand_dims(x_train,axis=-1)

  datagen = ImageDataGenerator(
          zoom_range = 0.1, # Aleatory zoom
          rotation_range= 45, 
          width_shift_range=0.2,  # horizontal shift
          height_shift_range=0.2,  # vertical shift
          horizontal_flip=True,  
          vertical_flip=True)

  datagen.fit(x_train_tf)

  count = 0

  new_x,new_y = [],[]

  for batch in datagen.flow(x_train_tf,y_train, batch_size=batch_size, save_format='png'):
      count += 1
      for index in range(batch_size):
        new_x.append(batch[0][index][:,:,0])
        new_y.append(batch[1][index])
        plt.show()
      if count >= loop_number:
          break
  
  return  new_x,new_y
