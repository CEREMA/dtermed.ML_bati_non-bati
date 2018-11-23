from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 

print('******************************************************')
print('*************** Initialisation du test ***************')
print('******************************************************')


# step 1: load data

img_width = 150
img_height = 150
# saisir le chemin où sont stockées vos images pour l'entrainement et la validation du modèle
train_data_dir = 'C:/_IA/bati_pas_bati/data/carreaux/50mx50m/geoportail/train'
valid_data_dir = 'C:/_IA/bati_pas_bati/data/carreaux/50mx50m/geoportail/valid'

datagen = ImageDataGenerator(rescale = 1./255)

train_generator = datagen.flow_from_directory(directory=train_data_dir,
											   target_size=(img_width,img_height),
											   classes=['bati','non_bati'],
											   class_mode='binary',
											   batch_size=32)

validation_generator = datagen.flow_from_directory(directory=valid_data_dir,
											   target_size=(img_width,img_height),
											   classes=['bati','non_bati'],
											   class_mode='binary',
											   batch_size=16)

# step-2 : build model

model =Sequential()

model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
#model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print('model complied!!')

print('starting training....')
training = model.fit_generator(generator=train_generator, steps_per_epoch=5000 // 16,epochs=1,validation_data=validation_generator,validation_steps=832//16)

print('training finished!!')

print('saving weights to bati_non-bati_geoportail_test_3.h5')
# saisir le chemin où seront stockés les poids et modèle du modèle qui vient d'être entraîné
model.save_weights('C:/_IA/bati_pas_bati/py_geoportail/model/bati_non-bati_geoportail_test_3.h5')
model.save('C:/_IA/bati_pas_bati/py_geoportail/model/bati_non-bati_geoportail_test_3.model')

print('all weights saved successfully !!')

input("Fin des traitements")