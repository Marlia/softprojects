#%%Импорт библиотек
import numpy as np
import tensorflow as tf
import keras


#%%Загрузка изображений
imgs = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=60, ).flow_from_directory("cell_images", target_size = (64,64),shuffle=True)


#%%Архитектура нейронной сети


input_layer = keras.layers.Input(shape=(64,64,3))

l = keras.layers.Conv2D(32, (3,3),activation="relu", padding="same")(input_layer)
l = keras.layers.MaxPool2D((2,2))(l)
l = keras.layers.BatchNormalization()(l)
l = keras.layers.Dropout(rate = 0.2)(l)

l = keras.layers.Conv2D(32, (3,3),activation="relu", padding="same")(l)
l = keras.layers.MaxPooling2D((2,2))(l)
l = keras.layers.BatchNormalization()(l)
l = keras.layers.Dropout(rate = 0.2)(l)
l = keras.layers.Flatten()(l)

l = keras.layers.Dense(512, activation="relu")(l)
l = keras.layers.BatchNormalization()(l)
l = keras.layers.Dropout(rate = 0.2)(l)
l = keras.layers.Dense(256, activation="relu")(l)
l = keras.layers.BatchNormalization()(l)
l = keras.layers.Dropout(rate = 0.2)(l)

output_layer = keras.layers.Dense(2,activation="sigmoid")(l)
model = keras.Model(input_layer, output_layer)
model.summary()
#%%
optimazer = keras.optimizers.Adam(0.001)
loss = keras.losses.BinaryCrossentropy()
model.compile(optimazer, loss, metrics=["accuracy"])
#%%
model.fit(imgs, epochs=20, batch_size=10 ,shuffle=True)
model.save("NN_for_binaryclassification_of_blood_cell.h5")
#%%

model = keras.models.load_model("NN_for_binaryclassification_of_blood_cell.h5")


#%%
test_imgs = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=60, ).flow_from_directory("cell_images_test", target_size = (64,64),shuffle=True)

model.evaluate(test_imgs)
 