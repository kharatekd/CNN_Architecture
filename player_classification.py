from google.colab import drive
drive.mount('/content/drive')
//Import Libarires
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import ResNet50  # Import ResNet50 model

test_directory = '/content/drive/MyDrive/playerdata/dataset/test'
train_directory = '/content/drive/MyDrive/playerdata/dataset/train'

resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

resnet_model.trainable = True   # True
set_trainable = False
for layer in resnet_model.layers:
  if layer.name == 'block5_conv1':
    set_trainable = True
  if set_trainable:
    layer.trainable = True
  else:
    layer.trainable = False

# for layer in conv_base.layers:
#   print(layer.name,layer.trainable)

resnet_model.summary()

model = Sequential()

model.add(resnet_model)
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

train_ds = keras.utils.image_dataset_from_directory(
    directory=train_directory,
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224,224)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory=test_directory,
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(224,224)
)

# Normalize
def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

model.compile(
    optimizer=keras.optimizers.RMSprop(lr=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
  )
history = model.fit(train_ds,epochs=10,validation_data=validation_ds)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

import cv2
test_img = cv2.imread(r'/content/drive/MyDrive/playerdata/dataset/test/dhoni/d18.jpg')

from matplotlib import pyplot as plt
plt.imshow(test_img)

test_img.shape

test_img = cv2.resize(test_img,(224,224))

test_input = test_img.reshape((1,224,224,3))

result = model.predict(test_input)
result

if result[0][0]>=0.5:
    prediction="dhoni"
else:
    prediction="kohli"
print(prediction)


