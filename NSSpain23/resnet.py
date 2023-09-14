import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add 
from tensorflow.keras.models import Model
from tensorflow.keras import activations

batch_size = 64
# Load and normalize data set
(train_im, train_lab), (test_im, test_lab) = tf.keras.datasets.cifar10.load_data()
train_im, test_im = train_im/255.0 , test_im/255.0
train_lab_categorical = tf.keras.utils.to_categorical(
    train_lab, num_classes=10, dtype='uint8')
train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator()
train_set_data = train_DataGen.flow(train_im, train_lab_categorical, batch_size=batch_size, shuffle=True)

def res_identity(x, filters): 
  x_skip = x
  f1, f2 = filters

  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False)(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False)(x)
  x = BatchNormalization()(x)

  x = Add()([x, x_skip])
  x = Activation(activations.relu)(x)

  return x

def res_conv(x, s, filters):
  x_skip = x
  f1, f2 = filters

  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', use_bias=False)(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=False)(x)
  x = BatchNormalization()(x)

  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', use_bias=False)(x_skip)
  x_skip = BatchNormalization()(x_skip)

  x = Add()([x, x_skip])
  x = Activation(activations.relu)(x)

  return x

input = Input(shape=(train_im.shape[1], train_im.shape[2], train_im.shape[3]), batch_size=batch_size)
x = Conv2D(2, kernel_size=(7, 7), strides=(2, 2), use_bias=False)(input)
x = BatchNormalization()(x)
x = Activation(activations.relu)(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = res_conv(x, s=1, filters=(64, 256))
x = res_identity(x, filters=(64, 256))
x = res_identity(x, filters=(64, 256))
x = res_conv(x, s=2, filters=(128, 512))
x = res_identity(x, filters=(128, 512))
x = res_identity(x, filters=(128, 512))
x = res_identity(x, filters=(128, 512))
x = res_conv(x, s=2, filters=(256, 1024))
x = res_identity(x, filters=(256, 1024))
x = res_identity(x, filters=(256, 1024))
x = res_identity(x, filters=(256, 1024))
x = res_identity(x, filters=(256, 1024))
x = res_identity(x, filters=(256, 1024))
x = res_conv(x, s=2, filters=(512, 2048))
x = res_identity(x, filters=(512, 2048))
x = res_identity(x, filters=(512, 2048))

x = AveragePooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=input, outputs=x, name='Resnet50')
opt = tf.keras.optimizers.legacy.SGD()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE), optimizer=opt, metrics=['acc'])
model.fit(train_set_data, epochs=5, steps_per_epoch=train_im.shape[0]/batch_size)
