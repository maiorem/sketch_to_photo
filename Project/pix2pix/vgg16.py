import tensorflow as tf
from tensorflow.keras.applications import VGG16, EfficientNetB1
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

TRAINING_DIR = "./data/sketch/0"

training_datagen = ImageDataGenerator(rescale=1./255, 
                                horizontal_flip=True,
                                vertical_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                rotation_range=5,
                                zoom_range=1.2,
                                shear_range=0.7,
                                fill_mode='nearest', 
                                validation_split=0.2
                                )
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical',
        subset='training')

valid_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(256, 256),
        batch_size=32,
        subset='validation'
    )

vgg16=VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
vgg16.trainable=False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(4, activation="softmax"))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)
reduce_lr=ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2)
ck=ModelCheckpoint('./ck/vgg16-{epoch:02d}-{val_loss:.4f}.hdf5', save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
model.fit(train_generator,
            steps_per_epoch=len(train_generator),
            validation_data=(valid_generator),
            validation_steps=len(valid_generator),
            epochs=100,
            verbose=1,
            callbacks=[earlystopping, reduce_lr, ck]
            )



model.save('category_4.h5')
