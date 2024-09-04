import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def alexnet(input_shape=(224, 224, 3), num_classes=2):
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True, 
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory('path_to_your_dataset', 
                                                    target_size=(224, 224), 
                                                    batch_size=32, 
                                                    class_mode='categorical', 
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory('path_to_your_dataset', 
                                                         target_size=(224, 224), 
                                                         batch_size=32, 
                                                         class_mode='categorical', 
                                                         subset='validation')

model = alexnet(input_shape=(224, 224, 3), num_classes=train_generator.num_classes)

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(train_generator, 
                    epochs=25, 
                    validation_data=validation_generator, 
                    steps_per_epoch=len(train_generator), 
                    validation_steps=len(validation_generator))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory('path_to_test_dataset',
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')
