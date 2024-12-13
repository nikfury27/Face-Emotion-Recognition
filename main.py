from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns


train_data_dir = 'data/train/'
validation_data_dir = 'data/test/'

# Data Augmentation with more aggressive augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Increased rotation range
    shear_range=0.4,    # Increased shear range
    zoom_range=0.4,     # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,  # Increased batch size for faster convergence
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,  # Increased batch size
    class_mode='categorical',
    shuffle=False) #Shuffle set to False for confusion matrix

# Class labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img, label = train_generator.__next__()

# Model with adjusted architecture
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))  # Increased dropout

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))  # Increased dropout

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))  # Increased dropout

model.add(Flatten())
model.add(Dense(1024, activation='relu'))  # Increased dense layer size
model.add(Dropout(0.4))  # Increased dropout

model.add(Dense(7, activation='softmax'))

# Using Adam optimizer with a smaller learning rate
optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Count number of images in training and test directories
train_path = "data/train/"
test_path = "data/test"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

print(f"Training Images: {num_train_imgs}")
print(f"Testing Images: {num_test_imgs}")

# Adjusting epochs based on the complexity of the model
epochs = 100  # Increased epochs for better training

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=num_train_imgs // 32,  # Adjusted for increased batch size
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_test_imgs // 32)  # Adjusted for increased batch size

# Save the model after training
model.save('model_file_improvedd.h5')

# Plotting Accuracy and Loss
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Reset the generator
validation_generator.reset()

# Predict on the entire validation set
predictions = model.predict(validation_generator, verbose=1)  # Remove 'steps' to predict all samples

# Extract the predicted and true classes
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Check if the lengths are consistent
if len(predicted_classes) != len(true_classes):
    print(f"Length mismatch: Predicted: {len(predicted_classes)}, True: {len(true_classes)}")
    min_length = min(len(predicted_classes), len(true_classes))
    predicted_classes = predicted_classes[:min_length]
    true_classes = true_classes[:min_length]

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification Report
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("Classification Report:\n")
print(report)
