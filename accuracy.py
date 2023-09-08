from cgi import test
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.applications import ResNet101, vgg16
import tensorflow as tf
import pandas as pd 
import numpy as np
from keras.models import load_model

test_gen = ImageDataGenerator(
    preprocessing_function = vgg16.preprocess_input
)

model = load_model("Vehicles.h5")

test_images = test_gen.flow_from_directory(
    directory= "test",
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False)

loss, accuracy = model.evaluate(test_images)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")