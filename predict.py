from keras.models import load_model
from keras.applications import vgg16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

vehicle = {0: "bike",1: "boat",2: "bus",3: "car",4: "cycle",5: "helicopter",6: "plane",7: "scooty",8: "train",9: "truck"}

model = load_model("Vehicles.h5")

style = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)

img = style.flow_from_directory("prediction", (224,224))

prediction = model.predict(img)

print(prediction)
print(vehicle[np.argmax(prediction)])
