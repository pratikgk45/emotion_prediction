import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FacialExpressionModel:
    emotion_dict = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

    def __init__(self, model_weights_file):
        self.loaded_model = Sequential()
        
        self.loaded_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.loaded_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.loaded_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.loaded_model.add(Dropout(0.25))

        self.loaded_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.loaded_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.loaded_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.loaded_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.loaded_model.add(Dropout(0.25))

        self.loaded_model.add(Flatten())
        self.loaded_model.add(Dense(1024, activation='relu'))
        self.loaded_model.add(Dropout(0.5))
        self.loaded_model.add(Dense(7, activation='softmax'))

        # Load the model weights
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        # Ensure the image is the correct shape
        if img.shape != (1, 48, 48, 1):
            raise ValueError("Input image must be of shape (1, 48, 48, 1)")

        # Predict the emotion
        preds = self.loaded_model.predict(img)
        return FacialExpressionModel.emotion_dict[int(np.argmax(preds))]

# Example usage (ensure the model_weights_file path is correct):
# model = FacialExpressionModel('path_to_weights_file.h5')
# img = np.random.rand(1, 48, 48, 1)  # Example input image
# emotion = model.predict_emotion(img)
# print(emotion)
