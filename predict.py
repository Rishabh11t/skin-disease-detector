import io
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model

def predict_image(model_path, file, target_size=(128, 128), categories=None):
    # Load the model
    model = load_model(model_path)
    
    # Convert FileStorage to BytesIO
    img_bytes = file.read()  # Read the file content
    img = io.BytesIO(img_bytes)  # Convert to BytesIO object
    
    # Load image from the byte stream
    image = load_img(img, target_size=target_size)  # Use the BytesIO stream here
    
    # Preprocess the image
    image_array = img_to_array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    
    # Return the predicted category
    return categories[predicted_class] if categories else predicted_class
