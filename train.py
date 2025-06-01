import os
from data_preprocessing import load_data
from model import create_model

def train_model(dataset_path, model_save_path="skin_disease_detector.h5", epochs=10):
    train_generator, validation_generator = load_data(dataset_path)
    model = create_model(num_classes=train_generator.num_classes)
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    model.save(model_save_path)
    print("Model saved to", model_save_path)
    return history

if __name__ == "__main__":
    dataset_path = "Dataset"
    train_model(dataset_path)
