import json
import os
import tensorflow as tf
from src.preprocessing import preprocess_image

CLASS_NAMES_PATH = "models/class_names.json"


def load_class_names(path: str = CLASS_NAMES_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
        
    return [
        "Tomato_Bacterial_Spot",
        "Tomato_Early_Blight",
        "Tomato_Healthy",
        "Tomato_Late_Blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_Leaf_Spot",
        "Tomato_Spider_Mites_Two_Spotted_Spider_Mite",
        "Tomato_Target_Spot",
        "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato_Tomato_mosaic_virus",
    ]


def save_class_names(class_names, path: str = CLASS_NAMES_PATH):
    with open(path, "w") as f:
        json.dump(class_names, f)


def load_model(path: str = "models/model_v1.h5"):
    return tf.keras.models.load_model(path)


def predict_single(model, file_path: str, class_names_path: str = CLASS_NAMES_PATH):
    class_names = load_class_names(class_names_path)
    x = preprocess_image(file_path)[None, ...]
    probs = model.predict(x, verbose=0)[0]
    idx = int(probs.argmax())
    return class_names[idx], float(probs[idx])
