import os
import shutil
import time


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware


from src.prediction import load_model, predict_single, save_class_names
from src.preprocessing import load_dataset_from_folder
from src.model import build_model, get_callbacks
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


app = FastAPI(title="Plant Disease Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/model_v1.h5"
CLASS_NAMES_PATH = "models/class_names.json"
TRAIN_DIR = "data/train"
UPLOAD_DIR = os.path.join(TRAIN_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
CONFIDENCE_THRESHOLD = 0.6

START_TIME = time.time()
model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@app.get("/health")
def health_check():
    return {
        "status": "up" if model is not None else "no_model_loaded",
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {
            "Error": "Model not loaded yet . Train or retrain first." 
        }

    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    label , confidence = predict_single(model, tmp_path, CLASS_NAMES_PATH)
    os.remove(tmp_path)

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "predicted_label": "Uncertain",
            "confidence": confidence,
            "message": "Low confidence — please upload a clearer photo of a tomato leaf.",
        }

    return {
        "predicted_label": label,
        "confidence": confidence,
    }


@app.post("/upload")
async def upload_bulk(class_name: str = Form(...), files: list[UploadFile] = File(...)):
    class_dir = os.path.join(UPLOAD_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    saved = []

    for f in files:
        dest = os.path.join(class_dir, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)

    return {
        "uploaded": len(saved), "class": class_name, "files": saved
    }



@app.post("/train")
def retrain_model():
    global model
    
    X, Y, class_names = load_dataset_from_folder(TRAIN_DIR)

    if len(class_names) < 2:
        return {
            "Error": "Need at least 2 classes with data to retrain."
        }

    y_cat = to_categorical(Y, num_classes=len(class_names))
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify= y_cat)


    new_model = build_model(num_classes=len(class_names))
    new_model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs=15,
        batch_size=32,
        callbacks= get_callbacks(MODEL_PATH),
        verbose=2
    )

    new_model.save(MODEL_PATH)
    save_class_names(class_names, path=CLASS_NAMES_PATH)
    model = new_model

    return {
        "status": "retrained",
        "samples_used": len(X),
        "classes": class_names,
    }
