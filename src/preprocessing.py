import os
import glob
import numpy as np
from PIL import Image

IMG_SIZE = (128, 128)


def preprocess_image(file_path: str) -> np.ndarray:

    img = Image.open(file_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def preprocess_batch(file_paths: list) -> np.ndarray:
    
    return np.array([preprocess_image(fp) for fp in file_paths])


def load_dataset_from_folder(root_dir: str, uploads_dirname: str = "uploads", exclude_dirs=None):

    exclude_dirs = set(exclude_dirs or ())
    class_to_files: dict[str, list[str]] = {}

    def add_class_dir(class_dir: str, class_name: str):
        files = [fp for fp in glob.glob(os.path.join(class_dir, "*")) if os.path.isfile(fp)]
        class_to_files.setdefault(class_name, []).extend(files)

    for entry in sorted(os.listdir(root_dir)):
        entry_path = os.path.join(root_dir, entry)
        if not os.path.isdir(entry_path) or entry in exclude_dirs:
            continue
        if entry == uploads_dirname:
            
            for sub_entry in sorted(os.listdir(entry_path)):
                sub_path = os.path.join(entry_path, sub_entry)
                if os.path.isdir(sub_path):
                    add_class_dir(sub_path, sub_entry)
        else:
            add_class_dir(entry_path, entry)

    class_names = sorted(class_to_files.keys())
    file_paths = []
    labels = []
    for idx, class_name in enumerate(class_names):
        for fp in class_to_files[class_name]:
            file_paths.append(fp)
            labels.append(idx)

    X = preprocess_batch(file_paths)
    y = np.array(labels)
    return X, y, class_names
