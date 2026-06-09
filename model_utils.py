import os
import glob
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def dataset_dir():
    for candidate in ['Dataset', 'dataset']:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("Dataset directory not found. Create 'Dataset/' or 'dataset/' with class subfolders.")


def load_image_paths_and_labels():
    root = Path(dataset_dir())
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    image_paths = []
    labels = []
    for cls in classes:
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for path in sorted(root.joinpath(cls).glob(ext)):
                image_paths.append(str(path))
                labels.append(cls)
    return image_paths, labels


def preprocess_image(path):
    img = load_img(path, target_size=(224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


def preprocess_image_array(img_array):
    arr = np.array(img_array, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


_FEATURE_EXTRACTOR = None

def load_feature_extractor():
    global _FEATURE_EXTRACTOR
    if _FEATURE_EXTRACTOR is None:
        _FEATURE_EXTRACTOR = MobileNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))
    return _FEATURE_EXTRACTOR


def extract_features_from_array(img_array):
    model = load_feature_extractor()
    arr = preprocess_image_array(img_array)
    return model.predict(arr, verbose=0)


def extract_features_from_paths(paths, batch_size=16):
    model = load_feature_extractor()
    features = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        batch = np.vstack([preprocess_image(p) for p in batch_paths])
        features.append(model.predict(batch, verbose=0))
    return np.vstack(features)
