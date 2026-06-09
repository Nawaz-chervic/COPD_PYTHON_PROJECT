# Evaluate saved model files on a common validation split and produce a comparison chart
import os
import glob
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from model_utils import load_image_paths_and_labels, preprocess_image, extract_features_from_array
import tensorflow as tf

MODELS_CONFIG_FILE = 'models_config.json'
MODELS_ACC_FILE = 'models_accuracy.json'
MODEL_ACC_IMG = 'model_accuracies.png'


def load_model_config():
    if not os.path.exists(MODELS_CONFIG_FILE):
        return {}
    try:
        with open(MODELS_CONFIG_FILE, 'r') as f:
            data = json.load(f)
            mappings = {}
            for entry in data.get('models', []):
                file_name = entry.get('file')
                algo_name = entry.get('algorithm')
                if file_name and algo_name:
                    mappings[file_name] = algo_name
            return mappings
    except Exception:
        return {}


def algorithm_name(file_name, config_map):
    return config_map.get(file_name, os.path.splitext(os.path.basename(file_name))[0])


def label_to_int(label_text):
    return 0 if 'copd' in label_text.lower() else 1


paths, labels = load_image_paths_and_labels()
if len(paths) == 0:
    print('No images found in the dataset folders.')
    exit(1)

labels_int = np.array([label_to_int(l) for l in labels])
train_paths, val_paths, _, val_labels = train_test_split(
    paths, labels_int, test_size=0.2, random_state=42, stratify=labels_int
)

print(f'Evaluating on {len(val_paths)} validation images.')
val_raw_images = np.stack([
    img_to_array(load_img(path, target_size=(224, 224))) for path in val_paths
]).astype(np.float32)
val_images = val_raw_images / 255.0
val_features = extract_features_from_array(val_raw_images)

config_map = load_model_config()
model_files = sorted(glob.glob('*.h5')) + sorted(glob.glob('*.joblib')) + sorted(glob.glob('*.pkl'))
if len(model_files) == 0:
    print('No model files found in current directory.')
    exit(0)

results = []
for model_file in model_files:
    try:
        print(f'Loading {model_file}...')
        if model_file.lower().endswith('.h5'):
            model = tf.keras.models.load_model(model_file)
            preds = model.predict(val_images, verbose=0)
            y_pred = (preds.reshape(-1) > 0.5).astype(int)
        else:
            model = joblib.load(model_file)
            y_pred = model.predict(val_features)
        accuracy = float((y_pred == val_labels).mean())
        algo = algorithm_name(model_file, config_map)
        print(f' -> {model_file} ({algo}): {accuracy*100:.2f}%')
        results.append({'model_file': model_file, 'algorithm': algo, 'accuracy': accuracy})
    except Exception as e:
        print(f'Failed to evaluate {model_file}: {e}')

with open(MODELS_ACC_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved accuracies to {MODELS_ACC_FILE}')

if len(results) > 0:
    names = [r['algorithm'] for r in results]
    accs = [r['accuracy'] * 100 for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, accs, color=plt.cm.tab10.colors[:len(names)])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison')
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f'{val:.1f}%', ha='center')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(MODEL_ACC_IMG)
    print(f'Saved comparison chart to {MODEL_ACC_IMG}')
else:
    print('No valid evaluation results to plot.')
