import json
import os
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from model_utils import load_image_paths_and_labels, extract_features_from_paths


MODEL_CONFIG_PATH = 'models_config.json'


def label_to_int(label_text):
    return 0 if 'copd' in label_text.lower() else 1


def save_pipeline(model, filename):
    print(f'Saving {filename}')
    joblib.dump(model, filename)


def update_model_config(new_models):
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
    else:
        config_data = {"models": []}

    model_map = {entry.get('file'): entry for entry in config_data.get('models', []) if entry.get('file')}
    for filename, algorithm in new_models.items():
        model_map[filename] = {"file": filename, "algorithm": algorithm}

    config_data['models'] = list(model_map.values())
    with open(MODEL_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)


def build_and_train_models(train_features, train_labels, class_weights):
    return {
        'svm_model.joblib': GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, class_weight=class_weights, random_state=42)),
            param_grid={
                'svc__C': [1.0, 5.0, 10.0],
                'svc__gamma': ['scale', 'auto'],
            },
            cv=3,
            n_jobs=-1,
            verbose=0,
        ),
        'knn_model.joblib': GridSearchCV(
            Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier())
            ]),
            param_grid={
                'scaler': [StandardScaler(), Normalizer()],
                'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
                'knn__weights': ['uniform', 'distance'],
                'knn__p': [1, 2],
                'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'knn__metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev', 'cosine'],
            },
            cv=3,
            n_jobs=-1,
            verbose=0,
        ),
        'rf_model.joblib': GridSearchCV(
            make_pipeline(StandardScaler(), RandomForestClassifier(class_weight=class_weights, random_state=42, n_jobs=-1)),
            param_grid={
                'randomforestclassifier__n_estimators': [200, 300, 400, 500],
                'randomforestclassifier__max_depth': [None, 10, 20, 30],
                'randomforestclassifier__max_features': ['sqrt', 'log2'],
            },
            cv=3,
            n_jobs=-1,
            verbose=0,
        ),
        'lr_model.joblib': GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000, class_weight=class_weights, solver='saga', random_state=42)),
            param_grid={
                'logisticregression__C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'logisticregression__penalty': ['l2'],
            },
            cv=3,
            n_jobs=-1,
            verbose=0,
        ),
    }


def main():
    paths, labels = load_image_paths_and_labels()
    if len(paths) == 0:
        raise RuntimeError('No images found in the dataset folders.')

    y = np.array([label_to_int(label) for label in labels])
    train_paths, val_paths, train_y, val_y = train_test_split(
        paths, y, test_size=0.2, random_state=42, stratify=y
    )

    print('Extracting features for training data...')
    train_features = extract_features_from_paths(train_paths)
    print('Extracting features for validation data...')
    val_features = extract_features_from_paths(val_paths)

    class_weights = {
        0: float(len(y)) / (2 * np.sum(y == 0)),
        1: float(len(y)) / (2 * np.sum(y == 1)),
    }
    print(f'Class weights: {class_weights}')

    models = build_and_train_models(train_features, train_y, class_weights)
    accuracies = {}

    for filename, pipeline in models.items():
        print(f'Training {filename}...')
        pipeline.fit(train_features, train_y)
        save_pipeline(pipeline, filename)

        y_pred = pipeline.predict(val_features)
        accuracy = accuracy_score(val_y, y_pred)
        accuracies[filename] = accuracy
        print(f'{filename} validation accuracy: {accuracy:.4f}')
        print(classification_report(val_y, y_pred, target_names=['COPD', 'NORMAL']))

    new_config = {
        'svm_model.joblib': 'SVM',
        'knn_model.joblib': 'KNN',
        'rf_model.joblib': 'Random Forest',
        'lr_model.joblib': 'Logistic Regression',
    }
    update_model_config(new_config)
    print('Updated model configuration in', MODEL_CONFIG_PATH)
    print('Training complete. Saved models:')
    for filename, accuracy in accuracies.items():
        print(f'  {filename}: {accuracy:.4f}')


if __name__ == '__main__':
    import os
    main()
