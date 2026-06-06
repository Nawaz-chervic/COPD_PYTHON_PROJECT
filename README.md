# COPD Detection Python Project

## Overview

This project is a COPD detection system built with Python, TensorFlow, and Streamlit. It includes:

- `app.py` — Streamlit web application with login and the COPD detection interface
- `train.py` — model training script using transfer learning on a local dataset
- `predict.py` — single image prediction script for `test_xray.jpg`
- `requirement.txt` — required Python packages

## Prerequisites

- Python 3.8+ installed
- `pip` available
- A working internet connection for installing dependencies

## Install dependencies

1. Open a terminal in the project folder. For example, navigate to your project folder using a relative or general path:

```powershell
cd path\to\COPD_PYTHON_PROJECT
```

If you are already in the project folder, you can simply run:

```powershell
cd .
```

2. Install required packages:

```powershell
pip install -r requirement.txt
```

## Dataset setup

Create the following directory structure before training:

```text
COPD_PYTHON_PROJECT/
  dataset/
    COPD/
      <COPD X-ray images>
    NORMAL/
      <Normal X-ray images>
```

You can obtain image files from the Kaggle dataset:

- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

This project uses the smaller `test` folder structure for quick experiments, but using more training data will generally improve accuracy.

## Train the model

If you want to train a new model, place your dataset in the `dataset` folder with the class subfolders above and run:

```powershell
python train.py
```

This creates or updates `copd_model.h5` in the project root.

## Run the Streamlit app

After training and generating `copd_model.h5`, start the web application:

```powershell
streamlit run app.py
```

Then open the URL shown in the terminal, typically `http://localhost:8501`.

### Default login accounts

- Doctor: `doctor1` / `doctor123`
- Patient: `patient1` / `patient123`
- Patient: `patient2` / `patient456`

## Train the model

If you want to train a new model, place your dataset in a `dataset` folder with class subfolders and run:

```powershell
python train.py
```

This creates or updates `copd_model.h5` in the project root.

## Run a single image prediction

To predict with a single X-ray image, update `test_xray.jpg` or replace the file path in `predict.py`, then run:

```powershell
python predict.py
```

## Notes

- `app.py` uses `copd_model.h5` for predictions. If the model file is missing, the app will still launch but predictions may fail.
- `train.py` requires a `dataset` directory structured for Keras `flow_from_directory`.
- `predict.py` requires `copd_model.h5` and `test_xray.jpg`.

## File summary

- `app.py`: Streamlit app with login and prediction dashboard
- `train.py`: model training workflow with data augmentation and fine-tuning
- `predict.py`: single image prediction script
- `requirement.txt`: package dependencies
