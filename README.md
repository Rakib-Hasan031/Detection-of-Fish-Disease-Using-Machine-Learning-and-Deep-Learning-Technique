# Detection-of-Fish-Disease-Using-Machine-Learning-and-Deep-Learning-Technique

# Fish Disease Detection

A deep learning project for detecting fish diseases using computer vision and transfer learning with ResNet50.

## Overview

This project uses deep learning to classify fish diseases into 5 categories:
- Healthy Fish
- Motile Aeromonas Septicemia (MAS)
- Parasitic Diseases
- Epizootic Ulcerative Syndrome (EUS)
- Oxygen and Liver Diseases

## Features

- Transfer learning using ResNet50
- Data augmentation and preprocessing
- Training and evaluation metrics
- Confusion matrix visualization
- Classification reports
- Model checkpointing

## Project Structure

```
fish-disease-detection/
├── data/
│   ├── raw/              # Original images
│   └── processed/        # Processed images
├── models/               # Saved model checkpoints
├── notebooks/           # Jupyter notebooks
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model architecture and training
│   └── utils/           # Helper functions
├── tests/               # Unit tests
├── requirements.txt     # Dependencies
└── README.md           # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fish-disease-detection.git
cd fish-disease-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
```python
from src.data.loader import load_data
from src.data.preprocessor import preprocess_images

# Load and preprocess data
train_data, test_data = load_data()
```

2. Training:
```python
from src.models.trainer import train_model

# Train the model
model, history = train_model(train_data, test_data)
```

3. Evaluation:
```python
from src.models.evaluation import evaluate_model

# Evaluate the model
metrics = evaluate_model(model, test_data)
```
