# LUKE Fine-tuning for Relation Extraction

This repository contains code for fine-tuning the LUKE model for relation extraction on a TACRED dataset. The LUKE model is fine-tuned using PyTorch and PyTorch Lightning.

## Introduction

The project aims to extract relationships between entities mentioned in text data. It involves preprocessing the TACRED dataset, defining PyTorch datasets and dataloaders, training the LUKE model, evaluating its performance, and conducting inference on new data.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/abdullahakram07/Supervised_Relation_extraction.git
   cd luke-relation-extraction

## Requirements
To ensure the notebook functions properly, please install the necessary dependencies. The notebook contains the required packages, each specified where necessary.
  ```bash
- Python 3.7 or higher
- PyTorch
- Transformers
- PyTorch Lightning
- Pandas
- tqdm
- scikit-learn
- seaborn
- matplotlib
  ```
## Overview

### 1. Data Preprocessing and Exploration

1. Download the dataset
2. Explore dataset structure
3. Preprocess data for model input

### 2. Model Definition

1. Define LUKE model architecture
2. Define dataset and dataloaders
3. Implement training and validation procedures

### 3. Training and Validation

1. Train the model using PyTorch Lightning
2. Monitor training progress with metrics
3. Validate the model on a separate validation set

### 4. Visualization of Training and Validation Metrics

- Epochs
- Training Accuracy
- Training Loss
- Validation Accuracy
- Validation Loss

### 5. Evaluation
- Precision
- Recall
- F1-Score
- Support

### 6. Inference

- Conduct inference on new sentences
- Predict relations between entities

**Note:** To accomplish all these steps pleae follow instructions as mentioned in the notebook.


<h2>Visualization of Training and Validation Metrics</h2>

<div style="display: flex; flex-wrap: wrap;">
    <div style="width: 50%; text-align: center; padding-right: 20px;">
        <p><strong>1. W&B Epochs</strong></p>
        <img src="https://github.com/abdullahakram07/Supervised_Relation_extraction/blob/27f0bb8c42e5a86da6322b0384b2d46faa4f6664/visualization/W%26B_Epochs.png" width="400">
    </div>
    <div style="width: 50%; text-align: center; padding-left: 20px;">
        <p><strong>2. W&B Training Accuracy</strong></p>
        <img src="https://github.com/abdullahakram07/Supervised_Relation_extraction/blob/27f0bb8c42e5a86da6322b0384b2d46faa4f6664/visualization/W%26B_Training_Accuracy.png" width="400">
    </div>
    <div style="width: 50%; text-align: center; padding-right: 20px; margin-top: 20px;">
        <p><strong>3. W&B Training Loss</strong></p>
        <img src="https://github.com/abdullahakram07/Supervised_Relation_extraction/blob/27f0bb8c42e5a86da6322b0384b2d46faa4f6664/visualization/W%26B_Training_Loss.png" width="400">
    </div>
    <div style="width: 50%; text-align: center; padding-left: 20px; margin-top: 20px;">
        <p><strong>4. W&B Validation Accuracy</strong></p>
        <img src="https://github.com/abdullahakram07/Supervised_Relation_extraction/blob/27f0bb8c42e5a86da6322b0384b2d46faa4f6664/visualization/W%26B_Validation_Accuracy.png" width="400">
    </div>
    <div style="width: 50%; text-align: center; padding-right: 20px; margin-top: 20px;">
        <p><strong>5. W&B Validation Loss</strong></p>
        <img src="https://github.com/abdullahakram07/Supervised_Relation_extraction/blob/27f0bb8c42e5a86da6322b0384b2d46faa4f6664/visualization/W%26B_Validation_loss.png" width="400">
    </div>
</div>

## Evaluation Results
Evaluation results including the classification report and the individual metrics.

### Classification Report:

|   Relation  | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| founded_by    | 0.97      | 0.98   | 0.97     | 732     |
| acquired_by   | 0.93      | 0.99   | 0.96     | 499     |
| invested_in   | 0.95      | 0.92   | 0.93     | 504     |
| CEO_of        | 0.97      | 0.95   | 0.96     | 307     |
| subsidiary_of | 0.96      | 0.62   | 0.76     | 168     |
| partners_with | 0.92      | 0.97   | 0.94     | 126     |
| owned_by      | 0.65      | 0.96   | 0.78     | 71      |

- **Accuracy:** 0.94

### Metrics:

- **Precision:** 0.9435805769947277
- **Recall:** 0.9389281262982966
- **F1 Score:** 0.9376028223897241



