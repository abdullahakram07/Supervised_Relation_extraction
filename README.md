# LUKE Fine-tuning for Relation Extraction

This repository contains code for fine-tuning the LUKE model for relation extraction on a TACRED dataset. The LUKE model is fine-tuned using PyTorch and PyTorch Lightning.

## Notebook Content

- **1. Data Preprocessing and Exploration**
    - Download the dataset
    - Explore dataset structure
    - Preprocess data for model input

- **2. Model Definition**
    - Define LUKE model architecture
    - Define dataset and dataloaders
    - Implement training and validation procedures

- **3. Training and Validation**
    - Train the model using PyTorch Lightning
    - Monitor training progress with metrics
    - Validate the model on a separate validation set

- **4. Inference**
    - Conduct inference on new sentences
    - Predict relations between entities


## Introduction

The project aims to extract relationships between entities mentioned in text data. It involves preprocessing the TACRED dataset, defining PyTorch datasets and dataloaders, training the LUKE model, evaluating its performance, and conducting inference on new data.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/luke-relation-extraction.git
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



