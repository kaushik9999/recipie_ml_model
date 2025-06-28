# Recipe Generation ML Model

This project is an end-to-end pipeline for training, evaluating, and deploying a recipe generation model using the [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) architecture. The pipeline includes data preprocessing, model training, evaluation, and recipe generation, with support for experiment tracking and reproducibility using DVC.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Recipe Generation](#recipe-generation)
- [Configuration](#configuration)
- [Reproducibility with DVC](#reproducibility-with-dvc)
- [References](#references)

---

## Project Structure

```
recipie_ml_model/
│
├── configs/                # Configuration files
├── data/
│   └── processed/          # Processed data (train/test splits)
├── models/
│   └── flan_t5_base/       # Trained model and tokenizer artifacts
├── notebooks/              # Jupyter notebooks (demo, experiments)
├── scripts/                # Data processing, training, evaluation, inference scripts
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Project parameters (data, model, training, evaluation)
├── requirements.txt        # Python dependencies
└── .gitignore
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd recipie_ml_model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Install DVC for data and experiment tracking:**
   ```bash
   pip install dvc
   ```

---

## Data Pipeline

- **Raw Data:** Place your raw recipe data as `data/raw/recipes_data.csv`.
- **Preprocessing:** Run the preprocessing script to clean and split the data:
  ```bash
  python scripts/preprocess.py
  ```
  - Handles missing values, parses ingredients/directions, and creates train/test splits.
  - Outputs: `data/processed/train.csv`, `data/processed/test.csv`

---

## Model Training

- **Train the model:**
  ```bash
  python scripts/training_optimized.py
  ```
  - Uses the [Flan-T5-base](https://huggingface.co/google/flan-t5-base) model.
  - Trains on the processed data, supports chunked training for large datasets.
  - Saves checkpoints and final model to `models/flan_t5_base/`.

- **Configurable parameters:** See `params.yaml` for batch size, epochs, model name, etc.

---

## Evaluation

- **Evaluate the trained model:**
  ```bash
  python scripts/evaluate_model.py
  ```
  - Loads the trained model and test data.
  - Generates predictions and computes BLEU and ROUGE scores (see `scripts/metrics.py`).
  - Prints sample predictions and evaluation metrics.

---

## Recipe Generation

- **Generate a recipe using the trained model:**
  ```bash
  python scripts/generate_recipe.py --model_path models/flan_t5_base --recipe "Chicken Curry"
  ```
  - Supports options for output length, temperature, and beam search.
  - Outputs the generated recipe to the console.

---

## Configuration

- **params.yaml:** Central place for all data, model, training, and evaluation parameters.
- **requirements.txt:** All required Python packages.

---

## Reproducibility with DVC

- **DVC Pipeline:** The `dvc.yaml` file defines the data and model pipeline:
  - `preprocess`: Cleans and splits the data.
  - `train`: Trains the model.
  - `evaluate`: Evaluates the model.

- **Run the full pipeline:**
  ```bash
  dvc repro
  ```

---

## References

- [Flan-T5 Model (HuggingFace)](https://huggingface.co/docs/transformers/model_doc/flan-t5)
- [DVC Documentation](https://dvc.org/doc)
- [Transformers Library](https://github.com/huggingface/transformers)

---

**Note:**  
- Ensure your raw data is in the correct format and path.
- For large files or datasets, consider using [Git LFS](https://git-lfs.github.com/) or DVC remote storage. 