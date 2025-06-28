import os
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_colab_paths(base_path=""):
    """
    Adjust file paths for Google Colab if running from Google Drive.
    
    Args:
        base_path (str): Base path to ML_PIPELINE folder in Drive.
    
    Returns:
        dict: Updated file paths.
    """
    paths = {
        "train_file": os.path.join(base_path, "data/processed/train.csv"),
        "test_file": os.path.join(base_path, "data/processed/test.csv"),
        "model_dir": os.path.join(base_path, "models/flan_t5_small")
    }
    logger.info(f"Updated paths for Colab: {paths}")
    return paths

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded CSV from {file_path}, shape: {df.shape}")
    return df

def save_predictions(predictions, references, output_path):
    """
    Save predictions and references to a CSV for analysis.
    
    Args:
        predictions (list): List of generated texts.
        references (list): List of ground truth texts.
        output_path (str): Path to save the CSV.
    """
    df = pd.DataFrame({"predictions": predictions, "references": references})
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    # Example usage
    paths = setup_colab_paths("/content/drive/MyDrive/ML_PIPELINE")
    print(paths)