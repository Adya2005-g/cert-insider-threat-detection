import os
import logging
import pandas as pd


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ------------------------------
# Load Data Function
# ------------------------------
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is empty or invalid
    """

    # Check if file exists
    if not os.path.exists(file_path):
        logging.error(f"❌ File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Load CSV
        df = pd.read_csv(file_path)

        # Check if empty
        if df.empty:
            logging.warning("⚠️ Loaded file is empty")
            raise ValueError("Dataset is empty")

        logging.info(f"✅ Data loaded successfully from: {file_path}")
        logging.info(f"📊 Dataset shape: {df.shape}")

        return df

    except pd.errors.EmptyDataError:
        logging.error("❌ CSV file is empty or corrupted")
        raise

    except Exception as e:
        logging.error(f"❌ Error loading data: {str(e)}")
        raise