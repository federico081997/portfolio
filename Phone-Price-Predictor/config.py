from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "data" / "cleaned_phone_specs.csv"
MODEL_PATH = ROOT_DIR / "models" / "random_forest.joblib"
EXPECTED_COLUMNS_PATH = ROOT_DIR / "models" / "expected_columns.joblib"

DEFAULT_BRANDS = ["Apple", "Samsung"]

REQUIRED_COLUMNS = [
    "Company Name",
    "Model Name",
    "RAM_GB",
    "Battery_mAh",
    "Screen_Size_inches",
    "Weight_g",
    "Front_Camera_MP",
    "Back_Camera_MP",
]
