import os
import re
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint, uniform

DATA_PATH = os.path.join("data", "laptop_data.csv")
MODELS_DIR = "models"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.joblib")


def extract_ram_gb(ram_str):
    if pd.isna(ram_str):
        return 8
    match = re.search(r'(\d+)GB', str(ram_str))
    return int(match.group(1)) if match else 8


def extract_memory_gb(memory_str):
    """Handle multiple memory components like '512GB SSD + 1TB HDD'"""
    if pd.isna(memory_str):
        return 256
    memory_str = str(memory_str).upper()
    total = 0
    for tb in re.findall(r'(\d+(?:\.\d+)?)\s*TB', memory_str):
        total += float(tb) * 1024
    for gb in re.findall(r'(\d+(?:\.\d+)?)\s*GB', memory_str):
        total += float(gb)
    return int(total) if total > 0 else 256


def extract_memory_type(memory_str):
    if pd.isna(memory_str):
        return "SSD"
    memory_str = str(memory_str).upper()
    if "SSD" in memory_str and "HDD" in memory_str:
        return "Hybrid"
    elif "SSD" in memory_str:
        return "SSD"
    elif "HDD" in memory_str:
        return "HDD"
    elif "FLASH" in memory_str:
        return "Flash Storage"
    return "SSD"


def extract_weight_kg(weight_str):
    if pd.isna(weight_str):
        return 2.0
    match = re.search(r'(\d+(?:\.\d+)?)', str(weight_str))
    return float(match.group(1)) if match else 2.0


def extract_cpu_brand(cpu_str):
    if pd.isna(cpu_str):
        return "Intel"
    cpu_str = str(cpu_str).upper()
    if "INTEL" in cpu_str:
        return "Intel"
    elif "AMD" in cpu_str:
        return "AMD"
    elif any(x in cpu_str for x in ["APPLE", "M1", "M2", "M3"]):
        return "Apple"
    return "Intel"


def extract_cpu_speed(cpu_str):
    if pd.isna(cpu_str):
        return 2.5
    match = re.search(r'(\d+(?:\.\d+)?)\s*GHZ', str(cpu_str).upper())
    return float(match.group(1)) if match else 2.5


def extract_gpu_brand(gpu_str):
    if pd.isna(gpu_str):
        return "Intel"
    gpu_str = str(gpu_str).upper()
    if "NVIDIA" in gpu_str or "GEFORCE" in gpu_str:
        return "Nvidia"
    elif "AMD" in gpu_str or "RADEON" in gpu_str:
        return "AMD"
    elif "INTEL" in gpu_str:
        return "Intel"
    return "Intel"


def extract_resolution(screen_str):
    if pd.isna(screen_str):
        return (1920, 1080)
    match = re.search(r'(\d+)x(\d+)', str(screen_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return (1920, 1080)


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Feature engineering
    df['Ram_GB'] = df['Ram'].apply(extract_ram_gb)
    df['Memory_GB'] = df['Memory'].apply(extract_memory_gb)
    df['Memory_Type'] = df['Memory'].apply(extract_memory_type)
    df['Weight_kg'] = df['Weight'].apply(extract_weight_kg)
    df['Cpu_Brand'] = df['Cpu'].apply(extract_cpu_brand)
    df['Cpu_Speed_GHz'] = df['Cpu'].apply(extract_cpu_speed)
    df['Gpu_Brand'] = df['Gpu'].apply(extract_gpu_brand)
    df['Resolution_Width'], df['Resolution_Height'] = zip(*df['ScreenResolution'].apply(extract_resolution))

    features = [
        "Company", "TypeName", "OpSys", "Memory_Type",
        "Cpu_Brand", "Gpu_Brand", "Inches", "Ram_GB",
        "Memory_GB", "Weight_kg", "Cpu_Speed_GHz",
        "Resolution_Width", "Resolution_Height"
    ]
    X = df[features]
    y = np.log1p(df["Price"])  # log-transform target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_features = ["Company", "TypeName", "OpSys", "Memory_Type", "Cpu_Brand", "Gpu_Brand"]
    num_features = ["Inches", "Ram_GB", "Memory_GB", "Weight_kg", "Cpu_Speed_GHz", "Resolution_Width", "Resolution_Height"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features)
    ])

    # RandomizedSearch for better hyperparams
    rf = RandomForestRegressor(random_state=42)
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(10, 30),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5)
    }

    model = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=20, cv=3, n_jobs=-1, scoring="r2", random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print("Training model with RandomizedSearchCV...")
    pipeline.fit(X_train, y_train)

    y_train_pred = np.expm1(pipeline.predict(X_train))
    y_test_pred = np.expm1(pipeline.predict(X_test))
    y_train_true = np.expm1(y_train)
    y_test_true = np.expm1(y_test)

    def evaluate(true, pred, label):
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, pred)
        print(f"\n{label} Metrics:")
        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        return mae, rmse, r2

    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    evaluate(y_train_true, y_train_pred, "Training Set")
    evaluate(y_test_true, y_test_pred, "Test Set")
    print("="*50)

    # Save metrics to JSON for dashboard
    metrics_data = {
        "train": {
            "MAE": float(mean_absolute_error(y_train_true, y_train_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_train_true, y_train_pred))),
            "R2": float(r2_score(y_train_true, y_train_pred))
        },
        "test": {
            "MAE": float(mean_absolute_error(y_test_true, y_test_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test_true, y_test_pred))),
            "R2": float(r2_score(y_test_true, y_test_pred))
        }
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline.named_steps["preprocessor"], PREPROCESSOR_PATH)
    joblib.dump(pipeline.named_steps["model"].best_estimator_, MODEL_PATH)

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"✅ Saved metrics to {metrics_path}")
    print(f"✅ Saved model + preprocessor to '{MODELS_DIR}' folder.")


if __name__ == "__main__":
    main()
