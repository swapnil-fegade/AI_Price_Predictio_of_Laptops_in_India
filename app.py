import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = "models"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
DATA_PATH = os.path.join("data", "laptop_data.csv")

st.set_page_config(page_title=" AI-Powered Laptop Price Prediction", page_icon="💻")
st.title("AI-Powered Laptop Price Prediction System")
st.write("Predict laptop prices based on specifications using a trained Random Forest model.")

# =============================
# Load Artifacts
# =============================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model

# =============================
# Data Loading Helper
# =============================
def load_options():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        companies = sorted(df["Company"].dropna().unique().tolist())
        typenames = sorted(df["TypeName"].dropna().unique().tolist())
        opsys = sorted(df["OpSys"].dropna().unique().tolist())

        cpu_brands, gpu_brands = [], []
        for cpu in df["Cpu"].dropna():
            cpu_str = str(cpu).upper()
            if "INTEL" in cpu_str and "Intel" not in cpu_brands:
                cpu_brands.append("Intel")
            elif "AMD" in cpu_str and "AMD" not in cpu_brands:
                cpu_brands.append("AMD")
            elif any(x in cpu_str for x in ["APPLE", "M1", "M2", "M3"]) and "Apple" not in cpu_brands:
                cpu_brands.append("Apple")

        for gpu in df["Gpu"].dropna():
            gpu_str = str(gpu).upper()
            if any(x in gpu_str for x in ["NVIDIA", "GEFORCE"]) and "Nvidia" not in gpu_brands:
                gpu_brands.append("Nvidia")
            elif any(x in gpu_str for x in ["AMD", "RADEON"]) and "AMD" not in gpu_brands:
                gpu_brands.append("AMD")
            elif "INTEL" in gpu_str and "Intel" not in gpu_brands:
                gpu_brands.append("Intel")

        memory_types = ["SSD", "HDD", "Hybrid", "Flash Storage"]
        return companies, typenames, opsys, cpu_brands, gpu_brands, memory_types

    # fallback defaults
    return (
        ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI"],
        ["Ultrabook", "Notebook", "Gaming", "Workstation", "2 in 1 Convertible"],
        ["windows", "windows", "macOS", "No OS", "Linux", "Chrome OS"],
        ["Intel", "AMD", "Apple"],
        ["Intel", "AMD", "Nvidia"],
        ["SSD", "HDD", "Hybrid", "Flash Storage"],
    )

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("📦 Model Artifacts")
    st.write(f"Preprocessor: {'✅' if os.path.exists(PREPROCESSOR_PATH) else '❌'}")
    st.write(f"Model: {'✅' if os.path.exists(MODEL_PATH) else '❌'}")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
        st.write("**Model Performance Summary:**")
        train_r2 = metrics["train"]["R2"]
        test_r2 = metrics["test"]["R2"]
        gap = abs(train_r2 - test_r2)
        if gap < 0.05:
            gen_quality = "🟢 Excellent Generalization"
        elif gap < 0.12:
            gen_quality = "🟡 Moderate Overfitting"
        else:
            gen_quality = "🔴 High Overfitting"

        st.metric("Train R²", f"{train_r2:.3f}")
        st.metric("Test R²", f"{test_r2:.3f}")
        st.caption(gen_quality)
    else:
        st.warning("⚠️ Metrics not found. Retrain model to generate `metrics.json`.")

    st.caption("If missing, run: `python train_model.py`")

if not (os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH)):
    st.warning("⚠️ Model artifacts not found. Please run 'python train_model.py' first.")
    st.stop()

preprocessor, model = load_artifacts()
companies, typenames, opsys, cpu_brands, gpu_brands, memory_types = load_options()

# =============================
# UI Inputs
# =============================
st.header("Laptop Specifications")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Company", companies)
    typename = st.selectbox("Type", typenames)
    opsys_input = st.selectbox("Operating System", opsys)
    inches = st.slider("Screen Size (inches)", 11.0, 18.0, 15.6, 0.1)
    ram_gb = st.slider("RAM (GB)", 4, 64, 8, 4)
    memory_type = st.selectbox("Storage Type", memory_types)

with col2:
    cpu_brand = st.selectbox("CPU Brand", cpu_brands)
    cpu_speed = st.slider("CPU Speed (GHz)", 1.0, 5.0, 2.5, 0.1)
    gpu_brand = st.selectbox("GPU Brand", gpu_brands)
    memory_gb = st.slider("Storage (GB)", 128, 2048, 512, 128)
    weight_kg = st.slider("Weight (kg)", 0.5, 5.0, 2.0, 0.1)

st.subheader("Display Resolution")
colr1, colr2 = st.columns(2)
with colr1:
    resolution_width = st.number_input("Resolution Width (pixels)", 800, 7680, 1920, 160)
with colr2:
    resolution_height = st.number_input("Resolution Height (pixels)", 600, 4320, 1080, 90)

# =============================
# Prediction
# =============================
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "Company": company,
        "TypeName": typename,
        "OpSys": opsys_input,
        "Inches": inches,
        "Ram_GB": ram_gb,
        "Memory_GB": memory_gb,
        "Memory_Type": memory_type,
        "Cpu_Brand": cpu_brand,
        "Cpu_Speed_GHz": cpu_speed,
        "Gpu_Brand": gpu_brand,
        "Weight_kg": weight_kg,
        "Resolution_Width": resolution_width,
        "Resolution_Height": resolution_height
    }])

    X_transformed = preprocessor.transform(input_df)
    log_pred = model.predict(X_transformed)
    price_predicted = np.expm1(log_pred[0])

    st.success("✅ Prediction Complete!")
    st.metric("Estimated Laptop Price", f"₹{price_predicted:,.0f} INR")
    st.info(f"Predicted price (approx.): **₹{price_predicted:,.0f} INR**")

st.write("---")
st.caption("Made by **Swapnil Fegade** | AI-Powered Laptop Price Prediction System")
