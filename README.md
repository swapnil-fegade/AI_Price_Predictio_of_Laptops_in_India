# AI-Powered Laptop Price Prediction System

An intelligent machine learning system that predicts laptop prices based on technical specifications using a Random Forest Regressor. Built with Python, scikit-learn, and Streamlit for an interactive web interface.

## 🎯 Overview

This project uses machine learning to predict laptop prices (in INR) based on various specifications including company, type, screen resolution, CPU, RAM, memory, GPU, operating system, and weight. The model employs advanced feature engineering, hyperparameter optimization, and log-transformed targets for improved accuracy.

## 📊 Dataset
Dataset from kaggle "https://www.kaggle.com/datasets/mohammadkaiftahir/laptop-price-in-india"
The model uses `data/laptop_data.csv` with the following features:

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| **Company** | Categorical | Laptop manufacturer | Apple, Dell, HP, Lenovo, Asus, Acer, etc. |
| **TypeName** | Categorical | Laptop type/category | Ultrabook, Notebook, Gaming, Workstation, 2 in 1 Convertible |
| **Inches** | Numeric | Screen size in inches | 11.6, 13.3, 15.6, 17.3 |
| **ScreenResolution** | String | Display resolution | "Full HD 1920x1080", "IPS Panel Retina Display 2560x1600" |
| **Cpu** | String | CPU specification | "Intel Core i5 2.3GHz", "AMD Ryzen 1700 3GHz" |
| **Ram** | String | RAM size | "4GB", "8GB", "16GB" |
| **Memory** | String | Storage specification | "256GB SSD", "1TB HDD", "512GB SSD + 1TB HDD" |
| **Gpu** | String | GPU specification | "Intel HD Graphics 620", "Nvidia GeForce GTX 1060" |
| **OpSys** | Categorical | Operating system | Windows 10, macOS, Linux, No OS |
| **Weight** | String | Laptop weight | "1.37kg", "2.1kg", "2.5kg" |
| **Price** | Numeric | Price in INR (target variable) | 30,636, 71,378, 135,195 |

**Dataset Size**: ~1,305 rows

## 🏗️ Project Structure

```
.
├── app.py                 # Streamlit web application
├── train_model.py         # Model training script with hyperparameter optimization
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── data/
│   └── laptop_data.csv   # Laptop price dataset
└── models/               # Generated after training
    ├── preprocessor.joblib        # Saved preprocessor
    ├── random_forest_model.joblib # Trained model
    └── metrics.json              # Model performance metrics
```

## 🚀 Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Train the Model

Run the training script to train the model with hyperparameter optimization:

```bash
python train_model.py
```

### What the Training Script Does

1. **Loads and Preprocesses Data**
   - Reads `data/laptop_data.csv`
   - Performs feature engineering on text fields

2. **Feature Engineering**
   - **RAM**: Extracts numeric value (e.g., "8GB" → 8)
   - **Memory**: Extracts total storage size and type (handles hybrid storage like "512GB SSD + 1TB HDD")
   - **Weight**: Extracts numeric value in kg
   - **CPU**: Extracts brand (Intel/AMD/Apple) and speed (GHz)
   - **GPU**: Extracts brand (Intel/AMD/Nvidia)
   - **ScreenResolution**: Extracts width and height in pixels

3. **Data Preprocessing**
   - Applies log transformation to target variable (`log1p(Price)`) for better model performance
   - One-Hot Encodes categorical features
   - Standardizes numeric features

4. **Model Training**
   - Uses `RandomizedSearchCV` for hyperparameter optimization
   - Searches over:
     - `n_estimators`: 100-500
     - `max_depth`: 10-30
     - `min_samples_split`: 2-10
     - `min_samples_leaf`: 1-5
   - Performs 3-fold cross-validation
   - Trains Random Forest Regressor with best hyperparameters

5. **Evaluation**
   - Calculates performance metrics on training and test sets:
     - **MAE** (Mean Absolute Error)
     - **RMSE** (Root Mean Squared Error)
     - **R² Score** (Coefficient of Determination)

6. **Saves Artifacts**
   - Preprocessor to `models/preprocessor.joblib`
   - Trained model to `models/random_forest_model.joblib`
   - Metrics to `models/metrics.json`

### Expected Output

```
Loaded dataset with 1305 rows and 11 columns.
Training model with RandomizedSearchCV...

==================================================
MODEL PERFORMANCE METRICS
==================================================

Training Set Metrics:
MAE: 3950.38, RMSE: 6988.96, R²: 0.9644

Test Set Metrics:
MAE: 10193.75, RMSE: 17306.58, R²: 0.7923
==================================================
✅ Saved metrics to models/metrics.json
✅ Saved model + preprocessor to 'models' folder.
```

## 🖥️ Run the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Application Features

1. **Interactive Input Form**
   - Company selection
   - Laptop type selection
   - Operating system selection
   - Screen size slider (11.0 - 18.0 inches)
   - RAM slider (4 - 64 GB)
   - Storage type selection (SSD, HDD, Hybrid, Flash Storage)
   - CPU brand and speed
   - GPU brand
   - Storage capacity slider (128 - 2048 GB)
   - Weight slider (0.5 - 5.0 kg)
   - Display resolution inputs (width and height in pixels)

2. **Model Performance Dashboard** (Sidebar)
   - Displays model artifact status
   - Shows training and test R² scores
   - Indicates model generalization quality:
     - 🟢 Excellent Generalization (gap < 0.05)
     - 🟡 Moderate Overfitting (gap < 0.12)
     - 🔴 High Overfitting (gap ≥ 0.12)

3. **Price Prediction**
   - Real-time price prediction in INR
   - Clear, formatted output

## 🔧 Model Features

### Categorical Features (One-Hot Encoded)
- Company
- TypeName
- OpSys
- Memory_Type (SSD, HDD, Hybrid, Flash Storage)
- Cpu_Brand (Intel, AMD, Apple)
- Gpu_Brand (Intel, AMD, Nvidia)

### Numeric Features (Standardized)
- Inches (Screen size)
- Ram_GB
- Memory_GB (Total storage capacity)
- Weight_kg
- Cpu_Speed_GHz
- Resolution_Width (pixels)
- Resolution_Height (pixels)

### Target Variable
- **Price** (in INR) - Log-transformed during training for better performance

## 📊 Model Performance

### Current Performance Metrics

**Training Set:**
- MAE: 3,950.38 INR
- RMSE: 6,988.96 INR
- R²: 0.9644

**Test Set:**
- MAE: 10,193.75 INR
- RMSE: 17,306.58 INR
- R²: 0.7923

### Model Characteristics

- **Algorithm**: Random Forest Regressor
- **Hyperparameter Optimization**: RandomizedSearchCV (20 iterations, 3-fold CV)
- **Target Transformation**: Log transformation (`log1p`) for improved accuracy
- **Preprocessing**: One-Hot Encoding + Standard Scaling
- **Generalization**: Moderate overfitting (train-test R² gap: 0.172)

## 🎨 Key Features

### Advanced Feature Engineering

1. **Hybrid Storage Support**: Handles multiple storage components (e.g., "512GB SSD + 1TB HDD" → 1536 GB total)
2. **Memory Type Detection**: Automatically classifies storage as SSD, HDD, Hybrid, or Flash Storage
3. **Resolution Extraction**: Parses screen resolution strings to extract width and height
4. **CPU/GPU Brand Extraction**: Intelligently extracts brand information from specification strings

### Model Optimization

1. **Hyperparameter Tuning**: Uses RandomizedSearchCV for optimal model parameters
2. **Log Transformation**: Applies log transformation to target variable to handle price distribution
3. **Pipeline Architecture**: Uses scikit-learn Pipeline for clean preprocessing and modeling workflow

### User Experience

1. **Interactive UI**: Clean, intuitive Streamlit interface
2. **Real-time Metrics**: Displays model performance in sidebar
3. **Generalization Indicator**: Visual feedback on model quality
4. **Comprehensive Inputs**: All relevant laptop specifications can be input

## 📦 Dependencies

- **pandas** (≥2.0): Data manipulation and analysis
- **numpy** (≥1.24): Numerical computations
- **scikit-learn** (≥1.3): Machine learning algorithms and preprocessing
- **joblib** (≥1.2): Model serialization
- **streamlit** (≥1.29): Web application framework
- **scipy** (for RandomizedSearchCV): Statistical functions

## 🔍 Technical Details

### Feature Extraction Functions

- `extract_ram_gb()`: Extracts RAM size from strings like "8GB"
- `extract_memory_gb()`: Calculates total storage (handles TB and GB, multiple components)
- `extract_memory_type()`: Classifies storage type (SSD/HDD/Hybrid/Flash Storage)
- `extract_weight_kg()`: Extracts weight in kg
- `extract_cpu_brand()`: Identifies CPU manufacturer
- `extract_cpu_speed()`: Extracts CPU clock speed in GHz
- `extract_gpu_brand()`: Identifies GPU manufacturer
- `extract_resolution()`: Extracts screen resolution width and height

### Model Architecture

```
Input Features
    ↓
Feature Engineering
    ↓
Preprocessing Pipeline
    ├── Categorical: OneHotEncoder
    └── Numeric: StandardScaler
    ↓
Random Forest Regressor (with optimized hyperparameters)
    ↓
Log Inverse Transformation (expm1)
    ↓
Price Prediction (INR)
```

## 🛠️ Usage Example

1. **Train the model**:
   ```bash
   python train_model.py
   ```

2. **Launch the app**:
   ```bash
   streamlit run app.py
   ```

3. **Input specifications** in the web interface:
   - Select: Apple, Ultrabook, macOS
   - Set: 13.3 inches, 8GB RAM, 256GB SSD
   - Choose: Intel CPU (2.5 GHz), Intel GPU
   - Set: 1.5 kg, 2560x1600 resolution

4. **Click "Predict Price"** to get the estimated price

## 📝 Notes

- The model uses log-transformed targets, so predictions are transformed back using `expm1()`
- Prices are predicted directly in INR (no currency conversion needed)
- The model handles unknown categories gracefully using `handle_unknown="ignore"` in OneHotEncoder
- Model artifacts are cached in Streamlit for faster loading

## 👤 Author

**Swapnil Fegade**

AI-Powered Laptop Price Prediction System

## 📄 License

This project is for educational purposes.

---

**Note**: Make sure to train the model (`python train_model.py`) before running the Streamlit app, as the app requires the saved model artifacts.
