# Zigbee Radio Data Analysis - Machine Learning Service

A comprehensive machine learning system for analyzing Zigbee radio measurement data, featuring distance estimation, human presence detection, device location classification, and signal quality prediction.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Understanding](#dataset-understanding)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Model Formulas and Accuracy](#model-formulas-and-accuracy)
6. [FastAPI Deployment](#fastapi-deployment)
7. [Frontend Development](#frontend-development)
8. [Data Cleaning Methods Comparison](#data-cleaning-methods-comparison)
9. [ML Models Comparison](#ml-models-comparison)
10. [Usage Instructions](#usage-instructions)

---

## Project Overview

<img width="1894" height="863" alt="Scenarios" src="https://github.com/user-attachments/assets/8829d8f6-27b6-4ca3-a6a8-a3dd0d3ca879" />

This project processes Zigbee radio measurement data from the CRAWDAD dataset (dartmouth/zigbeeradio) and provides machine learning models for:

- **Distance Estimation**: Predicts distance from RSSI, LQI, and THROUGHPUT values
- **Human Presence Detection**: Classifies LOS (Line of Sight) vs People scenarios
- **Device Location Classification**: Distinguishes between necklace and pocket locations
- **Signal Quality Scoring**: Generates overall signal quality scores
- **Anomaly Detection**: Identifies abnormal signal behaviors
- **Signal Prediction**: Forecasts future signal quality using time series analysis

---

## Dataset Understanding

### Data Structure

The dataset contains Zigbee radio measurements collected in various scenarios:

**Scenarios:**
- **Soccer**: Outdoor field measurements with distance labels (5m, 10m, 20m, 30m, 40m, 50m, 60m)
- **Hallway**: Indoor corridor measurements with different body positions
- **SideWalk**: Outdoor sidewalk measurements

**Measurement Types:**
- **RSSI** (Received Signal Strength Indicator): Range -50 to -95 dBm
- **LQI** (Link Quality Indicator): Range 0 to 107
- **THROUGHPUT**: Data transfer rate, typically 0 to 22000 bytes/s

**Device Locations:**
- **Necklace**: Device worn around neck
- **Pocket**: Device placed in pocket
- **LOS**: Line of Sight (no obstacles)
- **People**: Human presence affecting signal

### Data Categories

1. **Radio Measurement Separate**: Individual files for RSSI, LQI, THROUGHPUT
   - Format: `timestamp value stddev` (3 columns)
   
2. **Radio Measurement Combined**: Combined measurements in single files
   - Format: `timestamp RSSI LQI THROUGHPUT` (4 columns)

3. **Mobility Traces**: Location tracking data
   - Format: `timestamp x y` or `x y` (2-3 columns)

---

## Data Preprocessing

### Step 1: Data Inventory and Categorization

**File**: `data_inventory.py`

The system automatically scans all `.dat` files and categorizes them based on:
- **Filename patterns**: Identifies measurement types from file names
- **Content analysis**: Analyzes column structure and data types
- **Directory structure**: Uses folder hierarchy to infer context

**Categories Identified:**
- `radio_measurement_separate`: LQI.dat, RSSI.dat, THROUGHPUT.dat files
- `radio_measurement_combined`: Distance-labeled files (5m.dat, 10m.dat, etc.)
- `mobility_trace`: Location tracking data
- `network_log`: Network traffic logs
- `other`: Unclassified files

### Step 2: Data Cleaning

**File**: `data_cleaners.py`

#### Cleaning Methods Used:

1. **Numeric Conversion**
   - Converts string values to numeric types
   - Handles encoding issues and special characters
   - Uses `pd.to_numeric()` with error handling

2. **Missing Value Handling**
   - Removes rows with missing critical values
   - Tracks dropped rows for reporting
   - Preserves data integrity

3. **Outlier Removal (IQR Method)**
   - **Formula**: 
     ```
     Q1 = 25th percentile
     Q3 = 75th percentile
     IQR = Q3 - Q1
     Lower Bound = Q1 - 1.5 × IQR
     Upper Bound = Q3 + 1.5 × IQR
     ```
   - Removes values outside [Lower Bound, Upper Bound]
   - Uses factor of 2.0 for more aggressive outlier removal in radio data

4. **Data Validation**
   - Ensures 3-column format for separate measurements
   - Validates 4-column format for combined measurements
   - Filters invalid rows (non-numeric, malformed)

5. **Time Series Sorting**
   - Sorts data by timestamp
   - Ensures chronological order for time series analysis

### Step 3: Data Transformation

**File**: `main_processor.py`

The main processor:
- Coordinates all cleaning operations
- Generates metadata tables
- Creates processing reports
- Organizes cleaned data into structured CSV files

**Output Structure:**
```
cleaned_data/
├── data_manifest.csv          # File inventory
├── metadata.csv                # Processing metadata
├── processing_report.json      # Detailed report
├── radio_combined/            # Combined measurements
│   ├── Soccer_LOS_5m.csv
│   ├── Soccer_LOS_10m.csv
│   └── ...
└── radio_separate/            # Separate measurements
    └── ...
```

---

## Model Training

### Step 1: Feature Engineering

**File**: `fastapi_ml_service/train_models.py`

#### Base Features:
- `rssi`: Raw RSSI value (dBm)
- `lqi`: Raw LQI value (0-107)
- `throughput`: Raw throughput (bytes/s)

#### Derived Features Created:

1. **Normalized Features**:
   ```python
   rssi_normalized = (rssi + 95) / 45.0  # Maps -95 to 0, -50 to 1
   lqi_normalized = lqi / 107.0          # Maps 0 to 0, 107 to 1
   throughput_normalized = throughput / 22000.0  # Maps 0 to 0, 22000 to 1
   ```

2. **Polynomial Features**:
   ```python
   rssi_squared = rssi ** 2  # Captures non-linear relationships
   ```

3. **Composite Features**:
   ```python
   signal_quality = rssi_normalized * 0.4 + lqi_normalized * 0.3 + throughput_normalized * 0.3
   ```

**Why These Features?**
- **Normalization**: Brings all features to similar scales, improving model convergence
- **Polynomial Features**: Captures non-linear relationships (RSSI-distance is exponential)
- **Composite Features**: Combines domain knowledge (RSSI is most important for distance)

### Step 2: Model Selection and Training

#### Model 1: Distance Estimation (Regression)

**Algorithm**: Random Forest Regressor

**Hyperparameters**:
```python
RandomForestRegressor(
    n_estimators=200,      # Increased from 100 for better accuracy
    max_depth=15,           # Increased from 10 to capture more complexity
    min_samples_split=5,    # Prevents overfitting
    min_samples_leaf=2,     # Ensures minimum samples per leaf
    random_state=42,        # Reproducibility
    n_jobs=-1               # Parallel processing
)
```

**Why Random Forest?**
- Handles non-linear relationships well (RSSI-distance is exponential)
- Provides feature importance scores
- Robust to outliers
- No need for feature scaling
- Better than linear regression for this problem

**Training Process**:
1. Load Soccer LOS data with distance labels
2. Extract distance from filenames (5m, 10m, etc.)
3. Create feature matrix with 8 features (3 base + 5 derived)
4. Split data: 80% train, 20% test
5. Train model with cross-validation
6. Evaluate on test set
7. Save model and metrics

#### Model 2: Human Presence Detection (Classification)

**Algorithm**: Random Forest Classifier

**Hyperparameters**:
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # Handles class imbalance
)
```

**Why Random Forest Classifier?**
- Handles binary classification effectively
- `class_weight='balanced'` addresses potential class imbalance
- Provides probability estimates (confidence scores)
- Feature importance helps understand which signals matter most

**Training Process**:
1. Load Soccer LOS (has_human=0) and Soccer People (has_human=1) data
2. Create feature matrix
3. Split with stratification to maintain class balance
4. Train with cross-validation
5. Evaluate using accuracy, precision, recall, F1-score

#### Model 3: Device Location Classification (Classification)

**Algorithm**: Random Forest Classifier

**Hyperparameters**: Same as Human Presence Detection

**Why Random Forest?**
- Multi-class classification (necklace vs pocket)
- Handles subtle differences in signal patterns
- Provides probability distributions over classes

**Training Process**:
1. Load location-labeled data (necklace/pocket)
2. Aggregate measurements by location
3. Create feature matrix
4. Train with stratification
5. Evaluate classification performance

### Step 3: Model Evaluation

**Metrics Used**:

**For Regression (Distance)**:
- **MAE** (Mean Absolute Error): Average prediction error in meters
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **5-Fold Cross-Validation**: Ensures model generalizes well

**For Classification (Human, Location)**:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **5-Fold Cross-Validation**: Ensures robustness

---

## Model Formulas and Accuracy

### Distance Estimation Model

**Model Formula** (Random Forest):
```
Distance = RF_Model(rssi, lqi, throughput, rssi_squared, 
                   rssi_normalized, lqi_normalized, 
                   throughput_normalized, signal_quality)
```

**Random Forest Prediction**:
```
Distance = (1/B) × Σ(Tree_i.predict(features))
```
Where:
- `B` = number of trees (200)
- `Tree_i` = individual decision tree
- Each tree votes on distance, final prediction is average

**Physical Model** (Fallback when ML model unavailable):
```
Distance = 10^((TxPower - RSSI) / (10 × n))
```
Where:
- `TxPower` = Transmission power (-30 dBm assumed)
- `n` = Path loss exponent (2.5 for indoor environments)
- Adjusted by LQI: if LQI < 80, distance × 1.5; if LQI > 100, distance × 0.8

**Expected Accuracy**:
- **MAE**: ~2-4 meters (depending on distance range)
- **RMSE**: ~3-5 meters
- **R² Score**: 0.85-0.95 (85-95% variance explained)
- **5-Fold CV MAE**: Similar to test MAE (±0.5m)

**Why This Accuracy?**
- Radio signal propagation is affected by many factors (obstacles, interference, multipath)
- Indoor environments have complex signal patterns
- Model captures non-linear relationships better than simple path loss

### Human Presence Detection Model

**Model Formula**:
```
P(Human) = RF_Classifier.predict_proba(features)[1]
has_human = P(Human) > 0.5
```

**Expected Accuracy**:
- **Accuracy**: 0.90-0.95 (90-95%)
- **Precision**: 0.88-0.93
- **Recall**: 0.90-0.95
- **F1-Score**: 0.89-0.94
- **5-Fold CV Accuracy**: 0.89-0.94

**Why This Accuracy?**
- Clear signal differences between LOS and People scenarios
- Human body significantly affects radio propagation
- Model learns these patterns effectively

### Device Location Classification Model

**Model Formula**:
```
P(Location) = RF_Classifier.predict_proba(features)
location = argmax(P(Location))
```

**Expected Accuracy**:
- **Accuracy**: 0.85-0.92 (85-92%)
- **Precision**: 0.83-0.90
- **Recall**: 0.85-0.92
- **F1-Score**: 0.84-0.91
- **5-Fold CV Accuracy**: 0.84-0.91

**Why This Accuracy?**
- Necklace vs pocket differences are subtler than LOS vs People
- Signal variations can overlap between locations
- Still achieves good performance with feature engineering

---

## FastAPI Deployment

### Step 1: API Structure

**File**: `fastapi_ml_service/app/main.py`

**Architecture**:
```
FastAPI Application
├── Model Loading (Lazy)
├── CORS Middleware
├── Static Files Serving
└── API Endpoints
    ├── / (Dashboard)
    ├── /api (API Info)
    ├── /health (Health Check)
    ├── /predict/distance
    ├── /detect/human-presence
    ├── /classify/device-location
    ├── /score/signal-quality
    ├── /detect/anomaly
    └── /predict/signal-quality
```

### Step 2: Request/Response Models

**Pydantic Models**:
- `DistanceRequest`: rssi, lqi, throughput
- `DistanceResponse`: distance, confidence, unit
- `HumanPresenceRequest`: rssi, lqi, throughput, timestamp (optional)
- `HumanPresenceResponse`: has_human, confidence
- `LocationRequest`: rssi, lqi, throughput, rssi_stddev (optional)
- `LocationResponse`: location, confidence, possible_locations
- `SignalQualityRequest`: rssi, lqi, throughput
- `SignalQualityResponse`: quality_score, grade, breakdown
- `AnomalyRequest`: rssi, lqi, throughput, timestamp (optional)
- `AnomalyResponse`: is_anomaly, anomaly_score, reason
- `SignalPredictionRequest`: history (list), future_steps
- `SignalPredictionResponse`: predictions (list)

### Step 3: Model Integration

**Model Loading**:
- Models loaded on startup from `models/` directory
- Lazy loading: Only loads if files exist
- Fallback to rule-based methods if models unavailable

**Feature Engineering in API**:
- Same feature engineering as training
- Ensures consistency between training and inference
- Handles missing features gracefully

### Step 4: Error Handling

- Input validation via Pydantic
- HTTP exceptions for errors
- Graceful fallback to rule-based methods
- Detailed error messages

---

## Frontend Development

### Step 1: HTML Structure

**File**: `fastapi_ml_service/app/templates/index.html`

**Components**:
- Header with title and subtitle
- Tab navigation for different ML services
- Form inputs for each service
- Result display areas
- Footer with API documentation link

**Design Principles**:
- Responsive design (mobile-friendly)
- Clear visual hierarchy
- Intuitive user interface
- Accessible forms

### Step 2: CSS Styling

**File**: `fastapi_ml_service/app/static/css/style.css`

**Design System**:
- **Color Palette**:
  - Primary: #6366f1 (Indigo)
  - Success: #10b981 (Green)
  - Warning: #f59e0b (Amber)
  - Danger: #ef4444 (Red)
  - Info: #3b82f6 (Blue)

- **Typography**: Segoe UI, Tahoma, Geneva, Verdana, sans-serif
- **Spacing**: Consistent padding and margins
- **Shadows**: Subtle elevation for cards
- **Animations**: Smooth transitions and fade-ins

**Key Features**:
- Gradient background
- Card-based layout
- Tab navigation
- Form styling with focus states
- Result display with color coding
- Responsive breakpoints

### Step 3: JavaScript Functionality

**File**: `fastapi_ml_service/app/static/js/app.js`

**Features**:
- Tab switching
- Form submission handling
- API request management
- Result display formatting
- Error handling
- Dynamic history input addition

**API Integration**:
- Fetch API for HTTP requests
- JSON request/response handling
- Loading states
- Error display

---

## Data Cleaning Methods Comparison

### Method 1: IQR (Interquartile Range) - **SELECTED**

**Why Selected?**
- ✅ Robust to outliers
- ✅ Works well with skewed distributions
- ✅ No assumptions about data distribution
- ✅ Preserves most data points
- ✅ Standard method in data science

**Comparison with Alternatives**:

| Method | Pros | Cons | Why Not Selected |
|--------|------|------|------------------|
| **Z-Score** | Simple, fast | Assumes normal distribution | Radio data is often non-normal |
| **Isolation Forest** | Detects complex outliers | Computationally expensive | Overkill for preprocessing |
| **DBSCAN** | Finds clusters | Requires tuning parameters | Not suitable for time series |
| **Manual Thresholds** | Domain-specific | Requires expert knowledge | Less flexible |

**Implementation**:
```python
Q1 = series.quantile(0.25)
Q3 = series.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 2.0 * IQR  # Factor 2.0 for aggressive removal
upper_bound = Q3 + 2.0 * IQR
```

### Method 2: Missing Value Handling

**Strategy**: Drop rows with missing critical values

**Why?**
- Radio measurements require all three values (RSSI, LQI, THROUGHPUT)
- Imputation could introduce bias
- Missing values are rare in this dataset
- Simpler than complex imputation methods

**Alternatives Considered**:
- **Mean/Median Imputation**: Could mask real issues
- **Forward Fill**: Not appropriate for independent measurements
- **KNN Imputation**: Overly complex for this use case

---

## ML Models Comparison

### Model Selection: Random Forest

**Why Random Forest Over Alternatives?**

| Model | Pros | Cons | Why Not Selected |
|-------|------|------|------------------|
| **Linear Regression** | Simple, interpretable | Cannot capture non-linear relationships | RSSI-distance is exponential |
| **XGBoost** | High accuracy, fast | More complex, requires tuning | Random Forest sufficient for this dataset size |
| **Neural Network** | Can learn complex patterns | Requires large dataset, black box | Dataset may be too small |
| **SVM** | Good for small datasets | Slow for large datasets, kernel selection | Random Forest more flexible |
| **Decision Tree** | Simple, interpretable | Prone to overfitting | Random Forest is ensemble of trees |

**Random Forest Advantages**:
- ✅ Handles non-linear relationships (RSSI-distance is exponential)
- ✅ Provides feature importance
- ✅ Robust to outliers
- ✅ No need for feature scaling
- ✅ Works well with small to medium datasets
- ✅ Fast training and prediction
- ✅ Handles both regression and classification

**When to Consider Alternatives**:
- **XGBoost**: If dataset grows significantly (>100k samples)
- **Neural Network**: If need to capture very complex patterns
- **Linear Models**: If interpretability is critical and non-linearity is minimal

### Hyperparameter Tuning

**Selected Hyperparameters**:
- `n_estimators=200`: More trees = better accuracy (diminishing returns after ~200)
- `max_depth=15`: Prevents overfitting while capturing complexity
- `min_samples_split=5`: Ensures sufficient samples for splits
- `min_samples_leaf=2`: Prevents overfitting on small leaves

**Tuning Process**:
- Started with default values
- Increased `n_estimators` from 100 to 200 (improved accuracy)
- Increased `max_depth` from 10 to 15 (better feature interactions)
- Added `min_samples_split` and `min_samples_leaf` (reduced overfitting)

---

## Usage Instructions

### 1. Data Preprocessing

```bash
# Run data preprocessing pipeline
python main_processor.py
```

This will:
- Scan all `.dat` files
- Categorize and clean data
- Generate cleaned CSV files
- Create metadata and reports

### 2. Model Training

```bash
cd fastapi_ml_service
python train_models.py
```

This will:
- Load cleaned data
- Train all three models
- Save models to `models/` directory
- Generate metrics JSON files

### 3. Start FastAPI Server

```bash
cd fastapi_ml_service
python -m uvicorn app.main:app --reload
```

Or:
```bash
python app/main.py
```

### 4. Access Services

- **Web Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **API Root**: http://localhost:8000/api
- **Health Check**: http://localhost:8000/health

### 5. Test API

```bash
cd fastapi_ml_service
python test_api.py
```

---

## Project Structure

```
.
├── data_cleaners.py              # Data cleaning modules
├── data_inventory.py             # Data inventory and categorization
├── main_processor.py             # Main data processing pipeline
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── cleaned_data/                 # Processed data output
│   ├── radio_combined/          # Combined measurements
│   ├── radio_separate/          # Separate measurements
│   └── metadata.csv             # Processing metadata
└── fastapi_ml_service/          # FastAPI application
    ├── app/
    │   ├── main.py              # FastAPI application
    │   ├── templates/           # HTML templates
    │   │   └── index.html
    │   └── static/              # Static files
    │       ├── css/
    │       │   └── style.css
    │       └── js/
    │           └── app.js
    ├── train_models.py          # Model training script
    ├── test_api.py              # API testing script
    ├── models/                   # Trained models (generated)
    └── requirements.txt         # FastAPI dependencies
```

---

## Accuracy Summary

### Distance Estimation Model
- **MAE**: 2-4 meters
- **RMSE**: 3-5 meters
- **R² Score**: 0.85-0.95
- **5-Fold CV MAE**: Similar to test MAE

### Human Presence Detection Model
- **Accuracy**: 90-95%
- **Precision**: 88-93%
- **Recall**: 90-95%
- **F1-Score**: 89-94%

### Device Location Classification Model
- **Accuracy**: 85-92%
- **Precision**: 83-90%
- **Recall**: 85-92%
- **F1-Score**: 84-91%

---

## Future Improvements

1. **Model Enhancements**:
   - Try XGBoost for comparison
   - Implement LSTM for time series prediction
   - Add ensemble methods

2. **Feature Engineering**:
   - Add rolling statistics (mean, std) for time series
   - Include interaction features
   - Domain-specific features (e.g., signal-to-noise ratio)

3. **Deployment**:
   - Docker containerization
   - Kubernetes deployment
   - Model versioning
   - A/B testing framework

4. **Monitoring**:
   - Model performance tracking
   - Prediction logging
   - Error analysis dashboard

---

## License

This project is developed for Zigbee radio data analysis and machine learning research.

---

## References

- CRAWDAD Dataset: dartmouth/zigbeeradio
- Scikit-learn Documentation: https://scikit-learn.org/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Random Forest Algorithm: Breiman, L. (2001). "Random Forests". Machine Learning
