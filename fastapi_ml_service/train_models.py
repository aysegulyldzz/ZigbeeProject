"""
Model training script
Trains ML models from Zigbee radio data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import json

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "cleaned_data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_soccer_data():
    """Load Soccer data (contains distance labels)."""
    combined_dir = DATA_DIR / "radio_combined"
    if not combined_dir.exists():
        print(f"Data directory not found: {combined_dir}")
        return None
    
    all_data = []
    
    # Read files from Soccer_LOS directory
    for file in combined_dir.glob("Soccer_LOS_*.csv"):
        # Extract distance from filename (e.g., Soccer_LOS_10m.csv -> 10)
        distance_str = file.stem.replace("Soccer_LOS_", "").replace("m", "")
        try:
            distance = float(distance_str)
            df = pd.read_csv(file)
            df['distance'] = distance
            all_data.append(df)
        except ValueError:
            continue
    
    if not all_data:
        print("Soccer LOS data not found")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def train_distance_model():
    """Trains distance estimation model"""
    print("=" * 60)
    print("TRAINING DISTANCE ESTIMATION MODEL")
    print("=" * 60)
    
    # Load data
    df = load_soccer_data()
    if df is None:
        print("Failed to load data, cannot train model")
        return
    
    print(f"Total samples: {len(df)}")
    
    # Feature engineering
    base_features = ['rssi', 'lqi', 'throughput']
    
    # Add derived features
    df['rssi_squared'] = df['rssi'] ** 2
    df['lqi_normalized'] = df['lqi'] / 107.0
    df['throughput_normalized'] = df['throughput'] / 22000.0
    df['rssi_normalized'] = (df['rssi'] + 95) / 45.0
    df['signal_quality'] = df['rssi_normalized'] * 0.4 + df['lqi_normalized'] * 0.3 + df['throughput_normalized'] * 0.3
    
    features_extended = base_features + ['rssi_squared', 'lqi_normalized', 'throughput_normalized', 'rssi_normalized', 'signal_quality']
    
    X = df[features_extended]
    y = df['distance']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train model with improved hyperparameters
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"MAE (Mean Absolute Error): {mae:.2f} meters")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} meters")
    print(f"R² Score: {r2:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    print(f"\n5-Fold CV MAE: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f}")
    
    # Feature importance
    print("\nFeature importance:")
    feature_importance = pd.DataFrame({
        'feature': features_extended,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # Save model
    model_path = MODELS_DIR / "distance_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save metrics
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'cv_mae_mean': float(-cv_scores.mean()),
        'cv_mae_std': float(cv_scores.std()),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    metrics_path = MODELS_DIR / "distance_model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    return metrics

def load_human_presence_data():
    """Load human presence data (LOS vs People)."""
    combined_dir = DATA_DIR / "radio_combined"
    if not combined_dir.exists():
        return None
    
    los_data = []
    people_data = []
    
    # Soccer/LOS data
    for file in combined_dir.glob("Soccer_LOS_*.csv"):
        df = pd.read_csv(file)
        df['has_human'] = 0  # LOS = no human
        los_data.append(df)
    
    # Soccer/people data
    for file in combined_dir.glob("Soccer_people_*.csv"):
        df = pd.read_csv(file)
        df['has_human'] = 1  # People = has human
        people_data.append(df)
    
    if not los_data and not people_data:
        return None
    
    all_data = []
    if los_data:
        all_data.extend(los_data)
    if people_data:
        all_data.extend(people_data)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def train_human_detector():
    """Trains human presence detection model"""
    print("\n" + "=" * 60)
    print("TRAINING HUMAN PRESENCE DETECTION MODEL")
    print("=" * 60)
    
    # Load data
    df = load_human_presence_data()
    if df is None:
        print("Failed to load data, cannot train model")
        return
    
    print(f"Total samples: {len(df)}")
    print(f"Has human: {df['has_human'].sum()}, No human: {(df['has_human']==0).sum()}")
    
    # Feature engineering
    base_features = ['rssi', 'lqi', 'throughput']
    
    # Add derived features
    df['rssi_normalized'] = (df['rssi'] + 95) / 45.0
    df['lqi_normalized'] = df['lqi'] / 107.0
    df['throughput_normalized'] = df['throughput'] / 22000.0
    df['rssi_squared'] = df['rssi'] ** 2
    df['signal_quality'] = df['rssi_normalized'] * 0.4 + df['lqi_normalized'] * 0.3 + df['throughput_normalized'] * 0.3
    
    features_extended = base_features + ['rssi_normalized', 'lqi_normalized', 'throughput_normalized', 'rssi_squared', 'signal_quality']
    
    X = df[features_extended]
    y = df['has_human']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train model with improved hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Human', 'Has Human']))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    print("\nFeature importance:")
    feature_importance = pd.DataFrame({
        'feature': features_extended,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # Save model
    model_path = MODELS_DIR / "human_detector.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    metrics_path = MODELS_DIR / "human_detector_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    return metrics

def load_location_data():
    """Load device location data (necklace vs pocket)."""
    separate_dir = DATA_DIR / "radio_separate"
    if not separate_dir.exists():
        return None
    
    necklace_data = []
    pocket_data = []
    
    # Necklace verileri
    for file in separate_dir.glob("*necklace*.csv"):
        df = pd.read_csv(file)
        df['location'] = 'necklace'
        necklace_data.append(df)
    
    # Pocket verileri
    for file in separate_dir.glob("*pocket*.csv"):
        df = pd.read_csv(file)
        df['location'] = 'pocket'
        pocket_data.append(df)
    
    if not necklace_data and not pocket_data:
        return None
    
    all_data = []
    if necklace_data:
        all_data.extend(necklace_data)
    if pocket_data:
        all_data.extend(pocket_data)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Özellikleri hazırla (RSSI, LQI, THROUGHPUT'u birleştir)
    # Her dosya farklı ölçüm tipi içeriyor, bunları birleştirmemiz gerekiyor
    # Basitleştirme: Her dosyadan ortalama değerleri al
    location_stats = []
    
    for location in ['necklace', 'pocket']:
        location_df = combined_df[combined_df['location'] == location]
        if len(location_df) > 0:
            # RSSI, LQI, THROUGHPUT dosyalarını ayır
            rssi_df = location_df[location_df['measurement_type'] == 'rssi']
            lqi_df = location_df[location_df['measurement_type'] == 'lqi']
            throughput_df = location_df[location_df['measurement_type'] == 'throughput']
            
            # Dosya bazında grupla ve ortalama al
            for file_path in location_df['file_path'].unique():
                file_df = location_df[location_df['file_path'] == file_path]
                
                rssi_mean = file_df[file_df['measurement_type'] == 'rssi']['value'].mean()
                lqi_mean = file_df[file_df['measurement_type'] == 'lqi']['value'].mean()
                throughput_mean = file_df[file_df['measurement_type'] == 'throughput']['value'].mean()
                
                if not (pd.isna(rssi_mean) or pd.isna(lqi_mean) or pd.isna(throughput_mean)):
                    location_stats.append({
                        'rssi': rssi_mean,
                        'lqi': lqi_mean,
                        'throughput': throughput_mean,
                        'location': location
                    })
    
    if not location_stats:
        return None
    
    return pd.DataFrame(location_stats)

def train_location_classifier():
    """Trains device location classification model"""
    print("\n" + "=" * 60)
    print("TRAINING DEVICE LOCATION CLASSIFICATION MODEL")
    print("=" * 60)
    
    # Load data
    df = load_location_data()
    if df is None:
        print("Failed to load data, cannot train model")
        return
    
    print(f"Total samples: {len(df)}")
    print(f"Necklace: {(df['location']=='necklace').sum()}, Pocket: {(df['location']=='pocket').sum()}")
    
    # Feature engineering
    base_features = ['rssi', 'lqi', 'throughput']
    
    # Add derived features
    df['rssi_normalized'] = (df['rssi'] + 95) / 45.0
    df['lqi_normalized'] = df['lqi'] / 107.0
    df['throughput_normalized'] = df['throughput'] / 22000.0
    df['rssi_squared'] = df['rssi'] ** 2
    df['signal_quality'] = df['rssi_normalized'] * 0.4 + df['lqi_normalized'] * 0.3 + df['throughput_normalized'] * 0.3
    
    features_extended = base_features + ['rssi_normalized', 'lqi_normalized', 'throughput_normalized', 'rssi_squared', 'signal_quality']
    
    X = df[features_extended]
    y = df['location']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train model with improved hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Feature importance
    print("\nFeature importance:")
    feature_importance = pd.DataFrame({
        'feature': features_extended,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # Save model
    model_path = MODELS_DIR / "location_classifier.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    metrics_path = MODELS_DIR / "location_classifier_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    print("Zigbee Radio ML training starting...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")

    # Train all models
    distance_metrics = train_distance_model()
    human_metrics = train_human_detector()
    location_metrics = train_location_classifier()

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    
    # Print summary
    if distance_metrics:
        print(f"\nDistance Model - MAE: {distance_metrics['mae']:.2f}m, R²: {distance_metrics['r2']:.4f}")
    if human_metrics:
        print(f"Human Detection Model - Accuracy: {human_metrics['accuracy']:.4f}")
    if location_metrics:
        print(f"Location Classification Model - Accuracy: {location_metrics['accuracy']:.4f}")

