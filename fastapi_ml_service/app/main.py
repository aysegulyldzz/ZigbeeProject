"""
FastAPI ML Service - Zigbee Radio Data Analysis
Main FastAPI application
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from jinja2 import Template
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os

# Model importları (eğer varsa)
try:
    from app.models.distance_predictor import DistancePredictor
    from app.models.human_detector import HumanDetector
    from app.models.location_classifier import LocationClassifier
    from app.models.signal_predictor import SignalPredictor
    from app.models.anomaly_detector import AnomalyDetector
except ImportError:
    # Model dosyaları henüz oluşturulmadıysa
    pass

app = FastAPI(
    title="Zigbee Radio ML Service",
    description="Machine Learning API service for Zigbee radio measurement data",
    version="1.0.0"
)

# Static files
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic modelleri
class DistanceRequest(BaseModel):
    rssi: float = Field(..., description="Received Signal Strength Indicator (dBm)", example=-75.5)
    lqi: float = Field(..., description="Link Quality Indicator (0-107)", example=105.2)
    throughput: float = Field(..., description="Data throughput (bytes/s)", example=20000.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "rssi": -75.5,
                "lqi": 105.2,
                "throughput": 20000.0
            }
        }

class DistanceResponse(BaseModel):
    distance: float
    confidence: float
    unit: str = "meters"

class HumanPresenceRequest(BaseModel):
    rssi: float = Field(..., description="Received Signal Strength Indicator (dBm)", example=-82.3)
    lqi: float = Field(..., description="Link Quality Indicator (0-107)", example=98.5)
    throughput: float = Field(..., description="Data throughput (bytes/s)", example=15000.0)
    timestamp: Optional[float] = Field(None, description="Timestamp (optional)", example=45.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "rssi": -82.3,
                "lqi": 98.5,
                "throughput": 15000.0,
                "timestamp": 45.0
            }
        }

class HumanPresenceResponse(BaseModel):
    has_human: bool
    confidence: float

class LocationRequest(BaseModel):
    rssi: float = Field(..., description="Received Signal Strength Indicator (dBm)", example=-78.2)
    lqi: float = Field(..., description="Link Quality Indicator (0-107)", example=103.5)
    throughput: float = Field(..., description="Data throughput (bytes/s)", example=19500.0)
    rssi_stddev: Optional[float] = Field(None, description="RSSI standard deviation (optional)", example=1.2)
    
    class Config:
        json_schema_extra = {
            "example": {
                "rssi": -78.2,
                "lqi": 103.5,
                "throughput": 19500.0,
                "rssi_stddev": 1.2
            }
        }

class LocationResponse(BaseModel):
    location: str
    confidence: float
    possible_locations: Optional[dict] = None

class SignalQualityRequest(BaseModel):
    rssi: float = Field(..., description="Received Signal Strength Indicator (dBm)", example=-75.5)
    lqi: float = Field(..., description="Link Quality Indicator (0-107)", example=105.2)
    throughput: float = Field(..., description="Data throughput (bytes/s)", example=20000.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "rssi": -75.5,
                "lqi": 105.2,
                "throughput": 20000.0
            }
        }

class SignalQualityResponse(BaseModel):
    quality_score: float
    grade: str
    breakdown: Optional[dict] = None

class AnomalyRequest(BaseModel):
    rssi: float = Field(..., description="Received Signal Strength Indicator (dBm)", example=-95.5)
    lqi: float = Field(..., description="Link Quality Indicator (0-107)", example=50.2)
    throughput: float = Field(..., description="Data throughput (bytes/s)", example=500.0)
    timestamp: Optional[float] = Field(None, description="Timestamp (optional)", example=30.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "rssi": -95.5,
                "lqi": 50.2,
                "throughput": 500.0,
                "timestamp": 30.0
            }
        }

class AnomalyResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    reason: Optional[str] = None

class SignalHistoryPoint(BaseModel):
    timestamp: float = Field(..., description="Timestamp", example=0.0)
    rssi: float = Field(..., description="Received Signal Strength Indicator (dBm)", example=-75.0)
    lqi: float = Field(..., description="Link Quality Indicator (0-107)", example=105.0)
    throughput: float = Field(..., description="Data throughput (bytes/s)", example=20000.0)

class SignalPredictionRequest(BaseModel):
    history: List[SignalHistoryPoint] = Field(..., description="Historical signal data points")
    future_steps: int = Field(3, description="Number of future steps to predict", example=3)
    
    class Config:
        json_schema_extra = {
            "example": {
                "history": [
                    {"timestamp": 0.0, "rssi": -75.0, "lqi": 105.0, "throughput": 20000.0},
                    {"timestamp": 15.0, "rssi": -78.0, "lqi": 103.0, "throughput": 19500.0},
                    {"timestamp": 30.0, "rssi": -80.0, "lqi": 101.0, "throughput": 19000.0}
                ],
                "future_steps": 3
            }
        }

class SignalPredictionResponse(BaseModel):
    predictions: List[dict]

# Model loading (lazy loading)
models = {}

def load_models():
    """Loads models (if available)"""
    models_dir = Path(__file__).parent.parent / "models"
    if models_dir.exists():
        # Distance prediction model
        distance_model_path = models_dir / "distance_model.pkl"
        if distance_model_path.exists():
            models['distance'] = joblib.load(distance_model_path)
        
        # Human detection model
        human_model_path = models_dir / "human_detector.pkl"
        if human_model_path.exists():
            models['human'] = joblib.load(human_model_path)
        
        # Location classification model
        location_model_path = models_dir / "location_classifier.pkl"
        if location_model_path.exists():
            models['location'] = joblib.load(location_model_path)
        
        # Anomaly detection model
        anomaly_model_path = models_dir / "anomaly_detector.pkl"
        if anomaly_model_path.exists():
            models['anomaly'] = joblib.load(anomaly_model_path)

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()
    print("FastAPI ML Service started!")
    print(f"Loaded models: {list(models.keys())}")

# Root endpoint - Web Dashboard
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    html_file = templates_path / "index.html"
    if html_file.exists():
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

# API Root endpoint (JSON)
@app.get("/api")
async def api_root():
    return {
        "message": "Zigbee Radio ML Service API",
        "version": "1.0.0",
        "endpoints": {
            "distance": "/predict/distance",
            "human_presence": "/detect/human-presence",
            "device_location": "/classify/device-location",
            "signal_quality": "/score/signal-quality",
            "anomaly": "/detect/anomaly",
            "signal_prediction": "/predict/signal-quality"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(models.keys())
    }

# 1. Mesafe Tahmini
@app.post(
    "/predict/distance",
    response_model=DistanceResponse,
    summary="Distance Estimation",
    description="""
    Estimates distance from RSSI, LQI, and THROUGHPUT values.
    
    **Parameters:**
    - **rssi**: Received Signal Strength Indicator (dBm) - Typically between -50 and -95
    - **lqi**: Link Quality Indicator - Between 0 and 107
    - **throughput**: Data transfer rate (bytes/s) - Typically between 0 and 22000
    
    **Returns:**
    - **distance**: Predicted distance (meters)
    - **confidence**: Confidence score (0-1)
    - **unit**: Unit (meters)
    """
)
async def predict_distance(request: DistanceRequest):
    try:
        # Use model if loaded
        if 'distance' in models:
            model = models['distance']
            # Prepare features with feature engineering
            rssi = request.rssi
            lqi = request.lqi
            throughput = request.throughput
            rssi_squared = rssi ** 2
            lqi_normalized = lqi / 107.0
            throughput_normalized = throughput / 22000.0
            rssi_normalized = (rssi + 95) / 45.0
            signal_quality = rssi_normalized * 0.4 + lqi_normalized * 0.3 + throughput_normalized * 0.3
            
            features = np.array([[rssi, lqi, throughput, rssi_squared, lqi_normalized, 
                                 throughput_normalized, rssi_normalized, signal_quality]])
            distance = model.predict(features)[0]
            confidence = 0.85  # Model confidence can be calculated
        else:
            # Simple physical model (path loss model)
            # d = 10^((TxPower - RSSI) / (10 * n))
            # n: path loss exponent (2-4)
            tx_power = -30  # dBm (example value)
            n = 2.5  # Path loss exponent
            distance = 10 ** ((tx_power - request.rssi) / (10 * n))
            # Adjust based on LQI
            if request.lqi < 80:
                distance *= 1.5
            elif request.lqi > 100:
                distance *= 0.8
            confidence = 0.75
        
        return DistanceResponse(
            distance=round(distance, 2),
            confidence=round(confidence, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# 2. İnsan Varlığı Tespiti
@app.post(
    "/detect/human-presence",
    response_model=HumanPresenceResponse,
    summary="Human Presence Detection",
    description="""
    Detects human presence from RSSI, LQI, and THROUGHPUT values.
    
    **Parameters:**
    - **rssi**: Received Signal Strength Indicator (dBm)
    - **lqi**: Link Quality Indicator (0-107)
    - **throughput**: Data transfer rate (bytes/s)
    - **timestamp**: Timestamp (optional)
    
    **Description:**
    - LOS (Line of Sight) means no human present
    - People scenario means human present
    - RSSI is generally lower when human is present
    
    **Returns:**
    - **has_human**: Is human present? (True/False)
    - **confidence**: Confidence score (0-1)
    """
)
async def detect_human_presence(request: HumanPresenceRequest):
    try:
        if 'human' in models:
            model = models['human']
            # Prepare features with feature engineering
            rssi = request.rssi
            lqi = request.lqi
            throughput = request.throughput
            rssi_normalized = (rssi + 95) / 45.0
            lqi_normalized = lqi / 107.0
            throughput_normalized = throughput / 22000.0
            rssi_squared = rssi ** 2
            signal_quality = rssi_normalized * 0.4 + lqi_normalized * 0.3 + throughput_normalized * 0.3
            
            features = np.array([[rssi, lqi, throughput, rssi_normalized, lqi_normalized, 
                                 throughput_normalized, rssi_squared, signal_quality]])
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            confidence = max(proba)
            has_human = bool(prediction)
        else:
            # Simple rule-based approach
            # Human presence generally results in lower RSSI, higher variance
            # LOS has higher and more stable RSSI
            if request.rssi < -85 and request.lqi < 95:
                has_human = True
                confidence = 0.75
            elif request.rssi > -75 and request.lqi > 100:
                has_human = False
                confidence = 0.80
            else:
                # Ambiguous case
                has_human = request.rssi < -80
                confidence = 0.60
        
        return HumanPresenceResponse(
            has_human=has_human,
            confidence=round(confidence, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

# 3. Cihaz Konumu Sınıflandırması
@app.post(
    "/classify/device-location",
    response_model=LocationResponse,
    summary="Device Location Classification",
    description="""
    Classifies device location (necklace vs pocket).
    
    **Parameters:**
    - **rssi**: Received Signal Strength Indicator (dBm)
    - **lqi**: Link Quality Indicator (0-107)
    - **throughput**: Data transfer rate (bytes/s)
    - **rssi_stddev**: RSSI standard deviation (optional)
    
    **Description:**
    - **Necklace**: On neck, generally better signal quality
    - **Pocket**: In pocket, signal may be weaker
    
    **Returns:**
    - **location**: Detected location ("necklace" or "pocket")
    - **confidence**: Confidence score (0-1)
    - **possible_locations**: Probability scores for each location
    """
)
async def classify_device_location(request: LocationRequest):
    try:
        if 'location' in models:
            model = models['location']
            # Prepare features with feature engineering
            rssi = request.rssi
            lqi = request.lqi
            throughput = request.throughput
            rssi_normalized = (rssi + 95) / 45.0
            lqi_normalized = lqi / 107.0
            throughput_normalized = throughput / 22000.0
            rssi_squared = rssi ** 2
            signal_quality = rssi_normalized * 0.4 + lqi_normalized * 0.3 + throughput_normalized * 0.3
            
            features = np.array([[rssi, lqi, throughput, rssi_normalized, lqi_normalized, 
                                 throughput_normalized, rssi_squared, signal_quality]])
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            confidence = max(proba)
            location = prediction
            possible_locations = dict(zip(model.classes_, proba))
        else:
            # Simple rule-based approach
            # Necklace generally has better signal quality
            if request.lqi > 100 and request.rssi > -80:
                location = "necklace"
                confidence = 0.80
            elif request.lqi < 95 or request.rssi < -85:
                location = "pocket"
                confidence = 0.75
            else:
                location = "necklace"  # Default
                confidence = 0.60
            
            possible_locations = {
                "necklace": confidence if location == "necklace" else 1 - confidence,
                "pocket": 1 - confidence if location == "necklace" else confidence
            }
        
        return LocationResponse(
            location=location,
            confidence=round(confidence, 2),
            possible_locations=possible_locations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

# 4. Sinyal Kalitesi Skorlama
@app.post(
    "/score/signal-quality",
    response_model=SignalQualityResponse,
    summary="Signal Quality Scoring",
    description="""
    Generates overall signal quality score from RSSI, LQI, and THROUGHPUT values.
    
    **Parameters:**
    - **rssi**: Received Signal Strength Indicator (dBm)
    - **lqi**: Link Quality Indicator (0-107)
    - **throughput**: Data transfer rate (bytes/s)
    
    **Returns:**
    - **quality_score**: Overall quality score (0-100)
    - **grade**: Quality level ("poor", "fair", "good", "excellent")
    - **breakdown**: Individual scores for each feature (rssi_score, lqi_score, throughput_score)
    
    **Scoring:**
    - 80-100: excellent
    - 60-79: good
    - 40-59: fair
    - 0-39: poor
    """
)
async def score_signal_quality(request: SignalQualityRequest):
    try:
        # RSSI scoring (-50: excellent, -95: poor)
        rssi_score = max(0, min(100, ((request.rssi + 95) / (-50 + 95)) * 100))
        
        # LQI scoring (107: excellent, 0: poor)
        lqi_score = (request.lqi / 107) * 100
        
        # THROUGHPUT scoring (22000: excellent, 0: poor)
        throughput_score = min(100, (request.throughput / 22000) * 100)
        
        # Weighted average
        quality_score = (rssi_score * 0.4 + lqi_score * 0.3 + throughput_score * 0.3)
        
        # Determine grade
        if quality_score >= 80:
            grade = "excellent"
        elif quality_score >= 60:
            grade = "good"
        elif quality_score >= 40:
            grade = "fair"
        else:
            grade = "poor"
        
        breakdown = {
            "rssi_score": round(rssi_score, 2),
            "lqi_score": round(lqi_score, 2),
            "throughput_score": round(throughput_score, 2)
        }
        
        return SignalQualityResponse(
            quality_score=round(quality_score, 2),
            grade=grade,
            breakdown=breakdown
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

# 5. Anomali Tespiti
@app.post(
    "/detect/anomaly",
    response_model=AnomalyResponse,
    summary="Anomaly Detection",
    description="""
    Detects abnormal signal behaviors.
    
    **Parameters:**
    - **rssi**: Received Signal Strength Indicator (dBm)
    - **lqi**: Link Quality Indicator (0-107)
    - **throughput**: Data transfer rate (bytes/s)
    - **timestamp**: Timestamp (optional)
    
    **Anomaly Criteria:**
    - Very low RSSI (< -95 dBm)
    - Very low LQI (< 50)
    - Very low THROUGHPUT (< 1000 bytes/s)
    - Combinations outside normal ranges (e.g., high RSSI but low throughput)
    
    **Returns:**
    - **is_anomaly**: Is anomaly present? (True/False)
    - **anomaly_score**: Anomaly score (0-1, higher = more abnormal)
    - **reason**: Anomaly reason description (if available)
    """
)
async def detect_anomaly(request: AnomalyRequest):
    try:
        if 'anomaly' in models:
            model = models['anomaly']
            features = np.array([[request.rssi, request.lqi, request.throughput]])
            anomaly_score = model.decision_function(features)[0]
            is_anomaly = model.predict(features)[0] == -1
            # Skoru 0-1 aralığına normalize et
            normalized_score = 1 / (1 + np.exp(-anomaly_score))
        else:
            # Simple rule-based anomaly detection
            anomaly_flags = []
            
            if request.rssi < -95:
                anomaly_flags.append("Very low RSSI")
            if request.lqi < 50:
                anomaly_flags.append("Very low LQI")
            if request.throughput < 1000:
                anomaly_flags.append("Very low throughput")
            
            # Inconsistency check
            if request.rssi < -90 and request.lqi > 100:
                anomaly_flags.append("Inconsistent RSSI-LQI")
            if request.rssi > -70 and request.throughput < 5000:
                anomaly_flags.append("Inconsistent RSSI-throughput")
            
            is_anomaly = len(anomaly_flags) > 0
            anomaly_score = min(1.0, len(anomaly_flags) * 0.3)
            reason = "; ".join(anomaly_flags) if anomaly_flags else None
        
        return AnomalyResponse(
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 2),
            reason=reason if 'reason' in locals() else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection error: {str(e)}")

# 6. Sinyal Kalitesi Tahmini (Time Series)
@app.post(
    "/predict/signal-quality",
    response_model=SignalPredictionResponse,
    summary="Signal Quality Prediction (Time Series)",
    description="""
    Predicts future signal quality from historical signal data.
    
    **Parameters:**
    - **history**: List of historical signal data (minimum 2 data points required)
      - Each data point contains: timestamp, rssi, lqi, throughput
    - **future_steps**: Number of future steps to predict (default: 3)
    
    **Example Usage:**
    ```json
    {
      "history": [
        {"timestamp": 0, "rssi": -75, "lqi": 105, "throughput": 20000},
        {"timestamp": 15, "rssi": -78, "lqi": 103, "throughput": 19500},
        {"timestamp": 30, "rssi": -80, "lqi": 101, "throughput": 19000}
      ],
      "future_steps": 3
    }
    ```
    
    **Returns:**
    - **predictions**: List of predicted values for future time steps
      - Each prediction contains: timestamp, rssi, lqi, throughput
    
    **Note:** Currently uses simple trend analysis. More accurate predictions can be made when real ML models (LSTM/GRU) are added.
    """
)
async def predict_signal_quality(request: SignalPredictionRequest):
    try:
        if len(request.history) < 2:
            raise HTTPException(
                status_code=400, 
                detail="At least 2 historical data points required"
            )
        
        # Convert history to DataFrame
        history_df = pd.DataFrame([h.dict() for h in request.history])
        
        # Simple linear trend prediction
        predictions = []
        last_timestamp = history_df['timestamp'].iloc[-1]
        time_step = history_df['timestamp'].diff().mean()
        
        # Calculate trends
        rssi_trend = history_df['rssi'].diff().mean()
        lqi_trend = history_df['lqi'].diff().mean()
        throughput_trend = history_df['throughput'].diff().mean()
        
        # Last values
        last_rssi = history_df['rssi'].iloc[-1]
        last_lqi = history_df['lqi'].iloc[-1]
        last_throughput = history_df['throughput'].iloc[-1]
        
        for i in range(1, request.future_steps + 1):
            future_timestamp = last_timestamp + (time_step * i)
            future_rssi = last_rssi + (rssi_trend * i)
            future_lqi = last_lqi + (lqi_trend * i)
            future_throughput = max(0, last_throughput + (throughput_trend * i))
            
            predictions.append({
                "timestamp": round(future_timestamp, 2),
                "rssi": round(future_rssi, 2),
                "lqi": round(future_lqi, 2),
                "throughput": round(future_throughput, 2)
            })
        
        return SignalPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

