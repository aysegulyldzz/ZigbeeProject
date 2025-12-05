# API Reference

Base URL: `/` on your local server (default `http://localhost:8000`)

Available endpoints

- `GET /` - Web dashboard (if `templates/index.html` exists)
- `GET /api` - Basic API information
- `GET /health` - Health check and loaded models

Prediction endpoints

- `POST /predict/distance` - Distance estimation
  - Request: `rssi`, `lqi`, `throughput`
  - Response: `distance`, `confidence`, `unit`

- `POST /detect/human-presence` - Human presence detection
  - Request: `rssi`, `lqi`, `throughput`, optional `timestamp`
  - Response: `has_human`, `confidence`

- `POST /classify/device-location` - Necklace vs Pocket classification
  - Request: `rssi`, `lqi`, `throughput`, optional `rssi_stddev`
  - Response: `location`, `confidence`, `possible_locations`

- `POST /score/signal-quality` - Signal quality scoring
  - Request: `rssi`, `lqi`, `throughput`
  - Response: `quality_score`, `grade`, `breakdown`

- `POST /detect/anomaly` - Anomaly detection
  - Request: `rssi`, `lqi`, `throughput`, optional `timestamp`
  - Response: `is_anomaly`, `anomaly_score`, `reason`

- `POST /predict/signal-quality` - Time-series prediction
  - Request: `history` (list of timestamped points), `future_steps`
  - Response: `predictions` (timestamps and predicted values)

Request and response models are defined using Pydantic in `fastapi_ml_service/app/main.py`.

Examples

Use the Swagger UI (`/docs`) to view and test request/response schemas interactively.
