# Zigbee Radio ML Service - FastAPI

Bu proje, Zigbee radyo ölçüm verileri için Machine Learning modellerini FastAPI ile sunan bir REST API servisidir.

## Özellikler

- ✅ **Mesafe Tahmini**: RSSI/LQI/THROUGHPUT'tan mesafe tahmini
- ✅ **İnsan Varlığı Tespiti**: LOS vs People ayrımı
- ✅ **Cihaz Konumu Sınıflandırması**: Necklace vs Pocket
- ✅ **Sinyal Kalitesi Skorlama**: 0-100 arası kalite skoru
- ✅ **Anomali Tespiti**: Anormal sinyal davranışları
- ✅ **Sinyal Tahmini**: Time series ile gelecek tahmini

## Kurulum

```bash
# Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

## Kullanım

### Sunucuyu Başlatma

```bash
cd fastapi_ml_service
python -m uvicorn app.main:app --reload
```

veya

```bash
python app/main.py
```

API dokümantasyonu: http://localhost:8000/docs

### API Endpoint'leri

#### 1. Mesafe Tahmini
```bash
curl -X POST "http://localhost:8000/predict/distance" \
  -H "Content-Type: application/json" \
  -d '{
    "rssi": -75.5,
    "lqi": 105.2,
    "throughput": 20000
  }'
```

#### 2. İnsan Varlığı Tespiti
```bash
curl -X POST "http://localhost:8000/detect/human-presence" \
  -H "Content-Type: application/json" \
  -d '{
    "rssi": -82.3,
    "lqi": 98.5,
    "throughput": 15000
  }'
```

#### 3. Cihaz Konumu Sınıflandırması
```bash
curl -X POST "http://localhost:8000/classify/device-location" \
  -H "Content-Type: application/json" \
  -d '{
    "rssi": -78.2,
    "lqi": 103.5,
    "throughput": 19500
  }'
```

#### 4. Sinyal Kalitesi Skorlama
```bash
curl -X POST "http://localhost:8000/score/signal-quality" \
  -H "Content-Type: application/json" \
  -d '{
    "rssi": -75.5,
    "lqi": 105.2,
    "throughput": 20000
  }'
```

#### 5. Anomali Tespiti
```bash
curl -X POST "http://localhost:8000/detect/anomaly" \
  -H "Content-Type: application/json" \
  -d '{
    "rssi": -95.5,
    "lqi": 50.2,
    "throughput": 500
  }'
```

#### 6. Sinyal Tahmini
```bash
curl -X POST "http://localhost:8000/predict/signal-quality" \
  -H "Content-Type: application/json" \
  -d '{
    "history": [
      {"timestamp": 0, "rssi": -75, "lqi": 105, "throughput": 20000},
      {"timestamp": 15, "rssi": -78, "lqi": 103, "throughput": 19500},
      {"timestamp": 30, "rssi": -80, "lqi": 101, "throughput": 19000}
    ],
    "future_steps": 3
  }'
```

## Model Eğitimi

Gerçek ML modellerini eğitmek için `train_models.py` scriptini kullanabilirsiniz (oluşturulacak).

```bash
python train_models.py
```

Eğitilmiş modeller `models/` klasörüne kaydedilir ve API otomatik olarak yükler.

## Docker ile Çalıştırma

```bash
docker build -t zigbee-ml-service .
docker run -p 8000:8000 zigbee-ml-service
```

## Geliştirme

### Yeni Model Ekleme

1. `app/models/` klasörüne yeni model dosyası ekleyin
2. `app/main.py` içinde endpoint ekleyin
3. Model eğitim scriptini güncelleyin

### Test

```bash
pytest tests/
```

## Lisans

Bu proje, Zigbee radyo veri analizi için geliştirilmiştir.

