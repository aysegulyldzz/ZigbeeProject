# Zigbee Radyo Verileri için ML Senaryoları ve FastAPI Uygulamaları

## Veri Seti Özeti

Veri setiniz şu özelliklere sahip:
- **RSSI** (Received Signal Strength Indicator): -50 ile -95 dBm arası
- **LQI** (Link Quality Indicator): 0-107 arası
- **THROUGHPUT**: 0-22000 bytes/s arası
- **Senaryolar**: Hallway (farklı vücutlar, insan geçişi, köşe dönüşleri), SideWalk, Soccer
- **Konumlar**: necklace, pocket, LOS (Line of Sight), people
- **Mesafe Verileri**: Soccer verilerinde 5m, 10m, 20m, 30m, 40m, 50m, 60m

---

## Önerilen ML Senaryoları

### 1. **Mesafe Tahmin Modeli** 📏
**Amaç**: RSSI ve LQI değerlerinden mesafe tahmini

**Kullanılacak Veriler**:
- Soccer/LOS verileri (mesafe bilgisi mevcut: 5m, 10m, 20m, 30m, 40m, 50m, 60m)
- Input: RSSI, LQI, THROUGHPUT
- Output: Mesafe (metre)

**Önerilen Modeller**:
- **Random Forest Regressor**: Non-linear ilişkileri yakalama
- **XGBoost Regressor**: Yüksek performans, feature importance
- **Neural Network**: Derin öğrenme ile kompleks pattern'ler
- **Polynomial Regression**: Basit ve hızlı

**Gerçek Kullanım Alanları**:
- 📍 **Indoor Positioning Systems (IPS)**: Bina içi konumlandırma
- 🏭 **IoT Asset Tracking**: Depo/üretim tesislerinde eşya takibi
- 🏥 **Hastane Hasta Takibi**: Tıbbi cihazların konumlandırılması
- 🛒 **Akıllı Mağazalar**: Müşteri davranış analizi

**API Endpoint Örneği**:
```
POST /predict/distance
{
  "rssi": -75.5,
  "lqi": 105.2,
  "throughput": 20000
}
→ {"distance": 12.5, "confidence": 0.89}
```

---

### 2. **İnsan Varlığı Tespiti** 👤
**Amaç**: LOS (Line of Sight) vs People verilerinden insan varlığını tespit etme

**Kullanılacak Veriler**:
- Soccer/LOS vs Soccer/people karşılaştırması
- Input: RSSI, LQI, THROUGHPUT, zaman
- Output: Binary classification (0: LOS, 1: People)

**Önerilen Modeller**:
- **Random Forest Classifier**: Feature importance ile hangi özelliklerin önemli olduğunu görebilme
- **XGBoost Classifier**: Yüksek doğruluk
- **SVM (Support Vector Machine)**: Küçük veri setlerinde iyi performans
- **Neural Network**: Kompleks pattern'leri öğrenme

**Gerçek Kullanım Alanları**:
- 🚪 **Akıllı Kapı Sistemleri**: İnsan yaklaştığında otomatik açılma
- 💡 **Enerji Yönetimi**: İnsan varlığına göre aydınlatma kontrolü
- 🏢 **Bina Yönetim Sistemleri**: Oda doluluk oranı takibi
- 🚨 **Güvenlik Sistemleri**: Yetkisiz giriş tespiti

**API Endpoint Örneği**:
```
POST /detect/human-presence
{
  "rssi": -82.3,
  "lqi": 98.5,
  "throughput": 15000,
  "timestamp": 45
}
→ {"has_human": true, "confidence": 0.92}
```

---

### 3. **Cihaz Konumu Sınıflandırması** 📱
**Amaç**: Necklace vs Pocket konumlarını ayırt etme

**Kullanılacak Veriler**:
- Hallway ve SideWalk verilerindeki necklace/pocket karşılaştırması
- Input: RSSI, LQI, THROUGHPUT, stddev
- Output: Cihaz konumu (necklace/pocket)

**Önerilen Modeller**:
- **Random Forest Classifier**: Feature importance
- **Gradient Boosting**: Yüksek doğruluk
- **Neural Network**: Non-linear ilişkiler

**Gerçek Kullanım Alanları**:
- 👕 **Akıllı Giyilebilir Cihazlar**: Cihazın vücut üzerindeki konumunu tespit
- 🏃 **Spor Uygulamaları**: Aktivite tipine göre cihaz konumu optimizasyonu
- 📊 **Kullanıcı Davranış Analizi**: Cihaz kullanım alışkanlıkları

**API Endpoint Örneği**:
```
POST /classify/device-location
{
  "rssi": -78.2,
  "lqi": 103.5,
  "throughput": 19500,
  "rssi_stddev": 1.2
}
→ {"location": "necklace", "confidence": 0.87}
```

---

### 4. **Sinyal Kalitesi Tahmini (Time Series)** 📈
**Amaç**: Gelecekteki RSSI/LQI/THROUGHPUT değerlerini tahmin etme

**Kullanılacak Veriler**:
- Tüm zaman serisi verileri
- Input: Geçmiş N zaman adımındaki RSSI, LQI, THROUGHPUT
- Output: Sonraki zaman adımındaki değerler

**Önerilen Modeller**:
- **LSTM (Long Short-Term Memory)**: Zaman serisi için ideal
- **GRU (Gated Recurrent Unit)**: LSTM'den daha hızlı
- **ARIMA**: Klasik zaman serisi modeli
- **Prophet**: Facebook'un zaman serisi modeli

**Gerçek Kullanım Alanları**:
- 📡 **Ağ Optimizasyonu**: Sinyal kalitesi düşmeden önce önlem alma
- 🔄 **Proaktif Bakım**: Cihaz arızalarını önceden tespit
- 📊 **Kapasite Planlama**: Ağ yükünü önceden tahmin etme
- ⚡ **Adaptif Güç Yönetimi**: Sinyal kalitesine göre güç ayarlama

**API Endpoint Örneği**:
```
POST /predict/signal-quality
{
  "history": [
    {"timestamp": 0, "rssi": -75, "lqi": 105, "throughput": 20000},
    {"timestamp": 15, "rssi": -78, "lqi": 103, "throughput": 19500},
    {"timestamp": 30, "rssi": -80, "lqi": 101, "throughput": 19000}
  ],
  "future_steps": 3
}
→ {
  "predictions": [
    {"timestamp": 45, "rssi": -82.5, "lqi": 99.2, "throughput": 18500},
    {"timestamp": 60, "rssi": -84.1, "lqi": 97.8, "throughput": 18000}
  ]
}
```

---

### 5. **Senaryo Sınıflandırması** 🏃
**Amaç**: Hallway, SideWalk, Soccer gibi farklı senaryoları ayırt etme

**Kullanılacak Veriler**:
- Tüm senaryo verileri
- Input: RSSI, LQI, THROUGHPUT, istatistiksel özellikler (mean, std, min, max)
- Output: Senaryo tipi (hallway, sidewalk, soccer)

**Önerilen Modeller**:
- **Random Forest Classifier**: Feature importance
- **XGBoost Classifier**: Yüksek doğruluk
- **SVM**: Küçük veri setlerinde iyi
- **Neural Network**: Kompleks pattern'ler

**Gerçek Kullanım Alanları**:
- 🏃 **Aktivite Tanıma**: Kullanıcının ne yaptığını tespit
- 🗺️ **Ortam Tanıma**: İç mekan vs dış mekan ayrımı
- 📊 **Veri Analizi**: Senaryo bazlı performans karşılaştırması

**API Endpoint Örneği**:
```
POST /classify/scenario
{
  "rssi_mean": -78.5,
  "lqi_mean": 102.3,
  "throughput_mean": 19500,
  "rssi_std": 2.1,
  "measurements": [...]
}
→ {"scenario": "hallway", "confidence": 0.91}
```

---

### 6. **Anomali Tespiti** 🚨
**Amaç**: Normal olmayan sinyal davranışlarını tespit etme

**Kullanılacak Veriler**:
- Tüm veri seti (normal davranış öğrenmek için)
- Input: RSSI, LQI, THROUGHPUT, zaman
- Output: Anomali skoru (0-1)

**Önerilen Modeller**:
- **Isolation Forest**: Hızlı ve etkili
- **One-Class SVM**: Küçük veri setlerinde iyi
- **Autoencoder (Neural Network)**: Kompleks pattern'ler
- **DBSCAN Clustering**: Density-based anomali tespiti

**Gerçek Kullanım Alanları**:
- 🚨 **Güvenlik**: Yetkisiz cihaz tespiti
- 🔧 **Arıza Tespiti**: Cihaz arızalarını erken tespit
- 📡 **Ağ Saldırı Tespiti**: Anormal trafik pattern'leri
- ⚠️ **Kalite Kontrol**: Üretim hatası tespiti

**API Endpoint Örneği**:
```
POST /detect/anomaly
{
  "rssi": -95.5,
  "lqi": 50.2,
  "throughput": 500,
  "timestamp": 30
}
→ {"is_anomaly": true, "anomaly_score": 0.87, "reason": "Low signal quality"}
```

---

### 7. **Sinyal Kalitesi Skorlama** ⭐
**Amaç**: RSSI, LQI, THROUGHPUT'u birleştirerek genel sinyal kalitesi skoru üretme

**Kullanılacak Veriler**:
- Tüm veri seti
- Input: RSSI, LQI, THROUGHPUT
- Output: 0-100 arası kalite skoru

**Önerilen Modeller**:
- **Ensemble Methods**: Birden fazla modeli birleştirme
- **Weighted Scoring**: Domain knowledge ile ağırlıklandırma
- **Neural Network**: End-to-end öğrenme

**Gerçek Kullanım Alanları**:
- 📊 **Ağ İzleme Dashboard**: Tek bir metrik ile durum görüntüleme
- 🔄 **Otomatik Yönlendirme**: En iyi sinyal kalitesine sahip cihaza yönlendirme
- 📈 **Performans Raporlama**: Kullanıcı dostu metrikler

**API Endpoint Örneği**:
```
POST /score/signal-quality
{
  "rssi": -75.5,
  "lqi": 105.2,
  "throughput": 20000
}
→ {"quality_score": 85.3, "grade": "excellent"}
```

---

## FastAPI Uygulama Mimarisi

### Önerilen Klasör Yapısı:
```
fastapi_ml_service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI uygulaması
│   ├── models/                 # ML modelleri
│   │   ├── distance_predictor.py
│   │   ├── human_detector.py
│   │   ├── location_classifier.py
│   │   ├── signal_predictor.py
│   │   └── anomaly_detector.py
│   ├── schemas/                # Pydantic modelleri
│   │   └── requests.py
│   ├── services/               # İş mantığı
│   │   └── ml_service.py
│   └── utils/                  # Yardımcı fonksiyonlar
│       └── data_loader.py
├── models/                     # Eğitilmiş model dosyaları
│   ├── distance_model.pkl
│   ├── human_detector.pkl
│   └── ...
├── requirements.txt
└── README.md
```

---

## Model Eğitimi için Öneriler

### Veri Hazırlama:
1. **Feature Engineering**:
   - Zaman bazlı özellikler (rolling mean, std, min, max)
   - Mesafe bilgisi (Soccer verilerinden)
   - Senaryo etiketleri (directory path'ten)

2. **Veri Bölme**:
   - Train: 70%
   - Validation: 15%
   - Test: 15%

3. **Cross-Validation**:
   - Time-series veriler için TimeSeriesSplit kullanın
   - 5-fold cross-validation

### Model Seçimi Stratejisi:
1. **Basit modellerle başlayın** (Linear Regression, Random Forest)
2. **Feature importance** analizi yapın
3. **Hyperparameter tuning** (GridSearch/RandomSearch)
4. **Ensemble methods** deneyin
5. **Deep learning** sadece yeterli veri varsa

---

## Performans Metrikleri

### Regression (Mesafe, Sinyal Tahmini):
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**

### Classification (İnsan Tespiti, Konum):
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

### Anomali Tespiti:
- **Precision@K**
- **AUC-ROC**
- **F1-Score**

---

## Gerçek Dünya Entegrasyonu

### IoT Cihaz Entegrasyonu:
- **MQTT Broker**: Gerçek zamanlı veri akışı
- **WebSocket**: Canlı tahminler
- **REST API**: Batch işlemler

### Deployment:
- **Docker**: Containerization
- **Kubernetes**: Scaling
- **Redis**: Model caching
- **PostgreSQL**: Tahmin geçmişi

### Monitoring:
- **Prometheus**: Metrik toplama
- **Grafana**: Dashboard
- **ELK Stack**: Log analizi

---

## Sonuç

Bu veri seti ile **7 farklı ML senaryosu** geliştirilebilir. En pratik ve değerli olanlar:

1. ✅ **Mesafe Tahmini** - Indoor positioning için kritik
2. ✅ **İnsan Varlığı Tespiti** - Akıllı bina uygulamaları
3. ✅ **Sinyal Kalitesi Tahmini** - Proaktif ağ yönetimi

Bu senaryoları FastAPI ile RESTful API olarak sunarak, IoT cihazlarından ve web uygulamalarından kolayca erişilebilir hale getirebilirsiniz.

