"""
Veri Temizleme ve Formatlama Modülleri
Her kategori için özel temizleme işlemleri
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


class BaseCleaner:
    """Temel temizleme sınıfı - tüm cleaner'lar için ortak fonksiyonlar"""
    
    def __init__(self):
        self.stats = {
            'total_rows': 0,
            'cleaned_rows': 0,
            'dropped_rows': 0,
            'errors': []
        }
    
    def clean_numeric_column(self, series: pd.Series, drop_na: bool = True) -> pd.Series:
        """Sayısal sütunu temizler"""
        # String'leri sayıya çevir
        series = pd.to_numeric(series, errors='coerce')
        
        if drop_na:
            self.stats['dropped_rows'] += series.isna().sum()
            series = series.dropna()
        
        return series
    
    def remove_outliers_iqr(self, series: pd.Series, factor: float = 1.5) -> pd.Series:
        """IQR yöntemiyle aykırı değerleri kaldırır"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        mask = (series >= lower_bound) & (series <= upper_bound)
        dropped = (~mask).sum()
        self.stats['dropped_rows'] += dropped
        
        return series[mask]
    
    def get_stats(self) -> Dict:
        """İstatistikleri döndürür"""
        return self.stats.copy()


class RadioMeasurementSeparateCleaner(BaseCleaner):
    """Ayrı radyo ölçüm dosyaları için temizleyici (LQI.dat, RSSI.dat, THROUGHPUT.dat)"""
    
    def __init__(self, measurement_type: str = 'auto'):
        super().__init__()
        self.measurement_type = measurement_type  # 'lqi', 'rssi', 'throughput', 'auto'
    
    def clean(self, input_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Ayrı radyo ölçüm dosyasını temizler ve CSV'ye dönüştürür
        
        Format: zaman değer stddev (3 sütun) veya değişken sütun sayısı
        Çıkış: timestamp, value, stddev, measurement_type
        """
        try:
            # Dosyayı satır satır oku ve sadece 3 sütunlu satırları al
            # (Bazı dosyalarda başlangıçta sadece zaman değerleri olabilir)
            valid_rows = []
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) == 3:  # Sadece 3 sütunlu satırları al
                            try:
                                # Tüm değerlerin sayısal olduğunu kontrol et
                                float(parts[0])
                                float(parts[1])
                                float(parts[2])
                                valid_rows.append(parts)
                            except ValueError:
                                continue
            
            if not valid_rows:
                raise ValueError("Hiç geçerli 3 sütunlu satır bulunamadı")
            
            # DataFrame oluştur
            df = pd.DataFrame(valid_rows, columns=['timestamp', 'value', 'stddev'])
            
            self.stats['total_rows'] = len(valid_rows)
            
            # Ölçüm tipini belirle
            if self.measurement_type == 'auto':
                filename = Path(input_path).name.lower()
                if 'lqi' in filename:
                    self.measurement_type = 'lqi'
                elif 'rssi' in filename:
                    self.measurement_type = 'rssi'
                elif 'throughput' in filename:
                    self.measurement_type = 'throughput'
                else:
                    self.measurement_type = 'unknown'
            
            # Sütunları temizle
            df['timestamp'] = self.clean_numeric_column(df['timestamp'], drop_na=False)
            df['value'] = self.clean_numeric_column(df['value'], drop_na=False)
            df['stddev'] = self.clean_numeric_column(df['stddev'], drop_na=False)
            
            # Geçersiz satırları kaldır
            initial_len = len(df)
            df = df.dropna()
            self.stats['dropped_rows'] += (initial_len - len(df))
            
            # Aykırı değerleri kaldır (value sütunu için)
            if len(df) > 10:  # Yeterli veri varsa
                initial_len = len(df)
                df = df[~df['value'].isna()]
                df['value'] = self.remove_outliers_iqr(df['value'], factor=2.0)
                df = df[~df['value'].isna()]
                self.stats['dropped_rows'] += (initial_len - len(df))
            
            # Ölçüm tipini ekle
            df['measurement_type'] = self.measurement_type
            
            # Sütun sırasını düzenle
            df = df[['timestamp', 'measurement_type', 'value', 'stddev']]
            
            # Zaman serisi için sırala
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.stats['cleaned_rows'] = len(df)
            
            # CSV'ye kaydet
            if output_path:
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"Temizlenmiş dosya kaydedildi: {output_path}")
            
            return df
            
        except Exception as e:
            error_msg = f"Hata ({input_path}): {str(e)}"
            self.stats['errors'].append(error_msg)
            print(error_msg)
            return pd.DataFrame()


class RadioMeasurementCombinedCleaner(BaseCleaner):
    """Birleşik radyo ölçüm dosyaları için temizleyici (5m.dat, 10m.dat, vb.)"""
    
    def clean(self, input_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Birleşik radyo ölçüm dosyasını temizler ve CSV'ye dönüştürür
        
        Format: zaman RSSI LQI THROUGHPUT (4 sütun)
        Çıkış: timestamp, rssi, lqi, throughput
        """
        try:
            # Dosyayı oku
            df = pd.read_csv(input_path, sep='\s+', header=None,
                           names=['timestamp', 'rssi', 'lqi', 'throughput'],
                           comment='#', skip_blank_lines=True)
            
            self.stats['total_rows'] = len(df)
            
            # Tüm sütunları temizle
            df['timestamp'] = self.clean_numeric_column(df['timestamp'], drop_na=False)
            df['rssi'] = self.clean_numeric_column(df['rssi'], drop_na=False)
            df['lqi'] = self.clean_numeric_column(df['lqi'], drop_na=False)
            df['throughput'] = self.clean_numeric_column(df['throughput'], drop_na=False)
            
            # Geçersiz satırları kaldır
            initial_len = len(df)
            df = df.dropna()
            self.stats['dropped_rows'] += (initial_len - len(df))
            
            # Aykırı değerleri kaldır (her ölçüm için ayrı ayrı)
            if len(df) > 10:
                for col in ['rssi', 'lqi', 'throughput']:
                    initial_len = len(df)
                    df[col] = self.remove_outliers_iqr(df[col], factor=2.0)
                    df = df[~df[col].isna()]
                    self.stats['dropped_rows'] += (initial_len - len(df))
            
            # Zaman serisi için sırala
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.stats['cleaned_rows'] = len(df)
            
            # CSV'ye kaydet
            if output_path:
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"Temizlenmiş dosya kaydedildi: {output_path}")
            
            return df
            
        except Exception as e:
            error_msg = f"Hata ({input_path}): {str(e)}"
            self.stats['errors'].append(error_msg)
            print(error_msg)
            return pd.DataFrame()


class MobilityTraceCleaner(BaseCleaner):
    """Mobilite izleri için temizleyici"""
    
    def clean(self, input_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Mobilite izi dosyasını temizler ve CSV'ye dönüştürür
        
        Format: x y veya timestamp x y (2-3 sütun)
        Çıkış: timestamp, x, y (veya latitude, longitude)
        """
        try:
            # Dosyayı oku - önce formatı tespit et
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                num_cols = len(first_line.split())
            
            if num_cols == 2:
                df = pd.read_csv(input_path, sep='\s+', header=None,
                               names=['x', 'y'],
                               comment='#', skip_blank_lines=True)
                # Timestamp ekle (sıralı indeks)
                df['timestamp'] = df.index
            elif num_cols == 3:
                df = pd.read_csv(input_path, sep='\s+', header=None,
                               names=['timestamp', 'x', 'y'],
                               comment='#', skip_blank_lines=True)
            else:
                raise ValueError(f"Beklenmeyen sütun sayısı: {num_cols}")
            
            self.stats['total_rows'] = len(df)
            
            # Sütunları temizle
            df['timestamp'] = self.clean_numeric_column(df['timestamp'], drop_na=False)
            df['x'] = self.clean_numeric_column(df['x'], drop_na=False)
            df['y'] = self.clean_numeric_column(df['y'], drop_na=False)
            
            # Geçersiz satırları kaldır
            initial_len = len(df)
            df = df.dropna()
            self.stats['dropped_rows'] += (initial_len - len(df))
            
            # Sütun sırasını düzenle
            df = df[['timestamp', 'x', 'y']]
            
            # Zaman serisi için sırala
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.stats['cleaned_rows'] = len(df)
            
            # CSV'ye kaydet
            if output_path:
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"Temizlenmiş dosya kaydedildi: {output_path}")
            
            return df
            
        except Exception as e:
            error_msg = f"Hata ({input_path}): {str(e)}"
            self.stats['errors'].append(error_msg)
            print(error_msg)
            return pd.DataFrame()


class NetworkLogCleaner(BaseCleaner):
    """Ağ log dosyaları için temizleyici"""
    
    def clean(self, input_path: str, output_path: str = None, 
              log_format: str = 'auto') -> pd.DataFrame:
        """
        Ağ log dosyasını temizler ve structured CSV'ye dönüştürür
        
        Format: Değişken (timestamp, IP, port, packet_size, vb.)
        Çıkış: Structured CSV
        """
        try:
            # Basit log formatı için - satır satır oku
            lines = []
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        lines.append(line)
            
            self.stats['total_rows'] = len(lines)
            
            # Log formatını otomatik tespit et ve parse et
            # Bu basit bir implementasyon - gerçek log formatına göre genişletilebilir
            parsed_data = []
            for i, line in enumerate(lines):
                try:
                    # Basit parsing - gerçek log formatına göre özelleştirilebilir
                    # Örnek: timestamp IP:port packet_size
                    parts = line.split()
                    if len(parts) >= 2:
                        parsed_data.append({
                            'line_number': i + 1,
                            'raw_line': line,
                            'timestamp': parts[0] if self._is_numeric(parts[0]) else None,
                            'data': ' '.join(parts[1:])
                        })
                except:
                    self.stats['dropped_rows'] += 1
                    continue
            
            df = pd.DataFrame(parsed_data)
            
            if len(df) == 0:
                raise ValueError("Hiç veri parse edilemedi")
            
            self.stats['cleaned_rows'] = len(df)
            
            # CSV'ye kaydet
            if output_path:
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"Temizlenmiş dosya kaydedildi: {output_path}")
            
            return df
            
        except Exception as e:
            error_msg = f"Hata ({input_path}): {str(e)}"
            self.stats['errors'].append(error_msg)
            print(error_msg)
            return pd.DataFrame()
    
    def _is_numeric(self, value: str) -> bool:
        """Bir değerin sayısal olup olmadığını kontrol eder"""
        try:
            float(value)
            return True
        except ValueError:
            return False

