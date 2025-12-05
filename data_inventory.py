"""
Veri Envanteri ve Manifest Oluşturma Modülü
Tüm .dat dosyalarını tarar, içeriklerini analiz eder ve kategorize eder.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import re


class DataInventory:
    """Veri seti envanteri ve kategorizasyon sınıfı"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.inventory = []
        self.categories = {
            'radio_measurement_separate': [],  # LQI.dat, RSSI.dat, THROUGHPUT.dat
            'radio_measurement_combined': []   # 5m.dat, 10m.dat gibi birleşik dosyalar
        }
    
    def scan_all_dat_files(self) -> List[Dict]:
        """Tüm .dat dosyalarını tarar ve temel bilgileri toplar"""
        dat_files = list(self.root_path.rglob("*.dat"))
        
        for file_path in dat_files:
            try:
                file_info = self._analyze_file(file_path)
                self.inventory.append(file_info)
            except Exception as e:
                print(f"Hata: {file_path} analiz edilemedi: {e}")
                self.inventory.append({
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'directory': str(file_path.parent),
                    'category': 'error',
                    'error': str(e)
                })
        
        return self.inventory
    
    def _analyze_file(self, file_path: Path) -> Dict:
        """Bir dosyayı analiz eder ve kategorisini belirler"""
        file_info = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'directory': str(file_path.relative_to(self.root_path)),
            'file_size': file_path.stat().st_size,
            'category': None,
            'format_type': None,
            'num_columns': None,
            'num_rows': None,
            'sample_data': None
        }
        
        # Dosya adına göre ilk tahmin
        category = self._categorize_by_filename(file_path.name)
        
        # Dosya içeriğini okuyarak doğrula
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines()[:50] if line.strip()]
            
            if lines:
                # İlk geçerli satırı analiz et
                first_line = lines[0]
                parts = first_line.split()
                
                file_info['num_columns'] = len(parts)
                file_info['num_rows'] = len([l for l in lines if l and not l.startswith('#')])
                file_info['sample_data'] = lines[:5]
                
                # İçeriğe göre kategori doğrulama
                category = self._categorize_by_content(lines, file_path.name)
                file_info['category'] = category
                file_info['format_type'] = self._determine_format_type(lines)
                
        except Exception as e:
            file_info['category'] = 'error'
            file_info['error'] = str(e)
        
        return file_info
    
    def _categorize_by_filename(self, filename: str) -> str:
        """Dosya adına göre kategori tahmini"""
        filename_lower = filename.lower()
        
        if any(x in filename_lower for x in ['lqi', 'rssi', 'throughput']):
            return 'radio_measurement_separate'
        elif re.match(r'^\d+m\.dat$', filename_lower):
            return 'radio_measurement_combined'
        else:
            return 'radio_measurement_separate'  # Default to separate for unknown files
    
    def _categorize_by_content(self, lines: List[str], filename: str) -> str:
        """Dosya içeriğine göre kategori belirleme"""
        if not lines:
            return 'other'
        
        # Dosya adına göre öncelik ver (özellikle radyo ölçümleri için)
        filename_lower = filename.lower()
        if any(x in filename_lower for x in ['lqi', 'rssi', 'throughput']):
            # Dosya adı radyo ölçümü gösteriyorsa, içerikte 3 sütunlu satır ara
            sample_lines = [l for l in lines if l and not l.startswith('#')]
            for line in sample_lines:
                parts = line.split()
                if len(parts) == 3:
                    # 3 sütunlu satır bulundu, radyo ölçümü olabilir
                    if all(self._is_numeric(p) for p in parts):
                        return 'radio_measurement_separate'
            # 3 sütunlu satır bulunamadı ama dosya adı radyo ölçümü gösteriyor
            # Yine de radyo ölçümü kategorisine koy (cleaner daha esnek)
            return 'radio_measurement_separate'
        
        # İlk birkaç satırı analiz et
        sample_lines = [l for l in lines[:10] if l and not l.startswith('#')]
        if not sample_lines:
            return 'other'
        
        # Sütun sayısına göre
        first_line_parts = sample_lines[0].split()
        num_cols = len(first_line_parts)
        
        # Sayısal değer kontrolü
        try:
            # Tüm değerlerin sayısal olup olmadığını kontrol et
            numeric_count = sum(1 for part in first_line_parts if self._is_numeric(part))
            
            if num_cols == 3 and numeric_count == 3:
                # 3 sütun: zaman, değer, standart_sapma (ayrı radyo ölçümleri)
                if any(x in filename_lower for x in ['lqi', 'rssi', 'throughput']):
                    return 'radio_measurement_separate'
            
            elif num_cols == 4 and numeric_count == 4:
                # 4 sütun: zaman, RSSI, LQI, THROUGHPUT (birleşik radyo ölçümleri)
                return 'radio_measurement_combined'
            
            elif num_cols == 2 and numeric_count == 2:
                # 2 sütun: muhtemelen koordinat - assign to radio_measurement_separate
                return 'radio_measurement_separate'
            
            # Log formatı kontrolü (metin içerikli)
            if any(char in sample_lines[0].lower() for char in [':', '[', ']', 'timestamp']):
                return 'radio_measurement_separate'
                
        except:
            pass
        
        return 'other'
    
    def _determine_format_type(self, lines: List[str]) -> str:
        """Dosya format tipini belirler"""
        if not lines:
            return 'unknown'
        
        sample = lines[0] if lines else ""
        parts = sample.split()
        
        if len(parts) == 3:
            return 'time_value_stddev'
        elif len(parts) == 4:
            return 'time_rssi_lqi_throughput'
        elif len(parts) == 2:
            return 'coordinate_pair'
        else:
            return 'custom'
    
    def _is_numeric(self, value: str) -> bool:
        """Bir değerin sayısal olup olmadığını kontrol eder"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def categorize_files(self):
        """Dosyaları kategorilere göre gruplar"""
        for item in self.inventory:
            category = item.get('category', 'radio_measurement_separate')
            if category in self.categories:
                self.categories[category].append(item)
            else:
                self.categories['radio_measurement_separate'].append(item)
    
    def generate_manifest(self, output_path: str = None) -> pd.DataFrame:
        """Manifest CSV dosyası oluşturur"""
        if not output_path:
            output_path = self.root_path / "data_manifest.csv"
        
        df = pd.DataFrame(self.inventory)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"Manifest oluşturuldu: {output_path}")
        print(f"Toplam dosya sayısı: {len(df)}")
        print(f"\nKategori dağılımı:")
        print(df['category'].value_counts())
        
        return df
    
    def get_category_summary(self) -> Dict:
        """Kategori özeti döndürür"""
        summary = {}
        for category, files in self.categories.items():
            summary[category] = {
                'count': len(files),
                'files': [f['file_name'] for f in files]
            }
        return summary

