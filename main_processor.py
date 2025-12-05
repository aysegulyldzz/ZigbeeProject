"""
Ana İşlem Scripti
Tüm modülleri birleştirerek veri setini işler
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime

from data_inventory import DataInventory
from data_cleaners import (
    RadioMeasurementSeparateCleaner,
    RadioMeasurementCombinedCleaner
)


class DataProcessor:
    """Ana veri işleme sınıfı"""
    
    def __init__(self, root_path: str, output_dir: str = None):
        self.root_path = Path(root_path)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.root_path / "cleaned_data"
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Kategori bazlı çıktı klasörleri
        self.category_dirs = {
            'radio_measurement_separate': self.output_dir / 'radio_separate',
            'radio_measurement_combined': self.output_dir / 'radio_combined'
        }
        
        for dir_path in self.category_dirs.values():
            dir_path.mkdir(exist_ok=True, parents=True)
        
        self.inventory = None
        self.metadata = []
        self.processing_report = {
            'start_time': datetime.now().isoformat(),
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'category_stats': {},
            'errors': []
        }
    
    def run_full_pipeline(self):
        """Tam işlem hattını çalıştırır"""
        print("=" * 60)
        print("VERİ İŞLEME HATTI BAŞLATILIYOR")
        print("=" * 60)
        
        # 1. Envanter oluştur
        print("\n[1/4] Veri envanteri oluşturuluyor...")
        inventory_obj = DataInventory(str(self.root_path))
        inventory_obj.scan_all_dat_files()
        inventory_obj.categorize_files()
        self.inventory = inventory_obj
        
        # Manifest kaydet
        manifest_path = self.output_dir / "data_manifest.csv"
        manifest_df = inventory_obj.generate_manifest(str(manifest_path))
        self.processing_report['total_files'] = len(manifest_df)
        
        # 2. Dosyaları kategorilere göre işle
        print("\n[2/4] Dosyalar kategorilere göre işleniyor...")
        self._process_by_category(inventory_obj)
        
        # 3. Metadata tablosu oluştur
        print("\n[3/4] Metadata tablosu oluşturuluyor...")
        self._generate_metadata_table()
        
        # 4. Rapor oluştur
        print("\n[4/4] İşlem raporu oluşturuluyor...")
        self._generate_processing_report()
        
        print("\n" + "=" * 60)
        print("İŞLEM TAMAMLANDI!")
        print("=" * 60)
        print(f"Çıktı klasörü: {self.output_dir}")
        print(f"İşlenen dosya sayısı: {self.processing_report['processed_files']}")
        print(f"Başarısız dosya sayısı: {self.processing_report['failed_files']}")
    
    def _process_by_category(self, inventory_obj: DataInventory):
        """Dosyaları kategorilere göre işler"""
        
        # Radyo ölçümleri - ayrı dosyalar
        print("\n  → Radyo ölçümleri (ayrı) işleniyor...")
        self._process_radio_separate(inventory_obj.categories['radio_measurement_separate'])
        
        # Radyo ölçümleri - birleşik dosyalar
        print("\n  → Radyo ölçümleri (birleşik) işleniyor...")
        self._process_radio_combined(inventory_obj.categories['radio_measurement_combined'])
    
    def _process_radio_separate(self, files: List[Dict]):
        """Ayrı radyo ölçüm dosyalarını işler"""
        cleaner = RadioMeasurementSeparateCleaner()
        
        for file_info in files:
            try:
                input_path = file_info['file_path']
                output_filename = self._generate_output_filename(file_info)
                output_path = self.category_dirs['radio_measurement_separate'] / output_filename
                
                df = cleaner.clean(input_path, str(output_path))
                
                if not df.empty:
                    self.metadata.append({
                        'original_file': file_info['file_path'],
                        'original_filename': file_info['file_name'],
                        'category': 'radio_measurement_separate',
                        'cleaned_file': str(output_path),
                        'cleaned_filename': output_filename,
                        'rows_processed': cleaner.stats['total_rows'],
                        'rows_cleaned': cleaner.stats['cleaned_rows'],
                        'rows_dropped': cleaner.stats['dropped_rows'],
                        'status': 'success'
                    })
                    self.processing_report['processed_files'] += 1
                else:
                    self._record_failure(file_info, 'Boş DataFrame döndü')
                    
            except Exception as e:
                self._record_failure(file_info, str(e))
    
    def _process_radio_combined(self, files: List[Dict]):
        """Birleşik radyo ölçüm dosyalarını işler"""
        cleaner = RadioMeasurementCombinedCleaner()
        
        for file_info in files:
            try:
                input_path = file_info['file_path']
                output_filename = self._generate_output_filename(file_info)
                output_path = self.category_dirs['radio_measurement_combined'] / output_filename
                
                df = cleaner.clean(input_path, str(output_path))
                
                if not df.empty:
                    self.metadata.append({
                        'original_file': file_info['file_path'],
                        'original_filename': file_info['file_name'],
                        'category': 'radio_measurement_combined',
                        'cleaned_file': str(output_path),
                        'cleaned_filename': output_filename,
                        'rows_processed': cleaner.stats['total_rows'],
                        'rows_cleaned': cleaner.stats['cleaned_rows'],
                        'rows_dropped': cleaner.stats['dropped_rows'],
                        'status': 'success'
                    })
                    self.processing_report['processed_files'] += 1
                else:
                    self._record_failure(file_info, 'Boş DataFrame döndü')
                    
            except Exception as e:
                self._record_failure(file_info, str(e))
    
    def _generate_output_filename(self, file_info: Dict) -> str:
        """Temizlenmiş dosya için isim oluşturur"""
        original_name = Path(file_info['file_name']).stem
        directory = file_info.get('directory', '')
        
        # Klasör yapısını dosya adına dahil et
        safe_dir = directory.replace('\\', '_').replace('/', '_').replace(' ', '_')
        if safe_dir:
            output_name = f"{safe_dir}_{original_name}.csv"
        else:
            output_name = f"{original_name}.csv"
        
        return output_name
    
    def _record_failure(self, file_info: Dict, error_msg: str):
        """Başarısız işlemi kaydeder"""
        self.metadata.append({
            'original_file': file_info['file_path'],
            'original_filename': file_info['file_name'],
            'category': file_info.get('category', 'unknown'),
            'cleaned_file': None,
            'cleaned_filename': None,
            'rows_processed': None,
            'rows_cleaned': None,
            'rows_dropped': None,
            'status': 'failed',
            'error': error_msg
        })
        self.processing_report['failed_files'] += 1
        self.processing_report['errors'].append({
            'file': file_info['file_path'],
            'error': error_msg
        })
    
    def _generate_metadata_table(self):
        """Metadata tablosu oluşturur"""
        metadata_df = pd.DataFrame(self.metadata)
        metadata_path = self.output_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False, encoding='utf-8-sig')
        print(f"Metadata tablosu kaydedildi: {metadata_path}")
    
    def _generate_processing_report(self):
        """İşlem raporu oluşturur"""
        # Kategori istatistikleri
        if self.metadata:
            metadata_df = pd.DataFrame(self.metadata)
            self.processing_report['category_stats'] = metadata_df.groupby('category').agg({
                'status': 'count',
                'rows_processed': 'sum',
                'rows_cleaned': 'sum',
                'rows_dropped': 'sum'
            }).to_dict('index')
        
        self.processing_report['end_time'] = datetime.now().isoformat()
        self.processing_report['duration_seconds'] = (
            datetime.fromisoformat(self.processing_report['end_time']) -
            datetime.fromisoformat(self.processing_report['start_time'])
        ).total_seconds()
        
        # JSON raporu kaydet
        report_path = self.output_dir / "processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.processing_report, f, indent=2, ensure_ascii=False)
        
        # İnsan okunabilir rapor
        report_txt_path = self.output_dir / "processing_report.txt"
        with open(report_txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("VERİ İŞLEME RAPORU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Başlangıç Zamanı: {self.processing_report['start_time']}\n")
            f.write(f"Bitiş Zamanı: {self.processing_report['end_time']}\n")
            f.write(f"Süre: {self.processing_report['duration_seconds']:.2f} saniye\n\n")
            f.write(f"Toplam Dosya Sayısı: {self.processing_report['total_files']}\n")
            f.write(f"İşlenen Dosya Sayısı: {self.processing_report['processed_files']}\n")
            f.write(f"Başarısız Dosya Sayısı: {self.processing_report['failed_files']}\n\n")
            
            f.write("Kategori İstatistikleri:\n")
            f.write("-" * 60 + "\n")
            for category, stats in self.processing_report['category_stats'].items():
                f.write(f"\n{category}:\n")
                f.write(f"  Dosya Sayısı: {stats.get('status', 0)}\n")
                f.write(f"  İşlenen Satır: {stats.get('rows_processed', 0)}\n")
                f.write(f"  Temizlenen Satır: {stats.get('rows_cleaned', 0)}\n")
                f.write(f"  Atılan Satır: {stats.get('rows_dropped', 0)}\n")
            
            if self.processing_report['errors']:
                f.write("\n\nHatalar:\n")
                f.write("-" * 60 + "\n")
                for error in self.processing_report['errors']:
                    f.write(f"\nDosya: {error['file']}\n")
                    f.write(f"Hata: {error['error']}\n")
        
        print(f"İşlem raporu kaydedildi: {report_path}")
        print(f"İnsan okunabilir rapor: {report_txt_path}")


if __name__ == "__main__":
    # Kullanım örneği
    root_path = r"C:\Users\ayild\Desktop\ACP Files"
    processor = DataProcessor(root_path)
    processor.run_full_pipeline()

