# Data and Processing

This project expects cleaned CSV files under `cleaned_data/` produced by the preprocessing pipeline.

Key locations
- `cleaned_data/data_manifest.csv` - inventory of processed files
- `cleaned_data/metadata.csv` - per-file metadata from preprocessing
- `cleaned_data/processing_report.json` - summary statistics and processing report
- `cleaned_data/radio_combined/` - combined measurements (e.g., `Soccer_LOS_10m.csv`)
- `cleaned_data/radio_separate/` - separate measurement files (RSSI, LQI, THROUGHPUT variants)

Processing pipeline
- `data_inventory.py` scans raw `.dat` files and builds the manifest.
- `data_cleaners.py` cleans, validates, and converts `.dat` files into standardized CSVs.
- `main_processor.py` orchestrates transformation and writes cleaned outputs.

If you need to re-run preprocessing:

```powershell
python .\main_processor.py
```

Notes on data quality
- Check `processing_report.json` for dropped rows and failures.
- For ML tasks, ensure enough samples per class and balanced datasets for classification.
