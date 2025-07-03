# CarML: Used Car Price Prediction for Kenya

This repository contains code, notebooks, and documentation for building an accurate and interpretable ensemble machine learning model for predicting used car prices in Kenya.

## How to Use in Google Colab

1. **Clone this repository in Colab:**
   ```python
   !git clone https://github.com/Riekobrian/carml.git
   %cd carml
   ```
2. **Install dependencies:**
   ```python
   !pip install -r requirements.txt
   ```
3. **Mount Google Drive for data access:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Set your data path, e.g.:
   DATA_PATH = '/content/drive/MyDrive/Alpha_GO_dataset_consolidation/data/final/'
   ```
4. **Run notebooks as usual.**

## Project Structure
- `data/` - Data files (not included in repo; upload to Google Drive)
- `models/` - Model artifacts (not included in repo)
- `notebooks/` - Jupyter/Colab notebooks
- `src/` - Source code modules
- `requirements.txt` - Python dependencies

## Notes
- Large files (data, models) are not tracked in git. Use Google Drive for storage.
- For Colab, always check that file paths point to your Google Drive mount.
