# Data Directory

This directory contains raw data files (CSV, JSON, Parquet).

**Note:** Raw data files are NOT tracked in git (see `.gitignore`).

## Data Sources

1. **LinkedIn Sales Navigator exports** - CSV files with profile data
2. **LemList API exports** - JSON files with campaign outcomes

## Usage

- Use `scripts/fetch_lemlist_data.py` to fetch data from LemList API
- Export LinkedIn profiles from Sales Navigator manually

## Data Privacy

⚠️ **IMPORTANT:** Ensure personal data is anonymized before storing in database.
- Hash or remove LinkedIn URLs
- Anonymize names if storing long-term
- GDPR compliance required
