# Freight Rate Prediction — SpotterLabs Assessment

Machine learning solution for predicting freight load rates.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

Place the Spotter data files in this directory:

- `train-test.csv`
- `validation.csv`
- `december-chart-inputs.csv` (empty `predicted_rate` column)

Then run:

```bash
python solution.py
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

This produces:

- `validation_predictions.csv` — predictions for all 12,000 validation loads
- `december-chart-inputs.csv` — filled December fixed-lane rates
- `scorer_results/candidate_december.png` — December prediction chart

## Approach

- **Split:** time-based — train Jan–Sep, hold out Oct, predict Nov–Dec
- **Model:** LightGBM + XGBoost ensemble (log-target), inverse-RMSE weighted
- **Hold-out:** RMSE ~$639, MAPE ~6.1%

## Repo contents

| File | Description |
|---|---|
| `solution.py` | Training, feature engineering, and prediction pipeline |
| `score.py` | Provided scorer (validates outputs + December chart) |
| `generate_report.py` | Builds `report.pdf` |
| `validation_predictions.csv` | Final validation predictions |
| `requirements.txt` | Python dependencies |
| `report.pdf` | Written report (split approach + December chart) |
