# SOFE 4620U - Building an ML Trading Agent for Tesla Stocks

## Team 18  
Naftanan Mamo (100822222)  
Calvin Reveredo (100825740)  
Rodney Stanislaus (100822918)

## Files Overview
This repository contains scripts and datasets for processing, analyzing, and predicting Tesla (TSLA) stock prices using historical data from 2010 to 2025. The agent provides evaluation metrics and simulates stock trading suggestions for a specific week in March 2025.

### Data Processing Scripts
- **check_close.py**: Checks if the `Close` and `Adj Close` columns in the original CSV files ever have different values. (They don't.)
- **combine_csv.py**: Combines the 2010-2022 and 2022-2025 CSV files into one dataset. Adjusts stock prices and volumes to account for the 3:1 stock split in August 2022.
- **process_csv.py**: Drops unwanted columns from the combined dataset.
- **process_extra.py**: Adjusts the 2022-2025 dataset by multiplying stock prices by 3 and dividing volumes by 3 to match the 2010-2022 format.

### Datasets
- **TSLA_Combined_processed.csv**: Final processed dataset for 2010-2025.
- **TSLA_Combined.csv**: Combined 2010-2025 dataset with all original headers.
- **TSLA_Part1.csv**: 2010-2022 dataset (same as `TSLA.csv`).
- **TSLA_Part2.csv**: 2022-2025 dataset (reformatted to match the 2010-2022 dataset).
- **TSLA_Post2022.csv**: Raw 2022-2025 dataset copied from Yahoo Finance.
- **TSLA.csv**: Same as `TSLA_Part1.csv`.

### Prediction Scripts
- **TSLA_prediction.py**: Establishes the prediction model and calculates evaluation metrics.
- **TSLA_prediction2.py**: Simulates the agent for making stock trading suggestions for the week of March 24 to March 28, 2025.

## Usage

1. **Data Preparation**:
   - Run `combine_csv.py` to combine and adjust the datasets.
   - Use `process_csv.py` to clean the combined dataset.
   - Use `process_extra.py` to adjust the 2022-2025 dataset if needed.

2. **Model Evaluation**:
   - Run `TSLA_prediction.py` to train the model and evaluate its performance.

3. **Agent Simulation**:
   - Run `TSLA_prediction2.py` to simulate the agent's trading suggestions for the specified week in March 2025.

## Notes
- The 2022-2025 dataset has already been adjusted for the 3:1 stock split. The scripts revert this adjustment to maintain consistency with the 2010-2022 dataset, and the combined dataset once again adjust to be accurate for the current market.
- Ensure all dependencies are installed before running the scripts.

## Dependencies
- Python 3.x
- Libraries: pandas, numpy, scikit-learn
