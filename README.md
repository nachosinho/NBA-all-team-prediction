# NBA Awards Prediction

Project for the Machine Learning course (Poznan University of Technology, Faculty of Automatic Control, Robotics and Electrical Engineering). It utilizes machine learning techniques to predict the rosters for prestigious end-of-regular-season NBA awards: All-NBA Teams and All-Rookie Teams for the 2025/2026 season.

## Author
Sebastian Nachowiak

## Requirements and Installation

The project requires a Python 3.12 environment.

Dependencies can be installed using the pip tool:
```bash
pip install pandas numpy scikit-learn lightgbm nba_api
```

## Project Structure

* `Data_prep.py` - script fetching data from the official NBA API (`leaguedashplayerstats` endpoint), performing an inner join of player stats packages (Base and Advanced) for 11 seasons (2015-2026) and generating the training dataset.
* `nba_labeled_dataset.csv` - dataset enriched with target class labels for All-NBA and All-Rookie awards.
* `Model_nba.py` - main script responsible for feature engineering, training hybrid models, and generating final predictions.

## Running the Project

### Step 1: Downloading data
To build the current dataset, run:
```bash
python Data_prep.py
``` 
The script will create files including `nba_raw_stats_2019_2026.csv` and the target labeled dataset `nba_labeled_dataset.csv`.

### Step 2: Running predictions
After generating the database, run the main algorithm, providing the output JSON file name as an argument:
```bash
python Model_nba.py output_file.json
``` 

As a result of the script execution, the machine learning pipeline is processed (feature engineering, training, prediction), and the predicted rosters will be printed in the console and saved in the specified JSON output file.

## Architecture and Feature Engineering

Modeling is based on advanced statistics processing and a hybrid approach:

* **Hybrid models:** The project trains in parallel a ranking model for establishing the hierarchy within a season (LightGBM with the `lambdarank` objective function, optimizing the NDCG metric) and a classic regression model assigning an absolute player value (HistGradientBoostingRegressor). The final prediction is a weighted average of the normalized scores of both estimators. 
* **All-Rookie Model:** A separate modeling pipeline was implemented to estimate the best rookies, significantly improving accuracy compared to a joint prediction model.
* **Z-score per season standardization:** To mitigate the phenomenon of stats inflation and favoring recent years, a Z-score indicator for the 37 most important metrics is applied, calculated individually within each season.
* **Expert features and ranking inversion:** The `prepare_features` function processes raw stats into synthetic indicators (e.g., `MVP_PROFILE`, `SCORING_LOAD`, defensive efficiency) and extracts availability features. Additionally, league rankings from the API are inverted so that a higher numerical value represents a better result.
* **Season weighting:** To better generalize the latest game trends and the evolution of journalists' voting, the most recent seasons (e.g., 2023-24, 2024-25) are assigned weights of 2.25 during algorithm training.
* **Soft Eligibility (Availability criterion):** Due to the new rule requiring 65 games played, an exponential soft penalty for absences was implemented. This restriction promotes available players in the ranking (multiplier 1.0 for players with >= 65 games) without completely eliminating players who slightly missed the requirements (e.g., 64 games played).
