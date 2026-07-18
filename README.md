# NBA-all-team-prediction

**NBA-all-team-prediction** is a machine learning project focused on forecasting the 25 players who will be selected for the All-NBA (First, Second, Third) and All-Rookie (First, Second) teams for the 2025/2026 season. 

The dataset is built using the `nba_api` to collect base and advanced player statistics across 11 seasons (2015–2026). To ensure accurate predictions across different eras of play, the data undergoes rigorous feature engineering, including per-season Z-score normalization to handle statistical inflation, and the implementation of custom expert features that reflect voting criteria. The pipeline also dynamically applies the NBA's 65-game eligibility rule, carefully accounting for official exceptions to avoid falsely filtering out legitimate candidates.

At its core, the project utilizes a hybrid modeling strategy. It concurrently trains a **LightGBM (LambdaRank)** model to accurately sort players within a season, and a **HistGradientBoosting Regressor** to evaluate their absolute performance tier. By blending these predictions and assigning higher training weights to recent seasons, the model effectively adapts to modern basketball trends and voting logic
