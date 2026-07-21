import json
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

DATA_PATH = "nba_labeled_dataset.csv"
TARGET_SEASON = "2025-26"
TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
GP_PENALTY_POWER = 6
NBA_RANKER_WEIGHT = 0.65
ROOKIE_RANKER_WEIGHT = 0.60

# Normalizuje wyniki modeli do zakresu 0-1 przed ich połączeniem
def safe_minmax(values):
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if len(values) == 0 or np.nanmax(values) - np.nanmin(values) < 1e-12:
        return np.zeros(values.shape[0])
    return MinMaxScaler().fit_transform(values).flatten()

# Nadaje większą wagę nowszym sezonom podczas treningu
def season_weights(seasons):
    weights = np.ones(len(seasons), dtype=float)
    weights[seasons.isin(["2022-23"])] = 1.5
    weights[seasons.isin(["2023-24", "2024-25", "2025-26"])] = 2.25
    return weights

# Usuwa błędne wartości i uzupełnia braki medianą
def clean_numeric(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))
    return df

#Dodaje nowe cechy do zbioru danych, jeśli wymagane kolumny istnieją
def add_existing_features(df, features):
    for name, func in features.items():
        try:
            df[name] = func(df)
        except KeyError:
            pass

#PRZYGOTOWANIE CECH EKSPERCKICH

def prepare_features(df):
    df = clean_numeric(df.copy().sort_values("SEASON_ID").reset_index(drop=True))
    print("Inżynieria cech: bazowe, głosowanie, defense/winning...")

    rank_features = [c for c in df.columns if c.endswith("_RANK")]
    for col in rank_features:
        df[f"{col}_SCORE"] = df.groupby("SEASON_ID")[col].transform("max") + 1 - df[col]

    add_existing_features(df, {
        "TD3_per_game": lambda x: x["TD3"] / (x["GP"] + 1e-6),  # triple-double na mecz
        "DD2_per_game": lambda x: x["DD2"] / (x["GP"] + 1e-6),  # double-double na mecz
        "PTS_REB_AST": lambda x: x["PTS"] + x["REB"] + x["AST"],  # punkty + zbiórki + asysty
        "STOCKS": lambda x: x["STL"] + x["BLK"],  # Wkład defensywny (steals + blocks)
        "HEALTH_SCORE": lambda x: x["GP"] / (x.groupby("SEASON_ID")["GP"].transform("max") + 1e-6),# dostępność
        "AVAILABILITY": lambda x: x["GP"] * x["MIN"],  # łączna liczba minut w sezonie
        "SCORING_LOAD": lambda x: x["PTS"] * x["USG_PCT"],  # obciążenie punktowaniem
        "CREATION": lambda x: x["PTS"] + 1.5 * x["AST"],  # kreowanie ofensywy
        "TWO_WAY": lambda x: x["PTS"] + x["REB"] + x["AST"] + 2.0 * (x["STL"] + x["BLK"]),# wpływ po obu stronach parkietu
        "EFF_VOLUME": lambda x: x["PTS"] * x["TS_PCT"],  # efektywne zdobywanie punktów
        "TEAM_STAR": lambda x: x["W_PCT"] * x["PIE"],  # wpływ dla mocnej drużyny
        "MVP_PROFILE": lambda x: x["PIE"] * x["W_PCT"] * np.log1p(x["GP"]),  # profil kandydata do MVP
        "HIGH_USAGE_WINNING": lambda x: x["USG_PCT"] * x["W_PCT"],  # lider ofensywy w wygrywającym zespole
        "ON_BALL_VALUE": lambda x: x["PTS"] + 1.2 * x["AST"] - 0.7 * x["TOV"],  # wartość zawodnika z piłką
        "SCORING_EFFICIENCY_LOAD": lambda x: x["PTS"] * x["TS_PCT"] * x["USG_PCT"],  # efektywność przy dużej roli
        "PLUS_MINUS_WINNING": lambda x: x["PLUS_MINUS"] * x["W_PCT"],  # wpływ na sukces drużyny
        "FANTASY_AVAILABILITY": lambda x: x["NBA_FANTASY_PTS"] * np.log1p(x["GP"]),# produkcja przy uwzględnieniu dostępności
        "GP_65_RATIO": lambda x: np.minimum(x["GP"] / 65.0, 1.0),  # dtopień spełnienia progu 65 meczów
        "GP_65_FLAG": lambda x: (x["GP"] >= 65).astype(int),  # spełnia próg 65 meczów
        "GP_60_FLAG": lambda x: (x["GP"] >= 60).astype(int),  # spełnia próg 60 meczów
 })


    core_stats = [
        "PTS", "AST", "REB", "STL", "BLK", "MIN", "USG_PCT", "PIE", "OFF_RATING", "DEF_RATING",
        "NET_RATING", "TS_PCT", "EFG_PCT", "W_PCT", "TD3_per_game", "DD2_per_game", "PTS_REB_AST",
        "STOCKS", "HEALTH_SCORE", "AVAILABILITY", "SCORING_LOAD", "CREATION", "TWO_WAY", "EFF_VOLUME",
        "TEAM_STAR", "MVP_PROFILE", "HIGH_USAGE_WINNING", "ON_BALL_VALUE", "SCORING_EFFICIENCY_LOAD",
        "PLUS_MINUS_WINNING", "FANTASY_AVAILABILITY", "GP_65_RATIO",
    ]
    core_stats = [c for c in core_stats if c in df.columns]

    print("Generowanie z-score per sezon...")
    for stat in core_stats:
        mean = df.groupby("SEASON_ID")[stat].transform("mean")
        std = df.groupby("SEASON_ID")[stat].transform("std")
        df[f"{stat}_zscore"] = (df[stat] - mean) / (std + 1e-6)

    extra = [
        "W_PCT", "GP", "MIN", "TS_PCT", "EFG_PCT", "USG_PCT", "PIE", "TD3_per_game", "DD2_per_game",
        "PTS_REB_AST", "STOCKS", "HEALTH_SCORE", "AVAILABILITY", "SCORING_LOAD", "CREATION", "TWO_WAY",
        "EFF_VOLUME", "TEAM_STAR", "MVP_PROFILE", "HIGH_USAGE_WINNING", "ON_BALL_VALUE", "SCORING_EFFICIENCY_LOAD",
        "PLUS_MINUS_WINNING", "FANTASY_AVAILABILITY", "GP_65_RATIO", "GP_65_FLAG", "GP_60_FLAG",
    ]
    feature_cols = rank_features + [f"{c}_SCORE" for c in rank_features] + [f"{c}_zscore" for c in core_stats] + extra
    feature_cols = [c for c in dict.fromkeys(feature_cols) if c in df.columns]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median(numeric_only=True)).fillna(0)
    return df, feature_cols

# Trenuje model rankingowy LightGBM oraz model regresyjny dla All-NBA lub All-Rookie
def train_models(df_train, feature_cols, target, rookie=False):
    X, y = df_train[feature_cols], df_train[target]
    groups = df_train.groupby("SEASON_ID", sort=False).size().values
    weights = season_weights(df_train["SEASON_ID"])

    if rookie:
        ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
        regressor = HistGradientBoostingRegressor(max_iter=100, max_depth=4, learning_rate=0.05, random_state=42)
    else:
        ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", n_estimators=220, learning_rate=0.035, max_depth=5, num_leaves=21, min_child_samples=20, subsample=0.90, colsample_bytree=0.90, random_state=42, verbose=-1)
        regressor = HistGradientBoostingRegressor(max_iter=220, max_depth=5, learning_rate=0.035, l2_regularization=0.02, random_state=42)

    ranker.fit(X, y, group=groups, sample_weight=weights)
    regressor.fit(X, y, sample_weight=weights)
    return ranker, regressor

# Łączy predykcje modelu rankingowego i regresyjnego w jeden końcowy wynik
def blended_score(ranker, regressor, X, ranker_weight):
    return ranker_weight * safe_minmax(ranker.predict(X)) + (1.0 - ranker_weight) * safe_minmax(regressor.predict(X))

# Miękka kara za niespełnienie progu 65 rozegranych meczów
def soft_eligibility(scores, gp):
    return np.asarray(scores, dtype=float) * (np.clip(gp.values / 65.0, 0.0, 1.0) ** GP_PENALTY_POWER)

# Wybór 15 najbardziej prawdopodobnych kandydatów do All-NBA z odrzuceniem graczy o słabszym profilu statystycznym
def all_nba_candidate_pool(players):
    pool = players.head(25).copy()
    remove = (
        ((pool["PTS"] < 20.0) & (pool["USG_PCT"] < 0.235)) |
        ((pool["PTS"] < 20.0) & (pool["REB"] >= 7.0) & (pool["AST"] < 7.0) & (pool["W_PCT"] < 0.75))
    ) & ~(
        ((pool["W_PCT"] >= 0.70) & (pool["PIE"] >= 0.145)) |
        (pool["AST"] >= 7.5) |
        ((pool["W_PCT"] >= 0.70) & (pool["REB"] >= 8.0) & (pool["BLK"] >= 1.0))
    )
    selected = pool.loc[~remove].copy()
    if len(selected) < 15:
        selected = pd.concat([selected, players.loc[~players["PLAYER_NAME"].isin(selected["PLAYER_NAME"])].head(15 - len(selected))], ignore_index=True)
    return selected.head(15).copy()

# Tworzy końcowe drużyny All-NBA na podstawie wyniku modelu oraz dodatkowych premiowanych cech
def build_all_nba_teams(players):
    selected = all_nba_candidate_pool(players)
    selected["first_team_score"] = (
        selected["pred_nba_score"] + 1.30 * selected["W_PCT"] + 0.060 * selected["PTS"] +
        0.500 * selected["PIE"] + 0.200 * selected["USG_PCT"] + 0.035 * selected["AST"] +
        np.where((selected["W_PCT"] >= 0.70) & (selected["PTS"] >= 23.0), 0.18, 0.0) +
        np.where((selected["AST"] >= 8.5) & (selected["PTS"] >= 23.0), 0.12, 0.0)
    )
    first = selected.sort_values("first_team_score", ascending=False).head(5)
    rest = selected.loc[~selected["PLAYER_NAME"].isin(first["PLAYER_NAME"])].sort_values("pred_nba_score", ascending=False)
    return {
        "first all-nba team": first["PLAYER_NAME"].tolist(),
        "second all-nba team": rest.iloc[:5]["PLAYER_NAME"].tolist(),
        "third all-nba team": rest.iloc[5:10]["PLAYER_NAME"].tolist(),
    }


def main():
    if len(sys.argv) != 2:
        print("Użycie: python model_nba_155142.py /sciezka/do/pliku/Nazwisko_Imie.json")
        sys.exit(1)

    df, feature_cols = prepare_features(pd.read_csv(DATA_PATH))
    train = df[df["SEASON_ID"].isin(TRAIN_SEASONS)].copy().sort_values("SEASON_ID")
    test = df[df["SEASON_ID"] == TARGET_SEASON].copy().reset_index(drop=True)
    if test.empty:
        raise ValueError(f"Brak danych dla sezonu testowego: {TARGET_SEASON}")

    print(f"Liczba cech: {len(feature_cols)}")
    print(f"Trening: {TRAIN_SEASONS}")
    print(f"Sezon testowy: {TARGET_SEASON}")

    X_test = test[feature_cols]
    meta_cols = ["PLAYER_NAME", "is_rookie", "GP", "W_PCT", "PIE", "USG_PCT", "PTS", "REB", "AST", "BLK"]
    pred = test[[c for c in meta_cols if c in test.columns]].copy()
    pred["eligible_for_nba"] = (test["GP"] >= 65).astype(int)

    print("\nTrenowanie modeli All-NBA...")
    nba_ranker, nba_regressor = train_models(train, feature_cols, "all_nba_target")
    pred["pred_nba_score"] = soft_eligibility(blended_score(nba_ranker, nba_regressor, X_test, NBA_RANKER_WEIGHT), test["GP"])

    print("Trenowanie modeli All-Rookie...")
    rookies_train = train[train["is_rookie"] == 1].copy()
    rookie_ranker, rookie_regressor = train_models(rookies_train, feature_cols, "all_rookie_target", rookie=True)
    pred["pred_rookie_score"] = blended_score(rookie_ranker, rookie_regressor, X_test, ROOKIE_RANKER_WEIGHT)

    all_nba = pred.sort_values("pred_nba_score", ascending=False).reset_index(drop=True)
    rookies = pred[pred["is_rookie"] == 1].sort_values("pred_rookie_score", ascending=False).reset_index(drop=True)
    results = build_all_nba_teams(all_nba)
    results.update({
        "first rookie all-nba team": rookies.iloc[:5]["PLAYER_NAME"].tolist(),
        "second rookie all-nba team": rookies.iloc[5:10]["PLAYER_NAME"].tolist(),
    })

    print("\nTOP 25 All-NBA według modelu:")
    print(all_nba[["PLAYER_NAME", "pred_nba_score", "GP", "W_PCT", "PIE", "USG_PCT", "PTS", "REB", "AST", "BLK", "eligible_for_nba"]].head(25).to_string(index=False))
    print("\nTOP 12 All-Rookie według modelu:")
    print(rookies[["PLAYER_NAME", "pred_rookie_score", "GP"]].head(12).to_string(index=False))

    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nPredykcje zapisane do pliku: {sys.argv[1]}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
