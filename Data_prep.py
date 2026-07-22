import os
import time
import json
import sqlite3
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguedashplayerstats

# --- Config ---
SEASONS = ["2015-16","2016-17","2017-18","2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
RAW_CSV_PATH = "nba_raw_stats_2019_2026.csv"
LABELED_CSV_PATH = "nba_labeled_dataset.csv"


# Historical data
HISTORICAL_AWARDS = {
    "2010-11": {
        "first_all_nba": ["Derrick Rose", "Kobe Bryant", "LeBron James", "Kevin Durant", "Dwight Howard"],
        "second_all_nba": ["Russell Westbrook", "Dwyane Wade", "Dirk Nowitzki", "Amar'e Stoudemire", "Pau Gasol"],
        "third_all_nba": ["Chris Paul", "Manu Ginóbili", "LaMarcus Aldridge", "Zach Randolph", "Al Horford"],
        "first_rookie": ["Blake Griffin", "John Wall", "Landry Fields", "DeMarcus Cousins", "Gary Neal"],
        "second_rookie": ["Greg Monroe", "Wesley Johnson", "Eric Bledsoe", "Derrick Favors", "Paul George"]
    },
    "2011-12": {
        "first_all_nba": ["Chris Paul", "Kobe Bryant", "LeBron James", "Kevin Durant", "Dwight Howard"],
        "second_all_nba": ["Tony Parker", "Russell Westbrook", "Blake Griffin", "Kevin Love", "Andrew Bynum"],
        "third_all_nba": ["Dwyane Wade", "Rajon Rondo", "Carmelo Anthony", "Dirk Nowitzki", "Tyson Chandler"],
        "first_rookie": ["Kyrie Irving", "Ricky Rubio", "Kenneth Faried", "Klay Thompson", "Kawhi Leonard"],
        "second_rookie": ["Isaiah Thomas", "MarShon Brooks", "Chandler Parsons", "Tristan Thompson", "Derrick Williams"]
    },
    "2012-13": {
        "first_all_nba": ["Chris Paul", "Kobe Bryant", "LeBron James", "Kevin Durant", "Tim Duncan"],
        "second_all_nba": ["Tony Parker", "Russell Westbrook", "Carmelo Anthony", "Blake Griffin", "Marc Gasol"],
        "third_all_nba": ["Dwyane Wade", "James Harden", "Paul George", "David Lee", "Dwight Howard"],
        "first_rookie": ["Damian Lillard", "Bradley Beal", "Anthony Davis", "Dion Waiters", "Harrison Barnes"],
        "second_rookie": ["Andre Drummond", "Jonas Valančiūnas", "Michael Kidd-Gilchrist", "Kyle Singler", "Tyler Zeller"]
    },
    "2013-14": {
        "first_all_nba": ["Chris Paul", "James Harden", "LeBron James", "Kevin Durant", "Joakim Noah"],
        "second_all_nba": ["Tony Parker", "Stephen Curry", "Blake Griffin", "Kevin Love", "Dwight Howard"],
        "third_all_nba": ["Damian Lillard", "Goran Dragić", "Paul George", "LaMarcus Aldridge", "Al Jefferson"],
        "first_rookie": ["Michael Carter-Williams", "Victor Oladipo", "Trey Burke", "Mason Plumlee", "Tim Hardaway Jr."],
        "second_rookie": ["Kelly Olynyk", "Giannis Antetokounmpo", "Gorgui Dieng", "Cody Zeller", "Steven Adams"]
    },
    "2014-15": {
        "first_all_nba": ["Stephen Curry", "James Harden", "LeBron James", "Anthony Davis", "Marc Gasol"],
        "second_all_nba": ["Chris Paul", "Russell Westbrook", "LaMarcus Aldridge", "DeMarcus Cousins", "Pau Gasol"],
        "third_all_nba": ["Kyrie Irving", "Klay Thompson", "Blake Griffin", "Tim Duncan", "DeAndre Jordan"],
        "first_rookie": ["Andrew Wiggins", "Nikola Mirotić", "Nerlens Noel", "Elfrid Payton", "Jordan Clarkson"],
        "second_rookie": ["Marcus Smart", "Zach LaVine", "Bojan Bogdanović", "Jusuf Nurkić", "Langston Galloway"]
    },
    "2015-16": {
        "first_all_nba": ["Stephen Curry", "Russell Westbrook", "LeBron James", "Kawhi Leonard", "DeAndre Jordan"],
        "second_all_nba": ["Chris Paul", "Damian Lillard", "Kevin Durant", "Draymond Green", "DeMarcus Cousins"],
        "third_all_nba": ["Kyle Lowry", "Klay Thompson", "Paul George", "LaMarcus Aldridge", "Andre Drummond"],
        "first_rookie": ["Karl-Anthony Towns", "Kristaps Porziņģis", "Devin Booker", "Nikola Jokić", "Jahlil Okafor"],
        "second_rookie": ["Justise Winslow", "D'Angelo Russell", "Emmanuel Mudiay", "Myles Turner", "Willie Cauley-Stein"]
    },
    "2016-17": {
        "first_all_nba": ["James Harden", "Russell Westbrook", "LeBron James", "Kawhi Leonard", "Anthony Davis"],
        "second_all_nba": ["Stephen Curry", "Isaiah Thomas", "Giannis Antetokounmpo", "Kevin Durant", "Rudy Gobert"],
        "third_all_nba": ["John Wall", "DeMar DeRozan", "Jimmy Butler III", "Draymond Green", "DeAndre Jordan"],
        "first_rookie": ["Malcolm Brogdon", "Dario Šarić", "Joel Embiid", "Buddy Hield", "Willy Hernangómez"],
        "second_rookie": ["Jamal Murray", "Jaylen Brown", "Marquese Chriss", "Brandon Ingram", "Yogi Ferrell"]
    },
    "2017-18": {
        "first_all_nba": ["Damian Lillard", "James Harden", "LeBron James", "Kevin Durant", "Anthony Davis"],
        "second_all_nba": ["Russell Westbrook", "DeMar DeRozan", "Giannis Antetokounmpo", "LaMarcus Aldridge", "Joel Embiid"],
        "third_all_nba": ["Stephen Curry", "Victor Oladipo", "Jimmy Butler III", "Paul George", "Karl-Anthony Towns"],
        "first_rookie": ["Ben Simmons", "Donovan Mitchell", "Jayson Tatum", "Kyle Kuzma", "Lauri Markkanen"],
        "second_rookie": ["Dennis Smith Jr.", "Lonzo Ball", "John Collins", "Bogdan Bogdanović", "Josh Jackson"]
    },
    "2018-19": {
        "first_all_nba": ["Stephen Curry", "James Harden", "Giannis Antetokounmpo", "Paul George", "Nikola Jokić"],
        "second_all_nba": ["Damian Lillard", "Kyrie Irving", "Kawhi Leonard", "Kevin Durant", "Joel Embiid"],
        "third_all_nba": ["Russell Westbrook", "Kemba Walker", "Blake Griffin", "LeBron James", "Rudy Gobert"],
        "first_rookie": ["Luka Dončić", "Trae Young", "Deandre Ayton", "Jaren Jackson Jr.", "Marvin Bagley III"],
        "second_rookie": ["Shai Gilgeous-Alexander", "Collin Sexton", "Landry Shamet", "Mitchell Robinson", "Kevin Huerter"]
    },
    "2019-20": {
        "first_all_nba": ["LeBron James", "Giannis Antetokounmpo", "Anthony Davis", "James Harden", "Luka Dončić"],
        "second_all_nba": ["Kawhi Leonard", "Nikola Jokić", "Pascal Siakam", "Damian Lillard", "Chris Paul"],
        "third_all_nba": ["Jayson Tatum", "Jimmy Butler III", "Rudy Gobert", "Ben Simmons", "Russell Westbrook"],
        "first_rookie": ["Ja Morant", "Kendrick Nunn", "Brandon Clarke", "Zion Williamson", "Eric Paschall"],
        "second_rookie": ["Tyler Herro", "Terence Davis", "Coby White", "P.J. Washington", "Rui Hachimura"]
    },
    "2020-21": {
        "first_all_nba": ["Nikola Jokić", "Giannis Antetokounmpo", "Kawhi Leonard", "Stephen Curry", "Luka Dončić"],
        "second_all_nba": ["Joel Embiid", "Julius Randle", "LeBron James", "Chris Paul", "Damian Lillard"],
        "third_all_nba": ["Rudy Gobert", "Jimmy Butler III", "Paul George", "Bradley Beal", "Kyrie Irving"],
        "first_rookie": ["LaMelo Ball", "Anthony Edwards", "Tyrese Haliburton", "Saddiq Bey", "Jae'Sean Tate"],
        "second_rookie": ["Immanuel Quickley", "Desmond Bane", "Isaiah Stewart", "Isaac Okoro", "Patrick Williams"]
    },
    "2021-22": {
        "first_all_nba": ["Nikola Jokić", "Giannis Antetokounmpo", "Jayson Tatum", "Luka Dončić", "Devin Booker"],
        "second_all_nba": ["Joel Embiid", "Ja Morant", "Kevin Durant", "Stephen Curry", "DeMar DeRozan"],
        "third_all_nba": ["Karl-Anthony Towns", "LeBron James", "Chris Paul", "Trae Young", "Pascal Siakam"],
        "first_rookie": ["Scottie Barnes", "Cade Cunningham", "Evan Mobley", "Franz Wagner", "Jalen Green"],
        "second_rookie": ["Herbert Jones", "Josh Giddey", "Bones Hyland", "Ayo Dosunmu", "Chris Duarte"]
    },
    "2022-23": {
        "first_all_nba": ["Joel Embiid", "Giannis Antetokounmpo", "Jayson Tatum", "Luka Dončić", "Shai Gilgeous-Alexander"],
        "second_all_nba": ["Nikola Jokić", "Donovan Mitchell", "Stephen Curry", "Jimmy Butler III", "Jaylen Brown"],
        "third_all_nba": ["Domantas Sabonis", "De'Aaron Fox", "Damian Lillard", "LeBron James", "Julius Randle"],
        "first_rookie": ["Paolo Banchero", "Jalen Williams", "Walker Kessler", "Keegan Murray", "Bennedict Mathurin"],
        "second_rookie": ["Jaden Ivey", "Jalen Duren", "Jabari Smith Jr.", "Jeremy Sochan", "Tari Eason"]
    },
    "2023-24": {
        "first_all_nba": ["Shai Gilgeous-Alexander", "Nikola Jokić", "Luka Dončić", "Giannis Antetokounmpo", "Jayson Tatum"],
        "second_all_nba": ["Jalen Brunson", "Anthony Davis", "Kevin Durant", "Kawhi Leonard", "Anthony Edwards"],
        "third_all_nba": ["Devin Booker", "Stephen Curry", "Tyrese Haliburton", "LeBron James", "Domantas Sabonis"],
        "first_rookie": ["Victor Wembanyama", "Chet Holmgren", "Brandon Miller", "Jaime Jaquez Jr.", "Brandin Podziemski"],
        "second_rookie": ["Dereck Lively II", "Amen Thompson", "Keyonte George", "Cason Wallace", "GG Jackson"]
    },
    "2024-25": {
        "first_all_nba": ["Nikola Jokić", "Giannis Antetokounmpo", "Jayson Tatum", "Shai Gilgeous-Alexander", "Donovan Mitchell"],
        "second_all_nba": ["Evan Mobley", "LeBron James", "Stephen Curry", "Anthony Edwards", "Jalen Brunson"],
        "third_all_nba": ["Karl-Anthony Towns", "Jalen Williams", "Cade Cunningham", "Tyrese Haliburton", "James Harden"],
        "first_rookie": ["Stephon Castle", "Zach Edey", "Zaccharie Risacher", "Alex Sarr", "Jaylen Wells"],
        "second_rookie": ["Matas Buzelis", "Bub Carrington", "Donovan Clingan", "Yves Missi", "Kel'el Ware"]
    }
}

#Downloads statistics for each season.
def fetch_season_data(season: str, measure_type: str) -> pd.DataFrame:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw_data = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                per_mode_detailed='PerGame',
                measure_type_detailed_defense=measure_type
            )
            return raw_data.get_data_frames()[0]
        except Exception as e:
            print(f"   [Attempt {attempt + 1}/{max_retries}] Download error ({measure_type}) dla {season}: {e}")
            time.sleep(5)
    raise RuntimeError(f"Failed to download {measure_type} dla sezonu {season} po {max_retries} próbach.")

#Retrieves and combines traditional and advanced statistics for the specified seasons.
def download_nba_data(seasons_list: list) -> pd.DataFrame:
    all_seasons_data = []

    for season in seasons_list:
        print(f"=== Downloading data for season: {season} ===")
        try:
            # 1. Traditional statistics
            df_traditional = fetch_season_data(season, 'Base')
            time.sleep(2.0)  # Safe delay for the NBA API

            # 2. Advanced statistics
            df_advanced = fetch_season_data(season, 'Advanced')

            # 3. Merging unique columns for Advanced
            advanced_cols = ['PLAYER_ID'] + [col for col in df_advanced.columns if col not in df_traditional.columns]
            df_season = pd.merge(df_traditional, df_advanced[advanced_cols], on='PLAYER_ID', how='inner')

            df_season['SEASON_ID'] = season
            all_seasons_data.append(df_season)
            print(f"Success! Downloaded {len(df_season)} players.")
            time.sleep(2.0)

        except Exception as e:
            print(f"!!! Critical error for season {season}: {e}. Skipping...")

    return pd.concat(all_seasons_data, ignore_index=True) if all_seasons_data else None

#Maps historical awards to the training dataset.
def build_target_labels(stats_csv_path: str, awards_dict: dict) -> pd.DataFrame:
    if not os.path.exists(stats_csv_path):
        print(f"Error: File not found {stats_csv_path}.")
        return None

    df = pd.read_csv(stats_csv_path)

    # Labels init
    df['all_nba_target'] = 0
    df['all_rookie_target'] = 0
    df['is_rookie'] = 0

    # Maps awards
    for season, awards in awards_dict.items():
        season_mask = df['SEASON_ID'] == season
        if not season_mask.any():
            continue

        # All-NBA: 3 = 1st, 2 = 2nd, 1 = 3rd
        df.loc[season_mask & df['PLAYER_NAME'].isin(awards.get("first_all_nba", [])), 'all_nba_target'] = 3
        df.loc[season_mask & df['PLAYER_NAME'].isin(awards.get("second_all_nba", [])), 'all_nba_target'] = 2
        df.loc[season_mask & df['PLAYER_NAME'].isin(awards.get("third_all_nba", [])), 'all_nba_target'] = 1

        # All-Rookie: 2 = 1st, 1 = 2nd
        rookie_1st = awards.get("first_rookie", [])
        rookie_2nd = awards.get("second_rookie", [])

        df.loc[season_mask & df['PLAYER_NAME'].isin(rookie_1st + rookie_2nd), 'is_rookie'] = 1
        df.loc[season_mask & df['PLAYER_NAME'].isin(rookie_1st), 'all_rookie_target'] = 2
        df.loc[season_mask & df['PLAYER_NAME'].isin(rookie_2nd), 'all_rookie_target'] = 1

    # Marks rookies (players that appear first time in 25/26 season)
    current_season_mask = df['SEASON_ID'] == "2025-26"
    past_players = df[df['SEASON_ID'] != "2025-26"]['PLAYER_NAME'].unique()
    df.loc[current_season_mask & ~df['PLAYER_NAME'].isin(past_players), 'is_rookie'] = 1

    # Display the class distribution (diagnostic).
    train_mask = df['SEASON_ID'] != "2025-26"
    print("\n[All-NBA distribution (Train)]:\n", df[train_mask]['all_nba_target'].value_counts())
    print("\n[All-Rookie distribution (Train)]:\n", df[train_mask]['all_rookie_target'].value_counts())

    return df


if __name__ == "__main__":
    # 1. Data download
    raw_dataset = download_nba_data(SEASONS)
    if raw_dataset is not None:
        raw_dataset.to_csv(RAW_CSV_PATH, index=False)
        print(f"\nRaw statistics saved to {RAW_CSV_PATH}. Shape: {raw_dataset.shape}")

    # 2. Data labeling
    labeled_dataset = build_target_labels(RAW_CSV_PATH, HISTORICAL_AWARDS)
    if labeled_dataset is not None:
        labeled_dataset.to_csv(LABELED_CSV_PATH, index=False)
        print(f"Success! Labeled dataset saved to {LABELED_CSV_PATH}")
