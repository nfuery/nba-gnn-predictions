import os
import time
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams
from collections import defaultdict
import requests
from nba_api.stats.library.parameters import SeasonAll


# Parameters
SEASON = '2024-25'
MIN_MINUTES_PLAYED = 10
OUTPUT_DIR = 'data'

# Feature Selection
boxscore_fields = ['PTS', 'AST', 'REB', 'TO', 'STL', 'BLK', 'PLUS_MINUS']
selected_fields = boxscore_fields

# Prepare Outputs
X_seq = []
G_seq = []
player_id2name = {}
player_id2team = {}
player_id2position = {}

# Get all games from the season
nba_teams = teams.get_teams()
gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=SEASON, league_id_nullable='00')
games = gamefinder.get_data_frames()[0]
games.drop_duplicates('GAME_ID', keep='first', inplace=True)

print(f"Total games fetched: {len(games)}")

game_ids = games['GAME_ID'].tolist()

# Minutes are displyed as a weird format from NBA API
# Instead of MM:SS, it is MM.000000::SS
# This function parses the minute strings
def parse_minutes(min_str):
    try:
        if isinstance(min_str, str):
            if ':' in min_str:
                parts = min_str.split(':')
                if len(parts) == 2:
                    mins = int(float(parts[0]))
                    secs = int(float(parts[1]))
                    return mins * 60 + secs
            return int(float(min_str) * 60)
        elif isinstance(min_str, (float, int)):
            return int(min_str * 60)
    except Exception as e:
        print(f"Failed to parse MIN: {min_str} ({e})")
    return 0

# Loop through games and collect data
for game_id in game_ids:
    try:
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        player_stats = boxscore.get_data_frames()[0]
        
        # Filter players with MIN minutes played
        player_stats['MIN'] = player_stats['MIN'].apply(parse_minutes)
        player_stats = player_stats[player_stats['MIN'] >= MIN_MINUTES_PLAYED]

        if len(player_stats) < 2:
            continue  # skip games with too few players (doesn't happen often but safeguard)

        # Build graph: complete graph of all players in game
        G = nx.complete_graph(player_stats['PLAYER_ID'].tolist())
        G_seq.append(G)

        # Extract features per player
        player_vectors = []
        player_ids = []

        for _, row in player_stats.iterrows():
            pid = row['PLAYER_ID']
            player_ids.append(pid)
            
            features = [row.get(field, 0) for field in selected_fields]
            player_vectors.append(features)

            # Save metadata
            player_id2name[pid] = row['PLAYER_NAME']
            player_id2team[pid] = row['TEAM_ABBREVIATION']
            pos = row['START_POSITION']
            # One-hot encode position
            encoded_pos = [int(p in pos) for p in ['F', 'G', 'C']]
            player_id2position[pid] = encoded_pos

        # Stack into fixed-shape matrix
        df = pd.DataFrame(player_vectors, index=player_ids, columns=selected_fields)
        df = df.reindex(sorted(player_id2name.keys()), fill_value=0)
        X_seq.append(df.values)

        print(f"Processed game {game_id} with {len(player_stats)} players")
        time.sleep(5.0)  # rate limit NBA API, increased from 0.6s to 5.0s due to rate limiting issues

    except Exception as e:
        print(f"Failed to process game {game_id}: {e}")
        continue

# Pad X_seq to same shape (num_players x features)
max_players = max(x.shape[0] for x in X_seq)
feature_dim = X_seq[0].shape[1]
X_seq_padded = np.zeros((len(X_seq), max_players, feature_dim))
for i, x in enumerate(X_seq):
    X_seq_padded[i, :x.shape[0], :] = x

# Save data
os.makedirs(OUTPUT_DIR, exist_ok=True)
pickle.dump(X_seq_padded, open(f"{OUTPUT_DIR}/X_seq.pkl", "wb"))
pickle.dump(G_seq, open(f"{OUTPUT_DIR}/G_seq.pkl", "wb"))
pickle.dump(player_id2name, open(f"{OUTPUT_DIR}/player_id2name.pkl", "wb"))
pickle.dump(player_id2team, open(f"{OUTPUT_DIR}/player_id2team.pkl", "wb"))
pickle.dump(player_id2position, open(f"{OUTPUT_DIR}/player_id2position.pkl", "wb"))