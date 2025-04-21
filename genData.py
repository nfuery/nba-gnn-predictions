import os
import time
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from nba_api.stats.static import teams
from substitution_overlap_graph import build_overlap_graph
from collections import defaultdict

SEASON = '2024-25'
THRESHOLD_SECONDS = 600
OUTPUT_DIR = 'data'
PARTIAL_SAVE_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pkl")

# Feature Selection
boxscore_fields = ['PTS', 'AST', 'REB', 'TO', 'STL', 'BLK', 'PLUS_MINUS']

# Load previous progress if available
if os.path.exists(PARTIAL_SAVE_PATH):
    with open(PARTIAL_SAVE_PATH, "rb") as f:
        checkpoint = pickle.load(f)
        X_seq = checkpoint["X_seq"]
        G_seq = checkpoint["G_seq"]
        player_id2name = checkpoint["player_id2name"]
        player_id2team = checkpoint["player_id2team"]
        player_id2position = checkpoint["player_id2position"]
        processed_game_ids = checkpoint["processed_game_ids"]
    print(f"Resuming from checkpoint with {len(processed_game_ids)} games processed")
else:
    X_seq = []
    G_seq = []
    player_id2name = {}
    player_id2team = {}
    player_id2position = {}
    processed_game_ids = set()

# Get all games and group by game day
gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=SEASON, league_id_nullable='00')
games = gamefinder.get_data_frames()[0]
games.drop_duplicates('GAME_ID', keep='first', inplace=True)
print(f"Total games fetched: {len(games)}")

date_column = 'GAME_DATE' if 'GAME_DATE' in games.columns else 'GAME_DATE_EST'
games_by_day = games.groupby(date_column)

# Loop through each game day
for game_date, games_on_day in games_by_day:
    day_graphs = []
    player_vectors = {}

    for _, row in games_on_day.iterrows():
        game_id = row['GAME_ID']
        if game_id in processed_game_ids:
            continue

        try:
            print(f"Processing game {game_id} on {game_date}")
            G = build_overlap_graph(game_id, threshold_seconds=THRESHOLD_SECONDS)
            if G.number_of_edges() == 0:
                print(f"Skipping game {game_id} — no valid player overlaps")
                continue
            day_graphs.append(G)

            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            stats_df = boxscore.get_data_frames()[0]

            for _, s_row in stats_df.iterrows():
                pid = s_row['PLAYER_ID']
                stats = [s_row.get(f, 0) for f in boxscore_fields]
                player_vectors[pid] = stats
                player_id2name[pid] = s_row['PLAYER_NAME']
                player_id2team[pid] = s_row['TEAM_ABBREVIATION']
                pos = s_row['START_POSITION']
                if not pos or pos == '' or pd.isna(pos):
                    player_id2position[pid] = [0, 0, 0]  # No position
                elif set(pos) == {'F', 'C'}:
                    player_id2position[pid] = [1, 0, 1]  # F/C
                elif set(pos) == {'F', 'G'}:
                    player_id2position[pid] = [1, 1, 0]  # F/G
                else:
                    player_id2position[pid] = [int(p in pos) for p in ['F', 'G', 'C']]

            processed_game_ids.add(game_id)
            time.sleep(3.0)

        except Exception as e:
            print(f"Failed game {game_id}: {e}")
            continue

    if not day_graphs:
        continue

    G_day = nx.disjoint_union_all(day_graphs)
    G_seq.append(G_day)

    sorted_pids = list(G_day.nodes())
    X_day = np.array([player_vectors.get(pid, [0]*len(boxscore_fields)) for pid in sorted_pids])
    X_seq.append(X_day)

    # Save checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PARTIAL_SAVE_PATH, "wb") as f:
        pickle.dump({
            "X_seq": X_seq,
            "G_seq": G_seq,
            "player_id2name": player_id2name,
            "player_id2team": player_id2team,
            "player_id2position": player_id2position,
            "processed_game_ids": processed_game_ids
        }, f)

pickle.dump(X_seq, open(f"{OUTPUT_DIR}/X_seq.pkl", "wb"))
pickle.dump(G_seq, open(f"{OUTPUT_DIR}/G_seq.pkl", "wb"))
pickle.dump(player_id2name, open(f"{OUTPUT_DIR}/player_id2name.pkl", "wb"))
pickle.dump(player_id2team, open(f"{OUTPUT_DIR}/player_id2team.pkl", "wb"))
pickle.dump(player_id2position, open(f"{OUTPUT_DIR}/player_id2position.pkl", "wb"))

print("Saved X_seq, G_seq, and player metadata as day-aligned cluster graphs for the 2024–25 season.")
