from nba_api.stats.endpoints import playbyplayv2
from collections import defaultdict
import networkx as nx
import pandas as pd

def build_overlap_graph(game_id, threshold_seconds=600):
    pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
    df = pbp.get_data_frames()[0]

    on_court = set()
    player_times = defaultdict(list)

    def parse_time(period, clock_str):
        if isinstance(clock_str, str):
            mins, secs = map(int, clock_str.split(':'))
            period_offset = (period - 1) * 12 * 60
            return period_offset + (12 * 60 - (mins * 60 + secs))
        return 0

    for i, row in df.iterrows():
        period = row['PERIOD']
        clock = row['PCTIMESTRING']
        event_type = row['EVENTMSGTYPE']
        action = row['HOMEDESCRIPTION'] or row['VISITORDESCRIPTION'] or ""
        current_time = parse_time(period, clock)

        if event_type == 8 and 'FOR' in action:
            entering = row['PLAYER1_ID']
            replaced = row['PLAYER2_ID']

            if pd.notna(entering):
                on_court.add(entering)
                player_times[entering].append([current_time, None])

            if pd.notna(replaced):
                if replaced in player_times and player_times[replaced][-1][1] is None:
                    player_times[replaced][-1][1] = current_time
                else:
                    on_court.discard(replaced)

    if df.empty:
        return nx.Graph()

    final_time = parse_time(df.iloc[-1]['PERIOD'], df.iloc[-1]['PCTIMESTRING'])
    for pid in on_court:
        if player_times[pid][-1][1] is None:
            player_times[pid][-1][1] = final_time

    players = list(player_times.keys())
    G = nx.Graph()
    G.add_nodes_from(players)

    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            p1, p2 = players[i], players[j]
            overlap = 0
            for start1, end1 in player_times[p1]:
                for start2, end2 in player_times[p2]:
                    if end1 is None or end2 is None:
                        continue
                    latest_start = max(start1, start2)
                    earliest_end = min(end1, end2)
                    delta = earliest_end - latest_start
                    if delta > 0:
                        overlap += delta
            if overlap >= threshold_seconds:
                G.add_edge(p1, p2, shared_seconds=overlap)

    print(f"Finished game {game_id} — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G
