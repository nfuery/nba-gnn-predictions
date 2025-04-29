import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from gatv2tcn import GATv2TCN
import pickle
import matplotlib.colors as mcolors

# === Load Test Data ===
X_test = torch.load('data/X_test.pt')
y_test = torch.load('data/y_test.pt')
g_test = torch.load('data/g_test.pt')
h_test = torch.load('data/h_test.pt')
team_tensor = torch.load('data/team_tensor.pt')
position_tensor = torch.load('data/position_tensor.pt')
scaler = pickle.load(open('data/scaler.pkl', 'rb'))

player_prediction_metrics = ['PTS', 'AST', 'REB', 'TO', 'STL', 'BLK']

# === Reload Saved Model + Embeddings ===
team_embedding_in = team_tensor.shape[-1]
position_embedding_in = position_tensor.shape[-1]
team_embedding_out = 8
position_embedding_out = 8

model_in = y_test.shape[-1] + team_embedding_out + position_embedding_out

model = GATv2TCN(
    in_channels=model_in,
    out_channels=6,
    len_input=10,
    len_output=1,
    temporal_filter=64,
    out_gatv2conv=32,
    dropout_tcn=0.25,
    dropout_gatv2conv=0.5,
    head_gatv2conv=8
)

team_embedding = nn.Linear(team_embedding_in, team_embedding_out)
position_embedding = nn.Linear(position_embedding_in, position_embedding_out)

model.load_state_dict(torch.load('model/gatv2tcn-team-position-embedding/saved_astgcn.pth'))
team_embedding.load_state_dict(torch.load('model/gatv2tcn-team-position-embedding/team_embedding.pth'))
position_embedding.load_state_dict(torch.load('model/gatv2tcn-team-position-embedding/position_embedding.pth'))

model.eval()
team_embedding.eval()
position_embedding.eval()

# === Make Prediction ===
i = 145
SEQ_LENGTH = 10
pts_index = player_prediction_metrics.index('PTS')

team_embedding_vector = team_embedding(team_tensor)
position_embedding_vector = position_embedding(position_tensor)

y_test_mask = h_test[i].unique()
X_list = []
G_list = []
for j in range(SEQ_LENGTH):
    X_list.append(torch.cat([X_test[i][:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
    G_list.append(g_test[i][j])

x = torch.stack(X_list, dim=-1)
x = x[None, :, :, :]

with torch.no_grad():
    y_pred = model(x, G_list)[0]

true_pts_std = y_test[i][y_test_mask][:, pts_index].detach().numpy()
pred_pts_std = y_pred[y_test_mask][:, pts_index].detach().numpy()

# === UNSTANDARDIZE (de-normalize) ===
# scaler.mean_ and scaler.scale_ are arrays for all features
pts_mean = scaler.mean_[pts_index]
pts_std = scaler.scale_[pts_index]

true_pts = true_pts_std * pts_std + pts_mean
pred_pts = pred_pts_std * pts_std + pts_mean

# === Load mappings ===
player_id2name = pickle.load(open('data/player_id2name.pkl', 'rb'))
player_id2team = pickle.load(open('data/player_id2team.pkl', 'rb'))

# Build list of player_ids
player_ids = sorted(player_id2name.keys())

# Map based on y_test_mask
player_names = [player_id2name.get(player_ids[idx.item()], f"Player {j}") for j, idx in enumerate(y_test_mask)]
player_teams = [player_id2team.get(player_ids[idx.item()], "Unknown") for idx in y_test_mask]

# # === Plotting ===
# plt.figure(figsize=(16, 7))
# bar_width = 0.4
# indices = np.arange(len(true_pts))

# for idx in range(len(true_pts)):
#     plt.bar(indices[idx], true_pts[idx], width=bar_width, color="blue")
#     plt.bar(indices[idx] + bar_width, pred_pts[idx], width=bar_width, color="lightblue", alpha=0.8)

# === Sort by Predicted Points ===
# Get sorting indices based on descending predicted points
sort_indices = np.argsort(-true_pts)

# Reorder everything
pred_pts = pred_pts[sort_indices]
true_pts = true_pts[sort_indices]
player_names = [player_names[idx] for idx in sort_indices]
player_teams = [player_teams[idx] for idx in sort_indices]

# === Plotting (after sort) ===
plt.figure(figsize=(16, 7))
bar_width = 0.4
indices = np.arange(len(true_pts))

for idx in range(len(true_pts)):
    plt.bar(indices[idx], true_pts[idx], width=bar_width, color="blue")
    plt.bar(indices[idx] + bar_width, pred_pts[idx], width=bar_width, color="lightblue", alpha=0.8)

plt.xlabel('Player')
plt.ylabel('Points')
plt.title('Predicted vs. Actual Points')
plt.xticks(indices + bar_width / 2, player_names, rotation=45, ha='right')
plt.legend(['Actual', 'Predicted'])
plt.grid(axis='y')
plt.tight_layout()
plt.show()
