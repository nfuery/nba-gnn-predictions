import copy
import os
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# LR scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from IPython import display
from torch_geometric.nn import GATConv
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from gatv2tcn import ASTGCN, GATv2TCN, GATv2Conv
from sklearn import preprocessing
# Standard Scaler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


SEQ_LENGTH = 10
OFFSET = 1
player_boxscore_fields = ['PTS', 'AST', 'REB', 'TO', 'STL', 'BLK', 'PLUS_MINUS']
player_boxscore_tracking_fields = ['TCHS', 'PASS', 'DIST']
player_boxscore_advanced_fields = ['PACE', 'USG_PCT', 'TS_PCT']
player_prediction_metrics = ['PTS', 'AST', 'REB', 'TO', 'STL', 'BLK']
player_prediction_metrics_index = [
    (player_boxscore_fields + player_boxscore_tracking_fields + player_boxscore_advanced_fields).index(metric) for
    metric in player_prediction_metrics]


def fill_zeros_with_last(seq):
    seq_ffill = np.zeros_like(seq)
    for i in range(seq.shape[1]):
        arr = seq[:, i]
        prev = np.arange(len(arr))
        prev[arr == 0] = 0
        prev = np.maximum.accumulate(prev)
        seq_ffill[:, i] = arr[prev]

    return seq_ffill

def construct_input_sequences_and_output(z, seq_length=10, offset=1):
    # Check if z is already a numpy array
    if isinstance(z, (np.ndarray, np.generic)):
        # For NumPy arrays, use sliding_window_view
        if offset == 0:
            x = sliding_window_view(z, seq_length, axis=0)
        else:
            x = sliding_window_view(z[:-offset], seq_length, axis=0)
        y = z[seq_length+offset-1:]
    else:
        # For lists of tensors or other objects, manually create sequences
        x = []
        for i in range(len(z) - seq_length - offset + 1):
            x.append(z[i:i+seq_length])
        y = z[seq_length+offset-1:]
    
    return x, y

def create_dataset():
    # X_seq breakdown
    # 3D NumPy array of shape (num_games, num_players, num_features)
        # num_games -> each entry in the first dimension corresponds to a snapshot of a day's worth of games (single game day)
        # num_players -> all players appeared across any game (aligned by player id)
        # num_features -> stats (pts, ast, reb, to, stl, blk, +-)
    # X_seq[i] is a snapshot of all tracked players' features
        # E.g. X_seq[0] -> stats from 10/25 games, X_seq[1] -> stats from 10/27 games
            # Then X_seq[0][12] and X_seq[1][12] are the same player on diff game days
                # If they didnt play, row will be all 0s or missing (I THINK)
    X_seq = pd.read_pickle('data/X_seq.pkl')
    G_seq = pd.read_pickle('data/G_seq.pkl')
    player_id_to_team = pd.read_pickle('data/player_id2team.pkl')
    player_id_to_position = pd.read_pickle('data/player_id2position.pkl')

    le = preprocessing.LabelEncoder()
    df_id2team = pd.DataFrame.from_dict(player_id_to_team, orient='index').apply(le.fit_transform)
    enc = preprocessing.OneHotEncoder()
    enc.fit(df_id2team)
    onehotlabels = enc.transform(df_id2team).toarray()
    # team_onehot_seq = np.broadcast_to(onehotlabels, (X_seq.shape[0], X_seq.shape[1], onehotlabels.shape[-1]))
    team_tensor = Variable(torch.FloatTensor(onehotlabels))
    position_tensor = Variable(torch.FloatTensor(np.stack(list(player_id_to_position.values()), axis=0)))

    Xs = np.zeros_like(X_seq)
    for i in range(X_seq.shape[1]):
        Xs[:, i, :] = fill_zeros_with_last(X_seq[:, i, :])

    # Standardization
    # We do this to ensure each feature has a mean of 0 and sd of 1
    # Therefore all features contribute equally to the model training
    num_times, num_nodes, num_features = Xs.shape
    scaler = StandardScaler()

    # Determine split point in original sequence for fitting scaler (avoiding val/test leakage)
    # Val starts at index 41 in X_in. X_in sequences use SEQ_LENGTH=10 points.
    # Sequence X_in[41] uses Xs[41] to Xs[41+10-1].
    # So, we can safely fit the scaler on data up to index 40 of Xs.
    train_data_limit_idx = 41

    # Reshape data (time * nodes, features) for fitting
    Xs_train_flat = Xs[:train_data_limit_idx].reshape(-1, num_features)
    scaler.fit(Xs_train_flat)

    # Reshape all data and transform
    Xs_flat = Xs.reshape(-1, num_features)
    Xs_scaled_flat = scaler.transform(Xs_flat)

    # Reshape to original shape
    Xs_scaled = Xs_scaled_flat.reshape(num_times, num_nodes, num_features)

    Gs = []
    c = 0
    for g in G_seq:
        c += 1
        print(c)
        node_dict = {node: i for i, node in enumerate(g.nodes())}
        edges = np.array([edge.split(' ') for edge in nx.generate_edgelist(g)])[:, :2].astype(int).T
        edges = np.vectorize(node_dict.__getitem__)(edges)
        Gs.append(torch.LongTensor(np.hstack((edges, edges[[1, 0]]))))

    # Here we use the scaled data from above
    X_in, X_out = construct_input_sequences_and_output(Xs_scaled, seq_length=SEQ_LENGTH, offset=OFFSET)

    G_in, G_out = construct_input_sequences_and_output(Gs, seq_length=SEQ_LENGTH, offset=OFFSET)
    X_in = Variable(torch.FloatTensor(X_in))
    X_out = Variable(torch.FloatTensor(X_out))

    X_train, X_val, X_test = X_in[:31], X_in[41:41+16], X_in[41+26:]
    y_train, y_val, y_test = X_out[:31], X_out[41:41+16], X_out[41+26:]

    # Splitting data
    g_train = G_in[:31]
    g_val = G_in[41:41+16] 
    g_test = G_in[41+26:]
    h_train = G_out[:31]
    h_val = G_out[41:41+16]
    h_test = G_out[41+26:]

    
    print(X_train.shape, X_val.shape, X_test.shape)
    print(f"g_train length: {len(g_train)}, g_val length: {len(g_val)}, g_test length: {len(g_test)}")
    print(f"h_train length: {len(h_train)}, y_train length: {len(y_train)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test, h_train, h_val, h_test, team_tensor, position_tensor, scaler #, team_train, team_val, team_test


X_train, X_val, X_test, y_train, y_val, y_test, g_train, g_val, g_test, h_train, h_val, h_test, team_tensor, position_tensor, scaler = create_dataset()
team_embedding_in = team_tensor.shape[-1]
team_embedding_out = 8
team_embedding = nn.Linear(team_embedding_in, team_embedding_out) # nn.Embedding(num_embeddings=team_embedding_in, embedding_dim=team_embedding_out)

position_embedding_in = position_tensor.shape[-1]
position_embedding_out = 8
position_embedding = nn.Linear(position_embedding_in, position_embedding_out) # nn.Embedding(num_embeddings=position_embedding_in, embedding_dim=position_embedding_out)

model_in = y_train.shape[-1] + team_embedding_out + position_embedding_out

# model = ASTGCN(nb_block=2, 
#                in_channels=model_in, 
#                K=3,
#                nb_chev_filter=1, 
#                nb_time_filter=64, 
#                time_strides=1, 
#                num_for_predict=6,
#                len_input=10, 
#                num_of_vertices=582, 
#                nb_gatv2conv=16, 
#                dropout_gatv2conv=0.25,
#                head_gatv2conv=4)

model = GATv2TCN(in_channels=model_in,
        out_channels=6,
        len_input=10,
        len_output=1,
        temporal_filter=64,
        out_gatv2conv=32,
        dropout_tcn=0.25,
        dropout_gatv2conv=0.5,
        head_gatv2conv=8) # Was initially 4, changed to 8

model_name = 'gatv2tcn-team-position-embedding'

if not os.path.exists(f"model/{model_name}"):
    os.mkdir(f"model/{model_name}")
# model.load_state_dict(torch.load(f"model/{model_name}/saved_astgcn.pth"))
# team_embedding.load_state_dict(torch.load(f"model/{model_name}/saved_team.pth"))
# model.eval()
# team_embedding.eval()

parameters = list(model.parameters()) + list(team_embedding.parameters()) + list(position_embedding.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.01, weight_decay=0.001)

# Adding LR Scheduler
    # Dynamically adjusts learning rate during training
    #   LR: how fast a model adjusts its weights in response to the loss function gradient
# Patience subject to change
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=15, verbose=True)

min_val_loss = np.inf
min_val_iter = -1

# fig, ax = plt.subplots()
PT_INDEX = 0
EPOCHS = 300
# BATCH_SIZE = 20 --> Not really used anymore
train_loss_history = np.zeros(EPOCHS)
val_loss_history = np.zeros(EPOCHS)
for epoch in tqdm(range(EPOCHS)):
    epoch_train_loss_sum = 0.0
    model.train()

    # Ensuring embedding layers are in train mode
    team_embedding.train()
    position_embedding.train()

    team_embedding_vector = team_embedding(team_tensor)
    position_embedding_vector = position_embedding(position_tensor)

    optimizer.zero_grad() # Now at beginning of batch accumulation

    # Simple batch accumulation
        # Processing all samples and updating once

    for i in range(X_train.shape[0]): # Now using all training data instead of random sampling
        y_train_mask = h_train[i].unique()
        X_list = []
        G_list = []
        for j in range(SEQ_LENGTH):
            X_list.append(torch.cat([X_train[i][:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
            G_list.append(g_train[i][j])
        x = torch.stack(X_list, dim=-1)
        x = x[None, :, :, :] # Add batch dim
        
        # Forward pass
        x_astgat = model(x, G_list)[0, ...] # Remove batch dim from output

        # Changed from mse_loss to l1_loss (MAE loss)
        train_loss = F.l1_loss(x_astgat[y_train_mask], y_train[i][y_train_mask][:, player_prediction_metrics_index])
        train_loss.backward(retain_graph=True) # Accumulate gradients
        epoch_train_loss_sum += train_loss.item() # Store itemized loss

    # Update weights after processing all samples in the epoch
    optimizer.step()
    # optimizer.zero_grad() --> Removed to check gradients

    epoch_val_loss_sum = 0.0
    model.eval()
    team_embedding.eval()
    position_embedding.eval()
    # team_embedding_vector = team_embedding(team_tensor)
    # position_embedding_vector = position_embedding(position_tensor)


    with torch.no_grad(): # Ensure no gradients are computed during validation
        for i in range(X_val.shape[0]):
            y_val_mask = h_val[i].unique()
            X_list = []
            G_list = []
            for j in range(SEQ_LENGTH):
                X_list.append(torch.cat([X_val[i][:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
                G_list.append(g_val[i][j])
            x = torch.stack(X_list, dim=-1)
            x = x[None, :, :, :]
            x_astgat = model(x, G_list)[0, :, :]

            val_loss = F.l1_loss(x_astgat[y_val_mask], y_val[i][y_val_mask][:, player_prediction_metrics_index])
            #val_loss += F.mse_loss(x_astgat[y_val_mask], y_val[i][y_val_mask][:, player_prediction_metrics_index])
            epoch_val_loss_sum += val_loss.item()

    # Store epoch losses
    epoch_avg_train_loss = epoch_train_loss_sum / X_train.shape[0]
    epoch_avg_val_loss = epoch_val_loss_sum / X_val.shape[0]

    train_loss_history[epoch] = epoch_avg_train_loss
    val_loss_history[epoch] = epoch_avg_val_loss

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_avg_train_loss:.5f}, Val Loss: {epoch_avg_val_loss:.5f}, LR: {current_lr:.6f}")
    # print(f"validation loss: {val_loss.item()}, training loss: {train_loss.item()}")

    # if epoch % 5 == 0:
    #     if epoch == 0:
    #         ax.plot(y_val[i][y_val_mask, PT_INDEX].detach().numpy(), ls='-.', color='k', lw=1.5, label='real')
    #     if epoch >= 80:
    #         ax.lines.pop(1)
    #     for l in ax.lines[1:]:
    #         l.set_alpha(.3)
    #     ax.plot(x_astgat[y_val_mask, PT_INDEX].detach().numpy(), label=f'{epoch} ({train_loss.item()})')
    #     ax.set_title("Epoch: %d, loss: %1.5f" % (epoch, train_loss.item()))
    #     ax.legend(bbox_to_anchor=(1, 0.5))
    #     # display.clear_output(wait=True)
    #     # display.display(fig)

    # Step scheduler based on val loss
    scheduler.step(epoch_avg_val_loss)

    if min_val_loss > epoch_avg_val_loss:
        print(f"Validation Loss Decreased({min_val_loss:.5f}--->{epoch_avg_val_loss:.5f}) \t Saving The Model")
        min_val_loss = epoch_avg_val_loss
        min_val_iter = epoch
        # Saving State Dict
        torch.save(model.state_dict(), f"model/{model_name}/saved_astgcn.pth")
        torch.save(team_embedding.state_dict(), f"model/{model_name}/team_embedding.pth")
        torch.save(position_embedding.state_dict(), f"model/{model_name}/position_embedding.pth")

print(min_val_loss, min_val_iter)


#GATv2TCN
astgcn_test = copy.deepcopy(model)
astgcn_test.load_state_dict(torch.load(f"model/{model_name}/saved_astgcn.pth"))
astgcn_test.eval()

team_embedding_test = copy.deepcopy(team_embedding)
team_embedding_test.load_state_dict(torch.load(f"model/{model_name}/team_embedding.pth"))
team_embedding_test.eval()

position_embedding_test = copy.deepcopy(position_embedding)
position_embedding_test.load_state_dict(torch.load(f"model/{model_name}/position_embedding.pth"))
position_embedding_test.eval()

team_embedding_vector = team_embedding_test(team_tensor)
position_embedding_vector = position_embedding_test(position_tensor)

test_loss_l1 = 0.0
test_loss_rmse = 0.0
test_corr = 0.0
test_loss_mape = 0.0

for i in range(X_test.shape[0]):
    y_test_mask = h_test[i].unique()
    X_list = []
    G_list = []
    for j in range(SEQ_LENGTH):
        X_list.append(torch.cat([X_test[i][:, :, j], team_embedding_vector, position_embedding_vector], dim=1))
        G_list.append(g_test[i][j])
    x = torch.stack(X_list, dim=-1)
    x = x[None, :, :, :]
    x_astgcn = astgcn_test(x, G_list)[0, :, :]
    test_loss_rmse += root_mean_squared_error(x_astgcn[y_test_mask].detach().numpy(), y_test[i][y_test_mask][:, player_prediction_metrics_index].detach().numpy()) # torch.sqrt(F.mse_loss(x_astgcn[y_test_mask], y_test[i][y_test_mask][:, player_prediction_metrics_index]))
    test_loss_l1 += F.l1_loss(x_astgcn[y_test_mask], y_test[i][y_test_mask][:, player_prediction_metrics_index])
    test_loss_mape += mean_absolute_percentage_error(x_astgcn[y_test_mask].detach().numpy(), y_test[i][y_test_mask][:, player_prediction_metrics_index].detach().numpy()) # torch.sqrt(F.mse_loss(x_astgcn[y_test_mask], y_test[i][y_test_mask][:, player_prediction_metrics_index]))
    test_corr += torch.tanh(torch.mean(torch.stack([torch.arctanh(torch.corrcoef(torch.stack([x_astgcn[y_test_mask][:, metric_idx],
                             y_test[i][y_test_mask][:, player_prediction_metrics_index][:, metric_idx]], dim=0))[0, 1])
                            for metric_idx in range(len(player_prediction_metrics))])))
print(f"RMSE: {test_loss_rmse/X_test.shape[0]}, MAPE: {test_loss_mape/X_test.shape[0]}, CORR: {test_corr/X_test.shape[0]}, MAE: {test_loss_l1/X_test.shape[0]}")


player_id_to_team = pd.read_pickle('data/player_id2team.pkl')
from nba_api.stats.static import teams, players
nba_teams = teams.get_teams()
team_vec = team_embedding_vector.detach().numpy()
from pandas.plotting._matplotlib.style import get_standard_colors
from matplotlib.lines import Line2D
colors = get_standard_colors(num_colors=len(nba_teams))
markers = list(Line2D.markers.keys())[:len(nba_teams)+1]

fig, ax = plt.subplots()
for i, team in enumerate(nba_teams):
    player_in_team = [idx for idx, team_name in enumerate(player_id_to_team.values()) if team_name == team['nickname']]
    ax.plot(team_vec[player_in_team, 0], team_vec[player_in_team, 1], color=colors[i], marker=markers[i+1], label=team['nickname'])
    plt.text(team_vec[player_in_team, 0].mean(), team_vec[player_in_team, 1].mean(), team['nickname'])

player_id_to_position = pd.read_pickle('data/player_id2position.pkl')
position_vec = position_embedding_vector.detach().numpy()

fig, ax = plt.subplots()
position_dict = {(0, 0, 0): 'No position',
                 (0, 0, 1): 'C',
                 (0, 1, 0): 'G',
                 (1, 0, 0): 'F',
                 (1, 0, 1): 'F/C',
                 (1, 1, 0): 'F/G'}
for i, position in enumerate(np.unique(np.array(list(player_id_to_position.values())), axis=0)):
    player_at_position = [idx for idx, player_position in enumerate(player_id_to_position.values()) if (player_position==position).all()]
    label = position_dict[tuple(position)]
    ax.plot(position_vec[player_at_position, 0], position_vec[player_at_position, 1], color=colors[i], marker=markers[i+1], label=label)
ax.legend()
pass
