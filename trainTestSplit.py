import pickle
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
from torch.autograd import Variable
from sklearn import preprocessing

# Hyperparameters (must match training)
SEQ_LENGTH = 10
OFFSET = 1
train_data_limit_idx = 41

# Load X_seq and G_seq
X_seq = pickle.load(open('data/X_seq.pkl', 'rb'))  # shape: (num_games, num_players, num_features)
G_seq = pickle.load(open('data/G_seq.pkl', 'rb'))

# Fill zeros forward for missing games
def fill_zeros_with_last(seq):
    seq_ffill = np.zeros_like(seq)
    for i in range(seq.shape[1]):
        arr = seq[:, i]
        prev = np.arange(len(arr))
        prev[arr == 0] = 0
        prev = np.maximum.accumulate(prev)
        seq_ffill[:, i] = arr[prev]
    return seq_ffill

Xs = np.zeros_like(X_seq)
for i in range(X_seq.shape[1]):
    Xs[:, i, :] = fill_zeros_with_last(X_seq[:, i, :])

# Standardize
scaler = StandardScaler()
Xs_train_flat = Xs[:train_data_limit_idx].reshape(-1, Xs.shape[-1])
scaler.fit(Xs_train_flat)

Xs_flat = Xs.reshape(-1, Xs.shape[-1])
Xs_scaled_flat = scaler.transform(Xs_flat)
Xs_scaled = Xs_scaled_flat.reshape(Xs.shape)

# Reformat Graphs
Gs = []
for g in G_seq:
    node_dict = {node: i for i, node in enumerate(g.nodes())}
    edges = np.array([edge.split(' ') for edge in nx.generate_edgelist(g)])[:, :2].astype(int).T
    edges = np.vectorize(node_dict.__getitem__)(edges)
    Gs.append(torch.LongTensor(np.hstack((edges, edges[[1, 0]]))))

# Now reconstruct sequences
def construct_input_sequences_and_output(z, seq_length=10, offset=1):
    if isinstance(z, (np.ndarray, np.generic)):
        if offset == 0:
            x = sliding_window_view(z, seq_length, axis=0)
        else:
            x = sliding_window_view(z[:-offset], seq_length, axis=0)
        y = z[seq_length+offset-1:]
    else:
        x = []
        for i in range(len(z) - seq_length - offset + 1):
            x.append(z[i:i+seq_length])
        y = z[seq_length+offset-1:]
    return x, y

X_in, X_out = construct_input_sequences_and_output(Xs_scaled, seq_length=SEQ_LENGTH, offset=OFFSET)
G_in, G_out = construct_input_sequences_and_output(Gs, seq_length=SEQ_LENGTH, offset=OFFSET)

X_in = Variable(torch.FloatTensor(X_in))
X_out = Variable(torch.FloatTensor(X_out))

# Split like your training script
X_train, X_val, X_test = X_in[:31], X_in[41:41+16], X_in[41+26:]
y_train, y_val, y_test = X_out[:31], X_out[41:41+16], X_out[41+26:]

g_train = G_in[:31]
g_val = G_in[41:41+16] 
g_test = G_in[41+26:]

h_train = G_out[:31]
h_val = G_out[41:41+16]
h_test = G_out[41+26:]

# Save them
torch.save(X_test, 'data/X_test.pt')
torch.save(y_test, 'data/y_test.pt')
torch.save(g_test, 'data/g_test.pt')
torch.save(h_test, 'data/h_test.pt')

print("✅ Done rebuilding and saving X_test, y_test, g_test, h_test without regenerating from scratch!")

# Load mappings
player_id_to_team = pickle.load(open('data/player_id2team.pkl', 'rb'))
player_id_to_position = pickle.load(open('data/player_id2position.pkl', 'rb'))

# Build team_tensor
le = preprocessing.LabelEncoder()
df_id2team = pd.DataFrame.from_dict(player_id_to_team, orient='index').apply(le.fit_transform)
enc = preprocessing.OneHotEncoder()
enc.fit(df_id2team)
onehotlabels = enc.transform(df_id2team).toarray()
team_tensor = Variable(torch.FloatTensor(onehotlabels))

# Build position_tensor
position_tensor = Variable(torch.FloatTensor(np.stack(list(player_id_to_position.values()), axis=0)))

# Save
torch.save(team_tensor, 'data/team_tensor.pt')
torch.save(position_tensor, 'data/position_tensor.pt')

print("✅ Successfully saved team_tensor.pt and position_tensor.pt")