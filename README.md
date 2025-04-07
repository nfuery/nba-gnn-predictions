# GATv2-TCN for NBA Player Prop Prediction

Created by **Noah Fuery** and **Chris Ton-That**

This project leverages a hybrid model architecture combining **GATv2 (Graph Attention Network v2)** and **Temporal Convolutional Networks (TCN)** to predict NBA player performance statistics, particularly focusing on player props such as points, rebounds, assists, etc. The model incorporates **team and position embeddings** for improved player context.

---

## Project Structure

### `gatv2tcn.py`

This is the core model architecture file. It contains:

- `GATv2Conv`: A custom PyTorch Geometric implementation of the GATv2 layer with enhanced attention computation.
- `ChebConvAttention`, `SpatialAttention`, `TemporalAttention`: Layers inspired by ASTGCN for modeling spatiotemporal relationships.
- `ASTGCNBlock`: Combines spatial and temporal attention, GATv2Conv, and convolution layers.
- `GATv2TCN`: A lighter version focusing on GATv2 + TCN without Chebyshev layers.
- `ASTGCN`: Full attention-based spatial-temporal GCN model structure.

### `bets.py`

This script is for **running inference** on current player props (i.e., “Pick'em” predictions). It:

- Loads a trained model and embeddings.
- Feeds in recent game sequences.
- Predicts stats and evaluates against the target over/under lines.
- Outputs a detailed accuracy report and DataFrame summary.

### `teamPosResults.py`

This script performs **model evaluation and embedding visualization**:

- Generates **team vs team attention heatmaps** for specific matchups.
- Visualizes **team** and **position** embeddings in 2D.
- Reports RMSE, MAE, MAPE, and correlation on the test set.

### `teamPosTraining.py`

Handles **model training**, including:

- Custom data preprocessing pipeline.
- Train/validation/test splits with temporal graph snapshots.
- Team and position embedding layers.
- Training loop with dynamic learning rate scheduler.
- Saves best performing model and embeddings.

---

## Data Files

Expected pickle files in the `/data/` directory and root:

- `X_seq.pkl`: Player stat sequences over time.
- `G_seq.pkl`: Graph snapshots per game day, where nodes = players.
- `player_id2team.pkl`, `player_id2position.pkl`, `player_id2name.pkl`: Metadata mappings used for embedding and analysis.

---

## Model Highlights

- **Inputs**: 10-day rolling window of player stats + team & position embeddings.
- **Outputs**: Predictions for [PTS, AST, REB, TO, STL, BLK].
- **Evaluation**: Reports accuracy on daily “pick’em” props and standard metrics (MAE, RMSE, etc.)
- **Graph Construction**: Edges created if players share game time (>10 minutes) using disjoint complete subgraphs.

---

## Installation

```bash
pip install -r requirements.txt
```

Make sure you have:

- PyTorch with GPU support
- `torch_geometric`
- `nba_api`
- `networkx`, `matplotlib`, `scikit-learn`, `pandas`, `tqdm`

---

## Running the Project

**To evaluate predictions for today's props:**

```bash
python bets.py
```

**To train from scratch:**

```bash
python teamPosTraining.py
```

**To analyze results and generate visualizations:**

```bash
python teamPosResults.py
```

---

## Notes

- Make sure all pickled data files are correctly placed.
- Trained model weights will be saved under `model/gatv2tcn-team-position-embedding/`.
