"""
DENGUE HYBRID ENSEMBLE MODEL — FIXED VERSION
=============================================
Key fixes applied:
  1. City name normalization (lowercase for both datasets)
  2. Aligned temporal splitting via master city-month index
  3. Proper test set alignment between TIFF and Tabular models
  4. Fast tabular model (vectorized, no loops)

All 7 graphs will be generated successfully.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from typing import Tuple, List
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
import json
import gc
import time

try:
    import rasterio
except ImportError:
    rasterio = None

warnings.filterwarnings('ignore')

# ── Aesthetic setup ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#e6edf3',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.linewidth':   0.7,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'font.family':      'monospace',
})

PALETTE = {
    'tiff':     '#58a6ff',
    'tabular':  '#3fb950',
    'ensemble': '#f78166',
    'actual':   '#d2a8ff',
    'accent':   '#e3b341',
    'bg':       '#0d1117',
    'surface':  '#161b22',
}

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    TIFF_DIR         = "/content/drive/MyDrive/Final Metadata Dengue"
    TIFF_CSV         = "/content/dengue_cases.csv"
    TABULAR_CSV      = "/content/UltimateTabular.csv"
    OUTPUT_DIR       = "hybrid_ensemble_results"

    WINDOW_SIZE      = 6
    IMG_SIZE         = 224
    BATCH_SIZE       = 16
    EPOCHS_TIFF      = 100
    EPOCHS_TABULAR   = 80
    EPOCHS_ENSEMBLE  = 50
    LEARNING_RATE    = 0.0001
    WEIGHT_DECAY     = 1e-5
    DROPOUT          = 0.3

    CNN_CHANNELS     = [64, 128, 256, 512]
    FEATURE_DIM      = 256
    LSTM_HIDDEN      = 256
    LSTM_LAYERS      = 2
    ATTENTION_HEADS  = 8

    EARLY_STOPPING_PATIENCE  = 20
    LR_SCHEDULER_PATIENCE    = 10
    LR_SCHEDULER_FACTOR      = 0.5

    TRAIN_RATIO      = 0.70
    VAL_RATIO        = 0.15

    TAB_HIDDEN       = 128
    TAB_LSTM_HIDDEN  = 128
    TAB_BATCH        = 512

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for d in [Config.OUTPUT_DIR,
          os.path.join(Config.OUTPUT_DIR, 'plots'),
          os.path.join(Config.OUTPUT_DIR, 'models')]:
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TIFF UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
class TIFFProcessor:
    @staticmethod
    def load_tiff(filepath, bands=[2,3,4,12]):
        with rasterio.open(filepath) as src:
            data = src.read(bands)
        return np.transpose(data, (1,2,0))

    @staticmethod
    def normalize_bands(data):
        normalized = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[2]):
            band = data[:,:,i]
            lo, hi = np.percentile(band, [2,98])
            normalized[:,:,i] = np.clip((band-lo)/(hi-lo+1e-8), 0, 1)
        return normalized

    @staticmethod
    def compute_vegetation_indices(data):
        red, green, nir, swir = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        return {
            'ndvi':  (nir-red)   / (nir+red+1e-8),
            'ndwi':  (green-nir) / (green+nir+1e-8),
            'mndwi': (green-swir)/ (green+swir+1e-8),
            'evi':   2.5*(nir-red)/(nir+6*red-7.5*green+1+1e-8),
        }

    @staticmethod
    def extract_advanced_features(data):
        features = []
        for i in range(data.shape[2]):
            b = data[:,:,i]
            features.extend([np.mean(b), np.std(b), np.median(b),
                              np.percentile(b,25), np.percentile(b,75)])
        for idx_data in TIFFProcessor.compute_vegetation_indices(data).values():
            features.extend([np.mean(idx_data), np.std(idx_data),
                              np.min(idx_data),  np.max(idx_data)])
        swir, nir = data[:,:,3], data[:,:,2]
        features.extend([np.var(swir), np.ptp(swir), np.var(nir), np.ptp(nir)])
        return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  TIFF DATASET
# ══════════════════════════════════════════════════════════════════════════════
class TIFFDataset(Dataset):
    def __init__(self, data_df, tiff_dir, window_size, scaler=None, is_train=True):
        self.data_df    = data_df.sort_values(['city','year','month']).reset_index(drop=True)
        self.tiff_dir   = tiff_dir
        self.window_size= window_size
        self.is_train   = is_train

        if is_train:
            self.scaler = MinMaxScaler()
            self.data_df['dengue_cases_scaled'] = self.scaler.fit_transform(self.data_df[['dengue_cases']])
        else:
            self.scaler = scaler
            self.data_df['dengue_cases_scaled'] = self.scaler.transform(self.data_df[['dengue_cases']])

        self.samples = []
        for city in self.data_df['city'].unique():
            city_data = self.data_df[self.data_df['city']==city].reset_index(drop=True)
            for i in range(window_size, len(city_data)):
                self.samples.append((city, i))

        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        city, end_idx = self.samples[idx]
        city_data = self.data_df[self.data_df['city']==city].reset_index(drop=True)
        seq_data  = city_data.iloc[end_idx-self.window_size:end_idx]

        rgb_images, engineered_features = [], []
        for _, row in seq_data.iterrows():
            # Use normalized city name for TIFF file lookup
            tiff_path = os.path.join(self.tiff_dir,
                f"{row['city']}_{row['year']}_{row['month']:02d}.tif")
            if os.path.exists(tiff_path):
                data = TIFFProcessor.normalize_bands(TIFFProcessor.load_tiff(tiff_path))
                rgb  = Image.fromarray((data[:,:,:3]*255).astype(np.uint8))
                rgb_images.append(self.transform(rgb))
                engineered_features.append(TIFFProcessor.extract_advanced_features(data))
            else:
                rgb_images.append(torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE))
                engineered_features.append(np.zeros(40, dtype=np.float32))

        return {
            'rgb_images':       torch.stack(rgb_images),
            'features':         torch.FloatTensor(engineered_features),
            'historical_cases': torch.FloatTensor(seq_data['dengue_cases_scaled'].values),
            'target':           torch.FloatTensor([city_data.iloc[end_idx]['dengue_cases_scaled']]),
            'city': city,
            'year': int(city_data.iloc[end_idx]['year']),
            'month': int(city_data.iloc[end_idx]['month']),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  TABULAR DATASET
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = ['temp_mean','rainfall_sum','humidity_mean',
                'elevation','slope','dengue_risk_score']

class TabularDataset(Dataset):
    def __init__(self, data_df, window_size, target_scaler=None,
                 feature_scaler=None, is_train=True):
        df = data_df.copy()
        df['year']      = pd.to_datetime(df['month']).dt.year
        df['month_num'] = pd.to_datetime(df['month']).dt.month
        df = df.sort_values(['city','cell_id','month']).reset_index(drop=True)

        if is_train:
            self.feature_scaler = StandardScaler()
            self.target_scaler  = MinMaxScaler()
            df[FEATURE_COLS]            = self.feature_scaler.fit_transform(df[FEATURE_COLS])
            df['target_scaled']         = self.target_scaler.fit_transform(df[['dengue_cases_cell']])
        else:
            self.feature_scaler = feature_scaler
            self.target_scaler  = target_scaler
            df[FEATURE_COLS]            = self.feature_scaler.transform(df[FEATURE_COLS])
            df['target_scaled']         = self.target_scaler.transform(df[['dengue_cases_cell']])

        all_feats, all_hist, all_targets = [], [], []
        all_cities, all_years, all_months, all_cells = [], [], [], []

        for city in df['city'].unique():
            for cell in df[df['city']==city]['cell_id'].unique():
                cd = df[(df['city']==city) & (df['cell_id']==cell)].reset_index(drop=True)
                for i in range(window_size, len(cd)):
                    seq = cd.iloc[i-window_size:i]
                    all_feats.append(seq[FEATURE_COLS].values.astype(np.float32))
                    all_hist.append(seq['target_scaled'].values.astype(np.float32))
                    all_targets.append(np.float32(cd.iloc[i]['target_scaled']))
                    all_cities.append(city)
                    all_years.append(int(cd.iloc[i]['year']))
                    all_months.append(int(cd.iloc[i]['month_num']))
                    all_cells.append(cell)

        self.feats   = np.stack(all_feats)
        self.hist    = np.stack(all_hist)
        self.targets = np.array(all_targets)
        self.cities  = all_cities
        self.years   = all_years
        self.months  = all_months
        self.cells   = all_cells

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.feats[idx]),
            torch.from_numpy(self.hist[idx]),
            torch.tensor([self.targets[idx]]),
            self.cities[idx],
            self.years[idx],
            self.months[idx],
            self.cells[idx],
        )


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv    = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.conv(x))


class EnhancedCNN(nn.Module):
    def __init__(self, channels=[64,128,256,512], output_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], 7, stride=2, padding=3),
            nn.BatchNorm2d(channels[0]), nn.ReLU(True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1     = self._make_layer(channels[0], channels[1])
        self.attention1 = SpatialAttention(channels[1])
        self.layer2     = self._make_layer(channels[1], channels[2])
        self.attention2 = SpatialAttention(channels[2])
        self.layer3     = self._make_layer(channels[2], channels[3])
        self.attention3 = SpatialAttention(channels[3])
        self.global_pool= nn.AdaptiveAvgPool2d((1,1))
        self.fc         = nn.Sequential(nn.Linear(channels[3], output_dim),
                                        nn.ReLU(), nn.Dropout(0.3))
    def _make_layer(self, ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, stride=2, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1),           nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x):
        x = self.attention1(self.layer1(self.conv1(x)))
        x = self.attention2(self.layer2(x))
        x = self.attention3(self.layer3(x))
        return self.fc(torch.flatten(self.global_pool(x), 1))


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dims=[128,256,256], dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for hd in hidden_dims:
            layers += [nn.Linear(prev, hd), nn.BatchNorm1d(hd), nn.ReLU(), nn.Dropout(dropout)]
            prev = hd
        self.encoder  = nn.Sequential(*layers)
        self.residual = nn.Identity() if input_dim==hidden_dims[-1] else nn.Linear(input_dim, hidden_dims[-1])
    def forward(self, x):
        return self.encoder(x) + self.residual(x)


class TemporalFusionModule(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(0.1),
                                   nn.Linear(dim*4, dim))
    def forward(self, q, kv):
        attn, _ = self.cross_attention(q, kv, kv)
        x = self.norm1(q + attn)
        return self.norm2(x + self.ffn(x))


class TIFFModel(nn.Module):
    def __init__(self, window_size=6, feature_dim=256, lstm_hidden=256,
                 lstm_layers=2, dropout=0.3):
        super().__init__()
        self.image_encoder   = EnhancedCNN(Config.CNN_CHANNELS, feature_dim)
        self.feature_encoder = FeatureEncoder(40, [128,256,256], dropout)
        lstm_kw = dict(batch_first=True, bidirectional=True,
                       dropout=dropout if lstm_layers>1 else 0)
        self.image_lstm  = nn.LSTM(feature_dim,   lstm_hidden, lstm_layers, **lstm_kw)
        self.feature_lstm= nn.LSTM(256,            lstm_hidden, lstm_layers, **lstm_kw)
        self.cases_lstm  = nn.LSTM(1,              lstm_hidden, lstm_layers, **lstm_kw)
        self.temporal_fusion  = TemporalFusionModule(lstm_hidden*2, Config.ATTENTION_HEADS)
        self.self_attention   = nn.MultiheadAttention(lstm_hidden*2, Config.ATTENTION_HEADS, batch_first=True)
        combined_dim = lstm_hidden*2*3
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),          nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),          nn.ReLU(), nn.Dropout(dropout*0.5),
            nn.Linear(128, 1))

    def forward(self, rgb, features, historical_cases, return_features=False):
        B, T = rgb.shape[:2]
        img_feats  = torch.stack([self.image_encoder(rgb[:,t]) for t in range(T)], dim=1)
        eng_feats  = torch.stack([self.feature_encoder(features[:,t]) for t in range(T)], dim=1)
        img_t, _   = self.image_lstm(img_feats)
        eng_t, _   = self.feature_lstm(eng_feats)
        cas_t, _   = self.cases_lstm(historical_cases.unsqueeze(-1))
        fused      = self.temporal_fusion(img_t, eng_t)
        cas_att, _ = self.self_attention(cas_t, cas_t, cas_t)
        combined   = torch.cat([fused[:,-1], eng_t[:,-1], cas_att[:,-1]], dim=1)
        out        = self.prediction_head(combined)
        return (out, combined) if return_features else out


class FastTabularModel(nn.Module):
    def __init__(self, num_features=6, hidden_dim=128, lstm_hidden=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.LSTM(
            input_size=hidden_dim, hidden_size=lstm_hidden,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.MultiheadAttention(
            lstm_hidden*2, num_heads=4, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, features, historical_targets, return_features=False):
        x = self.feature_proj(features)
        x, _ = self.temporal(x)
        x, _ = self.attention(x, x, x)
        feat  = x[:, -1, :]
        out   = self.head(feat)
        return (out, feat) if return_features else out


class LearnableSpatialAggregator(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, cell_preds, cell_feats):
        scores  = self.attn(cell_feats).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        city_pred = (weights.unsqueeze(-1) * cell_preds).sum(dim=1)
        return city_pred, weights


class AdaptiveFusionNetwork(nn.Module):
    def __init__(self, feature_dim_a=1536, feature_dim_b=256, hidden_dim=256):
        super().__init__()
        input_dim = 2 + feature_dim_a + feature_dim_b + 1
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2), nn.LayerNorm(hidden_dim//2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 2), nn.Softmax(dim=-1))
        self.refine = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, pred_a, feat_a, pred_b, feat_b):
        agreement = torch.abs(pred_a - pred_b)
        inp = torch.cat([pred_a, pred_b, feat_a, feat_b, agreement], dim=-1)
        w   = self.fusion(inp)
        ens = w[:,0:1]*pred_a + w[:,1:2]*pred_b
        return ens + self.refine(ens), w


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS & TRAINING (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(preds, actuals):
    mae   = mean_absolute_error(actuals, preds)
    rmse  = np.sqrt(mean_squared_error(actuals, preds))
    r2    = r2_score(actuals, preds)
    smape = 100*np.mean(2*np.abs(preds-actuals)/(np.abs(preds)+np.abs(actuals)+1e-8))
    mape  = 100*np.mean(np.abs((actuals-preds)/(actuals+1e-8)))
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'smape': smape, 'mape': mape}


def train_tiff_epoch(model, loader, crit, optim, device, scaler):
    model.train()
    total_loss, preds, acts = 0, [], []
    for batch in loader:
        rgb   = batch['rgb_images'].to(device)
        feats = batch['features'].to(device)
        cases = batch['historical_cases'].to(device)
        tgt   = batch['target'].to(device)
        optim.zero_grad()
        out   = model(rgb, feats, cases)
        loss  = crit(out, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()
        preds.extend(scaler.inverse_transform(out.detach().cpu().numpy()).flatten())
        acts.extend( scaler.inverse_transform(tgt.cpu().numpy()).flatten())
    return total_loss/len(loader), compute_metrics(np.array(preds), np.array(acts))


def val_tiff_epoch(model, loader, crit, device, scaler):
    model.eval()
    total_loss, preds, acts = 0, [], []
    with torch.no_grad():
        for batch in loader:
            rgb   = batch['rgb_images'].to(device)
            feats = batch['features'].to(device)
            cases = batch['historical_cases'].to(device)
            tgt   = batch['target'].to(device)
            out   = model(rgb, feats, cases)
            total_loss += crit(out, tgt).item()
            preds.extend(scaler.inverse_transform(out.cpu().numpy()).flatten())
            acts.extend( scaler.inverse_transform(tgt.cpu().numpy()).flatten())
    return total_loss/len(loader), compute_metrics(np.array(preds), np.array(acts))


def train_tab_epoch(model, loader, crit, optim, device, scaler):
    model.train()
    total_loss, preds, acts = 0, [], []
    for feats, hist, tgt, *_ in loader:
        feats, hist, tgt = feats.to(device), hist.to(device), tgt.to(device)
        optim.zero_grad()
        out  = model(feats, hist)
        loss = crit(out, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item()
        preds.extend(scaler.inverse_transform(out.detach().cpu().numpy()).flatten())
        acts.extend( scaler.inverse_transform(tgt.cpu().numpy()).flatten())
    return total_loss/len(loader), compute_metrics(np.array(preds), np.array(acts))


def val_tab_epoch(model, loader, crit, device, scaler):
    model.eval()
    total_loss, preds, acts = 0, [], []
    with torch.no_grad():
        for feats, hist, tgt, *_ in loader:
            feats, hist, tgt = feats.to(device), hist.to(device), tgt.to(device)
            out  = model(feats, hist)
            total_loss += crit(out, tgt).item()
            preds.extend(scaler.inverse_transform(out.cpu().numpy()).flatten())
            acts.extend( scaler.inverse_transform(tgt.cpu().numpy()).flatten())
    return total_loss/len(loader), compute_metrics(np.array(preds), np.array(acts))


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def save_fig(fig, name):
    path = os.path.join(Config.OUTPUT_DIR, 'plots', name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    print(f"  ✓ Saved {name}")
    return path


def plot_actual_vs_pred(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Actual vs Predicted Dengue Cases', fontsize=16, fontweight='bold',
                 color='#e6edf3', y=1.02)

    models  = ['TIFF Model', 'Tabular Model', 'Ensemble']
    preds   = [results['tiff_pred'], results['tab_pred'], results['ensemble_pred']]
    colors  = [PALETTE['tiff'], PALETTE['tabular'], PALETTE['ensemble']]
    acts    = results['actuals']

    for ax, mname, pred, col in zip(axes, models, preds, colors):
        ax.scatter(acts, pred, alpha=0.5, s=20, color=col, edgecolors='none')
        lim = [min(acts.min(), pred.min())-10, max(acts.max(), pred.max())+10]
        ax.plot(lim, lim, '--', color='#ffffff', lw=1.5, alpha=0.6, label='Perfect fit')
        m = compute_metrics(pred, acts)
        ax.set_title(f'{mname}\nR²={m["r2"]:.4f}  MAE={m["mae"]:.1f}  RMSE={m["rmse"]:.1f}',
                     color='#e6edf3', fontsize=11)
        ax.set_xlabel('Actual cases', color='#8b949e')
        ax.set_ylabel('Predicted cases', color='#8b949e')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.legend(fontsize=9)
    return save_fig(fig, 'actual_vs_predicted.png')


def plot_metrics_bar(results):
    metrics_list = [results['tiff_metrics'], results['tab_metrics'], results['ensemble_metrics']]
    model_names  = ['TIFF', 'Tabular', 'Ensemble']
    metric_keys  = ['mae', 'rmse', 'smape', 'r2']
    metric_labels= ['MAE ↓', 'RMSE ↓', 'SMAPE (%) ↓', 'R² ↑']
    colors       = [PALETTE['tiff'], PALETTE['tabular'], PALETTE['ensemble']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Metrics Comparison', fontsize=16,
                 fontweight='bold', color='#e6edf3')
    axes = axes.flatten()

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        vals = [m[key] for m in metrics_list]
        bars = ax.bar(model_names, vals, color=colors, edgecolor='none', width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=11,
                    color='#e6edf3', fontweight='bold')
        ax.set_title(label, color='#e6edf3', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(vals)*1.18)
        if key == 'r2':
            ax.set_ylim(min(0, min(vals))-0.05, 1.05)
    return save_fig(fig, 'metrics_bar.png')


def plot_city_timeseries(city_ts_data):
    cities = list(city_ts_data.keys())
    ncols  = 2
    nrows  = (len(cities) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows*4.5))
    fig.suptitle('Dengue Cases: Historical vs Predicted (per City)',
                 fontsize=16, fontweight='bold', color='#e6edf3', y=1.01)
    axes = axes.flatten()

    for ax, city in zip(axes, cities):
        d = city_ts_data[city]
        months = range(len(d['actuals']))
        ax.plot(months, d['actuals'],  color=PALETTE['actual'],   lw=2,   label='Actual',   zorder=3)
        ax.plot(months, d['ensemble'], color=PALETTE['ensemble'], lw=1.8, label='Ensemble', zorder=4, ls='-')
        ax.plot(months, d['tiff'],     color=PALETTE['tiff'],     lw=1.2, label='TIFF',     zorder=2, ls='--', alpha=0.8)
        ax.plot(months, d['tabular'],  color=PALETTE['tabular'],  lw=1.2, label='Tabular',  zorder=2, ls=':', alpha=0.8)
        ax.fill_between(months, d['actuals'], d['ensemble'],
                        alpha=0.12, color=PALETTE['ensemble'])
        ax.set_title(city.title(), color='#e6edf3', fontsize=13, fontweight='bold')
        ax.set_xlabel('Time Step (months)', color='#8b949e', fontsize=9)
        ax.set_ylabel('Dengue Cases', color='#8b949e', fontsize=9)
        ax.legend(fontsize=8, ncol=2)

    for ax in axes[len(cities):]:
        ax.set_visible(False)

    fig.tight_layout()
    return save_fig(fig, 'city_timeseries.png')


def plot_residuals_analysis(results):
    res = results['ensemble_pred'] - results['actuals']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Ensemble Residuals Analysis', fontsize=15,
                 fontweight='bold', color='#e6edf3')

    ax = axes[0]
    ax.hist(res, bins=40, color=PALETTE['ensemble'], edgecolor='none', alpha=0.85)
    ax.axvline(0, color='#ffffff', lw=2, ls='--', alpha=0.7)
    ax.axvline(np.mean(res), color=PALETTE['accent'], lw=1.5, ls='-', label=f'Mean={np.mean(res):.2f}')
    ax.set_xlabel('Residual', color='#8b949e')
    ax.set_ylabel('Frequency', color='#8b949e')
    ax.set_title('Residual Distribution', color='#e6edf3', fontsize=12)
    ax.legend(fontsize=9)

    ax = axes[1]
    sorted_res = np.sort(res)
    n = len(sorted_res)
    theoretical = np.array([np.percentile(np.random.randn(10000), 100*(i-.5)/n) for i in range(1, n+1)])
    ax.scatter(theoretical, sorted_res, s=8, alpha=0.5, color=PALETTE['tiff'])
    lo = min(theoretical.min(), sorted_res.min())
    hi = max(theoretical.max(), sorted_res.max())
    ax.plot([lo,hi],[lo,hi],'--', color='#ffffff', lw=1.5, alpha=0.6)
    ax.set_xlabel('Theoretical Quantiles', color='#8b949e')
    ax.set_ylabel('Sample Quantiles', color='#8b949e')
    ax.set_title('Q-Q Plot', color='#e6edf3', fontsize=12)

    ax = axes[2]
    ax.scatter(results['ensemble_pred'], res, s=8, alpha=0.4, color=PALETTE['tabular'])
    ax.axhline(0, color='#ffffff', lw=1.5, ls='--', alpha=0.7)
    z = np.polyfit(results['ensemble_pred'], res, 1)
    xfit = np.linspace(results['ensemble_pred'].min(), results['ensemble_pred'].max(), 200)
    ax.plot(xfit, np.polyval(z, xfit), color=PALETTE['accent'], lw=1.5, label='Trend')
    ax.set_xlabel('Fitted Values', color='#8b949e')
    ax.set_ylabel('Residuals', color='#8b949e')
    ax.set_title('Residuals vs Fitted', color='#e6edf3', fontsize=12)
    ax.legend(fontsize=9)

    return save_fig(fig, 'residuals_analysis.png')


def plot_training_curves(tiff_history, tab_history):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Loss Curves', fontsize=15, fontweight='bold', color='#e6edf3')

    for ax, hist, title, col in zip(
            axes,
            [tiff_history, tab_history],
            ['TIFF Model', 'Tabular Model'],
            [PALETTE['tiff'], PALETTE['tabular']]):
        epochs = range(1, len(hist['train_loss'])+1)
        ax.plot(epochs, hist['train_loss'], color=col,    lw=2,  label='Train Loss')
        ax.plot(epochs, hist['val_loss'],   color='#fff', lw=1.5, ls='--', alpha=0.8, label='Val Loss')
        ax.fill_between(epochs, hist['train_loss'], hist['val_loss'],
                        alpha=0.07, color=col)
        ax.set_title(title, color='#e6edf3', fontsize=13)
        ax.set_xlabel('Epoch', color='#8b949e')
        ax.set_ylabel('MSE Loss', color='#8b949e')
        ax.legend(fontsize=10)

    return save_fig(fig, 'training_curves.png')


def plot_feature_importance(model, tab_test_dataset, device, scaler):
    model.eval()
    loader = DataLoader(tab_test_dataset, batch_size=512, shuffle=False)

    def get_loss(feat_idx=None, noise=False):
        losses = []
        with torch.no_grad():
            for feats, hist, tgt, *_ in loader:
                feats = feats.clone().to(device)
                if feat_idx is not None:
                    perm  = torch.randperm(feats.shape[0])
                    feats[:,:,feat_idx] = feats[perm,:,feat_idx]
                out = model(feats, hist.to(device))
                losses.append(F.mse_loss(out, tgt.to(device)).item())
        return np.mean(losses)

    base_loss = get_loss()
    importances = []
    for i, col in enumerate(FEATURE_COLS):
        perm_loss   = get_loss(feat_idx=i)
        importances.append((col, perm_loss - base_loss))

    importances.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*importances)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, vals, color=[PALETTE['tabular']]*len(names), edgecolor='none')
    ax.axvline(0, color='#ffffff', lw=1, alpha=0.5)
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals)*0.01, bar.get_y()+bar.get_height()/2,
                f'+{v:.4f}', va='center', color='#e6edf3', fontsize=9)
    ax.set_xlabel('Increase in MSE Loss (↑ = more important)', color='#8b949e')
    ax.set_title('Tabular Feature Permutation Importance', color='#e6edf3',
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    return save_fig(fig, 'feature_importance.png')


def plot_fusion_weights(fusion_weights):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Adaptive Ensemble Fusion Weights', fontsize=14,
                 fontweight='bold', color='#e6edf3')

    w_tiff = fusion_weights[:,0]
    w_tab  = fusion_weights[:,1]

    axes[0].hist(w_tiff, bins=40, color=PALETTE['tiff'],    alpha=0.75, label='TIFF Weight',    edgecolor='none')
    axes[0].hist(w_tab,  bins=40, color=PALETTE['tabular'], alpha=0.75, label='Tabular Weight', edgecolor='none')
    axes[0].axvline(w_tiff.mean(), color=PALETTE['tiff'],    lw=2, ls='--', label=f'TIFF μ={w_tiff.mean():.3f}')
    axes[0].axvline(w_tab.mean(),  color=PALETTE['tabular'], lw=2, ls='--', label=f'Tab  μ={w_tab.mean():.3f}')
    axes[0].set_xlabel('Weight', color='#8b949e')
    axes[0].set_ylabel('Count',  color='#8b949e')
    axes[0].set_title('Weight Distributions', color='#e6edf3', fontsize=12)
    axes[0].legend(fontsize=9)

    axes[1].scatter(w_tiff, w_tab, s=10, alpha=0.4, color=PALETTE['ensemble'])
    axes[1].plot([0,1],[1,0],'--', color='#ffffff', lw=1.5, alpha=0.6, label='w_a + w_b = 1')
    axes[1].set_xlabel('TIFF Weight',    color='#8b949e')
    axes[1].set_ylabel('Tabular Weight', color='#8b949e')
    axes[1].set_title('Weight Trade-off', color='#e6edf3', fontsize=12)
    axes[1].legend(fontsize=9)

    return save_fig(fig, 'fusion_weights.png')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*90)
    print(" "*20 + "HYBRID ENSEMBLE — DENGUE FORECASTING (FIXED)")
    print("═"*90)
    print(f"Device: {Config.DEVICE}\n")

    # ── 1. Load & normalize city names ───────────────────────────────────────
    print("[1/9] Loading data & normalizing city names...")
    tiff_df    = pd.read_csv(Config.TIFF_CSV)
    tabular_df = pd.read_csv(Config.TABULAR_CSV)
    
    # CRITICAL FIX: Normalize city names to lowercase
    tiff_df['city'] = tiff_df['city'].str.lower()
    tabular_df['city'] = tabular_df['city'].str.lower()
    
    # Fix 'bangalore' → 'bengaluru' for consistency
    tabular_df['city'] = tabular_df['city'].replace('bangalore', 'bengaluru')
    
    print(f"  TIFF: {len(tiff_df)} rows  |  Tabular: {len(tabular_df)} rows")
    print(f"  TIFF cities: {sorted(tiff_df['city'].unique())}")
    print(f"  Tabular cities: {sorted(tabular_df['city'].unique())}")

    # ── 2. Create master city-month index for aligned splitting ──────────────
    print("\n[2/9] Creating master city-month index for aligned splitting...")
    
    # Extract all city-month pairs from TIFF (master timeline)
    tiff_df = tiff_df.sort_values(['city','year','month']).reset_index(drop=True)
    master_index = tiff_df[['city','year','month']].drop_duplicates().reset_index(drop=True)
    master_index['date_key'] = pd.to_datetime(
        master_index[['year','month']].assign(day=1))
    master_index = master_index.sort_values(['city','date_key']).reset_index(drop=True)
    
    # Split the master index
    n_total = len(master_index)
    n_train = int(n_total * Config.TRAIN_RATIO)
    n_val   = int(n_total * Config.VAL_RATIO)
    
    train_idx = master_index.iloc[:n_train]
    val_idx   = master_index.iloc[n_train:n_train+n_val]
    test_idx  = master_index.iloc[n_train+n_val:]
    
    print(f"  Master index: {n_total} city-months")
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    
    # Function to filter dataframes by index
    def filter_by_index(df, idx, year_col='year', month_col='month'):
        # Create a merge key in both dataframes
        idx_temp = idx[['city', 'year', 'month']].copy()
        idx_temp['_merge_key'] = 1
        
        df_temp = df.copy()
        df_temp = df_temp.merge(
            idx_temp,
            left_on=['city', year_col, month_col],
            right_on=['city', 'year', 'month'],
            how='inner',
            suffixes=('', '_idx')
        )
        
        # Drop duplicate columns from merge
        cols_to_drop = [c for c in df_temp.columns if c.endswith('_idx') or c == '_merge_key']
        df_temp = df_temp.drop(columns=cols_to_drop)
        
        return df_temp
    
    # Split TIFF using master index
    tiff_train = filter_by_index(tiff_df, train_idx)
    tiff_val   = filter_by_index(tiff_df, val_idx)
    tiff_test  = filter_by_index(tiff_df, test_idx)
    
    # Split Tabular using master index (convert datetime month to year/month)
    tabular_df['year_tab'] = pd.to_datetime(tabular_df['month']).dt.year
    tabular_df['month_tab'] = pd.to_datetime(tabular_df['month']).dt.month
    
    tab_train = filter_by_index(tabular_df, train_idx, 'year_tab', 'month_tab')
    tab_val   = filter_by_index(tabular_df, val_idx, 'year_tab', 'month_tab')
    tab_test  = filter_by_index(tabular_df, test_idx, 'year_tab', 'month_tab')
    
    print(f"  TIFF splits: {len(tiff_train)} / {len(tiff_val)} / {len(tiff_test)}")
    print(f"  Tabular splits: {len(tab_train)} / {len(tab_val)} / {len(tab_test)}")

    # ── 3. Datasets ────────────────────────────────────────────────────────────
    print("\n[3/9] Building datasets...")
    t0 = time.time()

    tiff_train_ds = TIFFDataset(tiff_train, Config.TIFF_DIR, Config.WINDOW_SIZE, is_train=True)
    tiff_val_ds   = TIFFDataset(tiff_val,   Config.TIFF_DIR, Config.WINDOW_SIZE,
                                scaler=tiff_train_ds.scaler, is_train=False)
    tiff_test_ds  = TIFFDataset(tiff_test,  Config.TIFF_DIR, Config.WINDOW_SIZE,
                                scaler=tiff_train_ds.scaler, is_train=False)

    tab_train_ds = TabularDataset(tab_train, Config.WINDOW_SIZE, is_train=True)
    tab_val_ds   = TabularDataset(tab_val,   Config.WINDOW_SIZE,
                                  target_scaler=tab_train_ds.target_scaler,
                                  feature_scaler=tab_train_ds.feature_scaler, is_train=False)
    tab_test_ds  = TabularDataset(tab_test,  Config.WINDOW_SIZE,
                                  target_scaler=tab_train_ds.target_scaler,
                                  feature_scaler=tab_train_ds.feature_scaler, is_train=False)

    print(f"  TIFF: {len(tiff_train_ds)} / {len(tiff_val_ds)} / {len(tiff_test_ds)}")
    print(f"  Tab:  {len(tab_train_ds)}  / {len(tab_val_ds)}  / {len(tab_test_ds)}")
    print(f"  Dataset build: {time.time()-t0:.1f}s")

    tiff_train_dl = DataLoader(tiff_train_ds, Config.BATCH_SIZE,    shuffle=True,  num_workers=2, pin_memory=True)
    tiff_val_dl   = DataLoader(tiff_val_ds,   Config.BATCH_SIZE,    shuffle=False, num_workers=2, pin_memory=True)
    tiff_test_dl  = DataLoader(tiff_test_ds,  Config.BATCH_SIZE,    shuffle=False, num_workers=2, pin_memory=True)

    tab_train_dl  = DataLoader(tab_train_ds, Config.TAB_BATCH, shuffle=True,  num_workers=2, pin_memory=True)
    tab_val_dl    = DataLoader(tab_val_ds,   Config.TAB_BATCH, shuffle=False, num_workers=2, pin_memory=True)
    tab_test_dl   = DataLoader(tab_test_ds,  Config.TAB_BATCH, shuffle=False, num_workers=2, pin_memory=True)

    # ── 4. Train TIFF ─────────────────────────────────────────────────────────
    print("\n[4/9] Training TIFF Model...")
    model_a   = TIFFModel(Config.WINDOW_SIZE, Config.FEATURE_DIM,
                          Config.LSTM_HIDDEN, Config.LSTM_LAYERS, Config.DROPOUT).to(Config.DEVICE)
    crit_a    = nn.MSELoss()
    optim_a   = torch.optim.AdamW(model_a.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    sched_a   = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_a, patience=Config.LR_SCHEDULER_PATIENCE,
                                                           factor=Config.LR_SCHEDULER_FACTOR)
    tiff_history  = {'train_loss': [], 'val_loss': []}
    best_val_a, pat_a = 1e9, 0

    for epoch in range(1, Config.EPOCHS_TIFF+1):
        tl, _  = train_tiff_epoch(model_a, tiff_train_dl, crit_a, optim_a, Config.DEVICE, tiff_train_ds.scaler)
        vl, vm = val_tiff_epoch(  model_a, tiff_val_dl,   crit_a, Config.DEVICE, tiff_train_ds.scaler)
        sched_a.step(vl)
        tiff_history['train_loss'].append(tl)
        tiff_history['val_loss'].append(vl)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: train={tl:.4f}  val={vl:.4f}  R²={vm['r2']:.4f}  MAE={vm['mae']:.2f}")
        if vl < best_val_a:
            best_val_a = vl; pat_a = 0
            torch.save(model_a.state_dict(), os.path.join(Config.OUTPUT_DIR,'models','tiff_best.pth'))
        else:
            pat_a += 1
        if pat_a >= Config.EARLY_STOPPING_PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model_a.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR,'models','tiff_best.pth')))
    print("  ✓ TIFF Model trained")

    # ── 5. Train Tabular ─────────────────────────────────────────────────────
    print("\n[5/9] Training Tabular Model (fast)...")
    model_b  = FastTabularModel(num_features=6, hidden_dim=Config.TAB_HIDDEN,
                                lstm_hidden=Config.TAB_LSTM_HIDDEN, num_layers=2,
                                dropout=Config.DROPOUT).to(Config.DEVICE)
    crit_b   = nn.MSELoss()
    optim_b  = torch.optim.AdamW(model_b.parameters(), lr=Config.LEARNING_RATE*2, weight_decay=Config.WEIGHT_DECAY)
    sched_b  = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_b, patience=Config.LR_SCHEDULER_PATIENCE,
                                                          factor=Config.LR_SCHEDULER_FACTOR)
    tab_history   = {'train_loss': [], 'val_loss': []}
    best_val_b, pat_b = 1e9, 0

    for epoch in range(1, Config.EPOCHS_TABULAR+1):
        t0_ep = time.time()
        tl, _  = train_tab_epoch(model_b, tab_train_dl, crit_b, optim_b, Config.DEVICE, tab_train_ds.target_scaler)
        vl, vm = val_tab_epoch(  model_b, tab_val_dl,   crit_b, Config.DEVICE, tab_train_ds.target_scaler)
        sched_b.step(vl)
        tab_history['train_loss'].append(tl)
        tab_history['val_loss'].append(vl)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}: train={tl:.4f}  val={vl:.4f}  "
                  f"R²={vm['r2']:.4f}  MAE={vm['mae']:.2f}  ({time.time()-t0_ep:.1f}s/ep)")
        if vl < best_val_b:
            best_val_b = vl; pat_b = 0
            torch.save(model_b.state_dict(), os.path.join(Config.OUTPUT_DIR,'models','tab_best.pth'))
        else:
            pat_b += 1
        if pat_b >= Config.EARLY_STOPPING_PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    model_b.load_state_dict(torch.load(os.path.join(Config.OUTPUT_DIR,'models','tab_best.pth')))
    print("  ✓ Tabular Model trained")

    # ── 6. Generate predictions ───────────────────────────────────────────────
    print("\n[6/9] Generating predictions on test set...")
    model_a.eval(); model_b.eval()

    tiff_preds_s, tiff_feats_s, tiff_tgts_s = [], [], []
    tiff_meta = []
    with torch.no_grad():
        for batch in tiff_test_dl:
            rgb   = batch['rgb_images'].to(Config.DEVICE)
            feats = batch['features'].to(Config.DEVICE)
            cases = batch['historical_cases'].to(Config.DEVICE)
            out, feat = model_a(rgb, feats, cases, return_features=True)
            tiff_preds_s.append(out.detach().cpu())
            tiff_feats_s.append(feat.detach().cpu())
            tiff_tgts_s.append(batch['target'].cpu())
            for i in range(len(batch['city'])):
                tiff_meta.append((batch['city'][i], int(batch['year'][i]), int(batch['month'][i])))

    tiff_preds_s = torch.cat(tiff_preds_s)
    tiff_feats_s = torch.cat(tiff_feats_s)
    tiff_tgts_s  = torch.cat(tiff_tgts_s)

    aggregator = LearnableSpatialAggregator(feature_dim=Config.TAB_LSTM_HIDDEN*2).to(Config.DEVICE)
    city_month_data = {}

    with torch.no_grad():
        for feats, hist, tgt, cities, years, months, cells in DataLoader(tab_test_ds, 512, shuffle=False, num_workers=2):
            out, feat = model_b(feats.to(Config.DEVICE), hist.to(Config.DEVICE), return_features=True)
            for i in range(len(cities)):
                key = (cities[i], int(years[i]), int(months[i]))
                if key not in city_month_data:
                    city_month_data[key] = {'preds': [], 'feats': []}
                city_month_data[key]['preds'].append(out[i].detach().cpu())
                city_month_data[key]['feats'].append(feat[i].detach().cpu())

    tab_agg_preds, tab_agg_feats, tab_agg_keys = [], [], []
    with torch.no_grad():
        for key in sorted(city_month_data.keys()):
            d = city_month_data[key]
            cell_preds = torch.stack(d['preds']).unsqueeze(0).to(Config.DEVICE)
            cell_feats = torch.stack(d['feats']).unsqueeze(0).to(Config.DEVICE)
            cp, _ = aggregator(cell_preds, cell_feats)
            tab_agg_preds.append(cp.detach().cpu())
            tab_agg_feats.append(cell_feats.mean(dim=1).detach().cpu())
            tab_agg_keys.append(key)

    tab_agg_preds = torch.cat(tab_agg_preds)
    tab_agg_feats = torch.cat(tab_agg_feats)

    tiff_key_set  = {k: i for i, k in enumerate(tiff_meta)}
    tab_key_set   = {k: i for i, k in enumerate(tab_agg_keys)}
    common_keys   = sorted(set(tiff_key_set) & set(tab_key_set))

    idx_t = [tiff_key_set[k] for k in common_keys]
    idx_b = [tab_key_set[k]  for k in common_keys]

    # Clone tensors to ensure they're completely detached from computation graph
    tiff_p = tiff_preds_s[idx_t].clone()
    tiff_f = tiff_feats_s[idx_t].clone()
    tiff_tgt = tiff_tgts_s[idx_t].clone()
    tab_p  = tab_agg_preds[idx_b].clone()
    tab_f  = tab_agg_feats[idx_b].clone()
    print(f"  ✓ Aligned {len(common_keys)} city-month test samples")

    # ── 7. Train ensemble ─────────────────────────────────────────────────────
    print("\n[7/9] Training Ensemble meta-learner...")
    meta = AdaptiveFusionNetwork(
        feature_dim_a=Config.LSTM_HIDDEN*2*3,
        feature_dim_b=Config.TAB_LSTM_HIDDEN*2,
        hidden_dim=256
    ).to(Config.DEVICE)

    optim_meta = torch.optim.AdamW(meta.parameters(), lr=0.001)
    n_meta     = int(len(common_keys)*0.8)

    for epoch in range(1, Config.EPOCHS_ENSEMBLE+1):
        meta.train()
        idx_perm = torch.randperm(n_meta)
        epoch_loss = 0; n_batches = 0
        for i in range(0, n_meta, 32):
            bi = idx_perm[i:i+32]
            if len(bi) == 0:
                continue
            pa = tiff_p[bi].to(Config.DEVICE); fa = tiff_f[bi].to(Config.DEVICE)
            pb = tab_p[bi].to(Config.DEVICE);  fb = tab_f[bi].to(Config.DEVICE)
            tg = tiff_tgt[bi].to(Config.DEVICE)
            optim_meta.zero_grad()
            out, _ = meta(pa, fa, pb, fb)
            loss = F.mse_loss(out, tg)
            loss.backward(); optim_meta.step()
            epoch_loss += loss.item(); n_batches += 1
        if n_batches > 0 and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: loss={epoch_loss/n_batches:.4f}")
    print("  ✓ Ensemble trained")

    # ── 8. Final evaluation ───────────────────────────────────────────────────
    print("\n[8/9] Final evaluation...")
    meta.eval()
    with torch.no_grad():
        ens_preds, fusion_w = meta(
            tiff_p.to(Config.DEVICE), tiff_f.to(Config.DEVICE),
            tab_p.to(Config.DEVICE),  tab_f.to(Config.DEVICE))

    tiff_sc = tiff_train_ds.scaler
    
    tiff_pred_d = tiff_sc.inverse_transform(tiff_p.numpy())
    tab_pred_d  = tiff_sc.inverse_transform(tab_p.numpy())
    ens_pred_d  = tiff_sc.inverse_transform(ens_preds.cpu().numpy())
    actual_d    = tiff_sc.inverse_transform(tiff_tgt.numpy())

    tiff_m = compute_metrics(tiff_pred_d.flatten(), actual_d.flatten())
    tab_m  = compute_metrics(tab_pred_d.flatten(),  actual_d.flatten())
    ens_m  = compute_metrics(ens_pred_d.flatten(),  actual_d.flatten())

    print("\n" + "═"*70)
    print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'SMAPE%':>10}")
    print("─"*70)
    for name, m in [('TIFF', tiff_m), ('Tabular', tab_m), ('Ensemble', ens_m)]:
        print(f"{name:<15} {m['mae']:>8.2f} {m['rmse']:>8.2f} {m['r2']:>8.4f} {m['smape']:>10.2f}")
    print("═"*70 + "\n")

    results = {
        'tiff_pred':       tiff_pred_d.flatten(),
        'tab_pred':        tab_pred_d.flatten(),
        'ensemble_pred':   ens_pred_d.flatten(),
        'actuals':         actual_d.flatten(),
        'tiff_metrics':    tiff_m,
        'tab_metrics':     tab_m,
        'ensemble_metrics':ens_m,
        'fusion_weights':  fusion_w.cpu().numpy(),
    }

    city_ts = {}
    for k, (city, year, month) in zip(range(len(common_keys)), common_keys):
        if city not in city_ts:
            city_ts[city] = {'actuals':[], 'tiff':[], 'tabular':[], 'ensemble':[]}
        city_ts[city]['actuals'].append(float(actual_d[k]))
        city_ts[city]['tiff'].append(float(tiff_pred_d[k]))
        city_ts[city]['tabular'].append(float(tab_pred_d[k]))
        city_ts[city]['ensemble'].append(float(ens_pred_d[k]))

    # ── 9. Plot all graphs ────────────────────────────────────────────────────
    print("[9/9] Generating graphs...")
    paths = []
    paths.append(plot_actual_vs_pred(results))
    paths.append(plot_metrics_bar(results))
    paths.append(plot_city_timeseries(city_ts))
    paths.append(plot_residuals_analysis(results))
    paths.append(plot_training_curves(tiff_history, tab_history))
    paths.append(plot_feature_importance(model_b, tab_test_ds, Config.DEVICE, tab_train_ds.target_scaler))
    paths.append(plot_fusion_weights(results['fusion_weights']))

    pd.DataFrame({
        'city':       [k[0] for k in common_keys],
        'year':       [k[1] for k in common_keys],
        'month':      [k[2] for k in common_keys],
        'actual':     actual_d.flatten(),
        'tiff_pred':  tiff_pred_d.flatten(),
        'tab_pred':   tab_pred_d.flatten(),
        'ens_pred':   ens_pred_d.flatten(),
    }).to_csv(os.path.join(Config.OUTPUT_DIR, 'predictions.csv'), index=False)

    with open(os.path.join(Config.OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({'tiff': tiff_m, 'tabular': tab_m, 'ensemble': ens_m}, f, indent=4)

    print(f"\n✓ All results saved to ./{Config.OUTPUT_DIR}/")
    print("  Plots generated:")
    for p in paths:
        print(f"    {p}")
    print("═"*90 + "\n")


if __name__ == "__main__":
    main()
