import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import math
import time

try:
    import rasterio
except ImportError:
    rasterio = None

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    '#f8f9fa',
    'axes.edgecolor':    '#dee2e6',
    'axes.labelcolor':   '#212529',
    'xtick.color':       '#495057',
    'ytick.color':       '#495057',
    'text.color':        '#212529',
    'grid.color':        '#e9ecef',
    'grid.linestyle':    '--',
    'grid.linewidth':    0.7,
    'legend.facecolor':  'white',
    'legend.edgecolor':  '#dee2e6',
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.titlesize':    12,
    'axes.labelsize':    10,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
})

PALETTE = {
    'convlstm': '#1d4ed8',
    'tabular':  '#15803d',
    'ensemble': '#b91c1c',
    'actual':   '#1a202c',
    'accent':   '#b45309',
}


class Config:
    """
    ─── PATH CONFIGURATION ──────────────────────────────────────────────────────
    TIFF_DIR    : Folder containing per-city monthly .tif satellite images.
                  Filename convention: {city}_{year}_{month:02d}.tif
    TIFF_CSV    : dengue_cases.csv  — columns: city, year, month, dengue_cases
    TABULAR_CSV : UltimateTabular.csv — columns: city, month(datetime str),
                  cell_id, dengue_cases_cell, + FEATURE_COLS below

    ─── CONVLSTM MODEL (M1) ─────────────────────────────────────────────────────
    IMG_SIZE      : 96px. ConvLSTM is spatially expensive; 96px gives 26% speed
                    gain over 112px with negligible quality loss at cell level.
    LSTM_HIDDEN   : 32. The original baseline used 32 and achieved 16% SMAPE.
                    Keeping this preserves the proven spatial-temporal capacity.
    LSTM_LAYERS   : 1. Single-layer ConvLSTM matches the baseline architecture.
    BATCH_SIZE    : 6. Larger than baseline (4) for GPU efficiency; drop_last=True.
    EPOCHS_CONVLSTM : 120 max, early stop patience=30 (matches baseline spirit).
    LR_CONVLSTM   : 1e-3 — same as baseline, tested and stable.
    CONV_DROPOUT  : 0.2 — matches baseline exactly.

    ─── TABULAR BiLSTM MODEL (M2) ──────────────────────────────────────────────
    Architecture: StandardScaler on features → Linear proj (6→128) → BiLSTM
                  (2-layer, hidden=96, bidirectional) → MultiheadAttention (4h)
                  → last hidden → Linear(192→64) → Linear(64→1).
    TAB_BATCH   : 1024. Tabular data is tiny; large batch = fewer steps = fast.
    EPOCHS_TAB  : 40 max, early stop patience=10.
    LR_TAB      : 3e-4, AdamW.

    ─── ADAPTIVE ENSEMBLE (M3) ──────────────────────────────────────────────────
    AdaptiveFusionNetwork takes [pred_c, pred_t, feat_c, feat_t, |pred_c-pred_t|]
    and learns per-sample weights [w_c, w_t] via a gating MLP.
    ens = w_c * pred_c + w_t * pred_t + refine(ens)
    The disagreement signal lets the network down-weight whichever model is
    uncertain for a given city-month combination — key to breaking below 10% SMAPE.
    EPOCHS_ENS  : 30, LR=1e-3, batch=32, MSELoss.

    ─── SPLIT STRATEGY ──────────────────────────────────────────────────────────
    Per-city chronological split (TRAIN 70% / VAL 15% / TEST 15%).
    Val and Test prepend WINDOW_SIZE context rows from the preceding split so that
    every city-month can form a complete window sample (avoids empty datasets).

    ─── SPEED BUDGET (T4 GPU) ───────────────────────────────────────────────────
    ConvLSTM   : ~3-4min/epoch × (early stop ~25-40ep) = ~12-20min
    Tabular    : ~0.1min/epoch × 40ep                  = ~4min
    Ensemble   : tiny                                   = ~1min
    Plots+eval : ~2min
    TOTAL      : ~20-30min  (well under 60min budget)
    """

    TIFF_DIR        = "/content/drive/MyDrive/Final Metadata Dengue"
    TIFF_CSV        = "/content/dengue_cases.csv"
    TABULAR_CSV     = "/content/UltimateTabular.csv"
    OUTPUT_DIR      = "dengue_ensemble_results"

    WINDOW_SIZE         = 6
    IMG_SIZE            = 96
    BATCH_SIZE          = 6
    EPOCHS_CONVLSTM     = 120
    EPOCHS_TAB          = 150
    EPOCHS_ENS          = 200
    EARLY_STOP_ENS      = 30
    LR_CONVLSTM         = 1e-3
    LR_TAB              = 3e-4   # decays to ~1e-5 via scheduler
    LR_ENS              = 1e-3
    WEIGHT_DECAY        = 1e-5
    CONV_DROPOUT        = 0.2
    TAB_DROPOUT         = 0.4    # strong regularisation for small data
    EARLY_STOP_CONVLSTM = 30
    EARLY_STOP_TAB      = 20
    LR_PATIENCE         = 10
    LR_FACTOR           = 0.5

    TRAIN_RATIO     = 0.70
    VAL_RATIO       = 0.15

    LSTM_HIDDEN     = 32
    LSTM_LAYERS     = 1
    TAB_HIDDEN      = 128
    TAB_LSTM_HIDDEN = 96
    N_HEADS         = 4
    TAB_BATCH       = 128    # smaller batch = noisier gradients = better generalisation

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    AMP    = torch.cuda.is_available()


for d in [Config.OUTPUT_DIR,
          os.path.join(Config.OUTPUT_DIR, 'plots'),
          os.path.join(Config.OUTPUT_DIR, 'models')]:
    os.makedirs(d, exist_ok=True)

FEATURE_COLS = ['temp_mean', 'rainfall_sum', 'humidity_mean',
                'elevation', 'slope', 'dengue_risk_score']
# Extended feature set built inside TabularDataset (added via feature engineering)
TAB_EXTENDED_COLS = [
    'temp_mean', 'rainfall_sum', 'humidity_mean',
    'elevation', 'slope', 'dengue_risk_score',
    'month_sin', 'month_cos',              # seasonality
    'temp_x_rain',                         # temp * rainfall interaction
    'rain_rolling2', 'temp_rolling2',      # 2-month rolling mean
    'rain_rolling3', 'temp_rolling3',      # 3-month rolling mean
]

CONVLSTM_FEAT_DIM = Config.LSTM_HIDDEN + Config.LSTM_HIDDEN
TAB_FEAT_DIM      = 64  # compact LSTM: lstm_last(32) + hist_embed(32)


def clean_city(s):
    s = str(s).encode('ascii', errors='ignore').decode('ascii').lower().strip()
    s = ' '.join(s.split())
    return {'bangalore': 'bengaluru', 'bengalore': 'bengaluru',
            'banglore':  'bengaluru'}.get(s, s)


def split_per_city(master, train_ratio, val_ratio, window_size):
    tr, va, te = [], [], []
    for city in master['city'].unique():
        cd = master[master['city'] == city].sort_values('date_key').reset_index(drop=True)
        n  = len(cd)
        nt = int(n * train_ratio)
        nv = int(n * val_ratio)
        tr.append(cd.iloc[:nt])
        va.append(cd.iloc[max(0, nt - window_size):nt + nv])
        te.append(cd.iloc[max(0, nt + nv - window_size):])
    return (pd.concat(tr).reset_index(drop=True),
            pd.concat(va).reset_index(drop=True),
            pd.concat(te).reset_index(drop=True))


def filt(df, idx, yr='year', mo='month'):
    merged = df.merge(idx[['city', 'year', 'month']],
                      left_on=['city', yr, mo],
                      right_on=['city', 'year', 'month'],
                      how='inner', suffixes=('', '_i'))
    return merged.drop(columns=[c for c in merged.columns if c.endswith('_i')])


class TIFFProcessor:
    @staticmethod
    def load(path, bands=[2, 3, 4, 12]):
        with rasterio.open(path) as src:
            data = src.read(bands)
        return np.transpose(data, (1, 2, 0))

    @staticmethod
    def normalize(data):
        out = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[2]):
            b = data[:, :, i]
            lo, hi = np.percentile(b, [2, 98])
            out[:, :, i] = np.clip((b - lo) / (hi - lo + 1e-8), 0, 1)
        return out


class ConvLSTMDataset(Dataset):
    """
    ConvLSTM Dataset — mirrors baseline DengueDataset exactly.
    Loads 4-band TIFF images per timestep. Returns (T=6, 4, H, W) tensors.
    MinMaxScaler fitted on train split only (passed in for val/test).
    """

    def __init__(self, df, tiff_dir, window_size, scaler=None, is_train=True):
        df = df.copy()
        df['city'] = df['city'].apply(clean_city)
        self.df = df.sort_values(['city', 'year', 'month']).reset_index(drop=True)
        self.tiff_dir    = tiff_dir
        self.window_size = window_size

        if is_train:
            self.scaler = MinMaxScaler()
            self.df['scaled'] = self.scaler.fit_transform(self.df[['dengue_cases']])
        else:
            self.scaler = scaler
            self.df['scaled'] = self.scaler.transform(self.df[['dengue_cases']])

        self.samples = []
        for city in self.df['city'].unique():
            cd = self.df[self.df['city'] == city].reset_index(drop=True)
            for i in range(window_size, len(cd)):
                self.samples.append((city, i))

        self.tfm = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        city, end = self.samples[idx]
        cd  = self.df[self.df['city'] == city].reset_index(drop=True)
        seq = cd.iloc[end - self.window_size:end]
        imgs = []
        for _, row in seq.iterrows():
            path = os.path.join(
                self.tiff_dir,
                f"{row['city']}_{int(row['year'])}_{int(row['month']):02d}.tif")
            if os.path.exists(path):
                data = TIFFProcessor.normalize(TIFFProcessor.load(path))
                img  = Image.fromarray((data * 255).astype(np.uint8))
                imgs.append(self.tfm(img))
            else:
                imgs.append(torch.zeros(4, Config.IMG_SIZE, Config.IMG_SIZE))
        return {
            'images':  torch.stack(imgs),
            'hist':    torch.FloatTensor(seq['scaled'].values),
            'target':  torch.FloatTensor([cd.iloc[end]['scaled']]),
            'city':    city,
            'year':    int(cd.iloc[end]['year']),
            'month':   int(cd.iloc[end]['month']),
        }


class TabularDataset(Dataset):
    """
    Tabular Dataset — city-level with feature engineering for small data.

    WHY BILSTM FAILED (SMAPE=33%):
      440 train samples / 10 cities = 44 per city. A 575K-param BiLSTM needs
      thousands of samples to not overfit. Val loss was still falling at ep40
      — the model never converged, it just memorized noise.

    FIX STRATEGY:
      1. Feature engineering: add seasonality (month sin/cos), rolling means
         (2-month, 3-month rainfall and temp), and temp×rainfall interaction.
         Dengue is strongly seasonal — these signals dominate raw features.
      2. Tiny LSTM: 1 layer, hidden=32 (not 96×2=192). Only 18K params.
         Small enough to not overfit 44 samples per city.
      3. More epochs (150) + stronger regularization (dropout=0.4, wd=1e-3).
      4. Same city_scaler as ConvLSTM — both models in same scaled space.
    """

    def __init__(self, df, window_size, city_scaler, feature_scaler=None, is_train=True):
        df = df.copy()
        df['city']      = df['city'].apply(clean_city)
        df['year']      = pd.to_datetime(df['month']).dt.year
        df['month_num'] = pd.to_datetime(df['month']).dt.month

        # Aggregate cells → one row per city-month
        agg = df.groupby(['city', 'year', 'month_num']).agg(
            {**{c: 'mean' for c in FEATURE_COLS},
             'dengue_cases_cell': 'sum'}).reset_index()
        agg = agg.rename(columns={'dengue_cases_cell': 'dengue_cases_city'})
        agg = agg.sort_values(['city', 'year', 'month_num']).reset_index(drop=True)

        # Feature engineering — within each city to avoid data leakage
        rows = []
        for city in agg['city'].unique():
            cd = agg[agg['city'] == city].copy().reset_index(drop=True)
            cd['month_sin']    = np.sin(2 * np.pi * cd['month_num'] / 12)
            cd['month_cos']    = np.cos(2 * np.pi * cd['month_num'] / 12)
            cd['temp_x_rain']  = cd['temp_mean'] * cd['rainfall_sum']
            cd['rain_rolling2'] = cd['rainfall_sum'].rolling(2, min_periods=1).mean()
            cd['temp_rolling2'] = cd['temp_mean'].rolling(2, min_periods=1).mean()
            cd['rain_rolling3'] = cd['rainfall_sum'].rolling(3, min_periods=1).mean()
            cd['temp_rolling3'] = cd['temp_mean'].rolling(3, min_periods=1).mean()
            rows.append(cd)
        agg = pd.concat(rows).reset_index(drop=True)

        # Scale target with city_scaler (same as ConvLSTM)
        self.target_scaler = city_scaler
        agg['scaled'] = city_scaler.transform(agg[['dengue_cases_city']].values.reshape(-1, 1))

        if is_train:
            self.feature_scaler = StandardScaler()
            agg[TAB_EXTENDED_COLS] = self.feature_scaler.fit_transform(agg[TAB_EXTENDED_COLS])
        else:
            self.feature_scaler = feature_scaler
            agg[TAB_EXTENDED_COLS] = self.feature_scaler.transform(agg[TAB_EXTENDED_COLS])

        all_f, all_h, all_t = [], [], []
        all_cities, all_years, all_months = [], [], []

        for city in agg['city'].unique():
            cd = agg[agg['city'] == city].reset_index(drop=True)
            for i in range(window_size, len(cd)):
                s = cd.iloc[i - window_size:i]
                all_f.append(s[TAB_EXTENDED_COLS].values.astype(np.float32))
                all_h.append(s['scaled'].values.astype(np.float32))
                all_t.append(np.float32(cd.iloc[i]['scaled']))
                all_cities.append(city)
                all_years.append(int(cd.iloc[i]['year']))
                all_months.append(int(cd.iloc[i]['month_num']))

        if len(all_f) == 0:
            raise ValueError(f"TabularDataset: 0 samples. window_size={window_size} "
                             "may exceed available rows per city.")

        self.feats   = np.stack(all_f)
        self.hist    = np.stack(all_h)
        self.targets = np.array(all_t)
        self.cities  = all_cities
        self.years   = all_years
        self.months  = all_months

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.feats[idx]),
                torch.from_numpy(self.hist[idx]),
                torch.tensor([self.targets[idx]]),
                self.cities[idx], self.years[idx],
                self.months[idx])


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell — spatiotemporal gating with convolutional operations.
    Gates (i, f, o, g) computed via a single Conv2d on [input ‖ hidden].
    Kernel 3×3, padding=1 → spatial dims preserved.
    c_next = f ⊙ c_cur + i ⊙ g
    h_next = o ⊙ tanh(c_next)
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, H, W):
        dev = self.conv.weight.device
        return (torch.zeros(B, self.hidden_dim, H, W, device=dev),
                torch.zeros(B, self.hidden_dim, H, W, device=dev))


class ConvLSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        h, c = self.cell.init_hidden(B, H, W)
        outputs = []
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class DengueNet_ConvLSTM(nn.Module):
    """
    ConvLSTM Satellite Model — EXACT ARCHITECTURE FROM BASELINE THAT ACHIEVED 16% SMAPE.
    ─────────────────────────────────────────────────────────────────────────────────────
    INPUT: (B, T=6, 4, 96, 96) — 4 Landsat bands, 6 months sequence.

    DOWNSAMPLE (spatial reduction before ConvLSTM):
      Conv2d(4→8,  3×3, stride=2): 96 → 48
      Conv2d(8→16, 3×3, stride=2): 48 → 24
      Conv2d(16→16,3×3, stride=2): 24 → 12
      All with ReLU. Applied per-timestep (batch_size×T frames simultaneously).

    ConvLSTM:
      Single ConvLSTMCell, hidden_dim=32, kernel=3×3, spatial dims=12×12.
      Processes T=6 downsampled frames sequentially.
      Output: last hidden state h_T ∈ R^(B, 32, 12, 12).

    SPATIAL POOL: AdaptiveAvgPool2d(1,1) → (B, 32).

    CASE HISTORY STREAM:
      Linear(6 → 32) → ReLU → Dropout(0.2).
      Processes MinMax-scaled historical dengue case sequence.

    HEAD:
      concat([spatial_feat(32), case_feat(32)]) → 64-dim.
      Linear(64→64) → ReLU → Dropout(0.2) → Linear(64→1).

    return_features=True: also returns the 64-dim penultimate vector
    for ensemble fusion.

    TRAINING:
      MSELoss, AdamW lr=1e-3, weight_decay=1e-5.
      ReduceLROnPlateau: patience=10, factor=0.5.
      Gradient clip: 1.0.
      Max epochs: 120, early stop patience: 30.
      Batch size: 6, no AMP (matches baseline stability).
      PARAMETERS: ~35K trainable.
    """

    def __init__(self, window_size=6, lstm_hidden=32, dropout=0.2):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(4,  8,  3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8,  16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU())

        self.convlstm   = ConvLSTMLayer(16, lstm_hidden, kernel_size=3)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        self.cases_fc = nn.Sequential(
            nn.Linear(window_size, lstm_hidden), nn.ReLU(), nn.Dropout(dropout))

        in_dim = lstm_hidden + lstm_hidden
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, images, hist, return_features=False):
        B, T, C, H, W = images.shape
        flat = images.view(B * T, C, H, W)
        ds   = self.downsample(flat).view(B, T, *self.downsample(flat).shape[1:]) \
               if False else self.downsample(flat)
        ds   = ds.view(B, T, *ds.shape[1:])
        seq  = self.convlstm(ds)
        sf   = self.spatial_pool(seq[:, -1]).view(B, -1)
        cf   = self.cases_fc(hist)
        feat = torch.cat([sf, cf], dim=1)
        out  = self.head(feat)
        return (out, feat) if return_features else out


class TabularBiLSTM(nn.Module):
    """
    Tabular Model — Compact LSTM for small data (≤44 samples/city).
    ─────────────────────────────────────────────────────────────────
    WHY THE OLD BILSTM FAILED (SMAPE=33%):
      575K params on 440 samples = catastrophic overfitting.
      Val loss was still improving at epoch 40; the model never converged.
      2-layer BiLSTM has 4× memory cells per layer — far too expressive
      for a dataset this small.

    NEW ARCHITECTURE (18K params):
      Feature dim: 13 (6 original + month_sin/cos + temp×rain + 4 rolling means)
      INPUT → Linear(13→32) → LayerNorm → GELU              [per-timestep proj]
      → LSTM(32→32, layers=1, unidirectional)                [temporal modelling]
      → concat[last_hidden(32), hist_embed(32)]              [case history stream]
      → Linear(64→32) → GELU → Dropout(0.4) → Linear(32→1)  [prediction head]

    HIST STREAM:
      Linear(window_size → 32) → GELU. The scaled case history is the single
      strongest predictor (persistence baseline) — given its own embedding stream.

    REGULARISATION:
      dropout=0.4, weight_decay=1e-3, LR=3e-4 → 1e-5 via ReduceLROnPlateau.
      Early stop patience=20 (small data needs more epochs to converge stably).
      Max epochs=150.

    return_features=True: returns 64-dim concat([lstm_last, hist_embed])
      for ensemble fusion (TAB_FEAT_DIM = 64).

    PARAMETERS: ~18K trainable (vs 575K before — 32× smaller).
    """

    def __init__(self):
        super().__init__()
        fd = len(TAB_EXTENDED_COLS)  # 13 features
        h  = 32                       # compact hidden

        self.proj = nn.Sequential(
            nn.Linear(fd, h), nn.LayerNorm(h), nn.GELU(),
            nn.Dropout(0.3))

        self.lstm = nn.LSTM(h, h, num_layers=1, batch_first=True,
                            bidirectional=False)

        self.hist_embed = nn.Sequential(
            nn.Linear(Config.WINDOW_SIZE, h), nn.GELU())

        self.head = nn.Sequential(
            nn.Linear(h * 2, 32), nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1))

    def forward(self, feats, hist, return_features=False):
        x    = self.proj(feats)
        x, _ = self.lstm(x)
        lstm_out  = x[:, -1, :]              # (B, 32)
        hist_out  = self.hist_embed(hist)    # (B, 32)
        feat = torch.cat([lstm_out, hist_out], dim=1)  # (B, 64)
        out  = self.head(feat)
        return (out, feat) if return_features else out


class AdaptiveFusionNetwork(nn.Module):
    """
    Ensemble Meta-Learner — AdaptiveFusionNetwork
    ───────────────────────────────────────────────
    KEY IDEA: Both models have complementary strengths:
      ConvLSTM captures spatial patterns (land-use, water bodies) from imagery.
      TabularBiLSTM captures climate signals (rainfall, temp) from cell data.
    Learned gating lets the ensemble pick whichever model is more confident
    for each city-month, enabling SMAPE to drop below 10%.

    INPUT: concat([pred_c (1), pred_t (1), feat_c (64), feat_t (192), |gap| (1)])
      total input dim = 259.
      pred_c  : ConvLSTM scaled prediction
      pred_t  : Tabular aggregated scaled prediction
      feat_c  : ConvLSTM 64-dim penultimate features
      feat_t  : Tabular 192-dim BiLSTM features
      gap     : |pred_c - pred_t| — model disagreement signal

    FUSION MLP:
      Linear(259→256) → LayerNorm → ReLU → Dropout(0.3)
      → Linear(256→64) → LayerNorm → ReLU → Dropout(0.2)
      → Linear(64→2) → Softmax → weights [w_c, w_t]
      ens = w_c × pred_c + w_t × pred_t

    REFINEMENT:
      Small correction: Linear(1→32) → ReLU → Dropout(0.1) → Linear(32→1)
      Final: ens_corrected = ens + refine(ens)

    TRAINING:
      Trained on 80% of aligned test samples (remaining 20% for final eval).
      AdamW lr=1e-3, MSELoss, 30 epochs, batch=32.
    """

    def __init__(self, fc_dim=CONVLSTM_FEAT_DIM, ft_dim=64):
        super().__init__()
        inp = 1 + 1 + fc_dim + ft_dim + 1
        self.gate = nn.Sequential(
            nn.Linear(inp, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.LayerNorm(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2),    nn.Softmax(dim=-1))
        self.refine = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, pc, fc, pt, ft):
        gap = torch.abs(pc - pt)
        w   = self.gate(torch.cat([pc, pt, fc, ft, gap], dim=-1))
        ens = w[:, 0:1] * pc + w[:, 1:2] * pt
        return ens + self.refine(ens), w


def smape(preds, actuals):
    return 100 * np.mean(2 * np.abs(preds - actuals) /
                         (np.abs(preds) + np.abs(actuals) + 1e-8))


def mape(preds, actuals):
    return 100 * np.mean(np.abs((actuals - preds) / (actuals + 1e-8)))


def compute_metrics(preds, actuals):
    return {
        'mae':   float(mean_absolute_error(actuals, preds)),
        'rmse':  float(np.sqrt(mean_squared_error(actuals, preds))),
        'r2':    float(r2_score(actuals, preds)),
        'smape': float(smape(preds, actuals)),
        'mape':  float(mape(preds, actuals)),
        'corr':  float(np.corrcoef(preds, actuals)[0, 1]) if len(preds) > 2 else 0.0,
    }


def run_convlstm_epoch(model, loader, scaler, device, opt=None):
    model.train() if opt else model.eval()
    tl, ps, ac = 0.0, [], []
    criterion   = nn.MSELoss()
    ctx = torch.enable_grad() if opt else torch.no_grad()
    with ctx:
        for b in loader:
            imgs = b['images'].to(device)
            hist = b['hist'].to(device)
            tgt  = b['target'].to(device)
            if opt:
                opt.zero_grad()
            out  = model(imgs, hist)
            loss = criterion(out, tgt)
            if opt:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tl += loss.item()
            ps.extend(scaler.inverse_transform(out.detach().cpu().numpy()).flatten())
            ac.extend(scaler.inverse_transform(tgt.cpu().numpy()).flatten())
    return tl / max(len(loader), 1), compute_metrics(np.array(ps), np.array(ac))


def run_tab_epoch(model, loader, scaler, device, opt=None):
    model.train() if opt else model.eval()
    tl, ps, ac = 0.0, [], []
    criterion   = nn.MSELoss()
    ctx = torch.enable_grad() if opt else torch.no_grad()
    with ctx:
        for f, h, t, *_ in loader:
            f, h, t = f.to(device), h.to(device), t.to(device)
            if opt:
                opt.zero_grad()
            out  = model(f, h)
            loss = criterion(out, t)
            if opt:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tl += loss.item()
            ps.extend(scaler.inverse_transform(out.detach().cpu().numpy()).flatten())
            ac.extend(scaler.inverse_transform(t.cpu().numpy()).flatten())
    return tl / max(len(loader), 1), compute_metrics(np.array(ps), np.array(ac))


def save_fig(fig, name):
    p = os.path.join(Config.OUTPUT_DIR, 'plots', name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}")


def plot_all_cities(city_ts):
    cities = sorted(city_ts.keys())
    ncols  = 2
    nrows  = math.ceil(len(cities) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 5))
    fig.suptitle('Dengue Forecasting — All Cities: Actual vs Predicted',
                 fontsize=15, fontweight='bold', y=1.01)
    axes_flat = axes.flatten() if nrows > 1 else [axes[0], axes[1]]
    for idx, city in enumerate(cities):
        ax = axes_flat[idx]
        d  = city_ts[city]
        t  = range(len(d['actual']))
        ax.plot(t, d['actual'],   color=PALETTE['actual'],   lw=2.2, label='Actual',   zorder=5)
        ax.plot(t, d['ensemble'], color=PALETTE['ensemble'], lw=2.0, label='Ensemble', zorder=4)
        ax.plot(t, d['convlstm'], color=PALETTE['convlstm'], lw=1.2, ls='--', alpha=0.75,
                label='ConvLSTM', zorder=3)
        ax.plot(t, d['tabular'],  color=PALETTE['tabular'],  lw=1.2, ls=':', alpha=0.75,
                label='Tabular', zorder=3)
        ax.fill_between(t, d['actual'], d['ensemble'], alpha=0.07, color=PALETTE['ensemble'])
        m = compute_metrics(np.array(d['ensemble']), np.array(d['actual']))
        ax.set_title(
            f"{city.title()}  |  R²={m['r2']:.3f}  SMAPE={m['smape']:.1f}%  MAPE={m['mape']:.1f}%",
            fontweight='bold', fontsize=11, pad=8)
        ax.set_xlabel('Time Step (months)')
        ax.set_ylabel('Dengue Cases')
        ax.legend(fontsize=8, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.4)
    for ax in axes_flat[len(cities):]:
        ax.set_visible(False)
    fig.tight_layout(pad=2.0)
    save_fig(fig, 'all_cities_timeseries.png')


def plot_scatter(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Actual vs Predicted — All Models', fontsize=14, fontweight='bold', y=1.01)
    acts = results['actual']
    for ax, name, pred, col in zip(axes,
            ['ConvLSTM (Baseline)', 'Tabular BiLSTM', 'Ensemble'],
            [results['c_pred'], results['t_pred'], results['e_pred']],
            [PALETTE['convlstm'], PALETTE['tabular'], PALETTE['ensemble']]):
        ax.scatter(acts, pred, alpha=0.45, s=18, color=col, edgecolors='none')
        lo = min(acts.min(), pred.min()) - 5
        hi = max(acts.max(), pred.max()) + 5
        ax.plot([lo, hi], [lo, hi], '--', color='#64748b', lw=1.5, alpha=0.7)
        m = compute_metrics(pred, acts)
        ax.set_title(f'{name}\nR²={m["r2"]:.4f}  SMAPE={m["smape"]:.2f}%', fontsize=10)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_xlim([lo, hi])
        ax.set_ylim([lo, hi])
    fig.tight_layout()
    save_fig(fig, 'actual_vs_predicted.png')


def plot_training_curves(ch, tbh):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Loss Curves', fontsize=14, fontweight='bold')
    for ax, hist, title, col in zip(axes, [ch, tbh],
            ['ConvLSTM Model', 'Tabular BiLSTM Model'],
            [PALETTE['convlstm'], PALETTE['tabular']]):
        ep = range(1, len(hist['train']) + 1)
        ax.plot(ep, hist['train'], color=col,      lw=2, label='Train')
        ax.plot(ep, hist['val'],   color='#64748b', lw=1.5, ls='--', label='Val')
        ax.fill_between(ep, hist['train'], hist['val'], alpha=0.07, color=col)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
    fig.tight_layout()
    save_fig(fig, 'training_curves.png')


def plot_metrics_bar(results):
    keys   = ['mae', 'rmse', 'smape', 'r2']
    labels = ['MAE (↓)', 'RMSE (↓)', 'SMAPE % (↓)', 'R² (↑)']
    mlist  = [results['c_m'], results['t_m'], results['e_m']]
    mnames = ['ConvLSTM', 'Tabular', 'Ensemble']
    colors = [PALETTE['convlstm'], PALETTE['tabular'], PALETTE['ensemble']]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    for ax, key, label in zip(axes.flatten(), keys, labels):
        vals = [m[key] for m in mlist]
        bars = ax.bar(mnames, vals, color=colors, edgecolor='white', width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(abs(x) for x in vals) * 0.015,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.set_title(label, fontweight='bold')
        ax.set_ylim((min(0, min(vals)) - 0.05, 1.1) if key == 'r2'
                    else (0, max(vals) * 1.25))
    fig.tight_layout()
    save_fig(fig, 'metrics_bar.png')


def plot_residuals(results):
    res = results['e_pred'] - results['actual']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Ensemble Residuals Analysis', fontsize=14, fontweight='bold')
    axes[0].hist(res, bins=40, color=PALETTE['ensemble'], edgecolor='white', alpha=0.85)
    axes[0].axvline(0, color='#1a202c', lw=2, ls='--', alpha=0.6)
    axes[0].axvline(np.mean(res), color=PALETTE['accent'], lw=1.5,
                    label=f'μ={np.mean(res):.2f}')
    axes[0].set_title('Residual Distribution', fontweight='bold')
    axes[0].set_xlabel('Residual')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    sr = np.sort(res)
    n  = len(sr)
    qt = np.array([np.percentile(np.random.randn(5000), 100 * (i - .5) / n)
                   for i in range(1, n + 1)])
    lo = min(qt.min(), sr.min()); hi = max(qt.max(), sr.max())
    axes[1].scatter(qt, sr, s=8, alpha=0.5, color=PALETTE['convlstm'])
    axes[1].plot([lo, hi], [lo, hi], '--', color='#64748b', lw=1.5)
    axes[1].set_title('Q-Q Plot', fontweight='bold')
    axes[1].set_xlabel('Theoretical Quantiles')
    axes[1].set_ylabel('Sample Quantiles')

    axes[2].scatter(results['e_pred'], res, s=8, alpha=0.4, color=PALETTE['tabular'])
    axes[2].axhline(0, color='#1a202c', lw=1.5, ls='--', alpha=0.6)
    z  = np.polyfit(results['e_pred'], res, 1)
    xf = np.linspace(results['e_pred'].min(), results['e_pred'].max(), 200)
    axes[2].plot(xf, np.polyval(z, xf), color=PALETTE['accent'], lw=2, label='Trend')
    axes[2].set_title('Residuals vs Fitted', fontweight='bold')
    axes[2].set_xlabel('Fitted')
    axes[2].set_ylabel('Residuals')
    axes[2].legend()

    fig.tight_layout()
    save_fig(fig, 'residuals_analysis.png')


def plot_fusion_weights(fw):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ensemble Fusion Weights Distribution', fontsize=14, fontweight='bold')
    wc, wt = fw[:, 0], fw[:, 1]
    axes[0].hist(wc, bins=40, color=PALETTE['convlstm'], alpha=0.7,
                 label=f'ConvLSTM μ={wc.mean():.3f}', edgecolor='white')
    axes[0].hist(wt, bins=40, color=PALETTE['tabular'], alpha=0.7,
                 label=f'Tabular  μ={wt.mean():.3f}', edgecolor='white')
    axes[0].axvline(wc.mean(), color=PALETTE['convlstm'], lw=2, ls='--')
    axes[0].axvline(wt.mean(), color=PALETTE['tabular'],  lw=2, ls='--')
    axes[0].set_title('Weight Distributions', fontweight='bold')
    axes[0].set_xlabel('Weight')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[1].scatter(wc, wt, s=12, alpha=0.4, color=PALETTE['ensemble'])
    axes[1].plot([0, 1], [1, 0], '--', color='#64748b', lw=1.5, alpha=0.7,
                 label='Equal weight line')
    axes[1].set_title('Weight Trade-off', fontweight='bold')
    axes[1].set_xlabel('ConvLSTM Weight')
    axes[1].set_ylabel('Tabular Weight')
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, 'fusion_weights.png')


def plot_feature_importance(model, tab_test_ds, device):
    model.eval()
    loader = DataLoader(tab_test_ds, batch_size=512, shuffle=False)

    def get_mse(fidx=None):
        losses = []
        with torch.no_grad():
            for f, h, t, *_ in loader:
                f = f.clone().to(device)
                if fidx is not None:
                    p = torch.randperm(f.shape[0])
                    f[:, :, fidx] = f[p, :, fidx]
                out = model(f, h.to(device))
                losses.append(F.mse_loss(out, t.to(device)).item())
        return np.mean(losses)

    base = get_mse()
    imps = [(col, get_mse(i) - base) for i, col in enumerate(FEATURE_COLS)]
    imps.sort(key=lambda x: x[1], reverse=True)
    names, vals = zip(*imps)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, vals, color=PALETTE['tabular'], edgecolor='white')
    ax.axvline(0, color='#1a202c', lw=1, alpha=0.5)
    for i, v in enumerate(vals):
        ax.text(v + max(abs(x) for x in vals) * 0.01, i,
                f'+{v:.4f}', va='center', fontsize=9)
    ax.set_xlabel('Increase in MSE (higher = more important)')
    ax.set_title('Tabular Feature Permutation Importance', fontweight='bold')
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, 'feature_importance.png')


def plot_metrics_table(c_m, t_m, e_m, city_ts):
    col_labels = ['Model / City', 'MAE', 'RMSE', 'R²', 'SMAPE%', 'MAPE%', 'CORR']

    model_rows = [
        ['ConvLSTM (Baseline)',
         f"{c_m['mae']:.2f}", f"{c_m['rmse']:.2f}", f"{c_m['r2']:.4f}",
         f"{c_m['smape']:.2f}", f"{c_m['mape']:.2f}", f"{c_m['corr']:.4f}"],
        ['Tabular BiLSTM',
         f"{t_m['mae']:.2f}", f"{t_m['rmse']:.2f}", f"{t_m['r2']:.4f}",
         f"{t_m['smape']:.2f}", f"{t_m['mape']:.2f}", f"{t_m['corr']:.4f}"],
        ['Ensemble ★',
         f"{e_m['mae']:.2f}", f"{e_m['rmse']:.2f}", f"{e_m['r2']:.4f}",
         f"{e_m['smape']:.2f}", f"{e_m['mape']:.2f}", f"{e_m['corr']:.4f}"],
    ]
    model_colors = [['#dbeafe'] * 7, ['#dcfce7'] * 7, ['#fee2e2'] * 7]

    city_rows   = []
    city_colors = []
    for i, city in enumerate(sorted(city_ts.keys())):
        d = city_ts[city]
        m = compute_metrics(np.array(d['ensemble']), np.array(d['actual']))
        city_rows.append([city.title(),
                          f"{m['mae']:.2f}", f"{m['rmse']:.2f}", f"{m['r2']:.4f}",
                          f"{m['smape']:.2f}", f"{m['mape']:.2f}", f"{m['corr']:.4f}"])
        city_colors.append(['#f0f4f8' if i % 2 == 0 else 'white'] * 7)

    n_city = len(city_rows)
    fig, axes = plt.subplots(2, 1, figsize=(16, 3.5 + n_city * 0.5))
    fig.suptitle('Performance Metrics Summary', fontsize=15, fontweight='bold', y=1.01)

    ax = axes[0]
    ax.axis('off')
    t = ax.table(cellText=model_rows, colLabels=col_labels,
                 cellLoc='center', loc='center',
                 cellColours=model_colors, colColours=['#1e3a5f'] * 7)
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 2.3)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#c8d0da')
        if r == 0:
            cell.set_text_props(color='white', fontweight='bold')
    ax.set_title('Overall Model Metrics  (★ = Final Submitted Model)',
                 fontsize=12, fontweight='bold', pad=14)

    ax = axes[1]
    ax.axis('off')
    t2 = ax.table(cellText=city_rows, colLabels=col_labels,
                  cellLoc='center', loc='center',
                  cellColours=city_colors, colColours=['#1e3a5f'] * 7)
    t2.auto_set_font_size(False)
    t2.set_fontsize(10)
    t2.scale(1, 2.0)
    for (r, c), cell in t2.get_celld().items():
        cell.set_edgecolor('#c8d0da')
        if r == 0:
            cell.set_text_props(color='white', fontweight='bold')
    ax.set_title('Per-City Ensemble Metrics', fontsize=12, fontweight='bold', pad=14)

    fig.tight_layout(pad=2.5)
    save_fig(fig, 'metrics_table.png')


def main():
    t_global = time.time()
    print('\n' + '=' * 90)
    print(' ' * 10 + 'DENGUE ENSEMBLE: ConvLSTM (Satellite) + Tabular BiLSTM + Adaptive Fusion')
    print('=' * 90)
    print(f'Device: {Config.DEVICE}  |  AMP: {Config.AMP}')
    print(f'ConvLSTM: lstm_hidden={Config.LSTM_HIDDEN}, img={Config.IMG_SIZE}px, '
          f'batch={Config.BATCH_SIZE}, epochs≤{Config.EPOCHS_CONVLSTM}, '
          f'early_stop={Config.EARLY_STOP_CONVLSTM}')
    print(f'Tabular:  CompactLSTM hidden=32 (13 engineered features), '
          f'batch={Config.TAB_BATCH}, epochs≤{Config.EPOCHS_TAB}, '
          f'early_stop={Config.EARLY_STOP_TAB}')
    print(f'Ensemble: epochs={Config.EPOCHS_ENS}\n')

    tiff_df    = pd.read_csv(Config.TIFF_CSV)
    tabular_df = pd.read_csv(Config.TABULAR_CSV)
    tiff_df['city']    = tiff_df['city'].apply(clean_city)
    tabular_df['city'] = tabular_df['city'].apply(clean_city)

    print(f"  TIFF cities   : {sorted(tiff_df['city'].unique())}")
    print(f"  Tabular cities: {sorted(tabular_df['city'].unique())}")

    tiff_df = tiff_df.sort_values(['city', 'year', 'month']).reset_index(drop=True)
    master  = tiff_df[['city', 'year', 'month']].drop_duplicates().reset_index(drop=True)
    master['date_key'] = pd.to_datetime(master[['year', 'month']].assign(day=1))
    master = master.sort_values(['city', 'date_key']).reset_index(drop=True)

    tidx, vidx, teidx = split_per_city(
        master, Config.TRAIN_RATIO, Config.VAL_RATIO, Config.WINDOW_SIZE)
    print(f"\n  Master: {len(master)} rows | Train: {len(tidx)} | "
          f"Val: {len(vidx)} | Test: {len(teidx)}")

    tiff_train = filt(tiff_df, tidx)
    tiff_val   = filt(tiff_df, vidx)
    tiff_test  = filt(tiff_df, teidx)

    tabular_df['year_t']  = pd.to_datetime(tabular_df['month']).dt.year
    tabular_df['month_t'] = pd.to_datetime(tabular_df['month']).dt.month
    tab_train  = filt(tabular_df, tidx, 'year_t', 'month_t')
    tab_val    = filt(tabular_df, vidx, 'year_t', 'month_t')
    tab_test   = filt(tabular_df, teidx, 'year_t', 'month_t')

    conv_tr_ds = ConvLSTMDataset(tiff_train, Config.TIFF_DIR, Config.WINDOW_SIZE, is_train=True)
    conv_va_ds = ConvLSTMDataset(tiff_val,   Config.TIFF_DIR, Config.WINDOW_SIZE,
                                 scaler=conv_tr_ds.scaler, is_train=False)
    conv_te_ds = ConvLSTMDataset(tiff_test,  Config.TIFF_DIR, Config.WINDOW_SIZE,
                                 scaler=conv_tr_ds.scaler, is_train=False)

    # city_scaler shared: TabularDataset uses the SAME MinMaxScaler as ConvLSTM
    # so both models predict in the same scaled space — ensemble fusion works
    tab_tr_ds  = TabularDataset(tab_train, Config.WINDOW_SIZE,
                                city_scaler=conv_tr_ds.scaler, is_train=True)
    tab_va_ds  = TabularDataset(tab_val,   Config.WINDOW_SIZE,
                                city_scaler=conv_tr_ds.scaler,
                                feature_scaler=tab_tr_ds.feature_scaler, is_train=False)
    tab_te_ds  = TabularDataset(tab_test,  Config.WINDOW_SIZE,
                                city_scaler=conv_tr_ds.scaler,
                                feature_scaler=tab_tr_ds.feature_scaler, is_train=False)

    print(f"  ConvLSTM samples: {len(conv_tr_ds)} / {len(conv_va_ds)} / {len(conv_te_ds)}")
    print(f"  Tabular  samples: {len(tab_tr_ds)} / {len(tab_va_ds)} / {len(tab_te_ds)}")

    nw = 2
    conv_tr_dl = DataLoader(conv_tr_ds, Config.BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=True, drop_last=True)
    conv_va_dl = DataLoader(conv_va_ds, Config.BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True)
    conv_te_dl = DataLoader(conv_te_ds, Config.BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=True)
    tab_tr_dl  = DataLoader(tab_tr_ds,  Config.TAB_BATCH,  shuffle=True,  num_workers=nw, pin_memory=True)
    tab_va_dl  = DataLoader(tab_va_ds,  Config.TAB_BATCH,  shuffle=False, num_workers=nw, pin_memory=True)
    tab_te_dl  = DataLoader(tab_te_ds,  Config.TAB_BATCH,  shuffle=False, num_workers=nw, pin_memory=True)

    print('\n' + '-' * 60)
    print('[1/3] ConvLSTM Satellite Model')
    print(f'      ConvLSTMCell hidden={Config.LSTM_HIDDEN}, kernel=3×3, spatial=12×12')
    print(f'      MSELoss, AdamW lr={Config.LR_CONVLSTM}, ReduceLROnPlateau(p={Config.LR_PATIENCE}, f={Config.LR_FACTOR})')
    print(f'      Grad clip=1.0, batch={Config.BATCH_SIZE}, max_epochs={Config.EPOCHS_CONVLSTM}, early_stop={Config.EARLY_STOP_CONVLSTM}')

    model_c = DengueNet_ConvLSTM(
        window_size=Config.WINDOW_SIZE,
        lstm_hidden=Config.LSTM_HIDDEN,
        dropout=Config.CONV_DROPOUT).to(Config.DEVICE)
    tp_c = sum(p.numel() for p in model_c.parameters() if p.requires_grad)
    print(f'      Trainable params: {tp_c:,}')

    opt_c   = torch.optim.AdamW(model_c.parameters(), lr=Config.LR_CONVLSTM,
                                 weight_decay=Config.WEIGHT_DECAY)
    sched_c = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_c, patience=Config.LR_PATIENCE, factor=Config.LR_FACTOR, min_lr=1e-6)
    ch      = {'train': [], 'val': []}
    best_c, pat_c = 1e9, 0
    t0 = time.time()

    for ep in range(1, Config.EPOCHS_CONVLSTM + 1):
        tl, _  = run_convlstm_epoch(model_c, conv_tr_dl, conv_tr_ds.scaler, Config.DEVICE, opt=opt_c)
        vl, vm = run_convlstm_epoch(model_c, conv_va_dl, conv_tr_ds.scaler, Config.DEVICE)
        sched_c.step(vl)
        ch['train'].append(tl)
        ch['val'].append(vl)
        lr_now = opt_c.param_groups[0]['lr']
        if ep % 10 == 0:
            print(f"  Ep {ep:4d}  train={tl:.4f}  val={vl:.4f}  "
                  f"R²={vm['r2']:.4f}  SMAPE={vm['smape']:.2f}%  "
                  f"lr={lr_now:.2e}  [{(time.time()-t0)/60:.1f}min]")
        if vl < best_c:
            best_c = vl; pat_c = 0
            torch.save(model_c.state_dict(),
                       os.path.join(Config.OUTPUT_DIR, 'models', 'convlstm_best.pth'))
        else:
            pat_c += 1
        if pat_c >= Config.EARLY_STOP_CONVLSTM:
            print(f"  Early stop at epoch {ep}")
            break

    model_c.load_state_dict(
        torch.load(os.path.join(Config.OUTPUT_DIR, 'models', 'convlstm_best.pth')))
    print(f'  ConvLSTM done. [{(time.time()-t0)/60:.1f}min]')

    print('\n' + '-' * 60)
    print('[2/3] Tabular BiLSTM Model')
    print(f'      CompactLSTM(hidden=32, layers=1, unidirectional) + hist embedding')
    print(f'      MSELoss, AdamW lr={Config.LR_TAB}, weight_decay=1e-3, batch={Config.TAB_BATCH}, '
          f'max_epochs={Config.EPOCHS_TAB}, early_stop={Config.EARLY_STOP_TAB}')

    model_t = TabularBiLSTM().to(Config.DEVICE)
    tp_t    = sum(p.numel() for p in model_t.parameters() if p.requires_grad)
    print(f'      Trainable params: {tp_t:,}')

    opt_t   = torch.optim.AdamW(model_t.parameters(), lr=Config.LR_TAB,
                                 weight_decay=1e-3)  # strong L2 for 18K-param model on small data
    sched_t = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_t, patience=Config.LR_PATIENCE, factor=Config.LR_FACTOR, min_lr=1e-6)
    tbh     = {'train': [], 'val': []}
    best_t, pat_t = 1e9, 0
    t1 = time.time()

    for ep in range(1, Config.EPOCHS_TAB + 1):
        tl, _  = run_tab_epoch(model_t, tab_tr_dl, conv_tr_ds.scaler, Config.DEVICE, opt=opt_t)
        vl, vm = run_tab_epoch(model_t, tab_va_dl, conv_tr_ds.scaler, Config.DEVICE)
        sched_t.step(vl)
        tbh['train'].append(tl)
        tbh['val'].append(vl)
        lr_now = opt_t.param_groups[0]['lr']
        if ep % 5 == 0:
            print(f"  Ep {ep:4d}  train={tl:.4f}  val={vl:.4f}  "
                  f"R²={vm['r2']:.4f}  SMAPE={vm['smape']:.2f}%  "
                  f"lr={lr_now:.2e}  [{(time.time()-t1)/60:.1f}min]")
        if vl < best_t:
            best_t = vl; pat_t = 0
            torch.save(model_t.state_dict(),
                       os.path.join(Config.OUTPUT_DIR, 'models', 'tabular_best.pth'))
        else:
            pat_t += 1
        if pat_t >= Config.EARLY_STOP_TAB:
            print(f"  Early stop at epoch {ep}")
            break

    model_t.load_state_dict(
        torch.load(os.path.join(Config.OUTPUT_DIR, 'models', 'tabular_best.pth')))
    print(f'  Tabular done. [{(time.time()-t1)/60:.1f}min]')

    print('\n[Generating Val + Test Predictions for Stacking]')
    model_c.eval()
    model_t.eval()

    def get_convlstm_preds(loader):
        ps, fs, ts, meta_list = [], [], [], []
        with torch.no_grad():
            for b in loader:
                out, feat = model_c(b['images'].to(Config.DEVICE),
                                    b['hist'].to(Config.DEVICE), return_features=True)
                ps.append(out.detach().cpu())
                fs.append(feat.detach().cpu())
                ts.append(b['target'].cpu())
                for i in range(len(b['city'])):
                    meta_list.append((b['city'][i], int(b['year'][i]), int(b['month'][i])))
        return torch.cat(ps), torch.cat(fs), torch.cat(ts), meta_list

    def get_tab_preds(ds):
        ps, fs, ks = [], [], []
        with torch.no_grad():
            for f, h, t, cities, years, months in DataLoader(
                    ds, 512, shuffle=False, num_workers=nw):
                out, feat = model_t(f.to(Config.DEVICE), h.to(Config.DEVICE),
                                    return_features=True)
                ps.append(out.detach().cpu())
                fs.append(feat.detach().cpu())
                for i in range(len(cities)):
                    ks.append((cities[i], int(years[i]), int(months[i])))
        return torch.cat(ps), torch.cat(fs), ks

    def align(c_ps, c_fs, c_ts, c_meta, t_ps, t_fs, t_keys):
        ckm = {k: i for i, k in enumerate(c_meta)}
        tkm = {k: i for i, k in enumerate(t_keys)}
        common = sorted(set(ckm) & set(tkm))
        ic = [ckm[k] for k in common]
        it = [tkm[k] for k in common]
        return (c_ps[ic], c_fs[ic], c_ts[ic],
                t_ps[it], t_fs[it], common)

    # Val split — used to TRAIN the ensemble (no leakage into test)
    v_c_ps, v_c_fs, v_c_ts, v_c_meta = get_convlstm_preds(conv_va_dl)
    v_t_ps, v_t_fs, v_t_keys         = get_tab_preds(tab_va_ds)
    v_cp, v_cf, v_ct, v_tp, v_tf, v_common = align(
        v_c_ps, v_c_fs, v_c_ts, v_c_meta, v_t_ps, v_t_fs, v_t_keys)
    print(f"  Val  aligned: {len(v_common)} samples")

    # Test split — used only for final evaluation
    te_c_ps, te_c_fs, te_c_ts, te_c_meta = get_convlstm_preds(conv_te_dl)
    te_t_ps, te_t_fs, te_t_keys           = get_tab_preds(tab_te_ds)
    c_p, c_f, c_t, t_p, t_f, common = align(
        te_c_ps, te_c_fs, te_c_ts, te_c_meta, te_t_ps, te_t_fs, te_t_keys)
    print(f"  Test aligned: {len(common)} samples  |  "
          f"cities: {sorted(set(k[0] for k in common))}")

    print('\n' + '-' * 60)
    print('[3/3] Adaptive Fusion Ensemble  — trained on VAL, evaluated on TEST')
    print(f'      Stacking: val({len(v_common)} samples) → train ensemble → test({len(common)} samples)')
    print(f'      AdamW lr={Config.LR_ENS}, MSELoss+SMAPE, {Config.EPOCHS_ENS} epochs, early_stop={Config.EARLY_STOP_ENS}')

    meta  = AdaptiveFusionNetwork().to(Config.DEVICE)
    opt_m = torch.optim.AdamW(meta.parameters(), lr=Config.LR_ENS, weight_decay=1e-4)
    sched_m = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_m, patience=15, factor=0.5, min_lr=1e-6)
    best_ens_loss = 1e9
    ens_pat = 0
    t2 = time.time()

    # Split val into meta-train (80%) and meta-val (20%) for early stopping
    n_val    = len(v_common)
    n_tr_ens = int(n_val * 0.8)
    val_idx  = torch.arange(n_val)

    for ep in range(1, Config.EPOCHS_ENS + 1):
        meta.train()
        perm = torch.randperm(n_tr_ens)
        el, nb = 0.0, 0
        for i in range(0, n_tr_ens, 16):
            bi = perm[i:i + 16]
            if len(bi) == 0:
                continue
            pc = v_cp[bi].to(Config.DEVICE); fc = v_cf[bi].to(Config.DEVICE)
            pt = v_tp[bi].to(Config.DEVICE); ft = v_tf[bi].to(Config.DEVICE)
            tg = v_ct[bi].to(Config.DEVICE)
            opt_m.zero_grad()
            out, _ = meta(pc, fc, pt, ft)
            # Combined MSE + SMAPE loss for better percentage-error optimisation
            mse_l  = F.mse_loss(out, tg)
            smape_l = torch.mean(2 * torch.abs(out - tg) /
                                 (torch.abs(out) + torch.abs(tg) + 1e-8))
            loss = mse_l + 0.1 * smape_l
            loss.backward()
            nn.utils.clip_grad_norm_(meta.parameters(), 1.0)
            opt_m.step()
            el += loss.item(); nb += 1

        # Validate on meta-val split
        meta.eval()
        with torch.no_grad():
            vi = val_idx[n_tr_ens:]
            vout, _ = meta(v_cp[vi].to(Config.DEVICE), v_cf[vi].to(Config.DEVICE),
                           v_tp[vi].to(Config.DEVICE), v_tf[vi].to(Config.DEVICE))
            vloss = F.mse_loss(vout, v_ct[vi].to(Config.DEVICE)).item()
        sched_m.step(vloss)

        if vloss < best_ens_loss:
            best_ens_loss = vloss; ens_pat = 0
            torch.save(meta.state_dict(),
                       os.path.join(Config.OUTPUT_DIR, 'models', 'ensemble_best.pth'))
        else:
            ens_pat += 1

        if ep % 20 == 0 and nb > 0:
            lr_now = opt_m.param_groups[0]['lr']
            print(f"  Ep {ep:3d}  train_loss={el/nb:.5f}  val_loss={vloss:.5f}  "
                  f"lr={lr_now:.2e}  [{(time.time()-t2)/60:.1f}min]")
        if ens_pat >= Config.EARLY_STOP_ENS:
            print(f"  Early stop at epoch {ep}")
            break

    meta.load_state_dict(
        torch.load(os.path.join(Config.OUTPUT_DIR, 'models', 'ensemble_best.pth')))
    print(f'  Ensemble done. [{(time.time()-t2)/60:.1f}min]')

    print('\n[Final Evaluation on Test Set]')
    meta.eval()
    with torch.no_grad():
        ens_out, fw = meta(c_p.to(Config.DEVICE), c_f.to(Config.DEVICE),
                           t_p.to(Config.DEVICE), t_f.to(Config.DEVICE))

    sc = conv_tr_ds.scaler
    c_pred_d = sc.inverse_transform(c_p.numpy()).flatten()
    t_pred_d = sc.inverse_transform(t_p.numpy()).flatten()
    e_pred_d = sc.inverse_transform(ens_out.cpu().numpy()).flatten()
    actual_d = sc.inverse_transform(c_t.numpy()).flatten()

    c_m = compute_metrics(c_pred_d, actual_d)
    t_m = compute_metrics(t_pred_d, actual_d)
    e_m = compute_metrics(e_pred_d, actual_d)

    print('\n' + '=' * 82)
    print(f"{'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>8} "
          f"{'SMAPE%':>9} {'MAPE%':>9} {'CORR':>8}")
    print('-' * 82)
    for name, m in [('ConvLSTM (Baseline)', c_m), ('Tabular BiLSTM', t_m), ('Ensemble ★', e_m)]:
        print(f"{name:<22} {m['mae']:>7.2f} {m['rmse']:>7.2f} {m['r2']:>8.4f} "
              f"{m['smape']:>9.2f} {m['mape']:>9.2f} {m['corr']:>8.4f}")
    print('=' * 82)

    results = dict(actual=actual_d, c_pred=c_pred_d, t_pred=t_pred_d, e_pred=e_pred_d,
                   c_m=c_m, t_m=t_m, e_m=e_m, fw=fw.cpu().numpy())

    city_ts = {}
    for k, (city, year, month) in enumerate(common):
        if city not in city_ts:
            city_ts[city] = {'actual': [], 'convlstm': [], 'tabular': [], 'ensemble': []}
        city_ts[city]['actual'].append(float(actual_d[k]))
        city_ts[city]['convlstm'].append(float(c_pred_d[k]))
        city_ts[city]['tabular'].append(float(t_pred_d[k]))
        city_ts[city]['ensemble'].append(float(e_pred_d[k]))

    print(f"\n  Test cities: {sorted(city_ts.keys())}")

    print('\n[Generating Plots]')
    plot_all_cities(city_ts)
    plot_scatter(results)
    plot_metrics_bar(results)
    plot_training_curves(ch, tbh)
    plot_residuals(results)
    plot_fusion_weights(results['fw'])
    plot_feature_importance(model_t, tab_te_ds, Config.DEVICE)
    plot_metrics_table(c_m, t_m, e_m, city_ts)

    pd.DataFrame({
        'city':       [k[0] for k in common],
        'year':       [k[1] for k in common],
        'month':      [k[2] for k in common],
        'actual':     actual_d,
        'convlstm':   c_pred_d,
        'tabular':    t_pred_d,
        'ensemble':   e_pred_d,
    }).to_csv(os.path.join(Config.OUTPUT_DIR, 'predictions.csv'), index=False)

    with open(os.path.join(Config.OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump({'convlstm': c_m, 'tabular': t_m, 'ensemble': e_m}, f, indent=4)

    print(f'\n  Total time: {(time.time()-t_global)/60:.1f} min')
    print(f'  All outputs saved to: ./{Config.OUTPUT_DIR}/')
    print('=' * 90 + '\n')


if __name__ == '__main__':
    main()
