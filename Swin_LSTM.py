import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import rasterio
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from typing import Tuple, List
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ====================== Configuration ======================
class Config:
    # Paths
    TIFF_DIR = "/content/drive/MyDrive/Final Metadata Dengue"
    CSV_PATH = "/content/dengue_cases.csv"
    OUTPUT_DIR = "results_swin"  # Changed output directory

    # Model parameters
    WINDOW_SIZE = 6
    IMG_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 150
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-5
    LSTM_HIDDEN = 256
    LSTM_LAYERS = 3
    DROPOUT = 0.3

    # Training
    EARLY_STOPPING_PATIENCE = 40
    LR_SCHEDULER_PATIENCE = 10
    LR_SCHEDULER_FACTOR = 0.5

    # Data split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    # TEST_RATIO = 0.15

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Visualization
    PLOT_EVERY_N_EPOCHS = 5
    SAVE_PLOTS = True


# Create output directory
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, 'plots'), exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUT_DIR, 'models'), exist_ok=True)


# ====================== Data Processing ======================
class TIFFProcessor:
    """Process TIFF satellite images"""

    @staticmethod
    def load_tiff(filepath: str, bands: List[int] = [2, 3, 4, 12]) -> np.ndarray:
        """Load specific bands from TIFF file"""
        with rasterio.open(filepath) as src:
            data = src.read(bands)
            data = np.transpose(data, (1, 2, 0))
        return data

    @staticmethod
    def normalize_bands(data: np.ndarray) -> np.ndarray:
        """Normalize each band to 0-1"""
        normalized = np.zeros_like(data, dtype=np.float32)
        for i in range(data.shape[2]):
            band = data[:, :, i]
            min_val, max_val = np.percentile(band, [2, 98])
            normalized[:, :, i] = np.clip((band - min_val) / (max_val - min_val + 1e-8), 0, 1)
        return normalized

    @staticmethod
    def extract_radiomics_features(swir_band: np.ndarray) -> np.ndarray:
        """Extract statistical features from SWIR band"""
        features = [
            np.mean(swir_band),
            np.std(swir_band),
            np.median(swir_band),
            np.percentile(swir_band, 25),
            np.percentile(swir_band, 75),
            np.min(swir_band),
            np.max(swir_band),
            np.var(swir_band),
            np.ptp(swir_band)
        ]
        return np.array(features, dtype=np.float32)


class DengueDataset(Dataset):
    """Dataset for dengue prediction"""

    def __init__(self, data_df: pd.DataFrame, tiff_dir: str, window_size: int,
                 scaler=None, is_train=True):
        self.data_df = data_df.sort_values(['city', 'year', 'month']).reset_index(drop=True)
        self.tiff_dir = tiff_dir
        self.window_size = window_size
        self.is_train = is_train

        if is_train:
            self.scaler = MinMaxScaler()
            self.data_df['dengue_cases_scaled'] = self.scaler.fit_transform(
                self.data_df[['dengue_cases']]
            )
        else:
            self.scaler = scaler
            self.data_df['dengue_cases_scaled'] = self.scaler.transform(
                self.data_df[['dengue_cases']]
            )

        self.samples = []
        for city in self.data_df['city'].unique():
            city_data = self.data_df[self.data_df['city'] == city].reset_index(drop=True)
            for i in range(window_size, len(city_data)):
                self.samples.append((city, i))

        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        city, end_idx = self.samples[idx]
        city_data = self.data_df[self.data_df['city'] == city].reset_index(drop=True)

        start_idx = end_idx - self.window_size
        sequence_data = city_data.iloc[start_idx:end_idx]

        rgb_images = []
        radiomics_features = []

        for _, row in sequence_data.iterrows():
            tiff_path = os.path.join(
                self.tiff_dir,
                f"{row['city']}_{row['year']}_{row['month']:02d}.tif"
            )

            if os.path.exists(tiff_path):
                data = TIFFProcessor.load_tiff(tiff_path)
                data = TIFFProcessor.normalize_bands(data)

                rgb = data[:, :, :3]
                rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
                rgb_tensor = self.transform(rgb_pil)
                rgb_images.append(rgb_tensor)

                swir = data[:, :, 3]
                features = TIFFProcessor.extract_radiomics_features(swir)
                radiomics_features.append(features)
            else:
                rgb_images.append(torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE))
                radiomics_features.append(np.zeros(9, dtype=np.float32))

        historical_cases = sequence_data['dengue_cases_scaled'].values
        target = city_data.iloc[end_idx]['dengue_cases_scaled']

        return {
            'rgb_images': torch.stack(rgb_images),
            'radiomics': torch.FloatTensor(radiomics_features),
            'historical_cases': torch.FloatTensor(historical_cases),
            'target': torch.FloatTensor([target]),
            'city': city,
            'year': int(city_data.iloc[end_idx]['year']),
            'month': int(city_data.iloc[end_idx]['month'])
        }


# ====================== Swin Transformer Architecture ======================
class WindowAttention(nn.Module):
    """Window-based multi-head self attention"""
    
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block"""
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, H, W):
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # Partition windows
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
    
    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer for feature extraction"""
    
    def __init__(self, img_size=224, patch_size=4, embed_dim=96, depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=layer_dim,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                ) for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            
            # Patch merging (except last layer)
            if i_layer < self.num_layers - 1:
                self.layers.append(nn.Sequential(
                    nn.LayerNorm(layer_dim * 4),
                    nn.Linear(layer_dim * 4, layer_dim * 2)
                ))
        
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)), 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256)
        )
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        
        # Process through Swin blocks
        layer_idx = 0
        for i_layer in range(self.num_layers):
            # Swin blocks
            layer = self.layers[layer_idx]
            for blk in layer:
                x = blk(x, H, W)
            layer_idx += 1
            
            # Patch merging (except last layer)
            if i_layer < self.num_layers - 1:
                # Reshape for merging
                x = x.view(B, H, W, -1)
                # Merge 2x2 patches
                x0 = x[:, 0::2, 0::2, :]
                x1 = x[:, 1::2, 0::2, :]
                x2 = x[:, 0::2, 1::2, :]
                x3 = x[:, 1::2, 1::2, :]
                x = torch.cat([x0, x1, x2, x3], -1)
                x = x.view(B, -1, x.shape[-1])
                H, W = H // 2, W // 2
                
                # Apply linear layer
                merge_layer = self.layers[layer_idx]
                x = merge_layer(x)
                layer_idx += 1
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class DengueNet_Swin(nn.Module):
    """DengueNet with Swin Transformer backbone"""

    def __init__(self, window_size, lstm_hidden=256, lstm_layers=3, dropout=0.3):
        super().__init__()

        self.swin = SwinTransformer(
            img_size=Config.IMG_SIZE,
            patch_size=4,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            dropout=dropout
        )

        self.radiomics_fc = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        self.swin_lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.radiomics_lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.cases_lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(lstm_hidden * 2, num_heads=8, batch_first=True)

        # Final prediction layers
        combined_size = lstm_hidden * 2 * 3
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, rgb_images, radiomics, historical_cases):
        batch_size, seq_len = rgb_images.shape[0], rgb_images.shape[1]

        # Extract spatial features with Swin Transformer
        swin_features = []
        for t in range(seq_len):
            feat = self.swin(rgb_images[:, t])
            swin_features.append(feat)
        swin_features = torch.stack(swin_features, dim=1)

        # Process radiomics features
        radiomics_list = []
        for t in range(seq_len):
            feat = self.radiomics_fc(radiomics[:, t])
            radiomics_list.append(feat)
        radiomics_features = torch.stack(radiomics_list, dim=1)

        # LSTM processing
        swin_out, _ = self.swin_lstm(swin_features)
        radiomics_out, _ = self.radiomics_lstm(radiomics_features)
        cases_out, _ = self.cases_lstm(historical_cases.unsqueeze(-1))

        # Apply attention
        swin_attn, _ = self.attention(swin_out, swin_out, swin_out)
        radiomics_attn, _ = self.attention(radiomics_out, radiomics_out, radiomics_out)
        cases_attn, _ = self.attention(cases_out, cases_out, cases_out)

        # Use last hidden state
        swin_final = swin_attn[:, -1, :]
        radiomics_final = radiomics_attn[:, -1, :]
        cases_final = cases_attn[:, -1, :]

        # Concatenate and predict
        combined = torch.cat([swin_final, radiomics_final, cases_final], dim=1)
        output = self.fc(combined)

        return output


# ====================== Visualization (same as before) ======================
class Visualizer:
    """Comprehensive visualization utilities"""

    @staticmethod
    def plot_training_history(history, save_path):
        """Plot training and validation loss curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss plot
        axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
        axes[0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Learning rate plot
        if 'learning_rate' in history:
            axes[1].plot(epochs, history['learning_rate'], 'g-^', linewidth=2, markersize=4)
            axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
            axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_predictions_vs_actual(predictions, actuals, cities, save_path):
        """Plot predicted vs actual values"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Overall scatter plot
        axes[0].scatter(actuals, predictions, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Add regression line
        z = np.polyfit(actuals, predictions, 1)
        p = np.poly1d(z)
        axes[0].plot(actuals, p(actuals), 'b-', linewidth=2, alpha=0.8, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

        axes[0].set_xlabel('Actual Dengue Cases', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Dengue Cases', fontsize=12, fontweight='bold')
        axes[0].set_title('Predicted vs Actual (All Cities)', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Calculate R²
        r2 = r2_score(actuals, predictions)
        axes[0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0].transAxes,
                    fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Residuals plot
        residuals = predictions - actuals
        axes[1].scatter(predictions, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Dengue Cases', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Distribution of residuals
        axes[2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[2].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[2].set_xlabel('Residuals', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[2].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')

        # Per-city performance
        unique_cities = np.unique(cities)
        city_mae = []
        for city in unique_cities:
            mask = cities == city
            city_mae.append(mean_absolute_error(actuals[mask], predictions[mask]))

        axes[3].bar(range(len(unique_cities)), city_mae, edgecolor='black', alpha=0.7)
        axes[3].set_xlabel('City', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('MAE', fontsize=12, fontweight='bold')
        axes[3].set_title('MAE by City', fontsize=14, fontweight='bold')
        axes[3].set_xticks(range(len(unique_cities)))
        axes[3].set_xticklabels(unique_cities, rotation=45, ha='right')
        axes[3].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_time_series(predictions, actuals, dates_info, save_path):
        """Plot time series predictions for each city"""
        cities = np.unique([d['city'] for d in dates_info])
        n_cities = len(cities)

        fig, axes = plt.subplots(int(np.ceil(n_cities/2)), 2, figsize=(20, 5*int(np.ceil(n_cities/2))))
        if n_cities == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, city in enumerate(cities):
            city_mask = [d['city'] == city for d in dates_info]
            city_indices = np.where(city_mask)[0]

            city_actuals = actuals[city_indices]
            city_predictions = predictions[city_indices]
            city_dates = [f"{dates_info[i]['year']}-{dates_info[i]['month']:02d}"
                         for i in city_indices]

            x = range(len(city_actuals))
            axes[idx].plot(x, city_actuals, 'b-o', label='Actual', linewidth=2, markersize=6)
            axes[idx].plot(x, city_predictions, 'r-s', label='Predicted', linewidth=2, markersize=6)
            axes[idx].fill_between(x, city_actuals, city_predictions, alpha=0.3)

            axes[idx].set_xlabel('Time', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Dengue Cases', fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{city} - Time Series Prediction', fontsize=13, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)

            # Show dates on x-axis (sample every n points to avoid crowding)
            step = max(1, len(city_dates) // 10)
            axes[idx].set_xticks(x[::step])
            axes[idx].set_xticklabels(city_dates[::step], rotation=45, ha='right')

            # Add MAE text
            mae = mean_absolute_error(city_actuals, city_predictions)
            axes[idx].text(0.02, 0.98, f'MAE: {mae:.2f}', transform=axes[idx].transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # Hide empty subplots
        for idx in range(n_cities, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_metrics_by_epoch(history, save_path):
        """Plot various metrics evolution"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontweight='bold')
        axes[0, 0].set_ylabel('Loss', fontweight='bold')
        axes[0, 0].set_title('Loss Evolution', fontweight='bold', fontsize=13)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MAE
        if 'train_mae' in history:
            axes[0, 1].plot(epochs, history['train_mae'], 'b-', label='Train', linewidth=2)
            axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Validation', linewidth=2)
            axes[0, 1].set_xlabel('Epoch', fontweight='bold')
            axes[0, 1].set_ylabel('MAE', fontweight='bold')
            axes[0, 1].set_title('MAE Evolution', fontweight='bold', fontsize=13)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # RMSE
        if 'train_rmse' in history:
            axes[1, 0].plot(epochs, history['train_rmse'], 'b-', label='Train', linewidth=2)
            axes[1, 0].plot(epochs, history['val_rmse'], 'r-', label='Validation', linewidth=2)
            axes[1, 0].set_xlabel('Epoch', fontweight='bold')
            axes[1, 0].set_ylabel('RMSE', fontweight='bold')
            axes[1, 0].set_title('RMSE Evolution', fontweight='bold', fontsize=13)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # R²
        if 'train_r2' in history:
            axes[1, 1].plot(epochs, history['train_r2'], 'b-', label='Train', linewidth=2)
            axes[1, 1].plot(epochs, history['val_r2'], 'r-', label='Validation', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontweight='bold')
            axes[1, 1].set_ylabel('R² Score', fontweight='bold')
            axes[1, 1].set_title('R² Score Evolution', fontweight='bold', fontsize=13)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def create_summary_report(results, history, save_path):
        """Create a comprehensive summary report"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('DengueNet-Swin Training Summary Report', fontsize=18, fontweight='bold', y=0.98)

        # Metrics table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('tight')
        ax1.axis('off')

        metrics_data = [
            ['Metric', 'Train', 'Validation', 'Test'],
            ['MAE', f"{results['train_mae']:.2f}", f"{results['val_mae']:.2f}", f"{results['test_mae']:.2f}"],
            ['RMSE', f"{results['train_rmse']:.2f}", f"{results['val_rmse']:.2f}", f"{results['test_rmse']:.2f}"],
            ['R² Score', f"{results['train_r2']:.4f}", f"{results['val_r2']:.4f}", f"{results['test_r2']:.4f}"],
            ['sMAPE', f"{results['train_smape']:.2f}%", f"{results['val_smape']:.2f}%", f"{results['test_smape']:.2f}%"]
        ]

        table = ax1.table(cellText=metrics_data, cellLoc='center', loc='center',
                         colWidths=[0.2, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, 5):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')

        # Training loss curve
        ax2 = fig.add_subplot(gs[1, 0])
        epochs = range(1, len(history['train_loss']) + 1)
        ax2.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        ax2.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Loss', fontweight='bold')
        ax2.set_title('Loss Curve', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # MAE curve
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(epochs, history['train_mae'], 'b-', label='Train', linewidth=2)
        ax3.plot(epochs, history['val_mae'], 'r-', label='Val', linewidth=2)
        ax3.set_xlabel('Epoch', fontweight='bold')
        ax3.set_ylabel('MAE', fontweight='bold')
        ax3.set_title('MAE Curve', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # R² curve
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(epochs, history['train_r2'], 'b-', label='Train', linewidth=2)
        ax4.plot(epochs, history['val_r2'], 'r-', label='Val', linewidth=2)
        ax4.set_xlabel('Epoch', fontweight='bold')
        ax4.set_ylabel('R²', fontweight='bold')
        ax4.set_title('R² Curve', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Prediction scatter
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.scatter(results['test_actuals'], results['test_predictions'], alpha=0.6, s=50)
        min_val = min(results['test_actuals'].min(), results['test_predictions'].min())
        max_val = max(results['test_actuals'].max(), results['test_predictions'].max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax5.set_xlabel('Actual', fontweight='bold')
        ax5.set_ylabel('Predicted', fontweight='bold')
        ax5.set_title('Test Predictions', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Residuals
        ax6 = fig.add_subplot(gs[2, 1])
        residuals = results['test_predictions'] - results['test_actuals']
        ax6.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax6.set_xlabel('Residuals', fontweight='bold')
        ax6.set_ylabel('Frequency', fontweight='bold')
        ax6.set_title('Residual Distribution', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        # City-wise performance
        ax7 = fig.add_subplot(gs[2, 2])
        cities = list(results['city_metrics'].keys())
        mae_values = [results['city_metrics'][city]['mae'] for city in cities]
        ax7.bar(range(len(cities)), mae_values, edgecolor='black', alpha=0.7)
        ax7.set_xlabel('City', fontweight='bold')
        ax7.set_ylabel('MAE', fontweight='bold')
        ax7.set_title('MAE by City', fontweight='bold')
        ax7.set_xticks(range(len(cities)))
        ax7.set_xticklabels(cities, rotation=45, ha='right')
        ax7.grid(True, alpha=0.3, axis='y')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# ====================== Training Functions ======================
def compute_metrics(predictions, actuals):
    """Compute comprehensive metrics"""
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    smape = 100 * np.mean(2 * np.abs(predictions - actuals) /
                          (np.abs(predictions) + np.abs(actuals) + 1e-8))
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'smape': smape}


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_actuals = []

    for batch in loader:
        rgb = batch['rgb_images'].to(device)
        radiomics = batch['radiomics'].to(device)
        cases = batch['historical_cases'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        output = model(rgb, radiomics, cases)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pred_denorm = scaler.inverse_transform(output.detach().cpu().numpy())
        actual_denorm = scaler.inverse_transform(target.cpu().numpy())
        all_predictions.extend(pred_denorm.flatten())
        all_actuals.extend(actual_denorm.flatten())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_actuals))
    return avg_loss, metrics


def validate_epoch(model, loader, criterion, device, scaler):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch['rgb_images'].to(device)
            radiomics = batch['radiomics'].to(device)
            cases = batch['historical_cases'].to(device)
            target = batch['target'].to(device)

            output = model(rgb, radiomics, cases)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred_denorm = scaler.inverse_transform(output.cpu().numpy())
            actual_denorm = scaler.inverse_transform(target.cpu().numpy())
            all_predictions.extend(pred_denorm.flatten())
            all_actuals.extend(actual_denorm.flatten())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_actuals))
    return avg_loss, metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, scaler):
    """Enhanced training loop"""
    best_val_loss = float('inf')
    patience_counter = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
        'train_rmse': [], 'val_rmse': [],
        'train_r2': [], 'val_r2': [],
        'train_smape': [], 'val_smape': [],
        'learning_rate': []
    }

    print("\n" + "="*80)
    print(f"{'EPOCH':^8} | {'TRAIN LOSS':^12} | {'VAL LOSS':^12} | {'TRAIN MAE':^12} | {'VAL MAE':^12} | {'VAL R²':^12}")
    print("="*80)

    for epoch in range(epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, scaler)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_metrics['mae'])
        history['val_mae'].append(val_metrics['mae'])
        history['train_rmse'].append(train_metrics['rmse'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['train_r2'].append(train_metrics['r2'])
        history['val_r2'].append(val_metrics['r2'])
        history['train_smape'].append(train_metrics['smape'])
        history['val_smape'].append(val_metrics['smape'])
        history['learning_rate'].append(current_lr)

        print(f"{epoch+1:^8} | {train_loss:^12.4f} | {val_loss:^12.4f} | {train_metrics['mae']:^12.2f} | {val_metrics['mae']:^12.2f} | {val_metrics['r2']:^12.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, os.path.join(Config.OUTPUT_DIR, 'models', 'best_model.pth'))
            print(f"  ✓ Model saved (Val Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.2f})")
        else:
            patience_counter += 1

        if (epoch + 1) % Config.PLOT_EVERY_N_EPOCHS == 0:
            Visualizer.plot_training_history(history,
                os.path.join(Config.OUTPUT_DIR, 'plots', f'training_history_epoch_{epoch+1}.png'))

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break

    print("="*80)
    return history


def evaluate_model(model, loader, device, scaler):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_actuals = []
    all_cities = []
    all_dates = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch['rgb_images'].to(device)
            radiomics = batch['radiomics'].to(device)
            cases = batch['historical_cases'].to(device)
            target = batch['target'].to(device)

            output = model(rgb, radiomics, cases)

            pred_denorm = scaler.inverse_transform(output.cpu().numpy())
            actual_denorm = scaler.inverse_transform(target.cpu().numpy())

            all_predictions.extend(pred_denorm.flatten())
            all_actuals.extend(actual_denorm.flatten())
            all_cities.extend(batch['city'])
            all_dates.extend([{'city': c, 'year': y.item(), 'month': m.item()}
                            for c, y, m in zip(batch['city'], batch['year'], batch['month'])])

    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)
    cities = np.array(all_cities)

    metrics = compute_metrics(predictions, actuals)

    city_metrics = {}
    unique_cities = np.unique(cities)
    for city in unique_cities:
        mask = cities == city
        city_metrics[city] = compute_metrics(predictions[mask], actuals[mask])

    return {
        'predictions': predictions,
        'actuals': actuals,
        'cities': cities,
        'dates': all_dates,
        'metrics': metrics,
        'city_metrics': city_metrics
    }


# ====================== Main Execution ======================
def main():
    print("\n" + "="*80)
    print(" "*20 + "DENGUENET-SWIN TRANSFORMER TRAINING")
    print("="*80)

    print("\n[1/7] Loading data...")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"  ✓ Loaded {len(df)} records from {df['city'].nunique()} cities")

    # Split data TEMPORALLY (no data leakage)
    print("\n[2/7] Splitting data...")
    df = df.sort_values(['year', 'month', 'city']).reset_index(drop=True)
    
    # Create temporal identifier for splitting
    df['time_id'] = df['year'] * 12 + df['month']
    unique_times = sorted(df['time_id'].unique())
    
    # Split by time periods (not by rows)
    n_times = len(unique_times)
    train_end_idx = int(n_times * Config.TRAIN_RATIO)
    val_end_idx = int(n_times * (Config.TRAIN_RATIO + Config.VAL_RATIO))
    
    train_time_cutoff = unique_times[train_end_idx - 1] if train_end_idx > 0 else unique_times[0]
    val_time_cutoff = unique_times[val_end_idx - 1] if val_end_idx < n_times else unique_times[-1]
    
    # Split ALL cities by the same time cutoffs
    train_df = df[df['time_id'] <= train_time_cutoff].copy()
    val_df = df[(df['time_id'] > train_time_cutoff) & (df['time_id'] <= val_time_cutoff)].copy()
    test_df = df[df['time_id'] > val_time_cutoff].copy()
    
    # Clean up temporary column
    train_df = train_df.drop('time_id', axis=1)
    val_df = val_df.drop('time_id', axis=1)
    test_df = test_df.drop('time_id', axis=1)

    print(f"  ✓ Train: {len(train_df)} samples (up to {train_time_cutoff//12}-{train_time_cutoff%12:02d})")
    print(f"  ✓ Val: {len(val_df)} samples (up to {val_time_cutoff//12}-{val_time_cutoff%12:02d})")
    print(f"  ✓ Test: {len(test_df)} samples (after {val_time_cutoff//12}-{val_time_cutoff%12:02d})")

    print("\n[3/7] Creating datasets...")
    train_dataset = DengueDataset(train_df, Config.TIFF_DIR, Config.WINDOW_SIZE, is_train=True)
    val_dataset = DengueDataset(val_df, Config.TIFF_DIR, Config.WINDOW_SIZE,
                                scaler=train_dataset.scaler, is_train=False)
    test_dataset = DengueDataset(test_df, Config.TIFF_DIR, Config.WINDOW_SIZE,
                                 scaler=train_dataset.scaler, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"  ✓ Train batches: {len(train_loader)}")

    print("\n[4/7] Initializing Swin Transformer model...")
    model = DengueNet_Swin(
        window_size=Config.WINDOW_SIZE,
        lstm_hidden=Config.LSTM_HIDDEN,
        lstm_layers=Config.LSTM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Device: {Config.DEVICE}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                                  weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=Config.LR_SCHEDULER_FACTOR,
        patience=Config.LR_SCHEDULER_PATIENCE
    )

    print("\n[5/7] Training model...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer,
                         scheduler, Config.EPOCHS, Config.DEVICE, train_dataset.scaler)

    print("\n[6/7] Evaluating model...")
    checkpoint = torch.load(
        os.path.join(Config.OUTPUT_DIR, 'models', 'best_model.pth'),
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    train_results = evaluate_model(model, train_loader, Config.DEVICE, train_dataset.scaler)
    val_results = evaluate_model(model, val_loader, Config.DEVICE, train_dataset.scaler)
    test_results = evaluate_model(model, test_loader, Config.DEVICE, train_dataset.scaler)

    print("\n" + "="*80)
    print(" "*30 + "FINAL RESULTS")
    print("="*80)
    print(f"\n{'Metric':<15} {'Train':<15} {'Validation':<15} {'Test':<15}")
    print("-"*60)
    print(f"{'MAE':<15} {train_results['metrics']['mae']:<15.2f} {val_results['metrics']['mae']:<15.2f} {test_results['metrics']['mae']:<15.2f}")
    print(f"{'RMSE':<15} {train_results['metrics']['rmse']:<15.2f} {val_results['metrics']['rmse']:<15.2f} {test_results['metrics']['rmse']:<15.2f}")
    print(f"{'R²':<15} {train_results['metrics']['r2']:<15.4f} {val_results['metrics']['r2']:<15.4f} {test_results['metrics']['r2']:<15.4f}")
    print(f"{'sMAPE':<15} {train_results['metrics']['smape']:<15.2f} {val_results['metrics']['smape']:<15.2f} {test_results['metrics']['smape']:<15.2f}")
    print("="*80)

    print("\n[7/7] Generating visualizations...")
    Visualizer.plot_training_history(history,
        os.path.join(Config.OUTPUT_DIR, 'plots', 'final_training_history.png'))
    Visualizer.plot_metrics_by_epoch(history,
        os.path.join(Config.OUTPUT_DIR, 'plots', 'metrics_evolution.png'))
    Visualizer.plot_predictions_vs_actual(
        test_results['predictions'], test_results['actuals'], test_results['cities'],
        os.path.join(Config.OUTPUT_DIR, 'plots', 'predictions_vs_actual.png'))
    Visualizer.plot_time_series(
        test_results['predictions'], test_results['actuals'], test_results['dates'],
        os.path.join(Config.OUTPUT_DIR, 'plots', 'time_series_predictions.png'))

    combined_results = {
        'train_mae': train_results['metrics']['mae'],
        'train_rmse': train_results['metrics']['rmse'],
        'train_r2': train_results['metrics']['r2'],
        'train_smape': train_results['metrics']['smape'],
        'val_mae': val_results['metrics']['mae'],
        'val_rmse': val_results['metrics']['rmse'],
        'val_r2': val_results['metrics']['r2'],
        'val_smape': val_results['metrics']['smape'],
        'test_mae': test_results['metrics']['mae'],
        'test_rmse': test_results['metrics']['rmse'],
        'test_r2': test_results['metrics']['r2'],
        'test_smape': test_results['metrics']['smape'],
        'test_predictions': test_results['predictions'],
        'test_actuals': test_results['actuals'],
        'city_metrics': test_results['city_metrics']
    }

    Visualizer.create_summary_report(combined_results, history,
        os.path.join(Config.OUTPUT_DIR, 'plots', 'summary_report.png'))

    # Save results
    results_df = pd.DataFrame({
        'city': test_results['cities'],
        'year': [d['year'] for d in test_results['dates']],
        'month': [d['month'] for d in test_results['dates']],
        'actual': test_results['actuals'],
        'predicted': test_results['predictions'],
        'error': test_results['predictions'] - test_results['actuals'],
        'abs_error': np.abs(test_results['predictions'] - test_results['actuals'])
    })
    results_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'detailed_predictions.csv'), index=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(Config.OUTPUT_DIR, 'training_history.csv'), index=False)

    with open(os.path.join(Config.OUTPUT_DIR, 'metrics_summary.json'), 'w') as f:
        summary = {
            'train': {k: float(v) for k, v in train_results['metrics'].items()},
            'validation': {k: float(v) for k, v in val_results['metrics'].items()},
            'test': {k: float(v) for k, v in test_results['metrics'].items()},
            'per_city': {city: {k: float(v) for k, v in metrics.items()} 
                        for city, metrics in test_results['city_metrics'].items()}
        }
        json.dump(summary, f, indent=4)

    print("\n" + "="*80)
    print(" "*25 + "TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {Config.OUTPUT_DIR}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()