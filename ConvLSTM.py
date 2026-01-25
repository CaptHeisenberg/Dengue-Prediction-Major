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


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ====================== Configuration ======================
class Config:
    # Paths
    TIFF_DIR = "/content/drive/MyDrive/Final Metadata Dengue"
    CSV_PATH = "/content/dengue_cases.csv"
    OUTPUT_DIR = "results_convlstm"

    # Model parameters
    WINDOW_SIZE = 6
    IMG_SIZE = 96  # Balanced size for better features
    BATCH_SIZE = 4  # Increased from 2
    GRADIENT_ACCUMULATION_STEPS = 1  # Disabled for stability
    EPOCHS = 150
    LEARNING_RATE = 0.001  # Earlier 0.01
    WEIGHT_DECAY = 1e-5
    LSTM_HIDDEN = 32  # Drastically reduced
    LSTM_LAYERS = 1  # Simplified
    DROPOUT = 0.2  # Reduced
    USE_MIXED_PRECISION = False  # Disabled for stability

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
    """Dataset for dengue prediction - ConvLSTM version"""

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
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        city, end_idx = self.samples[idx]
        city_data = self.data_df[self.data_df['city'] == city].reset_index(drop=True)

        start_idx = end_idx - self.window_size
        sequence_data = city_data.iloc[start_idx:end_idx]

        # Load all 4 bands for spatiotemporal processing
        spatiotemporal_images = []

        for _, row in sequence_data.iterrows():
            tiff_path = os.path.join(
                self.tiff_dir,
                f"{row['city']}_{row['year']}_{row['month']:02d}.tif"
            )

            if os.path.exists(tiff_path):
                data = TIFFProcessor.load_tiff(tiff_path)
                data = TIFFProcessor.normalize_bands(data)
                # Convert to PIL and then tensor
                img_pil = Image.fromarray((data * 255).astype(np.uint8))
                img_tensor = self.transform(img_pil)
                spatiotemporal_images.append(img_tensor)
            else:
                spatiotemporal_images.append(torch.zeros(4, Config.IMG_SIZE, Config.IMG_SIZE))

        historical_cases = sequence_data['dengue_cases_scaled'].values
        target = city_data.iloc[end_idx]['dengue_cases_scaled']

        return {
            'spatiotemporal_images': torch.stack(spatiotemporal_images),  # (T, C, H, W)
            'historical_cases': torch.FloatTensor(historical_cases),
            'target': torch.FloatTensor([target]),
            'city': city,
            'year': int(city_data.iloc[end_idx]['year']),
            'month': int(city_data.iloc[end_idx]['month'])
        }


# ====================== Model Architecture ======================
class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell that performs convolution in LSTM gates"""

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )


class ConvLSTM(nn.Module):
    """ConvLSTM module with multiple layers"""

    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, batch_first=True, bias=True):
        super().__init__()

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers
        if not isinstance(kernel_sizes, list):
            kernel_sizes = [kernel_sizes] * num_layers

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dims[i],
                kernel_size=kernel_sizes[i],
                bias=bias
            ))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if self.batch_first:
            # (B, T, C, H, W) -> (T, B, C, H, W)
            input_tensor = input_tensor.transpose(0, 1)

        b, _, h, w = input_tensor[0].size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        seq_len = input_tensor.size(0)
        cur_layer_input = input_tensor
        layer_output_list = []
        layer_state_list = []

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[t], (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=0)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            layer_state_list.append((h, c))

        if self.batch_first:
            layer_output_list = [out.transpose(0, 1) for out in layer_output_list]

        return layer_output_list, layer_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class DengueNet_ConvLSTM(nn.Module):
    """Simplified DengueNet with ConvLSTM for spatiotemporal processing"""

    def __init__(self, window_size, lstm_hidden=32, lstm_layers=1, dropout=0.2):
        super().__init__()

        # More aggressive downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),   # 96 -> 48
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 48 -> 24
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), # 24 -> 12
            nn.ReLU()
        )

        # Single ConvLSTM layer
        self.convlstm = ConvLSTM(
            input_dim=16,
            hidden_dims=[lstm_hidden],
            kernel_sizes=[3],
            num_layers=1,
            batch_first=True
        )

        # Global pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Simplified historical cases processing
        self.cases_fc = nn.Sequential(
            nn.Linear(window_size, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Calculate flattened size
        convlstm_features = lstm_hidden
        cases_features = lstm_hidden

        # Much simpler prediction layers
        self.fc = nn.Sequential(
            nn.Linear(convlstm_features + cases_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, spatiotemporal_images, historical_cases):
        batch_size, seq_len = spatiotemporal_images.shape[0], spatiotemporal_images.shape[1]

        # Downsample all frames
        flat_images = spatiotemporal_images.view(batch_size * seq_len, *spatiotemporal_images.shape[2:])
        downsampled_flat = self.downsample(flat_images)
        downsampled_seq = downsampled_flat.view(batch_size, seq_len, *downsampled_flat.shape[1:])

        # ConvLSTM processing
        layer_output_list, _ = self.convlstm(downsampled_seq)
        convlstm_out = layer_output_list[-1][:, -1, :, :, :]  # Last timestep
        
        # Global pooling
        convlstm_features = self.spatial_pool(convlstm_out)
        convlstm_features = convlstm_features.view(batch_size, -1)

        # Simple historical cases processing
        cases_features = self.cases_fc(historical_cases)

        # Concatenate and predict
        combined = torch.cat([convlstm_features, cases_features], dim=1)
        output = self.fc(combined)

        return output


# ====================== Training & Evaluation ======================
def compute_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """Compute comprehensive evaluation metrics"""
    mae = float(mean_absolute_error(actuals, predictions))
    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    r2 = float(r2_score(actuals, predictions))
    
    # sMAPE
    smape = float(100 * np.mean(2 * np.abs(predictions - actuals) / (np.abs(predictions) + np.abs(actuals) + 1e-8)))
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'smape': smape}


def train_epoch(model, loader, criterion, optimizer, device, scaler_data):
    """Train for one epoch - simplified"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_actuals = []

    for batch in loader:
        images = batch['spatiotemporal_images'].to(device)
        cases = batch['historical_cases'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        output = model(images, cases)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()

        pred_denorm = scaler_data.inverse_transform(output.detach().cpu().numpy())
        actual_denorm = scaler_data.inverse_transform(target.cpu().numpy())
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
            images = batch['spatiotemporal_images'].to(device)
            cases = batch['historical_cases'].to(device)
            target = batch['target'].to(device)

            output = model(images, cases)
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
    """Enhanced training loop with comprehensive logging"""
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
        # Training
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validation
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, scaler)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Store history
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

        # Print progress
        print(f"{epoch+1:^8} | {train_loss:^12.4f} | {val_loss:^12.4f} | {train_metrics['mae']:^12.2f} | {val_metrics['mae']:^12.2f} | {val_metrics['r2']:^12.4f}")

        # Save best model
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

        # Early stopping
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
            images = batch['spatiotemporal_images'].to(device)
            cases = batch['historical_cases'].to(device)
            target = batch['target'].to(device)

            output = model(images, cases)

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

    # Overall metrics
    metrics = compute_metrics(predictions, actuals)

    # Per-city metrics
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


# ====================== Visualization ======================
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
        """Plot predictions vs actual values with comprehensive views"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Scatter plot
        axes[0, 0].scatter(actuals, predictions, alpha=0.6, s=50)
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Cases', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Predicted Cases', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Predictions vs Actual', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # Residual plot
        residuals = predictions - actuals
        axes[0, 1].scatter(actuals, residuals, alpha=0.6, s=50)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Actual Cases', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Residuals', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # Residual distribution
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # MAE by city
        unique_cities = np.unique(cities)
        city_mae = []
        for city in unique_cities:
            mask = cities == city
            city_mae.append(mean_absolute_error(actuals[mask], predictions[mask]))

        axes[1, 1].bar(range(len(unique_cities)), city_mae, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('City', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('MAE', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('MAE by City', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(range(len(unique_cities)))
        axes[1, 1].set_xticklabels(unique_cities, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

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
    def plot_metrics_evolution(history, save_path):
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
    def create_summary_report(train_results, val_results, test_results, history, save_path):
        """Create a comprehensive summary report"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('DengueNet-ConvLSTM Training Summary Report', fontsize=18, fontweight='bold', y=0.98)

        # Metrics table
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('tight')
        ax1.axis('off')

        metrics_data = [
            ['Metric', 'Train', 'Validation', 'Test'],
            ['MAE', f"{train_results['metrics']['mae']:.2f}", f"{val_results['metrics']['mae']:.2f}", f"{test_results['metrics']['mae']:.2f}"],
            ['RMSE', f"{train_results['metrics']['rmse']:.2f}", f"{val_results['metrics']['rmse']:.2f}", f"{test_results['metrics']['rmse']:.2f}"],
            ['R² Score', f"{train_results['metrics']['r2']:.4f}", f"{val_results['metrics']['r2']:.4f}", f"{test_results['metrics']['r2']:.4f}"],
            ['sMAPE', f"{train_results['metrics']['smape']:.2f}%", f"{val_results['metrics']['smape']:.2f}%", f"{test_results['metrics']['smape']:.2f}%"]
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
        ax5.scatter(test_results['actuals'], test_results['predictions'], alpha=0.6, s=50)
        min_val = min(test_results['actuals'].min(), test_results['predictions'].min())
        max_val = max(test_results['actuals'].max(), test_results['predictions'].max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax5.set_xlabel('Actual', fontweight='bold')
        ax5.set_ylabel('Predicted', fontweight='bold')
        ax5.set_title('Test Predictions', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Residuals
        ax6 = fig.add_subplot(gs[2, 1])
        residuals = test_results['predictions'] - test_results['actuals']
        ax6.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax6.set_xlabel('Residuals', fontweight='bold')
        ax6.set_ylabel('Frequency', fontweight='bold')
        ax6.set_title('Residual Distribution', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        # City-wise performance
        ax7 = fig.add_subplot(gs[2, 2])
        cities = list(test_results['city_metrics'].keys())
        mae_values = [test_results['city_metrics'][city]['mae'] for city in cities]
        ax7.bar(range(len(cities)), mae_values, edgecolor='black', alpha=0.7)
        ax7.set_xlabel('City', fontweight='bold')
        ax7.set_ylabel('MAE', fontweight='bold')
        ax7.set_title('MAE by City', fontweight='bold')
        ax7.set_xticks(range(len(cities)))
        ax7.set_xticklabels(cities, rotation=45, ha='right')
        ax7.grid(True, alpha=0.3, axis='y')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# ====================== Main Execution ======================
def main():
    print("\n" + "="*80)
    print(" "*25 + "ConvLSTM TRAINING")
    print("="*80)

    # Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv(Config.CSV_PATH)
    print(f"  ✓ Loaded {len(df)} records from {df['city'].nunique()} cities")

    # Split data TEMPORALLY (no data leakage)
    print("\n[2/6] Splitting data...")
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

    # Create datasets
    print("\n[3/6] Creating datasets...")
    train_dataset = DengueDataset(train_df, Config.TIFF_DIR, Config.WINDOW_SIZE, is_train=True)
    val_dataset = DengueDataset(val_df, Config.TIFF_DIR, Config.WINDOW_SIZE,
                                scaler=train_dataset.scaler, is_train=False)
    test_dataset = DengueDataset(test_df, Config.TIFF_DIR, Config.WINDOW_SIZE,
                                 scaler=train_dataset.scaler, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches: {len(val_loader)}")
    print(f"  ✓ Test batches: {len(test_loader)}")

    # Initialize model
    print("\n[4/6] Initializing ConvLSTM model...")
    model = DengueNet_ConvLSTM(
        window_size=Config.WINDOW_SIZE,
        lstm_hidden=Config.LSTM_HIDDEN,
        lstm_layers=Config.LSTM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total parameters: {total_params:,}")
    print(f"  ✓ Trainable parameters: {trainable_params:,}")
    print(f"  ✓ Device: {Config.DEVICE}")
    print(f"  ✓ Mixed precision: {Config.USE_MIXED_PRECISION}")
    print(f"  ✓ Gradient accumulation steps: {Config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"  ✓ Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                                  weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=Config.LR_SCHEDULER_FACTOR,
        patience=Config.LR_SCHEDULER_PATIENCE
    )

    # Train
    print("\n[5/6] Training model...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer,
                         scheduler, Config.EPOCHS, Config.DEVICE, train_dataset.scaler)

    # Load best model
    print("\n[6/6] Evaluating model...")
    checkpoint = torch.load(
        os.path.join(Config.OUTPUT_DIR, 'models', 'best_model.pth'),
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  ✓ Loaded best model from epoch {checkpoint['epoch']+1}")

    # Evaluate
    train_results = evaluate_model(model, train_loader, Config.DEVICE, train_dataset.scaler)
    val_results = evaluate_model(model, val_loader, Config.DEVICE, train_dataset.scaler)
    test_results = evaluate_model(model, test_loader, Config.DEVICE, train_dataset.scaler)

    # Print results
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
            'train': train_results['metrics'],
            'validation': val_results['metrics'],
            'test': test_results['metrics'],
            'per_city': {city: metrics for city, metrics in test_results['city_metrics'].items()}
        }
        json.dump(summary, f, indent=4)

    # Visualizations
    print("\nGenerating comprehensive visualizations...")
    
    Visualizer.plot_training_history(history,
        os.path.join(Config.OUTPUT_DIR, 'plots', 'final_training_history.png'))
    
    Visualizer.plot_predictions_vs_actual(
        test_results['predictions'], test_results['actuals'], test_results['cities'],
        os.path.join(Config.OUTPUT_DIR, 'plots', 'predictions_vs_actual.png'))
    
    Visualizer.plot_time_series(
        test_results['predictions'], test_results['actuals'], test_results['dates'],
        os.path.join(Config.OUTPUT_DIR, 'plots', 'time_series_predictions.png'))
    
    Visualizer.plot_metrics_evolution(history,
        os.path.join(Config.OUTPUT_DIR, 'plots', 'metrics_evolution.png'))
    
    Visualizer.create_summary_report(train_results, val_results, test_results, history,
        os.path.join(Config.OUTPUT_DIR, 'plots', 'summary_report.png'))
    
    print("  ✓ All visualizations generated successfully!")

    print("\n" + "="*80)
    print(" "*25 + "TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {Config.OUTPUT_DIR}/")
    print(f"  - Training history: {Config.OUTPUT_DIR}/training_history.csv")
    print(f"  - Detailed predictions: {Config.OUTPUT_DIR}/detailed_predictions.csv")
    print(f"  - Metrics summary: {Config.OUTPUT_DIR}/metrics_summary.json")
    print(f"  - Plots: {Config.OUTPUT_DIR}/plots/")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()