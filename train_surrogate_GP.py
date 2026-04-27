# train_surrogate.py
import numpy as np
import pandas as pd
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SURROGATE MODEL TRAINING")
print("=" * 60)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv("bandpass_dataset.csv")
print(f"Loaded {len(df)} samples")

# Filter out failed simulations (cost = 1e6)
df_valid = df[df['cost'] < 1e5].copy()
print(f"Valid samples: {len(df_valid)} ({100*len(df_valid)/len(df):.1f}%)")

X = df_valid.iloc[:, :-1].values  # R1-R6, C1-C2
y = df_valid.iloc[:, -1].values   # cost

print(f"Features shape: {X.shape}")
print(f"Target range: [{y.min():.4g}, {y.max():.4g}]")

# Scale inputs (CRITICAL for GP performance(matern kernal))
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Log-transform targets for better GP fit (costs are often log-normal)
y_log = np.log1p(y)

# TRAIN MULTIPLE MODELS

print("\n" + "-" * 40)
print("Training Gaussian Process...")
print("-" * 40)

kernel = (
    ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
    Matern(
        length_scale=np.ones(8),
        nu=2.5,
        length_scale_bounds=(1e-2, 1e2)
    ) +
    WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1.0))
)

start_time = time.time()
gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    alpha=1e-6,
    random_state=42
)
gp.fit(X_scaled, y_log)
gp_time = time.time() - start_time
print(f"GP trained in {gp_time:.1f}s")

# Evaluate GP
y_pred_gp = np.expm1(gp.predict(X_scaled))          #converts back to original space
gp_rmse = np.sqrt(np.mean((y - y_pred_gp) ** 2))
gp_r2 = 1 - np.sum((y - y_pred_gp) ** 2) / np.sum((y - np.mean(y)) ** 2)
print(f"GP - RMSE: {gp_rmse:.4g}, R²: {gp_r2:.4f}")

print("\n" + "-" * 40)
print("Training Random Forest (backup model)...")
print("-" * 40)

start_time = time.time()
rf = RandomForestRegressor(                                     #random forest 
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_scaled, y)
rf_time = time.time() - start_time
print(f"RF trained in {rf_time:.1f}s")

y_pred_rf = rf.predict(X_scaled)
rf_rmse = np.sqrt(np.mean((y - y_pred_rf) ** 2))
rf_r2 = 1 - np.sum((y - y_pred_rf) ** 2) / np.sum((y - np.mean(y)) ** 2)
print(f"RF - RMSE: {rf_rmse:.4g}, R²: {rf_r2:.4f}")

print("\n" + "-" * 40)
print("Training Gradient Boosting (backup model)...")
print("-" * 40)

start_time = time.time()
gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=6,                                            #gradient boosting
    learning_rate=0.1,
    min_samples_split=5,
    random_state=42
)
gb.fit(X_scaled, y)
gb_time = time.time() - start_time
print(f"GB trained in {gb_time:.1f}s")

y_pred_gb = gb.predict(X_scaled)
gb_rmse = np.sqrt(np.mean((y - y_pred_gb) ** 2))
gb_r2 = 1 - np.sum((y - y_pred_gb) ** 2) / np.sum((y - np.mean(y)) ** 2)
print(f"GB - RMSE: {gb_rmse:.4g}, R²: {gb_r2:.4f}")

# CROSS-VALIDATION
print("\n" + "-" * 40)
print("Cross-validation (5-fold)...")
print("-" * 40)

cv_scores_gp = cross_val_score(gp, X_scaled, y_log, cv=5, scoring='neg_root_mean_squared_error')
cv_rmse_gp = -cv_scores_gp.mean()
print(f"GP CV RMSE (log): {cv_rmse_gp:.4g}")

cv_scores_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
cv_rmse_rf = -cv_scores_rf.mean()
print(f"RF CV RMSE: {cv_rmse_rf:.4g}")

# SAVE MODEL
print("\n" + "-" * 40)
print("Saving models...")
print("-" * 40)

# Find best sample in dataset
best_idx = np.argmin(y)
best_cost = y[best_idx]
best_params = X[best_idx]

model_data = {
    # Models
    "gp": gp,
    "rf": rf,
    "gb": gb,
    "scaler": scaler,
    
    # Training data for warm-start
    "X_raw": X,
    "y_raw": y,
    "X_scaled": X_scaled,
    "y_log": y_log,
    
    # Best found during training
    "best_params": best_params,
    "best_cost": best_cost,
    
    # Model metrics
    "gp_r2": gp_r2,
    "gp_rmse": gp_rmse,
    "rf_r2": rf_r2,
    "rf_rmse": rf_rmse,
    "gb_r2": gb_r2,
    "n_samples": len(y),
    
    # Use log transform for GP
    "use_log_transform": True
}

joblib.dump(model_data, "surrogate_model.pkl")

print("\n" + "=" * 60)
print("MODEL SAVED: surrogate_model.pkl")
print("=" * 60)
print(f"Training samples: {len(y)}")
print(f"Best cost in dataset: {best_cost:.6g}")
print(f"")
print(f"Model Performance:")
print(f"  GP  - R²: {gp_r2:.4f}, RMSE: {gp_rmse:.4g}")
print(f"  RF  - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.4g}")
print(f"  GB  - R²: {gb_r2:.4f}, RMSE: {gb_rmse:.4g}")
print(f"")
print(f"Ready for Bandpass_BO.py!")
print("=" * 60)
