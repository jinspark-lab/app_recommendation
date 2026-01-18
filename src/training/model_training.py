"""Model training module using LightGBM."""

import pickle
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split


def load_training_data(data_path: Path) -> pd.DataFrame:
    """
    Load training dataset.
    
    Args:
        data_path: Path to training dataset pickle file
    
    Returns:
        Training DataFrame
    """
    print(f"\n{'='*70}")
    print(f"Loading training dataset from: {data_path}")
    print(f"{'='*70}")
    
    df = pd.read_pickle(data_path)
    
    print(f"Dataset loaded:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns[:10])}... ({len(df.columns)} total)")
    
    return df


def prepare_features_and_labels(df: pd.DataFrame, label_type: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and labels for training.
    
    Args:
        df: Training DataFrame
        label_type: Type of label ('clicks', 'carts', or 'orders')
    
    Returns:
        Tuple of (features_df, labels_series)
    """
    # Get label column
    label_col = f'label_{label_type}'
    
    # Drop metadata and label columns to get features
    feature_cols = [col for col in df.columns 
                   if col not in ['session', 'aid', 'type', 'ts', 
                                  'label_clicks', 'label_carts', 'label_orders']]
    
    X = df[feature_cols]
    y = df[label_col]
    
    print(f"\nPrepared features for '{label_type}':")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Samples: {len(df):,}")
    print(f"  - Positive ratio: {y.sum() / len(y) * 100:.2f}%")
    
    return X, y


def train_lgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label_type: str
) -> lgb.Booster:
    """
    Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        label_type: Type of label being trained
    
    Returns:
        Trained LightGBM model
    """
    print(f"\n{'='*70}")
    print(f"Training LightGBM model for '{label_type}'")
    print(f"{'='*70}")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Set parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'max_depth': -1,
        'min_data_in_leaf': 20,
    }
    
    print(f"\nTraining parameters:")
    for key, value in params.items():
        print(f"  - {key}: {value}")
    
    # Train model
    print(f"\nStarting training...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    
    print(f"\nTraining completed!")
    print(f"  - Best iteration: {model.best_iteration}")
    print(f"  - Best score: {model.best_score}")
    
    return model


def save_model(model: lgb.Booster, output_path: Path, label_type: str) -> None:
    """
    Save trained model.
    
    Args:
        model: Trained LightGBM model
        output_path: Path to save model
        label_type: Type of label
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    with open(output_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nModel saved to: {output_path}")
    print(f"  - Label type: {label_type}")
    print(f"  - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def train_all_models(
    training_data_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, lgb.Booster]:
    """
    Train models for all label types.
    
    Args:
        training_data_path: Path to training dataset
        output_dir: Directory to save models
        test_size: Validation split ratio
        random_state: Random seed
    
    Returns:
        Dictionary of label_type -> trained model
    """
    # Load data
    df = load_training_data(training_data_path)
    
    # Train model for each label type
    models = {}
    label_types = ['clicks', 'carts', 'orders']
    
    for label_type in label_types:
        print(f"\n{'='*70}")
        print(f"Processing label type: {label_type.upper()}")
        print(f"{'='*70}")
        
        # Prepare features and labels
        X, y = prepare_features_and_labels(df, label_type)
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nData split:")
        print(f"  - Train: {len(X_train):,} samples")
        print(f"  - Validation: {len(X_val):,} samples")
        
        # Train model
        model = train_lgbm_model(X_train, y_train, X_val, y_val, label_type)
        
        # Save model
        model_path = output_dir / f"lgbm_model_{label_type}.pkl"
        save_model(model, model_path, label_type)
        
        models[label_type] = model
    
    print(f"\n{'='*70}")
    print(f"All models trained successfully!")
    print(f"{'='*70}")
    
    return models
