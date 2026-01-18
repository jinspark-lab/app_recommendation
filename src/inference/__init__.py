"""Inference and recommendation generation module."""

import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_models(model_dir: Path) -> Dict[str, lgb.Booster]:
    """
    Load trained models.
    
    Args:
        model_dir: Directory containing trained models
    
    Returns:
        Dictionary of label_type -> model
    """
    print(f"\n{'='*70}")
    print(f"Loading trained models from: {model_dir}")
    print(f"{'='*70}")
    
    models = {}
    for label_type in ['clicks', 'carts', 'orders']:
        model_path = model_dir / f"lgbm_model_{label_type}.pkl"
        
        with open(model_path, "rb") as f:
            models[label_type] = pickle.load(f)
        
        print(f"  - Loaded {label_type} model")
    
    return models


def predict_scores(
    training_df_path: Path,
    models: Dict[str, lgb.Booster]
) -> pd.DataFrame:
    """
    Generate prediction scores for all events.
    
    Args:
        training_df_path: Path to training dataset
        models: Dictionary of trained models
    
    Returns:
        DataFrame with predictions
    """
    print(f"\n{'='*70}")
    print(f"Generating predictions")
    print(f"{'='*70}")
    
    # Load training data
    df = pd.read_pickle(training_df_path)
    
    # Get feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['session', 'aid', 'type', 'ts', 
                                  'label_clicks', 'label_carts', 'label_orders']]
    
    X = df[feature_cols]
    
    # Predict for each model
    predictions = df[['session', 'aid', 'type', 'ts']].copy()
    
    for label_type, model in models.items():
        print(f"  - Predicting {label_type}...")
        predictions[f'score_{label_type}'] = model.predict(X)
    
    print(f"\nPredictions generated for {len(predictions):,} events")
    
    return predictions


def generate_top_k_recommendations(
    predictions: pd.DataFrame,
    k: int = 20,
    label_weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Generate top-k recommendations per session.
    
    Args:
        predictions: DataFrame with prediction scores
        k: Number of recommendations per session
        label_weights: Weights for combining scores (default: equal weight)
    
    Returns:
        DataFrame with top-k recommendations per session
    """
    print(f"\n{'='*70}")
    print(f"Generating Top-{k} recommendations per session")
    print(f"{'='*70}")
    
    # Default weights if not provided
    if label_weights is None:
        label_weights = {'clicks': 0.1, 'carts': 0.3, 'orders': 0.6}
    
    print(f"Score weights: {label_weights}")
    
    # Calculate combined score
    predictions['combined_score'] = (
        predictions['score_clicks'] * label_weights['clicks'] +
        predictions['score_carts'] * label_weights['carts'] +
        predictions['score_orders'] * label_weights['orders']
    )
    
    # Group by session and get top-k
    recommendations = []
    
    for session_id, group in predictions.groupby('session'):
        # Sort by combined score and get top-k
        top_k = group.nlargest(k, 'combined_score')
        
        # Create recommendation string (space-separated aids)
        aids = ' '.join(map(str, top_k['aid'].values))
        
        recommendations.append({
            'session': session_id,
            'labels': aids
        })
    
    result_df = pd.DataFrame(recommendations)
    
    print(f"\nGenerated recommendations for {len(result_df):,} sessions")
    print(f"  - Top-{k} items per session")
    
    return result_df


def save_recommendations(
    recommendations: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save recommendations to CSV file.
    
    Args:
        recommendations: DataFrame with recommendations
        output_path: Path to save CSV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    recommendations.to_csv(output_path, index=False)
    
    print(f"\nRecommendations saved to: {output_path}")
    print(f"  - Sessions: {len(recommendations):,}")
    print(f"  - File size: {output_path.stat().st_size / 1024:.2f} KB")


def generate_and_save_recommendations(
    training_df_path: Path,
    model_dir: Path,
    output_dir: Path,
    top_k: int = 20,
    label_weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Complete pipeline: load models, predict, generate top-k, and save.
    
    Args:
        training_df_path: Path to training dataset
        model_dir: Directory containing trained models
        output_dir: Directory to save output
        top_k: Number of recommendations per session
        label_weights: Weights for combining scores
    
    Returns:
        DataFrame with recommendations
    """
    # Load models
    models = load_models(model_dir)
    
    # Generate predictions
    predictions = predict_scores(training_df_path, models)
    
    # Generate top-k recommendations
    recommendations = generate_top_k_recommendations(
        predictions, 
        k=top_k, 
        label_weights=label_weights
    )
    
    # Save to CSV
    output_path = output_dir / "recommendations.csv"
    save_recommendations(recommendations, output_path)
    
    return recommendations
