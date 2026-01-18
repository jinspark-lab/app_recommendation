"""Main application entry point."""

import sys
import pickle
from pathlib import Path
from typing import Dict
import numpy as np

# Add parent directory to path to enable absolute imports
sys.path.insert(0, str(Path(__file__).parent))

from data import load_sessions_from_file
from features import (
    ItemEmbeddingTrainer,
    SessionEmbeddingCalculator,
    create_item_feature_table,
    create_session_feature_table,
    save_feature_table,
    get_session_last_items,
    create_training_dataframe,
    save_training_dataframe,
)
from training import train_all_models
from inference import generate_and_save_recommendations
from entity.session import Session


def save_embeddings(embeddings: Dict, output_path: Path, name: str) -> None:
    """
    Save embeddings to file.
    
    Args:
        embeddings: Dictionary of embeddings
        output_path: Path to save
        name: Name for logging
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n{name} saved to: {output_path}")
    print(f"  - Count: {len(embeddings):,}")
    print(f"  - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def extract_item_embeddings(
    train_file: Path,
    embedding_dim: int = 128,
    max_lines: int | None = 50000
) -> Dict[int, np.ndarray]:
    """
    Extract item embeddings from training data.
    
    Args:
        train_file: Path to training data file
        embedding_dim: Dimension of embeddings
        max_lines: Maximum lines to process
    
    Returns:
        Dictionary of item_id -> embedding vector
    """
    print(f"\n{'='*70}")
    print(f"Extracting item embeddings from: {train_file}")
    print(f"Embedding dimension: {embedding_dim} | Max lines: {max_lines:,}")
    print(f"{'='*70}")
    
    # Initialize trainer
    trainer = ItemEmbeddingTrainer(embedding_dim=embedding_dim)
    
    # Load sessions and add to trainer
    for session_batch in load_sessions_from_file(
        file_path=train_file,
        chunk_size=5000,
        max_lines=max_lines
    ):
        for session in session_batch:
            trainer.add_session(session)
    
    # Train embeddings
    print(f"\n{'='*70}")
    trainer.train(window_size=5, negative_samples=5)
    
    print(f"{'='*70}")
    print(f"Item embeddings extracted: {len(trainer.item_embeddings):,} items")
    
    return trainer.item_embeddings


def extract_session_embeddings(
    train_file: Path,
    item_embeddings: Dict[int, np.ndarray],
    n_recent: int = 10,
    max_lines: int | None = 50000
) -> Dict[int, np.ndarray]:
    """
    Extract session embeddings from training data using item embeddings.
    
    Args:
        train_file: Path to training data file
        item_embeddings: Dictionary of item_id -> embedding vector
        n_recent: Number of recent items to average
        max_lines: Maximum lines to process
    
    Returns:
        Dictionary of session_id -> embedding vector
    """
    print(f"\n{'='*70}")
    print(f"Extracting session embeddings from: {train_file}")
    print(f"Recent items window: {n_recent} | Max lines: {max_lines:,}")
    print(f"{'='*70}")
    
    # Initialize calculator
    calculator = SessionEmbeddingCalculator(item_embeddings)
    session_embeddings = {}
    
    # Process sessions
    total_sessions = 0
    valid_embeddings = 0
    
    for session_batch in load_sessions_from_file(
        file_path=train_file,
        chunk_size=5000,
        max_lines=max_lines
    ):
        for session in session_batch:
            total_sessions += 1
            embedding = calculator.calculate_session_embedding(session, n_recent)
            if embedding is not None:
                session_embeddings[session.session] = embedding
                valid_embeddings += 1
    
    print(f"\n{'='*70}")
    print(f"Session embeddings extracted: {valid_embeddings:,} / {total_sessions:,} sessions")
    print(f"Coverage: {valid_embeddings/total_sessions*100:.1f}%")
    
    return session_embeddings


def main():
    """Main function."""
    print("App Recommendation System - Embedding Extraction")
    print("=" * 70)
    
    # Define paths
    train_file = Path("../data/train.jsonl")
    feature_dir = Path("../feature")
    item_embeddings_path = feature_dir / "item_embeddings.pkl"
    session_embeddings_path = feature_dir / "session_embeddings.pkl"
    item_feature_table_path = feature_dir / "item_feature_table.pkl"
    session_feature_table_path = feature_dir / "session_feature_table.pkl"
    training_df_path = feature_dir / "training_dataset.pkl"
    
    # Check if train file exists
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        return
    
    max_lines = 1000  # Process first 10,000 lines for testing
    
    # Extract item embeddings
    item_embeddings = extract_item_embeddings(
        train_file=train_file,
        embedding_dim=128,
        max_lines=max_lines
    )
    
    # Save item embeddings
    save_embeddings(item_embeddings, item_embeddings_path, "Item embeddings")
    
    # Extract session embeddings
    session_embeddings = extract_session_embeddings(
        train_file=train_file,
        item_embeddings=item_embeddings,
        n_recent=10,
        max_lines=max_lines
    )
    
    # Save session embeddings
    save_embeddings(session_embeddings, session_embeddings_path, "Session embeddings")
    
    # Create Item Feature Table
    item_feature_table = create_item_feature_table(
        train_file=train_file,
        item_embeddings=item_embeddings,
        max_lines=max_lines
    )
    save_feature_table(item_feature_table, item_feature_table_path, "Item Feature Table")
    
    # Create Session Feature Table
    session_feature_table = create_session_feature_table(
        train_file=train_file,
        session_embeddings=session_embeddings,
        max_lines=max_lines
    )
    save_feature_table(session_feature_table, session_feature_table_path, "Session Feature Table")
    
    # Get session last items for labeling
    session_last_items = get_session_last_items(
        train_file=train_file,
        max_lines=max_lines
    )
    
    # Create Training DataFrame
    training_df = create_training_dataframe(
        item_feature_table_path=item_feature_table_path,
        session_feature_table_path=session_feature_table_path,
        session_last_items=session_last_items
    )
    save_training_dataframe(training_df, training_df_path)
    
    # Train LightGBM models
    model_dir = Path("../models")
    trained_models = train_all_models(
        training_data_path=training_df_path,
        output_dir=model_dir,
        test_size=0.2,
        random_state=42
    )
    
    # Generate Top-20 recommendations
    output_dir = Path("../output")
    recommendations = generate_and_save_recommendations(
        training_df_path=training_df_path,
        model_dir=model_dir,
        output_dir=output_dir,
        top_k=20,
        label_weights={'clicks': 0.1, 'carts': 0.3, 'orders': 0.6}
    )
    
    print("\n" + "=" * 70)
    print("Pipeline completed!")
    print(f"  - Item embeddings: {item_embeddings_path}")
    print(f"  - Session embeddings: {session_embeddings_path}")
    print(f"  - Item feature table: {item_feature_table_path}")
    print(f"  - Session feature table: {session_feature_table_path}")
    print(f"  - Training dataset: {training_df_path}")
    print(f"  - Trained models: {model_dir}")
    print(f"    - Clicks model: {model_dir / 'lgbm_model_clicks.pkl'}")
    print(f"    - Carts model: {model_dir / 'lgbm_model_carts.pkl'}")
    print(f"    - Orders model: {model_dir / 'lgbm_model_orders.pkl'}")
    print(f"  - Recommendations: {output_dir / 'recommendations.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
