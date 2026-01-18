"""Training dataset creation module."""

import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from entity.session import Session
from data import load_sessions_from_file


def get_session_last_items(
    train_file: Path,
    max_lines: int | None = None
) -> Dict[int, Dict[str, int]]:
    """
    Get last item aid for each session by event type.
    
    Args:
        train_file: Path to training data
        max_lines: Maximum lines to process
    
    Returns:
        Dictionary of session_id -> {event_type: last_aid}
    """
    print(f"\n{'='*70}")
    print(f"Extracting last items per session")
    print(f"{'='*70}")
    
    session_last_items = {}
    
    for session_batch in load_sessions_from_file(train_file, chunk_size=5000, max_lines=max_lines):
        for session in session_batch:
            session_id = session.session
            
            # Group events by type and get last aid for each type
            events_by_type = {}
            for event in session.events:
                if event.type not in events_by_type:
                    events_by_type[event.type] = []
                events_by_type[event.type].append(event.aid)
            
            # Store last aid for each event type
            session_last_items[session_id] = {
                event_type: aids[-1] for event_type, aids in events_by_type.items()
            }
    
    print(f"Extracted last items for {len(session_last_items):,} sessions")
    
    return session_last_items


def create_training_dataframe(
    item_feature_table_path: Path,
    session_feature_table_path: Path,
    session_last_items: Dict[int, Dict[str, int]]
) -> pd.DataFrame:
    """
    Create training DataFrame by joining item and session feature tables.
    
    Args:
        item_feature_table_path: Path to item feature table
        session_feature_table_path: Path to session feature table
        session_last_items: Dictionary of session_id -> {event_type: last_aid}
    
    Returns:
        Training DataFrame with features and labels
    """
    print(f"\n{'='*70}")
    print(f"Creating Training DataFrame")
    print(f"{'='*70}")
    
    # Load feature tables
    print("Loading feature tables...")
    with open(item_feature_table_path, "rb") as f:
        item_feature_table = pickle.load(f)
    
    with open(session_feature_table_path, "rb") as f:
        session_feature_table = pickle.load(f)
    
    print(f"  - Item feature records: {len(item_feature_table):,}")
    print(f"  - Session feature records: {len(session_feature_table):,}")
    
    # Create dictionaries for fast lookup
    print("\nIndexing feature tables...")
    item_features = {}  # (session_id, aid) -> item_embedding
    for event_record, item_embedding in item_feature_table:
        key = (event_record.session_id, event_record.aid)
        item_features[key] = item_embedding
    
    session_features = {}  # session_id -> session_embedding
    for event_record, session_embedding in session_feature_table:
        session_features[event_record.session_id] = session_embedding
    
    # Build training records
    print("\nBuilding training records...")
    training_records = []
    
    for event_record, item_embedding in item_feature_table:
        session_id = event_record.session_id
        aid = event_record.aid
        event_type = event_record.event_type
        
        # Get session embedding
        if session_id not in session_features:
            continue
        session_embedding = session_features[session_id]
        
        # Create labels for clicks, carts, orders
        labels = {}
        for label_type in ['clicks', 'carts', 'orders']:
            if session_id in session_last_items and label_type in session_last_items[session_id]:
                last_aid = session_last_items[session_id][label_type]
                labels[f'label_{label_type}'] = 1 if aid == last_aid else 0
            else:
                labels[f'label_{label_type}'] = 0
        
        # Combine features
        record = {
            'session': session_id,
            'aid': aid,
            'type': event_type,
            'ts': event_record.ts,
            **labels
        }
        
        # Add item embedding features
        for i, val in enumerate(item_embedding):
            record[f'item_emb_{i}'] = val
        
        # Add session embedding features
        for i, val in enumerate(session_embedding):
            record[f'session_emb_{i}'] = val
        
        training_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(training_records)
    
    print(f"\nTraining DataFrame created:")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Label distribution:")
    for label_col in ['label_clicks', 'label_carts', 'label_orders']:
        if label_col in df.columns:
            positive = df[label_col].sum()
            print(f"    - {label_col}: {positive:,} positive ({positive/len(df)*100:.2f}%)")
    
    return df


def save_training_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save training DataFrame to file.
    
    Args:
        df: Training DataFrame
        output_path: Path to save
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle for fast loading
    df.to_pickle(output_path)
    
    print(f"\nTraining DataFrame saved to: {output_path}")
    print(f"  - Shape: {df.shape}")
    print(f"  - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
