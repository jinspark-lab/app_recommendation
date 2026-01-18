"""Feature table creation module."""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from entity.session import Session
from data import load_sessions_from_file


class EventRecord:
    """Single event record for feature table."""
    
    def __init__(self, session_id: int, aid: int, ts: int, event_type: str):
        self.session_id = session_id
        self.aid = aid
        self.ts = ts
        self.event_type = event_type
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session": self.session_id,
            "aid": self.aid,
            "ts": self.ts,
            "type": self.event_type
        }


def flatten_sessions_to_events(sessions: List[Session]) -> List[EventRecord]:
    """
    Flatten session list to event records.
    
    Args:
        sessions: List of Session objects
    
    Returns:
        List of EventRecord objects
    """
    events = []
    for session in sessions:
        for event in session.events:
            events.append(EventRecord(
                session_id=session.session,
                aid=event.aid,
                ts=event.ts,
                event_type=event.type
            ))
    return events


def create_item_feature_table(
    train_file: Path,
    item_embeddings: Dict[int, np.ndarray],
    max_lines: int | None = None
) -> List[Tuple[EventRecord, np.ndarray]]:
    """
    Create item feature table by joining events with item embeddings.
    
    Args:
        train_file: Path to training data
        item_embeddings: Dictionary of item_id -> embedding
        max_lines: Maximum lines to process
    
    Returns:
        List of (EventRecord, item_embedding) tuples
    """
    print(f"\n{'='*70}")
    print(f"Creating Item Feature Table")
    print(f"{'='*70}")
    
    item_feature_table = []
    total_events = 0
    matched_events = 0
    
    for session_batch in load_sessions_from_file(train_file, chunk_size=5000, max_lines=max_lines):
        events = flatten_sessions_to_events(session_batch)
        
        for event in events:
            total_events += 1
            
            # Join with item embedding
            if event.aid in item_embeddings:
                item_feature_table.append((event, item_embeddings[event.aid]))
                matched_events += 1
    
    print(f"\nItem Feature Table created:")
    print(f"  - Total events: {total_events:,}")
    print(f"  - Matched with embeddings: {matched_events:,}")
    print(f"  - Coverage: {matched_events/total_events*100:.1f}%")
    
    return item_feature_table


def create_session_feature_table(
    train_file: Path,
    session_embeddings: Dict[int, np.ndarray],
    max_lines: int | None = None
) -> List[Tuple[EventRecord, np.ndarray]]:
    """
    Create session feature table by joining events with session embeddings.
    
    Args:
        train_file: Path to training data
        session_embeddings: Dictionary of session_id -> embedding
        max_lines: Maximum lines to process
    
    Returns:
        List of (EventRecord, session_embedding) tuples
    """
    print(f"\n{'='*70}")
    print(f"Creating Session Feature Table")
    print(f"{'='*70}")
    
    session_feature_table = []
    total_events = 0
    matched_events = 0
    
    for session_batch in load_sessions_from_file(train_file, chunk_size=5000, max_lines=max_lines):
        events = flatten_sessions_to_events(session_batch)
        
        for event in events:
            total_events += 1
            
            # Join with session embedding
            if event.session_id in session_embeddings:
                session_feature_table.append((event, session_embeddings[event.session_id]))
                matched_events += 1
    
    print(f"\nSession Feature Table created:")
    print(f"  - Total events: {total_events:,}")
    print(f"  - Matched with embeddings: {matched_events:,}")
    print(f"  - Coverage: {matched_events/total_events*100:.1f}%")
    
    return session_feature_table


def save_feature_table(feature_table: List, output_path: Path, name: str) -> None:
    """
    Save feature table to file.
    
    Args:
        feature_table: List of feature records
        output_path: Path to save
        name: Name for logging
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "wb") as f:
        pickle.dump(feature_table, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n{name} saved to: {output_path}")
    print(f"  - Records: {len(feature_table):,}")
    print(f"  - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
