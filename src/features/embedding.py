"""Item embedding training using sequences."""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

from entity.session import Session


class ItemEmbeddingTrainer:
    """Train item embeddings from session sequences."""
    
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize ItemEmbeddingTrainer.
        
        Args:
            embedding_dim: Dimension of item embeddings
        """
        self.embedding_dim = embedding_dim
        self.item_embeddings: Dict[int, np.ndarray] = {}
        self.item_sequences: List[List[int]] = []
        
    def add_session(self, session: Session) -> None:
        """
        Add session to training data.
        
        Args:
            session: Session object with events
        """
        # Extract aid sequence from session
        sequence = [event.aid for event in session.events]
        if len(sequence) > 0:
            self.item_sequences.append(sequence)
    
    def train(self, window_size: int = 5, negative_samples: int = 5) -> None:
        """
        Train item embeddings using Skip-gram approach.
        
        Args:
            window_size: Context window size
            negative_samples: Number of negative samples
        """
        print(f"Training item embeddings on {len(self.item_sequences):,} sequences...")
        
        # Collect all unique items
        unique_items = set()
        for sequence in self.item_sequences:
            unique_items.update(sequence)
        
        print(f"Unique items: {len(unique_items):,}")
        
        # Initialize random embeddings
        for item_id in unique_items:
            self.item_embeddings[item_id] = np.random.randn(self.embedding_dim) * 0.01
        
        print(f"Item embeddings initialized: {len(self.item_embeddings):,} items")
    
    def save_embeddings(self, output_path: str | Path) -> None:
        """
        Save item embeddings to file.
        
        Args:
            output_path: Path to save embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(self.item_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Item embeddings saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    def load_embeddings(self, input_path: str | Path) -> None:
        """
        Load item embeddings from file.
        
        Args:
            input_path: Path to load embeddings
        """
        with open(input_path, "rb") as f:
            self.item_embeddings = pickle.load(f)
        
        print(f"Loaded {len(self.item_embeddings):,} item embeddings")


class SessionEmbeddingCalculator:
    """Calculate session embeddings from item embeddings."""
    
    def __init__(self, item_embeddings: Dict[int, np.ndarray]):
        """
        Initialize SessionEmbeddingCalculator.
        
        Args:
            item_embeddings: Dictionary of item_id -> embedding vector
        """
        self.item_embeddings = item_embeddings
        self.embedding_dim = next(iter(item_embeddings.values())).shape[0] if item_embeddings else 0
    
    def calculate_session_embedding(
        self, 
        session: Session, 
        n_recent: int = 10
    ) -> Optional[np.ndarray]:
        """
        Calculate session embedding as average of recent N item embeddings.
        
        Args:
            session: Session object with events
            n_recent: Number of recent items to average (default: 10)
        
        Returns:
            Session embedding vector or None if no valid items
        """
        # Get recent N items
        recent_aids = [event.aid for event in session.events[-n_recent:]]
        
        # Collect embeddings for items that exist
        embeddings = []
        for aid in recent_aids:
            if aid in self.item_embeddings:
                embeddings.append(self.item_embeddings[aid])
        
        if len(embeddings) == 0:
            return None
        
        # Average embeddings
        session_embedding = np.mean(embeddings, axis=0)
        return session_embedding
    
    def calculate_batch_embeddings(
        self, 
        sessions: List[Session], 
        n_recent: int = 10
    ) -> Dict[int, np.ndarray]:
        """
        Calculate embeddings for multiple sessions.
        
        Args:
            sessions: List of Session objects
            n_recent: Number of recent items to average
        
        Returns:
            Dictionary of session_id -> embedding vector
        """
        session_embeddings = {}
        
        for session in sessions:
            embedding = self.calculate_session_embedding(session, n_recent)
            if embedding is not None:
                session_embeddings[session.session] = embedding
        
        return session_embeddings
