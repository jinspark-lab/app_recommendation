"""Flask API for recommendation inference."""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import pickle
import lightgbm as lgb
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import load_models


app = Flask(__name__)

# Global variables for loaded models and data
MODELS: Dict[str, lgb.Booster] = {}
ITEM_EMBEDDINGS: Dict[int, np.ndarray] = {}
SESSION_EMBEDDINGS: Dict[int, np.ndarray] = {}
FEATURE_COLUMNS: List[str] = []


def initialize_models():
    """Load models and embeddings on startup."""
    global MODELS, ITEM_EMBEDDINGS, SESSION_EMBEDDINGS, FEATURE_COLUMNS
    
    model_dir = Path(__file__).parent.parent.parent / "models"
    feature_dir = Path(__file__).parent.parent.parent / "feature"
    
    # Load models
    print("Loading models...")
    MODELS = load_models(model_dir)
    
    # Load embeddings
    print("Loading embeddings...")
    with open(feature_dir / "item_embeddings.pkl", "rb") as f:
        ITEM_EMBEDDINGS = pickle.load(f)
    
    with open(feature_dir / "session_embeddings.pkl", "rb") as f:
        SESSION_EMBEDDINGS = pickle.load(f)
    
    # Determine feature columns from training data
    training_df_path = feature_dir / "training_dataset.pkl"
    if training_df_path.exists():
        df_sample = pd.read_pickle(training_df_path)
        FEATURE_COLUMNS = [col for col in df_sample.columns 
                          if col not in ['session', 'aid', 'type', 'ts', 
                                        'label_clicks', 'label_carts', 'label_orders']]
        print(f"Feature columns: {len(FEATURE_COLUMNS)}")
    
    print("Initialization complete!")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(MODELS) > 0,
        'embeddings_loaded': len(ITEM_EMBEDDINGS) > 0,
        'feature_columns': len(FEATURE_COLUMNS)
    })


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Generate recommendations for a session.
    
    Request body:
    {
        "session_id": int,
        "events": [
            {"aid": int, "ts": int, "type": str}
        ],
        "top_k": int (optional, default: 20),
        "weights": {
            "clicks": float,
            "carts": float,
            "orders": float
        } (optional)
    }
    
    Response:
    {
        "session_id": int,
        "recommendations": [aid1, aid2, ...],
        "scores": {
            "clicks": [score1, score2, ...],
            "carts": [score1, score2, ...],
            "orders": [score1, score2, ...],
            "combined": [score1, score2, ...]
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        session_id = data.get('session_id')
        events = data.get('events', [])
        top_k = data.get('top_k', 20)
        weights = data.get('weights', {
            'clicks': 0.1,
            'carts': 0.3,
            'orders': 0.6
        })
        
        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        
        if not events:
            return jsonify({'error': 'events list is required'}), 400
        
        # Use all items as candidates (not just from events)
        # This allows recommending new items that the user hasn't seen
        candidate_aids = list(ITEM_EMBEDDINGS.keys())
        
        # Calculate session embedding from events instead of using pre-stored
        # This enables real-time recommendations for new sessions
        event_aids = [event['aid'] for event in events]
        valid_event_embeddings = [ITEM_EMBEDDINGS[aid] for aid in event_aids 
                                  if aid in ITEM_EMBEDDINGS]
        
        if not valid_event_embeddings:
            return jsonify({
                'error': 'No valid items found in events'
            }), 404
        
        # Calculate session embedding as mean of recent N items
        n_recent = min(10, len(valid_event_embeddings))
        session_emb = np.mean(valid_event_embeddings[-n_recent:], axis=0)
        
        # Remove items already in events to avoid recommending them again
        exclude_aids = set(event_aids)
        candidate_aids = [aid for aid in candidate_aids if aid not in exclude_aids]
        
        # Limit candidates for performance (optional)
        # For production, consider using approximate nearest neighbors (ANN)
        if len(candidate_aids) > 10000:
            # Randomly sample candidates for faster inference
            import random
            candidate_aids = random.sample(candidate_aids, 10000)
        
        # Use pre-calculated session embedding if available (for comparison)
        # But we already calculated it above for real-time inference
        
        # Prepare features for each candidate
        features_list = []
        valid_aids = []
        
        for aid in candidate_aids:
            if aid not in ITEM_EMBEDDINGS:
                continue  # Skip items without embeddings
            
            item_emb = ITEM_EMBEDDINGS[aid]
            
            # Concatenate item and session embeddings
            features = np.concatenate([item_emb, session_emb])
            features_list.append(features)
            valid_aids.append(aid)
        
        if not features_list:
            return jsonify({
                'error': 'No valid items with embeddings found'
            }), 404
        
        # Create DataFrame
        X = pd.DataFrame(features_list, columns=FEATURE_COLUMNS[:len(features_list[0])])
        
        # Ensure all feature columns are present (pad with zeros if needed)
        for col in FEATURE_COLUMNS:
            if col not in X.columns:
                X[col] = 0
        
        X = X[FEATURE_COLUMNS]
        
        # Predict with each model
        scores = {}
        for label_type, model in MODELS.items():
            scores[label_type] = model.predict(X).tolist()
        
        # Calculate combined score
        combined_scores = []
        for i in range(len(valid_aids)):
            combined = (
                scores['clicks'][i] * weights['clicks'] +
                scores['carts'][i] * weights['carts'] +
                scores['orders'][i] * weights['orders']
            )
            combined_scores.append(combined)
        
        scores['combined'] = combined_scores
        
        # Get top-k
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        top_aids = [valid_aids[i] for i in top_indices]
        top_scores = {
            'clicks': [scores['clicks'][i] for i in top_indices],
            'carts': [scores['carts'][i] for i in top_indices],
            'orders': [scores['orders'][i] for i in top_indices],
            'combined': [scores['combined'][i] for i in top_indices]
        }
        
        return jsonify({
            'session_id': session_id,
            'recommendations': top_aids,
            'scores': top_scores
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_recommend', methods=['POST'])
def batch_recommend():
    """
    Generate recommendations for multiple sessions.
    
    Request body:
    {
        "sessions": [
            {
                "session_id": int,
                "events": [{"aid": int, "ts": int, "type": str}]
            }
        ],
        "top_k": int (optional),
        "weights": {...} (optional)
    }
    
    Response:
    {
        "results": [
            {
                "session_id": int,
                "recommendations": [aid1, aid2, ...]
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        sessions = data.get('sessions', [])
        top_k = data.get('top_k', 20)
        weights = data.get('weights', {
            'clicks': 0.1,
            'carts': 0.3,
            'orders': 0.6
        })
        
        if not sessions:
            return jsonify({'error': 'sessions list is required'}), 400
        
        results = []
        errors = []
        
        for session_data in sessions:
            session_id = session_data.get('session_id')
            events = session_data.get('events', [])
            
            # Make individual recommendation
            response = app.test_client().post(
                '/recommend',
                json={
                    'session_id': session_id,
                    'events': events,
                    'top_k': top_k,
                    'weights': weights
                }
            )
            
            if response.status_code == 200:
                result = response.get_json()
                results.append({
                    'session_id': result['session_id'],
                    'recommendations': result['recommendations']
                })
            else:
                errors.append({
                    'session_id': session_id,
                    'error': response.get_json().get('error', 'Unknown error')
                })
        
        return jsonify({
            'results': results,
            'errors': errors,
            'total': len(sessions),
            'success': len(results),
            'failed': len(errors)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/items/<int:aid>/embedding', methods=['GET'])
def get_item_embedding(aid: int):
    """Get embedding for a specific item."""
    if aid not in ITEM_EMBEDDINGS:
        return jsonify({'error': f'Item {aid} not found'}), 404
    
    embedding = ITEM_EMBEDDINGS[aid].tolist()
    
    return jsonify({
        'aid': aid,
        'embedding': embedding,
        'dimension': len(embedding)
    })


@app.route('/sessions/<int:session_id>/embedding', methods=['GET'])
def get_session_embedding(session_id: int):
    """Get embedding for a specific session."""
    if session_id not in SESSION_EMBEDDINGS:
        return jsonify({'error': f'Session {session_id} not found'}), 404
    
    embedding = SESSION_EMBEDDINGS[session_id].tolist()
    
    return jsonify({
        'session_id': session_id,
        'embedding': embedding,
        'dimension': len(embedding)
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about loaded data."""
    return jsonify({
        'models': list(MODELS.keys()),
        'total_items': len(ITEM_EMBEDDINGS),
        'total_sessions': len(SESSION_EMBEDDINGS),
        'embedding_dimension': len(next(iter(ITEM_EMBEDDINGS.values()))) if ITEM_EMBEDDINGS else 0,
        'feature_dimension': len(FEATURE_COLUMNS)
    })


if __name__ == '__main__':
    initialize_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
