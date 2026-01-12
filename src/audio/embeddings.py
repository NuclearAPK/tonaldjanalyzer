"""
Audio embeddings module using CLAP model for content-based similarity.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List
import warnings

# Suppress warnings during model loading
warnings.filterwarnings('ignore', category=UserWarning)

# Lazy loading of heavy dependencies
_model = None
_processor = None
_model_loaded = False
_model_available = True


def _load_model():
    """Lazy load CLAP model on first use."""
    global _model, _processor, _model_loaded, _model_available

    if _model_loaded:
        return _model_available

    try:
        from transformers import ClapModel, ClapProcessor
        import torch

        print("Loading CLAP model (first time may take a while)...")

        # Use smaller CLAP model
        model_name = "laion/smaller-clap-htsat-unfused"

        try:
            _processor = ClapProcessor.from_pretrained(model_name)
            _model = ClapModel.from_pretrained(model_name)
            _model.eval()

            # Use CPU for compatibility
            _model = _model.to('cpu')

            print("CLAP model loaded successfully")
            _model_available = True
        except Exception as e:
            # Fallback to larger model if smaller not available
            print(f"Smaller model not available, trying standard model...")
            model_name = "laion/clap-htsat-unfused"
            _processor = ClapProcessor.from_pretrained(model_name)
            _model = ClapModel.from_pretrained(model_name)
            _model.eval()
            _model = _model.to('cpu')
            _model_available = True

    except ImportError as e:
        print(f"CLAP dependencies not available: {e}")
        _model_available = False
    except Exception as e:
        print(f"Failed to load CLAP model: {e}")
        _model_available = False

    _model_loaded = True
    return _model_available


class AudioEmbeddings:
    """Extract audio embeddings using CLAP model."""

    EMBEDDING_DIM = 512  # CLAP embedding dimension

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        """Check if embedding extraction is available."""
        if self._available is None:
            self._available = _load_model()
        return self._available

    def extract_embedding(self, file_path: Path, audio_data: np.ndarray = None,
                          sr: int = 48000) -> Optional[np.ndarray]:
        """
        Extract audio embedding from file or audio data.

        Args:
            file_path: Path to audio file
            audio_data: Pre-loaded audio data (optional)
            sr: Sample rate (CLAP expects 48000)

        Returns:
            Embedding vector (512-dim) or None if failed
        """
        if not self.is_available():
            return None

        try:
            import torch
            import librosa

            # Load audio if not provided
            if audio_data is None:
                audio_data, orig_sr = librosa.load(str(file_path), sr=sr, mono=True)
            elif sr != 48000:
                # Resample to 48kHz for CLAP
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=48000)

            # Take a representative segment (middle 30 seconds for efficiency)
            target_length = 30 * 48000  # 30 seconds at 48kHz
            if len(audio_data) > target_length:
                start = (len(audio_data) - target_length) // 2
                audio_data = audio_data[start:start + target_length]

            # Process audio
            inputs = _processor(
                audios=audio_data,
                sampling_rate=48000,
                return_tensors="pt"
            )

            # Extract embedding
            with torch.no_grad():
                audio_features = _model.get_audio_features(**inputs)
                embedding = audio_features.squeeze().numpy()

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            return embedding.astype(np.float32)

        except Exception as e:
            print(f"Failed to extract embedding: {e}")
            return None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-100)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        # Cosine similarity (embeddings should already be normalized)
        similarity = np.dot(embedding1, embedding2)

        # Convert to 0-100 scale (cosine similarity ranges from -1 to 1)
        # For audio, typically ranges from 0 to 1, so we just scale
        score = max(0, similarity) * 100

        return float(score)

    def find_most_similar(self, master_embedding: np.ndarray,
                          track_embeddings: List[tuple]) -> List[tuple]:
        """
        Find tracks most similar to master by content.

        Args:
            master_embedding: Embedding of master track
            track_embeddings: List of (track_id, embedding) tuples

        Returns:
            List of (track_id, similarity_score) sorted by similarity
        """
        if master_embedding is None:
            return []

        results = []
        for track_id, embedding in track_embeddings:
            if embedding is not None:
                score = self.compute_similarity(master_embedding, embedding)
                results.append((track_id, score))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# Global instance
_embeddings_instance: Optional[AudioEmbeddings] = None


def get_audio_embeddings() -> AudioEmbeddings:
    """Get global audio embeddings instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = AudioEmbeddings()
    return _embeddings_instance
