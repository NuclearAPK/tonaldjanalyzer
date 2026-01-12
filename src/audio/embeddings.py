"""
Audio embeddings module using CLAP model for content-based similarity.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List
import warnings
import os
import sys

# Suppress warnings during model loading
warnings.filterwarnings('ignore', category=UserWarning)

# Lazy loading of heavy dependencies
_model = None
_processor = None
_model_loaded = False
_model_available = True
_load_error = None


def _fix_windows_dll_path():
    """Fix DLL loading issues on Windows by adding torch lib to DLL search path."""
    if sys.platform != 'win32':
        return

    try:
        import importlib.util

        # Find torch package location
        torch_spec = importlib.util.find_spec('torch')
        if torch_spec and torch_spec.origin:
            torch_path = Path(torch_spec.origin).parent
            lib_path = torch_path / 'lib'

            if lib_path.exists():
                # Add to DLL search path (Windows 8+)
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(str(lib_path))

                # Also add to PATH as fallback
                current_path = os.environ.get('PATH', '')
                if str(lib_path) not in current_path:
                    os.environ['PATH'] = str(lib_path) + os.pathsep + current_path
    except Exception:
        pass  # Silently ignore if fix fails


def _load_model():
    """Lazy load CLAP model on first use."""
    global _model, _processor, _model_loaded, _model_available, _load_error

    if _model_loaded:
        return _model_available

    _model_loaded = True  # Mark as attempted even if it fails

    try:
        # Set environment variable to help with DLL loading on Windows
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

        # Fix Windows DLL path before importing torch
        _fix_windows_dll_path()

        # Import torch after DLL path fix
        import torch

        from transformers import ClapModel, ClapProcessor

        print("Loading CLAP model (first time may take a while)...")

        model_name = "laion/clap-htsat-unfused"

        _processor = ClapProcessor.from_pretrained(model_name)
        _model = ClapModel.from_pretrained(model_name)
        _model.eval()

        # Use CPU for compatibility
        _model = _model.to('cpu')

        print("CLAP model loaded successfully")
        _model_available = True
        return True

    except ImportError as e:
        _load_error = f"CLAP dependencies not available: {e}"
        print(_load_error)
        _model_available = False
    except Exception as e:
        _load_error = f"Failed to load CLAP model: {e}"
        print(_load_error)
        _model_available = False

    return _model_available


def get_load_error() -> Optional[str]:
    """Get the error message if model loading failed."""
    return _load_error


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

            # Process audio using loaded model
            import torch

            inputs = _processor(
                audio=audio_data,
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
