"""
Audio embeddings module using CLAP model for content-based similarity.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List
import warnings
import os
import sys

from ..core.logger import log_error, log_info, log_warning

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
        log_info("CLAP model loaded successfully")
        _model_available = True
        return True

    except ImportError as e:
        _load_error = f"CLAP dependencies not available: {e}"
        print(_load_error)
        log_error(_load_error)
        _model_available = False
    except Exception as e:
        _load_error = f"Failed to load CLAP model: {e}"
        print(_load_error)
        log_error(_load_error)
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

    # Genre labels with detailed descriptions for better CLAP classification
    GENRE_PROMPTS = {
        "Drum And Bass": [
            "fast drum and bass music with rolling breakbeats and heavy bass at 170 BPM",
            "drum and bass electronic music with rapid drums and sub bass",
            "DnB music with amen breaks and reese bass",
            "jungle drum and bass with chopped breakbeats and dub bass",
            "liquid drum and bass with melodic synths and rolling drums",
            "neurofunk drum and bass with complex bass design"
        ],
        "Dubstep": [
            "dubstep music with heavy wobble bass drops and half-time drums",
            "electronic dubstep with aggressive bass wobbles and syncopated rhythms",
            "brostep music with massive bass drops"
        ],
        "House": [
            "house music with four on the floor beat and groovy bassline at 120 BPM",
            "electronic house music with steady kick drum and soulful vocals",
            "deep house music with warm pads and rolling bassline"
        ],
        "Techno": [
            "techno music with repetitive beats and industrial sounds at 130 BPM",
            "minimal techno with hypnotic rhythm and dark atmosphere",
            "hard techno with driving kick drums and acid synths"
        ],
        "Trance": [
            "trance music with euphoric melodies and uplifting synths at 140 BPM",
            "progressive trance with building energy and emotional breakdown",
            "uplifting trance music with epic melodies and arpeggios"
        ],
        "Hip Hop": [
            "hip hop music with boom bap drums and sampled beats at 90 BPM",
            "rap music with heavy bass and snare hits",
            "old school hip hop with breakbeats and scratching"
        ],
        "Trap": [
            "trap music with 808 bass, hi-hat rolls and snare at 140 BPM",
            "electronic trap with heavy sub bass and rapid hi-hats",
            "EDM trap with bass drops and triplet hi-hats"
        ],
        "Ambient": [
            "ambient music with atmospheric pads and minimal rhythm",
            "atmospheric electronic music with drones and textures",
            "chill ambient soundscape with reverb and delay"
        ],
        "Breakbeat": [
            "breakbeat music with chopped drum breaks and funky bassline at 130 BPM",
            "big beat electronic music with distorted breaks",
            "nu skool breaks with heavy drums and synths"
        ],
        "Hardstyle": [
            "hardstyle music with hard kick drum and reverse bass at 150 BPM",
            "hard dance music with distorted kicks and euphoric melodies",
            "rawstyle with aggressive kicks and dark atmosphere"
        ],
        "Progressive": [
            "progressive electronic music with evolving layers and long builds",
            "progressive house with melodic development and subtle changes",
            "progressive breaks with building tension and release"
        ],
        "Minimal": [
            "minimal techno with sparse elements and repetitive groove",
            "minimal electronic music with subtle variations and hypnotic rhythm",
            "micro house with clicks and cuts"
        ],
        "Electro": [
            "electro music with robotic sounds and syncopated rhythm",
            "electro house with saw wave bass and energetic drops",
            "electronic body music with mechanical beats"
        ]
    }

    # Mood prompts with descriptions
    MOOD_PROMPTS = {
        "Energetic": ["high energy dance music", "energetic and powerful electronic music", "upbeat and driving rhythm"],
        "Chill": ["relaxed chill music", "laid back and mellow electronic", "calm and soothing sounds"],
        "Dark": ["dark and ominous music", "heavy and sinister electronic", "dark underground sound"],
        "Melodic": ["melodic electronic music with beautiful harmonies", "melodic and emotional synths", "musical and tuneful"],
        "Aggressive": ["aggressive and intense music", "hard and aggressive electronic", "powerful and fierce sound"],
        "Uplifting": ["uplifting and euphoric music", "happy and positive electronic", "joyful and inspiring sound"],
        "Atmospheric": ["atmospheric and spacious music", "ambient and ethereal soundscape", "wide and immersive sound"],
        "Groovy": ["groovy dance music with funky rhythm", "groovy and funky electronic", "rhythmic and bouncy beat"],
        "Intense": ["intense and driving music", "relentless and powerful electronic", "hard hitting and impactful"],
        "Dreamy": ["dreamy and ethereal music", "floating and otherworldly sound", "soft and hazy electronic"]
    }

    def classify_style(self, file_path: Path, audio_data: np.ndarray = None,
                       sr: int = 48000, bpm: float = None) -> Optional[dict]:
        """
        Classify track style using CLAP zero-shot classification with ensemble prompts.

        Args:
            file_path: Path to audio file
            audio_data: Pre-loaded audio data (optional)
            sr: Sample rate
            bpm: Track BPM for genre correction (optional)

        Returns:
            Dictionary with 'genre' and 'mood' classifications
        """
        if not self.is_available():
            return None

        try:
            import librosa
            import torch

            # Load audio if not provided
            if audio_data is None:
                audio_data, _ = librosa.load(str(file_path), sr=sr, mono=True)

            # Take middle 30 seconds
            target_length = 30 * sr
            if len(audio_data) > target_length:
                start = (len(audio_data) - target_length) // 2
                audio_data = audio_data[start:start + target_length]

            # Get audio features
            audio_inputs = _processor(
                audio=audio_data,
                sampling_rate=sr,
                return_tensors="pt"
            )

            with torch.no_grad():
                audio_features = _model.get_audio_features(**audio_inputs)
                audio_features_norm = audio_features / audio_features.norm(dim=-1, keepdim=True)

                # Classify genre using ensemble of prompts
                genre_scores = {}
                for genre_name, prompts in self.GENRE_PROMPTS.items():
                    # Get features for all prompts of this genre
                    genre_inputs = _processor(text=prompts, return_tensors="pt", padding=True)
                    genre_features = _model.get_text_features(**genre_inputs)
                    genre_features_norm = genre_features / genre_features.norm(dim=-1, keepdim=True)

                    # Compute similarity and average across all prompts
                    similarities = (audio_features_norm @ genre_features_norm.T).squeeze()
                    avg_score = similarities.mean().item()
                    genre_scores[genre_name] = avg_score

                # Find best genre
                genre = max(genre_scores, key=genre_scores.get)

                # BPM-based genre correction: Breakbeat with BPM >= 160 (or half-time ~80+) is likely DnB
                # Check both original BPM and doubled BPM to catch half-time detection
                if genre == "Breakbeat" and bpm is not None:
                    if bpm >= 160 or (bpm >= 80 and bpm < 100):
                        genre = "Drum And Bass"

                # Classify mood using ensemble of prompts
                mood_scores = {}
                for mood_name, prompts in self.MOOD_PROMPTS.items():
                    mood_inputs = _processor(text=prompts, return_tensors="pt", padding=True)
                    mood_features = _model.get_text_features(**mood_inputs)
                    mood_features_norm = mood_features / mood_features.norm(dim=-1, keepdim=True)

                    similarities = (audio_features_norm @ mood_features_norm.T).squeeze()
                    avg_score = similarities.mean().item()
                    mood_scores[mood_name] = avg_score

                # Find best mood
                mood = max(mood_scores, key=mood_scores.get)

            return {
                'genre': genre,
                'mood': mood,
                'style': f"{genre} / {mood}"
            }

        except Exception as e:
            print(f"Failed to classify style: {e}")
            return None

    def classify_mood(self, file_path: Path, audio_data: np.ndarray = None,
                      sr: int = 48000) -> Optional[dict]:
        """
        Classify only the mood/character of a track (when genre is from metadata).

        Args:
            file_path: Path to audio file
            audio_data: Pre-loaded audio data (optional)
            sr: Sample rate

        Returns:
            Dictionary with 'mood' classification
        """
        if not self.is_available():
            return None

        try:
            import librosa
            import torch

            # Load audio if not provided
            if audio_data is None:
                audio_data, _ = librosa.load(str(file_path), sr=sr, mono=True)

            # Take middle 30 seconds
            target_length = 30 * sr
            if len(audio_data) > target_length:
                start = (len(audio_data) - target_length) // 2
                audio_data = audio_data[start:start + target_length]

            # Get audio features
            audio_inputs = _processor(
                audio=audio_data,
                sampling_rate=sr,
                return_tensors="pt"
            )

            with torch.no_grad():
                audio_features = _model.get_audio_features(**audio_inputs)
                audio_features_norm = audio_features / audio_features.norm(dim=-1, keepdim=True)

                # Classify mood using ensemble of prompts
                mood_scores = {}
                for mood_name, prompts in self.MOOD_PROMPTS.items():
                    mood_inputs = _processor(text=prompts, return_tensors="pt", padding=True)
                    mood_features = _model.get_text_features(**mood_inputs)
                    mood_features_norm = mood_features / mood_features.norm(dim=-1, keepdim=True)

                    similarities = (audio_features_norm @ mood_features_norm.T).squeeze()
                    avg_score = similarities.mean().item()
                    mood_scores[mood_name] = avg_score

                # Find best mood
                mood = max(mood_scores, key=mood_scores.get)

            return {'mood': mood}

        except Exception as e:
            print(f"Failed to classify mood: {e}")
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
