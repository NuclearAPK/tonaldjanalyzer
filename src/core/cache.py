"""
Track analysis cache using SQLite for persistent storage.
"""

import sqlite3
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager


class TrackCache:
    """SQLite-based cache for track analysis results."""

    # Default database location (user data directory)
    DEFAULT_DB_NAME = "tonal_cache.db"

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the cache.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Store in user's app data or next to the script
            db_path = Path(__file__).parent.parent.parent / self.DEFAULT_DB_NAME

        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS track_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    duration REAL,
                    bpm REAL,
                    key TEXT,
                    camelot TEXT,
                    bpm_multiplier REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path
                ON track_cache(file_path)
            """)
            conn.commit()

            # Migrations
            self._run_migrations(conn)

    def _run_migrations(self, conn):
        """Run database migrations for new columns."""
        cursor = conn.execute("PRAGMA table_info(track_cache)")
        columns = [row[1] for row in cursor.fetchall()]

        # Migration: add bpm_multiplier column
        if 'bpm_multiplier' not in columns:
            conn.execute("ALTER TABLE track_cache ADD COLUMN bpm_multiplier REAL DEFAULT 1.0")
            conn.commit()

        # Migration: add embedding column for content-based matching
        if 'embedding' not in columns:
            conn.execute("ALTER TABLE track_cache ADD COLUMN embedding BLOB")
            conn.commit()

        # Migration: add style column for AI-detected style
        if 'style' not in columns:
            conn.execute("ALTER TABLE track_cache ADD COLUMN style TEXT")
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """
        Compute a hash based on file path, size, and modification time.
        This is faster than hashing file contents.

        Args:
            file_path: Path to the file

        Returns:
            Hash string representing the file state
        """
        try:
            stat = file_path.stat()
            hash_input = f"{file_path.absolute()}|{stat.st_size}|{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except OSError:
            return ""

    def get_cached_result(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis result for a file.

        Returns cached data only if the file hasn't been modified since caching.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with analysis results or None if not cached/outdated
        """
        file_path = Path(file_path).absolute()
        current_hash = self._compute_file_hash(file_path)

        if not current_hash:
            return None

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT duration, bpm, key, camelot, bpm_multiplier, file_hash
                FROM track_cache
                WHERE file_path = ?
                """,
                (str(file_path),)
            )
            row = cursor.fetchone()

            if row and row['file_hash'] == current_hash:
                return {
                    'duration': row['duration'],
                    'bpm': row['bpm'],
                    'key': row['key'],
                    'camelot': row['camelot'],
                    'bpm_multiplier': row['bpm_multiplier'] or 1.0,
                    'error': None,
                    'from_cache': True
                }

        return None

    def save_result(self, file_path: Path, result: Dict[str, Any]):
        """
        Save analysis result to cache.

        Args:
            file_path: Path to the audio file
            result: Analysis result dictionary
        """
        file_path = Path(file_path).absolute()
        file_hash = self._compute_file_hash(file_path)

        if not file_hash:
            return

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO track_cache (file_path, file_hash, duration, bpm, key, camelot, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    duration = excluded.duration,
                    bpm = excluded.bpm,
                    key = excluded.key,
                    camelot = excluded.camelot,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    str(file_path),
                    file_hash,
                    result.get('duration'),
                    result.get('bpm'),
                    result.get('key'),
                    result.get('camelot')
                )
            )
            conn.commit()

    def save_bpm_multiplier(self, file_path: Path, multiplier: float):
        """
        Save BPM multiplier for a track (without changing other cached data).

        Args:
            file_path: Path to the audio file
            multiplier: BPM multiplier value (0.5, 1.0, or 2.0)
        """
        file_path = Path(file_path).absolute()

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE track_cache
                SET bpm_multiplier = ?, updated_at = CURRENT_TIMESTAMP
                WHERE file_path = ?
                """,
                (multiplier, str(file_path))
            )
            conn.commit()

    def save_embedding(self, file_path: Path, embedding: np.ndarray):
        """
        Save audio embedding for content-based matching.

        Args:
            file_path: Path to the audio file
            embedding: Embedding vector (numpy array)
        """
        file_path = Path(file_path).absolute()

        # Convert numpy array to bytes
        embedding_bytes = embedding.tobytes() if embedding is not None else None

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE track_cache
                SET embedding = ?, updated_at = CURRENT_TIMESTAMP
                WHERE file_path = ?
                """,
                (embedding_bytes, str(file_path))
            )
            conn.commit()

    def get_embedding(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Get cached embedding for a file.

        Args:
            file_path: Path to the audio file

        Returns:
            Embedding vector or None if not cached
        """
        file_path = Path(file_path).absolute()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT embedding FROM track_cache WHERE file_path = ?",
                (str(file_path),)
            )
            row = cursor.fetchone()

            if row and row['embedding']:
                # Convert bytes back to numpy array (float32, 512-dim)
                return np.frombuffer(row['embedding'], dtype=np.float32)

        return None

    def save_style(self, file_path: Path, style: str):
        """
        Save AI-detected style for a track.

        Args:
            file_path: Path to the audio file
            style: Style string (e.g., "Drum And Bass / Energetic")
        """
        file_path = Path(file_path).absolute()

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE track_cache
                SET style = ?, updated_at = CURRENT_TIMESTAMP
                WHERE file_path = ?
                """,
                (style, str(file_path))
            )
            conn.commit()

    def get_style(self, file_path: Path) -> Optional[str]:
        """
        Get cached style for a file.

        Args:
            file_path: Path to the audio file

        Returns:
            Style string or None if not cached
        """
        file_path = Path(file_path).absolute()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT style FROM track_cache WHERE file_path = ?",
                (str(file_path),)
            )
            row = cursor.fetchone()

            if row and row['style']:
                return row['style']

        return None

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get all cached embeddings.

        Returns:
            Dictionary mapping file_path -> embedding
        """
        embeddings = {}

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT file_path, embedding FROM track_cache WHERE embedding IS NOT NULL"
            )
            for row in cursor.fetchall():
                embeddings[row['file_path']] = np.frombuffer(
                    row['embedding'], dtype=np.float32
                )

        return embeddings

    def clear_ai_data(self, file_path: Path):
        """
        Clear AI analysis data (embedding and style) for a file without removing basic analysis.

        Args:
            file_path: Path to the audio file
        """
        file_path = Path(file_path).absolute()

        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE track_cache
                SET embedding = NULL, style = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE file_path = ?
                """,
                (str(file_path),)
            )
            conn.commit()

    def invalidate(self, file_path: Path):
        """
        Remove cached result for a file (force re-analysis on next access).

        Args:
            file_path: Path to the audio file
        """
        file_path = Path(file_path).absolute()

        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM track_cache WHERE file_path = ?",
                (str(file_path),)
            )
            conn.commit()

    def invalidate_all(self):
        """Remove all cached results."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM track_cache")
            conn.commit()

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM track_cache")
            row = cursor.fetchone()
            return {
                'total_entries': row['count'] if row else 0
            }


# Global cache instance
_cache_instance: Optional[TrackCache] = None


def get_cache() -> TrackCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TrackCache()
    return _cache_instance
