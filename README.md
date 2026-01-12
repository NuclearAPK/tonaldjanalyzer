# Tonal DJ - Track Compatibility Analyzer

[Русская версия](README_RU.md)

A desktop application for DJs that analyzes audio tracks and finds compatible matches based on musical key (Camelot Wheel notation) and BPM for seamless mixing.

## Features

- **BPM Detection** - Automatic tempo detection using beat tracking
- **Key Detection** - Musical key detection using Krumhansl-Schmuckler algorithm
- **Camelot Wheel** - Automatic conversion to Camelot notation for harmonic mixing
- **Compatibility Scoring** - Calculate match percentage between tracks based on key and BPM
- **BPM Multiplier** - Support for half-time (x0.5) and double-time (x2) tempo relationships
- **MP3 Metadata** - Read/write analysis results to MP3 ID3 tags
- **Local Cache** - SQLite database for fast loading of previously analyzed tracks
- **Audio Playback** - Built-in player to preview tracks
- **Drag & Drop** - Easy file loading
- **CSV Export** - Export track list with compatibility data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TonalDJPlugin.git
cd TonalDJPlugin
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Requirements

- Python 3.8+
- PyQt5
- librosa
- numpy
- scipy
- soundfile
- pygame
- mutagen

## Usage

1. **Load Tracks** - Click "Load Tracks" or drag & drop audio files (MP3, WAV, FLAC, OGG, M4A)
2. **Set Master Track** - Select a track and click "Set as Master" or use context menu
3. **View Compatibility** - All tracks will show compatibility percentage with the master track
4. **Sort by Match** - Click "Sort by Match" to order tracks by compatibility (toggle ascending/descending)
5. **Adjust BPM** - Right-click on a track to set BPM multiplier (x0.5, x1, x2) for tempo matching
6. **Play Tracks** - Double-click or use context menu to preview tracks
7. **Export** - Click "Export CSV" to save the analysis results

## Camelot Wheel

The Camelot Wheel is a tool that helps DJs mix tracks harmonically. Compatible keys are:
- Same key (e.g., 8A to 8A)
- Adjacent numbers (e.g., 8A to 7A or 9A)
- Same number, different letter (e.g., 8A to 8B - relative major/minor)

## Supported Audio Formats

- MP3
- WAV
- FLAC
- OGG
- M4A
- AAC

## License

MIT License
