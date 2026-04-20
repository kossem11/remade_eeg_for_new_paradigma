# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
PROCESED_DIR = BASE_DIR / "procesed"
LOADERS_DIR = BASE_DIR / "loaders"
EXECUTERS_DIR = BASE_DIR / "executers"

MAIN_RDATA_DIR = RAW_DATA_DIR / "Main_research"
OPEN_CLOSED_DIR = RAW_DATA_DIR / "Opened_closed_eyes"
EPOCHS_DIR = PROCESED_DIR / "epochs"

# Фиксационные маркеры – как они записаны в аннотациях
FIXATION_MARKERS = {
    'aggression': 'Stimulus/S 11',
    'checking': 'Stimulus/S 21',
    'contamination': 'Stimulus/S 31',
    'symmetry': 'Stimulus/S 41',
    'neutral': 'Stimulus/S 51',
}

# Диапазоны саккадических маркеров (номера)
SACCADE_RANGES = [
    (114, 117),
    (124, 127),
    (134, 137),
    (114, 147),
    (154, 157),
]

def expand_saccade_ranges(ranges, prefix='Saccade/S'):
    """Возвращает список строк вида 'Saccade/S114' для всех номеров в диапазонах."""
    markers = []
    for start, end in ranges:
        for num in range(start, end + 1):
            markers.append(f"{prefix}{num}")
    return markers

SACCADE_MARKERS = expand_saccade_ranges(SACCADE_RANGES)

FIXATION_EPOCHS = {
    'tmin': -0.2,
    'tmax': 0.8,
    'baseline': (None, 0),
}

SACCADE_EPOCHS = [
    {'name': 'window_1', 'tmin': -0.6,   'tmax': -0.240, 'baseline': None},
    {'name': 'window_2', 'tmin': -0.240, 'tmax': -0.010, 'baseline': None},
    {'name': 'window_3', 'tmin': -0.010, 'tmax': 0.010,  'baseline': None},
]

NORMALIZATION = {
    'method': 'zscore',
    'type': 'standard',
}