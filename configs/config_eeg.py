from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
PROCESED_DIR = BASE_DIR / "procesed"

MAIN_RDATA_DIR = RAW_DATA_DIR / "Main_research"
OPEN_CLOSED_DIR = RAW_DATA_DIR / "Opened_closed_eyes"
EPOCHS_DIR = PROCESED_DIR / "epochs"


FIXATION_MARKERS = {
    'aggression': 'S11',
    'checking': 'S21',
    'contamination': 'S31',
    'symmetry': 'S41',
    'neutral': 'S51',
}


SACCADE_RANGES = [
    (114, 117),    # 114, 115, 116, 117
    (124, 127),    # 124, 125, 126, 127
    (134, 137),    # 134, 135, 136, 137
    (114, 147),    # 114-147 
    (154, 157),    # 154, 155, 156, 157
]

# Если нужны отдельные маркеры вне диапазонов – добавьте сюда
SACCADE_EXTRA = []  # например, [200, 205]


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

