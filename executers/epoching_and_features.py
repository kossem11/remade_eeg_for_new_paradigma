# executers/epoching_and_features.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import mne
import numpy as np
import pandas as pd
import neurokit2 as nk
import antropy as ant
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from configs import config

# Создаём папку для результатов, если её нет
config.EPOCHS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Функция загрузки данных (поддержка .vhdr и .edf)
# ------------------------------------------------------------
def load_raw(subject_id, base_path):
    vhdr_files = list(base_path.glob(f"*{subject_id}*.vhdr"))
    edf_files = list(base_path.glob(f"*{subject_id}*.edf"))
    
    if vhdr_files:
        raw = mne.io.read_raw_brainvision(vhdr_files[0], preload=True)
    elif edf_files:
        raw = mne.io.read_raw_edf(edf_files[0], preload=True)
    else:
        raise FileNotFoundError(f"No EEG file found for subject {subject_id}")
    
    # Приводим в порядок типы каналов
    eeg_like = ['F9', 'P9', 'P10', 'F10']
    for ch in eeg_like:
        if ch in raw.ch_names:
            raw.set_channel_types({ch: 'eeg'})
    # Не-ЭЭГ каналы (blood_pressure, GSR) можно оставить как misc или удалить
    misc_ch = ['blood_pressure', 'GSR']
    raw.drop_channels([ch for ch in misc_ch if ch in raw.ch_names])
    return raw

# ------------------------------------------------------------
# 2. Предобработка
# ------------------------------------------------------------
def preprocess_raw(raw, apply_ica=True):
    """Фильтрация, удаление плохих каналов, референция, ICA."""
    raw.filter(l_freq=0.1, h_freq=40.0)
    raw.set_eeg_reference('average', projection=False)
    
    if apply_ica:
        ica = mne.preprocessing.ICA(n_components=20, random_state=42)
        ica.fit(raw)
        # Автоматическое удаление артефактов EOG/ECG можно добавить
        ica.apply(raw)
    return raw

# ------------------------------------------------------------
# 3. Получение событий из аннотаций
# ------------------------------------------------------------
def get_events_from_annotations(raw, marker_list):
    """Возвращает массив событий только для указанных маркеров."""
    events, event_id = mne.events_from_annotations(raw)
    # Оставляем только нужные маркеры
    valid_ids = {desc: idx for desc, idx in event_id.items() if desc in marker_list}
    if not valid_ids:
        return None, None
    # Фильтруем events
    mask = np.isin(events[:, 2], list(valid_ids.values()))
    events_filtered = events[mask]
    return events_filtered, valid_ids

# ------------------------------------------------------------
# 4. Нарезка эпох
# ------------------------------------------------------------
def create_epochs(raw, events, event_id, tmin, tmax, baseline, preload=True):
    """Обёртка для mne.Epochs с отловом ошибок."""
    try:
        epochs = mne.Epochs(
            raw, events, event_id=event_id,
            tmin=tmin, tmax=tmax, baseline=baseline,
            preload=preload, verbose=False
        )
        return epochs
    except Exception as e:
        print(f"Error creating epochs: {e}")
        return None

# ------------------------------------------------------------
# 5. Извлечение признаков (одна эпоха → словарь)
# ------------------------------------------------------------
def extract_features(epoch_data, sfreq):
    """
    epoch_data: array (n_channels, n_times)
    Возвращает словарь признаков, ключи содержат имя канала.
    """
    features = {}
    n_channels, n_times = epoch_data.shape

    # Частотные полосы
    freqs, psd_all = welch(epoch_data, fs=sfreq, nperseg=int(sfreq*2), axis=-1)
    theta_mask = (freqs >= 4) & (freqs <= 8)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask  = (freqs >= 13) & (freqs <= 30)
    gamma_mask = (freqs >= 30) & (freqs <= 40)

    for ch_idx in range(n_channels):
        ch_data = epoch_data[ch_idx]
        psd = psd_all[ch_idx]

        # --- Временные признаки ---
        features[f'ch{ch_idx}_mean'] = np.mean(ch_data)
        features[f'ch{ch_idx}_std'] = np.std(ch_data)
        features[f'ch{ch_idx}_peak_amp'] = np.max(np.abs(ch_data))
        features[f'ch{ch_idx}_peak_latency'] = np.argmax(np.abs(ch_data)) / sfreq * 1000
        features[f'ch{ch_idx}_rms'] = np.sqrt(np.mean(ch_data**2))
        features[f'ch{ch_idx}_auc'] = np.trapz(np.abs(ch_data))

        # Оконные амплитуды
        p300_start = int(0.25 * sfreq)
        p300_end   = int(0.50 * sfreq)
        p300_win = ch_data[p300_start:p300_end]
        features[f'ch{ch_idx}_p300_amp'] = np.max(p300_win) if len(p300_win) > 0 else np.nan

        n200_start = int(0.15 * sfreq)
        n200_end   = int(0.25 * sfreq)
        n200_win = ch_data[n200_start:n200_end]
        features[f'ch{ch_idx}_n200_amp'] = np.min(n200_win) if len(n200_win) > 0 else np.nan

        # --- Частотные признаки ---
        theta_power = np.mean(psd[theta_mask]) if any(theta_mask) else np.nan
        alpha_power = np.mean(psd[alpha_mask]) if any(alpha_mask) else np.nan
        beta_power  = np.mean(psd[beta_mask])  if any(beta_mask)  else np.nan
        gamma_power = np.mean(psd[gamma_mask]) if any(gamma_mask) else np.nan

        features[f'ch{ch_idx}_theta_power'] = theta_power
        features[f'ch{ch_idx}_alpha_power'] = alpha_power
        features[f'ch{ch_idx}_beta_power']  = beta_power
        features[f'ch{ch_idx}_gamma_power'] = gamma_power

        # Отношение с защитой от деления на 0
        if alpha_power and not np.isnan(alpha_power) and alpha_power != 0:
            features[f'ch{ch_idx}_theta_alpha_ratio'] = theta_power / alpha_power
        else:
            features[f'ch{ch_idx}_theta_alpha_ratio'] = np.nan

        # Пиковая частота в альфа-диапазоне
        if any(alpha_mask) and not np.all(np.isnan(psd[alpha_mask])):
            peak_alpha_idx = np.argmax(psd[alpha_mask])
            features[f'ch{ch_idx}_peak_alpha_freq'] = freqs[alpha_mask][peak_alpha_idx]
        else:
            features[f'ch{ch_idx}_peak_alpha_freq'] = np.nan

        # --- Нелинейные признаки (с обработкой исключений) ---
        try:
            features[f'ch{ch_idx}_sample_entropy'] = ant.sample_entropy(ch_data)
        except:
            features[f'ch{ch_idx}_sample_entropy'] = np.nan
        try:
            features[f'ch{ch_idx}_perm_entropy'] = ant.perm_entropy(ch_data, normalize=True)
        except:
            features[f'ch{ch_idx}_perm_entropy'] = np.nan
        try:
            features[f'ch{ch_idx}_hurst'] = nk.complexity_hurst(ch_data)[0]
        except:
            features[f'ch{ch_idx}_hurst'] = np.nan
        try:
            features[f'ch{ch_idx}_higuchi_fd'] = ant.higuchi_fd(ch_data)
        except:
            features[f'ch{ch_idx}_higuchi_fd'] = np.nan

        # --- Морфологические признаки ---
        features[f'ch{ch_idx}_skewness'] = skew(ch_data)
        features[f'ch{ch_idx}_kurtosis'] = kurtosis(ch_data)

    return features

# ------------------------------------------------------------
# 6. Обработка одного испытуемого
# ------------------------------------------------------------
def process_subject(subject_id, raw_path, output_dir):
    """Полный цикл для одного испытуемого."""
    print(f"Processing {subject_id}...")
    
    # Загрузка и препроцессинг
    raw = load_raw(subject_id, raw_path)
    raw = preprocess_raw(raw, apply_ica=True)
    sfreq = raw.info['sfreq']
    
    all_features = []      # список словарей признаков
    labels = []            # тип события (для классификации)
    epoch_counts = {}      # для отладки
    
    # ---------- Фиксации ----------
    fix_markers = list(config.FIXATION_MARKERS.values())
    fix_events, fix_id = get_events_from_annotations(raw, fix_markers)
    if fix_events is not None and len(fix_events) > 0:
        epochs_fix = create_epochs(
            raw, fix_events, fix_id,
            tmin=config.FIXATION_EPOCHS['tmin'],
            tmax=config.FIXATION_EPOCHS['tmax'],
            baseline=config.FIXATION_EPOCHS['baseline']
        )
        if epochs_fix is not None:
            data_fix = epochs_fix.get_data()  # (n_epochs, n_ch, n_times)
            for i_ep, ep_data in enumerate(data_fix):
                feat = extract_features(ep_data, sfreq)
                # Добавляем метку типа фиксации
                event_desc = list(fix_id.keys())[list(fix_id.values()).index(epochs_fix.events[i_ep, 2])]
                feat['event_type'] = event_desc
                feat['subject'] = subject_id
                feat['epoch_category'] = 'fixation'
                all_features.append(feat)
                labels.append(event_desc)
            epoch_counts['fixation'] = len(data_fix)
        else:
            epoch_counts['fixation'] = 0
    else:
        epoch_counts['fixation'] = 0
    
    # ---------- Саккады (три окна) ----------
    sacc_markers = config.SACCADE_MARKERS
    sacc_events, sacc_id = get_events_from_annotations(raw, sacc_markers)
    if sacc_events is not None and len(sacc_events) > 0:
        for window in config.SACCADE_EPOCHS:
            epochs_sacc = create_epochs(
                raw, sacc_events, sacc_id,
                tmin=window['tmin'],
                tmax=window['tmax'],
                baseline=window['baseline']
            )
            if epochs_sacc is not None:
                data_sacc = epochs_sacc.get_data()
                for i_ep, ep_data in enumerate(data_sacc):
                    feat = extract_features(ep_data, sfreq)
                    # Метка – номер саккадического маркера
                    event_id_rev = {v: k for k, v in sacc_id.items()}
                    marker_code = event_id_rev[sacc_events[i_ep, 2]]
                    feat['event_type'] = marker_code
                    feat['subject'] = subject_id
                    feat['epoch_category'] = f'saccade_{window["name"]}'
                    all_features.append(feat)
                    labels.append(marker_code)
                epoch_counts[f'saccade_{window["name"]}'] = len(data_sacc)
            else:
                epoch_counts[f'saccade_{window["name"]}'] = 0
    else:
        for window in config.SACCADE_EPOCHS:
            epoch_counts[f'saccade_{window["name"]}'] = 0
    
    # Превращаем в DataFrame
    df = pd.DataFrame(all_features)
    if df.empty:
        print(f"  No epochs extracted for {subject_id}")
        return None, epoch_counts

    # Определяем столбцы с признаками (исключаем метаинформацию)
    feature_cols = [c for c in df.columns if c not in ['event_type', 'subject', 'epoch_category']]

    # Заменяем бесконечности на NaN
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Заполняем NaN: если весь столбец пуст → 0, иначе средним
    for col in feature_cols:
        if df[col].isnull().all():
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Нормализация
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Сохраняем в CSV
    out_file = output_dir / f"{subject_id}_features.csv"
    df.to_csv(out_file, index=False)
    print(f"  Saved {len(df)} epochs to {out_file}")
    print(f"  Epoch counts: {epoch_counts}")
    return df, epoch_counts

# ------------------------------------------------------------
# 7. Главная функция – обработка всех файлов в папке
# ------------------------------------------------------------
def main():
    # Находим все уникальные идентификаторы испытуемых по именам файлов
    vhdr_files = list(config.MAIN_RDATA_DIR.glob("*.vhdr"))
    edf_files = list(config.MAIN_RDATA_DIR.glob("*.edf"))
    all_files = vhdr_files + edf_files
    
    # Извлекаем идентификатор (например, "ava1209") из имени файла
    # Предполагаем, что имя содержит что-то вроде "..._OCD_ava1209.edf_1_HC_corr_prepro..."
    subjects = set()
    for f in all_files:
        name = f.stem
        # Простой способ: ищем часть после "OCD_" и до следующего подчеркивания или точки
        parts = name.split('_')
        for part in parts:
            if part.startswith('ava'):   # по примеру
                subjects.add(part)
                break
        else:
            # Если не нашли, используем имя файла без расширения как ID
            subjects.add(f.stem)
    
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    # Обрабатываем каждого
    all_dfs = []
    for subj in tqdm(sorted(subjects)):
        df, _ = process_subject(subj, config.MAIN_RDATA_DIR, config.EPOCHS_DIR)
        if df is not None:
            all_dfs.append(df)
    
    # Объединяем все данные в один файл (опционально)
    '''
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_file = config.EPOCHS_DIR / "all_subjects_features.csv"
        combined.to_csv(combined_file, index=False)
        print(f"\nCombined features saved to {combined_file}")
        print(f"Total epochs: {len(combined)}")
        print(f"Columns: {list(combined.columns)}")
    else:
        print("No features extracted.")
    '''

if __name__ == "__main__":
    main()