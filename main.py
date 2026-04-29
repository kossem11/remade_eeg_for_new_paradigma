"""
Классификация HC/OCD.
Используются:
- все эпохи, кроме саккад (saccade_window_*), и одно выбранное окно саккад (saccade_window_x)
- признаки: _mean, _std, _peak_amp, _peak_latency, _auc + event_type
- модели: Random Forest, SVM, Logistic Regression
- вывод accuracy, confusion matrix и важности стимулов (коэффициенты LR, importance RF и permutation importance для SVM)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import EPOCHS_DIR
import matplotlib.pyplot as plt
import mne

#from optimizers.svm_optimized import OptimizedSVM, ParallelSVMEnsemble, Parallel
MNE_AVAILABLE = True
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost не установлен")

#EPOCHS_DIR = PROCESED_DIR / "epochs"           
SACCADE_WINDOW = "saccade_window_2" #саккадическое окно
FEATURE_SUFFIXES = ["_mean", "_std", "_auc"] # фитчи с каналов "_mean", "_std", "_peak_amp", "_peak_latency", "_auc"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_filter_data(data_dir, saccade_window):

    all_data = []
    for file_path in data_dir.glob("*.csv"):
        name = file_path.stem
        if "HC" in name:
            group = "HC"
        elif "OCD" in name:
            group = "OCD"
        else:
            continue
        
        df = pd.read_csv(file_path)
        
        mask_fix = ~df["epoch_category"].str.startswith("saccade_window_")
        # Строки с выбранным саккадическим окном
        mask_sacc = df["epoch_category"] == saccade_window
        df_filtered = df[mask_fix | mask_sacc].copy()
        
        if df_filtered.empty:
            continue

        feature_cols = [col for col in df_filtered.columns 
                        if any(col.endswith(suf) for suf in FEATURE_SUFFIXES)]
        feature_cols.append("event_type")
        
        X_part = df_filtered[feature_cols].copy()
        X_part["group"] = group
        all_data.append(X_part)
    
    if not all_data:
        raise ValueError(f"Не найдено данных в {data_dir}")
    
    full_data = pd.concat(all_data, ignore_index=True)
    return full_data



def prepare_features_labels(full_data):
    #  one-hot encoding для event_type.
    y = (full_data["group"] == "OCD").astype(int)
    X = full_data.drop(columns=["group"])
    X = pd.get_dummies(X, columns=["event_type"], prefix="event")
    return X, y



def train_evaluate_models(X, y, models):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        if name in ("SVM", "LogisticRegression"):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            trained_models[name] = (model, scaler)
        else:
            model.fit(X_train, y_train)
            #y_pred = model.predict(X_test)
            
            y_probs = model.predict_proba(X_test)[:, 1]
            threshold = 0.45 # порог HC/Ocd
            y_pred = (y_probs >= threshold).astype(int)

            trained_models[name] = (model, None)
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {"accuracy": acc, "confusion_matrix": cm}        
        print(f"\n{'='*40}\n{name}\n{'='*40}")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion matrix:")
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, target_names=["HC", "OCD"]))
    
    return results, trained_models, (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)



def print_stimulus_importance_rf(model, feature_names):
    importances = model.feature_importances_
    event_indices = [(i, name) for i, name in enumerate(feature_names) if name.startswith("event_")]
    if not event_indices:
        print("Нет признаков event_* для анализа важности.")
        return
    print("\n=== Важность стимулов (Random Forest) ===")
    for idx, name in sorted(event_indices, key=lambda x: importances[x[0]], reverse=True):
        print(f"{name}: {importances[idx]:.6f}")

def print_stimulus_coefficients_lr(model, feature_names, scaler=None):

    coef = model.coef_[0]
    event_indices = [(i, name) for i, name in enumerate(feature_names) if name.startswith("event_")]
    if not event_indices:
        print("Нет признаков event_* для анализа коэффициентов.")
        return
    sorted_events = sorted(event_indices, key=lambda x: abs(coef[x[0]]), reverse=True)
    print("\n=== Коэффициенты логистической регрессии для стимулов ===")
    for idx, name in sorted_events:
        print(f"{name}: {coef[idx]:.6f}")

def print_stimulus_importance_xgb(model, feature_names):
    """Feature importance для XGBoost (как у Random Forest)"""
    if not hasattr(model, 'feature_importances_'):
        print("Модель XGBoost не имеет feature_importances_")
        return
    importances = model.feature_importances_
    event_indices = [(i, name) for i, name in enumerate(feature_names) if name.startswith("event_")]
    if not event_indices:
        print("Нет признаков event_* для анализа важности.")
        return
    print("\n=== Важность стимулов (XGBoost) ===")
    for idx, name in sorted(event_indices, key=lambda x: importances[x[0]], reverse=True):
        print(f"{name}: {importances[idx]:.6f}")

"""
def print_stimulus_importance_svm(model, X_test, y_test, feature_names):
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy'
    )
    importances = perm_importance.importances_mean
    event_indices = [(i, name) for i, name in enumerate(feature_names) if name.startswith("event_")]
    if not event_indices:
        print("Нет признаков event_* для анализа важности.")
        return
    print("\n=== Важность стимулов (SVM, permutation importance) ===")
    for idx, name in sorted(event_indices, key=lambda x: importances[x[0]], reverse=True):
        print(f"{name}: {importances[idx]:.6f}")
"""

def optimize_xgboost(X_train, y_train, X_test, y_test):
    """Подбор гиперпараметров XGBoost с помощью GridSearchCV"""
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
    }
    
    xgb_base = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    grid = GridSearchCV(
        xgb_base, param_grid, cv=5,
        scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    print("\n=== Оптимизированный XGBoost ===")
    print(f"Лучшие параметры: {grid.best_params_}")
    print(f"Лучшая кросс-валидационная точность: {grid.best_score_:.4f}")
    
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Тестовая точность: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["HC", "OCD"]))
    
    return best

def aggregate_channel_importance(model, feature_names, channel_prefix='ch'):

    importances = model.feature_importances_
    ch_imp = {}
    for name, imp in zip(feature_names, importances):
        if name.startswith(channel_prefix):
            # Извлекаем номер канала из "ch123_mean"
            parts = name.split('_')
            ch_str = parts[0]          # "ch123"
            ch_num = int(ch_str[len(channel_prefix):])  # 123
            ch_imp[ch_num] = ch_imp.get(ch_num, 0.0) + imp
    return ch_imp

def plot_channel_importance_topomap(ch_importance, model_name, n_channels=64, 
                                    montage_name='GSN-HydroCel-64_1.0', save_path=None):

 
    importance_list = [ch_importance.get(i, 0.0) for i in range(n_channels)]
    
    # Загружаем montage и координаты
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        positions = montage.get_positions()['ch_pos']
        ch_names = list(positions.keys())[:n_channels]
        pos_2d = np.array([positions[name][:2] for name in ch_names])
    except Exception as e:
        print(f"Ошибка загрузки montage: {e}")
        # Fallback barplot
        plt.figure(figsize=(12,5))
        ch_list = sorted(ch_importance.keys())
        imp_vals = [ch_importance[c] for c in ch_list]
        plt.bar(ch_list, imp_vals)
        plt.xlabel('Channel index')
        plt.ylabel('Summed feature importance')
        plt.title(f'{model_name} - Channel importance (barplot)')
        if save_path:
            bar_path = str(save_path).replace('.png', '_bar.png')
            plt.savefig(bar_path)
            print(f"Сохранён barplot: {bar_path}")
        plt.show()
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        result = mne.viz.plot_topomap(importance_list, pos_2d, axes=ax, show=False,
                                      names=ch_names, cmap='RdBu_r')
    except Exception as e:
        print(f"Ошибка при построении топоплота: {e}")
        result = mne.viz.plot_topomap(importance_list, pos_2d, axes=ax, show=False, cmap='RdBu_r')
    
    if isinstance(result, tuple):
        im = result[0]
    else:
        im = result

    if hasattr(im, 'cmap'):
        cbar = plt.colorbar(im, ax=ax, label='Summed Feature Importance')
    else:
        print("Не удалось создать colorbar, но график построен.")
    
    ax.set_title(f'{model_name} - Channel importance (topographic map)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(str(save_path), dpi=150)
        print(f"Сохранено: {save_path}")
    plt.show()

def plot_channel_importance_for_models(trained_models, feature_names, 
                                       n_channels=64, save_dir=None):

    for name, (model, _) in trained_models.items():
        if hasattr(model, 'feature_importances_'):
            ch_imp = aggregate_channel_importance(model, feature_names)
            if ch_imp:
                save_path = None
                if save_dir:
                    save_path = Path(save_dir) / f"{name}_channel_importance.png"
                plot_channel_importance_topomap(ch_imp, name, n_channels=n_channels,
                                                save_path=save_path)
            else:
                print(f"Для модели {name} не найдены признаки каналов.")

if __name__ == "__main__":

    print("Загрузка данных...")
    full_data = load_and_filter_data(EPOCHS_DIR, SACCADE_WINDOW)
    print(f"Загружено строк: {len(full_data)}")
    print(f"Распределение групп:\n{full_data['group'].value_counts()}")
    X, y = prepare_features_labels(full_data)
    print(f"Размер X: {X.shape}")

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )

    results, trained_models, splits = train_evaluate_models(X, y, models)
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = splits
    feature_names = X.columns.tolist()

    #RandomForest
    if "RandomForest" in trained_models:
        rf_model = trained_models["RandomForest"][0]
        print_stimulus_importance_rf(rf_model, feature_names)
    
    #XGBoost
    if "XGBoost" in trained_models:
        xgb_model = trained_models["XGBoost"][0]
        print_stimulus_importance_xgb(xgb_model, feature_names)

    # Параметры: у вас 64 канала (ch0..ch63)
    N_CHANNELS = 64
    
    # Построим карты для RandomForest и XGBoost
    # Создадим папку для сохранения, например 'topomaps'
    save_dir = Path(__file__).parent / "topomaps"
    save_dir.mkdir(exist_ok=True)
    
    plot_channel_importance_for_models(trained_models, feature_names,
                                       n_channels=N_CHANNELS, save_dir=save_dir)

    '''
    # (Опционально) подбор гиперпараметров XGBoost
    if XGBOOST_AVAILABLE:
        print("\n" + "="*40)
        print("Запуск оптимизации XGBoost...")
        best_xgb = optimize_xgboost(X_train, y_train, X_test, y_test)
    '''

    '''
    # svm
    rf_model = trained_models["RandomForest"][0]
    print_stimulus_importance_rf(rf_model, feature_names)
    svm_model = trained_models["SVM"][0]  # Получаем модель
    importance_df = analyze_stimulus_impact(
        svm_model, 
        X_test,      
        y_test, 
        feature_names,
        top_k=20
    )
    if importance_df is not None:
        importance_df.to_csv("svm_stimulus_importance.csv", index=False)
        print("\n✓ Результаты сохранены в 'svm_stimulus_importance.csv'")'''