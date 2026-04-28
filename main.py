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

#from optimizers.svm_optimized import OptimizedSVM, ParallelSVMEnsemble, Parallel

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost не установлен")

#EPOCHS_DIR = PROCESED_DIR / "epochs"           
SACCADE_WINDOW = "saccade_window_2" #саккадическое окно
FEATURE_SUFFIXES = ["_mean", "_std", "_peak_amp", "_peak_latency", "_auc"] # фитчи с каналов
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
    
    # Масштабирование для SVM и LR
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