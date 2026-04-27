import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing
import os
import pandas as pd
from sklearn.inspection import permutation_importance

class ParallelSVMEnsemble:
    """
    Ансамбль SVM моделей, обучающихся параллельно на разных подвыборках данных
    """
    
    def __init__(self, n_estimators=None, n_jobs=-1, kernel='rbf', 
                 cache_size=1000, random_state=42, subsample_ratio=0.8, **kwargs):
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        self.n_estimators = n_estimators or self.n_jobs * 2
        self.kernel = kernel
        self.cache_size = cache_size
        self.random_state = random_state
        self.subsample_ratio = subsample_ratio
        self.kwargs = kwargs
        
        if 'random_state' in self.kwargs:
            del self.kwargs['random_state']

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        
        self.scaler = StandardScaler()
        self.models = []
        self.feature_names = None
        
    def _train_single_svm(self, X, y, model_id, subsample_indices=None):
        if subsample_indices is not None:
            X_sub = X[subsample_indices]
            y_sub = y[subsample_indices]
        else:
            X_sub = X
            y_sub = y
        
        model = SVC(
            kernel=self.kernel,
            cache_size=self.cache_size,
            random_state=self.random_state + model_id,
            probability=True,
            **self.kwargs
        )
        
        model.fit(X_sub, y_sub)
        return model
    
    def fit(self, X, y, feature_names=None):
        print(f"\n{'='*60}")
        print(f"ПАРАЛЛЕЛЬНОЕ ОБУЧЕНИЕ АНСАМБЛЯ SVM")
        print(f"{'='*60}")
        print(f"Моделей в ансамбле: {self.n_estimators}")
        print(f"Используется ядер: {self.n_jobs}")
        print(f"Доля данных на модель: {self.subsample_ratio}")
        print(f"Параметры SVM: kernel={self.kernel}, {self.kwargs}")
        
        self.feature_names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(X.shape[1])]
        X_scaled = self.scaler.fit_transform(X)

        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = y

        n_samples = len(X_scaled)
        subsample_size = int(n_samples * self.subsample_ratio)
        
        np.random.seed(self.random_state)
        bootstrap_indices = []
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, subsample_size, replace=True)
            bootstrap_indices.append(indices)
        
        self.models = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=10)(
            delayed(self._train_single_svm)(X_scaled, y_array, i, bootstrap_indices[i])
            for i in range(self.n_estimators)
        )
        
        print(f"\n✓ Обучение завершено. Успешно обучено {len(self.models)} моделей")
        return self
    
    def predict(self, X):
        if not self.models:
            raise ValueError("Модель не обучена. Сначала вызовите fit()")
        
        X_scaled = self.scaler.transform(X)
        predictions = []
        for model in self.models:
            pred = model.predict(X_scaled)
            predictions.append(pred)

        predictions_array = np.array(predictions)
        final_predictions = np.round(np.mean(predictions_array, axis=0)).astype(int)
        
        return final_predictions
    
    def predict_proba(self, X):
        if not self.models:
            raise ValueError("Модель не обучена. Сначала вызовите fit()")
        
        X_scaled = self.scaler.transform(X)
        
        probas = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)
                probas.append(proba)
        
        if probas:
            return np.mean(probas, axis=0)
        else:
            preds = self.predict(X)
            proba = np.zeros((len(preds), 2))
            proba[:, 0] = 1 - preds
            proba[:, 1] = preds
            return proba
    
    def compute_permutation_importance(self, X, y, n_repeats=10):
        print(f"\nВычисляем permutation importance для ансамбля...")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = y
        

        def importance_for_model(model):
            return permutation_importance(
                model, X_scaled, y_array,
                n_repeats=n_repeats,
                random_state=42,
                scoring='accuracy',
                n_jobs=1
            )

        results = []
        for model in self.models:
            try:
                result = importance_for_model(model)
                results.append(result)
            except Exception as e:
                print(f"Ошибка при вычислении важности для модели: {e}")
                continue
        
        if not results:
            print("Не удалось вычислить permutation importance")
            return None
        
        self.permutation_importance_ = {
            'importances_mean': np.mean([r.importances_mean for r in results], axis=0),
            'importances_std': np.mean([r.importances_std for r in results], axis=0)
        }
        
        return self.permutation_importance_
    
    def get_stimulus_importance(self, top_k=20):
        """event_*"""
        if not hasattr(self, 'permutation_importance_') or self.permutation_importance_ is None:
            print("Сначала выполните compute_permutation_importance()")
            return None

        event_indices = [(i, name) for i, name in enumerate(self.feature_names) 
                        if name.startswith("event_")]
        
        if not event_indices:
            print("Нет признаков event_* для анализа")
            return None
        
        importance_df = pd.DataFrame({
            'feature': [name for _, name in event_indices],
            'importance': [self.permutation_importance_['importances_mean'][i] for i, _ in event_indices],
            'importance_std': [self.permutation_importance_['importances_std'][i] for i, _ in event_indices]
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        print(f"\n=== Важность стимулов для SVM (top {min(top_k, len(importance_df))}) ===")
        for _, row in importance_df.head(top_k).iterrows():
            stimulus_name = row['feature'].replace('event_', '')
            print(f"{stimulus_name:40s} | {row['importance']:.6f} ± {row['importance_std']:.6f}")
        
        return importance_df
    
    def get_ensemble_stats(self):
        print(f"\n=== Статистика ансамбля SVM ===")
        print(f"Количество моделей: {len(self.models)}")
        print(f"Используется ядер: {self.n_jobs}")
        print(f"Доля данных на модель: {self.subsample_ratio}")
        print(f"Параметры SVM: {self.kwargs}")

        n_support = [model.n_support_ for model in self.models if hasattr(model, 'n_support_')]
        if n_support:
            avg_support = np.mean([sum(s) for s in n_support])
            print(f"Среднее количество опорных векторов на модель: {avg_support:.1f}")
        
        return {
            'n_models': len(self.models),
            'n_jobs': self.n_jobs,
            'avg_support_vectors': np.mean([sum(s) for s in n_support]) if n_support else None
        }



class OptimizedSVM(ParallelSVMEnsemble):
    """Алиас для ParallelSVMEnsemble"""
    def __init__(self, n_jobs=-1, kernel='rbf', cache_size=1000, random_state=42, **kwargs):

        super().__init__(
            n_estimators=2,
            n_jobs=n_jobs,
            kernel=kernel,
            cache_size=cache_size,
            random_state=random_state,
            subsample_ratio=0.8,
            **kwargs
        )


def analyze_stimulus_impact(svm_model, X_test, y_test, feature_names, top_k=20):
    print(f"\n{'='*60}")
    print(f"КОМПЛЕКСНЫЙ АНАЛИЗ ВЛИЯНИЯ СТИМУЛОВ")
    print(f"{'='*60}")
    
    svm_model.compute_permutation_importance(X_test, y_test, n_repeats=10)
    stimulus_importance = svm_model.get_stimulus_importance(top_k=top_k)
    svm_model.get_ensemble_stats()
    
    return stimulus_importance