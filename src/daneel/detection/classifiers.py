"""Classifier interfaces reserved for the detection course exercises."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, precision_score
)

class RandomForestClassifier:
    """Placeholder for the random-forest exercise."""

    def __init__(self, config: dict):
        if "classifier" in config:
            clf_cfg = config["classifier"]      ###goes into thios to find parameters name
        else:
            clf_cfg = config            ###goes into randoforest(input pars) dictionary
        self.csv_path = clf_cfg.get("path", "data/tess_data.csv")
        self.n_bins = clf_cfg.get("n_bins", 1000)
        self.samples_per_class = clf_cfg.get("samples_per_class", 350)
        self.random_state = clf_cfg.get("random_state", 42)
        self.use_scaler = clf_cfg.get("use_scaler", False)
        np.random.seed(self.random_state)

        self.rf_params = clf_cfg.get("rf", {})     ###params.yaml, distinctioin between rf and cnn

        self.scaler = None
        self.model = None
        self.best_threshold = 0.5    ##### default value, threshold at the beginning, after evaluate() will be changed
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def create_balanced_dataset(self, X, y, samples_per_class=350):
        X0 = X[y == 0]
        X1 = X[y == 1]

        def augment_to_target(X_orig, n_target):
            if len(X_orig) >= n_target:
                idx = np.random.choice(len(X_orig), n_target, replace=False)
                return X_orig[idx]
            
            X_result = [X_orig]
            while len(np.vstack(X_result)) < n_target:
                n_needed = n_target - len(np.vstack(X_result))
                idx = np.random.choice(len(X_orig), min(len(X_orig), n_needed))
                aug_type = np.random.rand()
                if aug_type < 0.25:
                    X_aug = X_orig[idx] + np.random.normal(0, 0.01, (len(idx), X_orig.shape[1]))
                elif aug_type < 0.5:
                    scale = 1.0 + np.random.uniform(-0.03, 0.03, (len(idx), 1))
                    X_aug = X_orig[idx] * scale
                elif aug_type < 0.75:
                    shifts = np.random.randint(-20, 20, len(idx))
                    X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
                else:
                    X_aug = X_orig[idx] * (1.0 + np.random.uniform(-0.02, 0.02, (len(idx), 1)))
                    X_aug += np.random.normal(0, 0.008, X_aug.shape)
                X_result.append(X_aug)
            
            X_final = np.vstack(X_result)
            return X_final[:n_target]
        
        X0_bal = augment_to_target(X0, samples_per_class)
        X1_bal = augment_to_target(X1, samples_per_class)
        X_bal = np.vstack([X0_bal, X1_bal])
        y_bal = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])
            
        idx = np.arange(len(X_bal))
        np.random.shuffle(idx)
        return X_bal[idx], y_bal[idx]

    def load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"File {self.csv_path} non trovato.")

        df = pd.read_csv(self.csv_path)
        flux_cols = [f'flux_{i:04d}' for i in range(self.n_bins)]
        flux_err_cols = [f'flux_err_{i:04d}' for i in range(self.n_bins)]
        X = df[flux_cols].values
        X_err = df[flux_err_cols].values
        y = df['label'].values

        X_train, X_test, y_train, y_test, _, _, _, _ = train_test_split(
            X, y, X_err, np.arange(len(y)),
            test_size=0.2,
            random_state=self.random_state,
            stratify=y
        )
        X_train, y_train = self.create_balanced_dataset(
            X_train, y_train, samples_per_class=self.samples_per_class
        )
        if self.use_scaler:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
    
    def build_random_forest(self):
        return SklearnRF(           ###da params.yaml
            n_estimators=self.rf_params.get("n_estimators", 500),
            max_depth=self.rf_params.get("max_depth", None),
            min_samples_leaf=self.rf_params.get("min_samples_leaf", 1),
            max_features=self.rf_params.get("max_features", "sqrt"),
            bootstrap=self.rf_params.get("bootstrap", True),
            class_weight=self.rf_params.get("class_weight", None),
            random_state=self.random_state,
            n_jobs=self.rf_params.get("n_jobs", -1),
            oob_score=self.rf_params.get("oob_score", True)
        )

    def train_model(self):
        self.model = self.build_random_forest()
        self.model.fit(self.X_train, self.y_train)

    def evaluate_with_optimal_threshold(self):
        proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        self.best_threshold = thresholds[best_idx]
        
        y_pred_best = (proba >= self.best_threshold).astype(int)
        
        auc = roc_auc_score(self.y_test, proba)
        acc_best = accuracy_score(self.y_test, y_pred_best)
        prec_planet = precision_score(self.y_test, y_pred_best, pos_label=1, zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred_best)

##############OUPUT e RUN

        print("\n" + "=" * 60)
        print("RANDOM FOREST EVALUATION STATISTICS (TASK G)")
        print("=" * 60)
        if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
                        print(f"OOB Score: {self.model.oob_score_:.4f}")
        print(f"Optimal Threshold: {self.best_threshold:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy (at optimal threshold): {acc_best:.4f}")
        print(f"Precision (Planet): {prec_planet:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
        print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
        print("\nClassification report:")
        print(classification_report(self.y_test, y_pred_best, target_names=['Non-Planet', 'Planet'], digits=4, zero_division=0))
        print("=" * 60 + "\n")

    def run(self):
        print("Loading data...")
        self.load_data()
        print("Training model...")
        self.train_model()
        print("Evaluating model...")
        self.evaluate_with_optimal_threshold()






#     def fit(self, X, y):
#         raise NotImplementedError("Random-forest training is a course exercise")


# class CNNClassifier:
#     """Placeholder for the CNN exercise."""

#     def fit(self, X, y):
#         raise NotImplementedError("CNN training is a course exercise")
