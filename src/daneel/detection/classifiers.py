"""Classifier interfaces reserved for the detection course exercises."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as SklearnTREE
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, precision_score
)


class RandomForestClassifier:
    """Placeholder for the random-forest exercise."""

    def __init__(self, config: dict):
        clf_cfg = config["classifier"]      ###goes into thios to find parameters name
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
        self.X_err_test = None  ####per i plot
        self.metadata_test = None ####per plot lightcurve

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

        metadata_cols = ['toi_name', 'tic', 'label', 'disp', 'period_d', 't0_bjd', 'dur_hr', 'sector']
        metadata = df[metadata_cols]

        ####didn't add x_err_train and idx_train
        
        X_train, X_test, y_train, y_test, _, X_err_test, _, idx_test = train_test_split(      
            X, y, X_err, np.arange(len(y)),
            test_size=0.2,
            random_state=self.random_state,
            stratify=y
        )
        if(self.rf_params.get("class_weight",None) ==None):
            X_train, y_train = self.create_balanced_dataset(
                X_train, y_train, samples_per_class=self.samples_per_class
            )
        if self.use_scaler:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.X_err_test = X_err_test
        self.metadata_test = metadata.iloc[idx_test].reset_index(drop=True)

    ########################Plot?!        
    
    def _plot_confusion_matrix(self, ax, y_true, y_pred, threshold):
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2),
               yticks=np.arange(2),
               xticklabels=['Non-Planet', 'Planet'],
               yticklabels=['Non-Planet', 'Planet'],
               xlabel='Predicted', ylabel='True',
               title=f'Confusion Matrix (threshold={threshold:.3f})')
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center')

    
    def _plot_roc_curve(self, ax, y_true, proba):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, linewidth=2)
        ax.plot([0,1], [0,1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve (AUC={auc:.4f})')
        ax.grid(alpha=0.3)

    def _plot_pr_curve(self, ax, y_true, proba):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(recall, precision, linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve (AP={ap:.4f})')
        ax.grid(alpha=0.3)

    def _plot_probability_histograms(self, ax, y_true, proba):
        ax.hist(proba[y_true==0], bins=30, alpha=0.6, label='Non-Planet', density=True)
        ax.hist(proba[y_true==1], bins=30, alpha=0.6, label='Planet', density=True)
        ax.set_xlabel('Predicted Probability (class=1)')
        ax.set_ylabel('Density')
        ax.set_title('Predicted Probability Distributions by Class')
        ax.legend()
        ax.grid(alpha=0.3)

    def plot_lightcurve_sample(self, idx, X_test_standardized, X_err_test, metadata_test, scaler=None, proba=None, y_true=None, y_pred=None, save_prefix='sample_lightcurve_RF'):
        # Make a single-figure plot for one sample (no subplots)
        x_std = X_test_standardized[idx].reshape(1, -1)
        if scaler is not None:
            x_orig = scaler.inverse_transform(x_std).flatten()
            yerr = X_err_test[idx]
        else:
            x_orig = x_std.flatten()
            yerr = X_err_test[idx]
        
        fig = plt.figure(figsize=(10, 4))
        ax = plt.gca()
        ax.errorbar(np.arange(len(x_orig)), x_orig, yerr=yerr, fmt='o', markersize=2, alpha=0.6)
        ax.axhline(np.median(x_orig), linestyle='--', linewidth=1)
        ax.set_xlabel('Time Bin')
        ax.set_ylabel('Flux')
        
        toi = metadata_test.loc[idx, 'toi_name']
        tic = metadata_test.loc[idx, 'tic']
        disp = metadata_test.loc[idx, 'disp']
        sector = metadata_test.loc[idx, 'sector']
        
        tstr = f'TOI {toi} (TIC {tic}, {disp}) - Sector {sector}'
        if proba is not None and y_true is not None and y_pred is not None:
            pred_str = 'Transit' if y_pred[idx]==1 else 'Non-Transit'
            true_str = 'Transit' if y_true[idx]==1 else 'Non-Transit'
            tstr += f'\nTrue: {true_str} | Pred: {pred_str} (p={proba[idx]:.3f})'
        
        ax.set_title(tstr)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = f"{save_prefix}_{idx}.png"
        plt.savefig(path, dpi=300)
        print(f"Saved: {path}")
        plt.close(fig)

    def plot_all_metrics(self, y_true, y_pred, proba, threshold, save_path='evaluation_grid.png'):
        # Crea una figura con una griglia 2x2 di assi
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
        # Appiattisce l'array degli assi per accedervi comodamente con un indice da 0 a 3
        ax1, ax2, ax3, ax4 = axes.flatten()
                
        # Chiama i singoli metodi passando l'asse corrispondente
        self._plot_confusion_matrix(ax1, y_true, y_pred, threshold)
        self._plot_roc_curve(ax2, y_true, proba)
        self._plot_pr_curve(ax3, y_true, proba)
        self._plot_probability_histograms(ax4, y_true, proba)
                
        # Sistema automaticamente gli spazi tra i sottografici
        plt.tight_layout()
                
        # Salva l'immagine finale unica
        plt.savefig(save_path, dpi=300)
        print(f"Saved combined plot: {save_path}")
        plt.close(fig)
    
    
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

    def evaluate_with_optimal_threshold(self,show,name):
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
        print("RANDOM FOREST EVALUATION STATISTICS")
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

        if(show):
            self.plot_all_metrics(self.y_test, y_pred_best, proba, self.best_threshold, save_path=f'RF_evaluation_grid_{name}.png')
       
    def run(self,show,name):
        print(f"Loading random forest data for {name}")
        self.load_data()
        print(f"Training random forest model for {name}")
        self.train_model()
        print(f"Evaluating random forest model for {name}") 
        self.evaluate_with_optimal_threshold(show,name)

    def evaluate_best_hyperparameters(self):
        param_grid = {
                        'n_estimators': [250,500],
                        'max_depth': [10,50],
                        'min_samples_leaf': [5,10,20], 
                        'max_features': ['sqrt', 'log2']
                     }
        self.load_data()
        self.model = self.build_random_forest()
        grid_search_rf = GridSearchCV(self.model,
                              param_grid=param_grid, 
                              cv=3,
                              scoring='recall',
                              return_train_score=True)

        grid_search_rf.fit(self.X_train, self.y_train)
        print("The best hyperparameters to use for random forest model are equal to:") 
        print(grid_search_rf.best_params_)
    
class TreemodelClassifier:
    ''' Defining the tree model class for classifier '''

    #initializing our class 

    def __init__(self, config: dict):
            clf_cfg = config["classifier"]      ###goes into thios to find parameters name
            self.csv_path = clf_cfg.get("path", "data/tess_data.csv")
            self.n_bins = clf_cfg.get("n_bins", 1000)
            self.samples_per_class = clf_cfg.get("samples_per_class", 350)
            self.random_state = clf_cfg.get("random_state", 42)
            self.use_scaler = clf_cfg.get("use_scaler", False)
            np.random.seed(self.random_state)
    
            self.tree_params = clf_cfg.get("tree", {})     ###params.yaml, distinctioin between the different classifers models
    
            self.scaler = None
            self.model = None
            self.best_threshold = 0.5    ##### default value, threshold at the beginning, after evaluate() will be changed
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.X_err_test = None  ####per i plot
            self.metadata_test = None ####per plot lightcurve
    
    #augumentation
    def create_balanced_dataset(self, X, y, samples_per_class=350):
            np.random.seed(42)
            tf.random.set_seed(42)
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
            np.random.seed(42)
            tf.random.set_seed(42)
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"File {self.csv_path} not found.")
    
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
            #if not specify by the user we artificially create a balanced class
            if(self.tree_params.get("class_weight",None) == None):
                X_train, y_train = self.create_balanced_dataset(
                        X_train, y_train, samples_per_class=self.samples_per_class
                    )
            if self.use_scaler:
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
    
            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test
    ########################Plot?!        
    
    def _plot_confusion_matrix(self, ax, y_true, y_pred, threshold):
            cm = confusion_matrix(y_true, y_pred)
            im = ax.imshow(cm, interpolation='nearest')
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(2),
                   yticks=np.arange(2),
                   xticklabels=['Non-Planet', 'Planet'],
                   yticklabels=['Non-Planet', 'Planet'],
                   xlabel='Predicted', ylabel='True',
                   title=f'Confusion Matrix (threshold={threshold:.3f})')
            total = cm.sum()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    count = cm[i, j]
                    pct = (count / total * 100) if total > 0 else 0.0
                    ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center')
    
        
    def _plot_roc_curve(self, ax, y_true, proba):
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc = roc_auc_score(y_true, proba)
            ax.plot(fpr, tpr, linewidth=2)
            ax.plot([0,1], [0,1], linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve (AUC={auc:.4f})')
            ax.grid(alpha=0.3)
    
    def _plot_pr_curve(self, ax, y_true, proba):
            precision, recall, _ = precision_recall_curve(y_true, proba)
            ap = average_precision_score(y_true, proba)
            ax.plot(recall, precision, linewidth=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curve (AP={ap:.4f})')
            ax.grid(alpha=0.3)
    
    def _plot_probability_histograms(self, ax, y_true, proba):
            ax.hist(proba[y_true==0], bins=30, alpha=0.6, label='Non-Planet', density=True)
            ax.hist(proba[y_true==1], bins=30, alpha=0.6, label='Planet', density=True)
            ax.set_xlabel('Predicted Probability (class=1)')
            ax.set_ylabel('Density')
            ax.set_title('Predicted Probability Distributions by Class')
            ax.legend()
            ax.grid(alpha=0.3)
    
    def plot_lightcurve_sample(self, idx, X_test_standardized, X_err_test, metadata_test, scaler=None, proba=None, y_true=None, y_pred=None, save_prefix='sample_lightcurve_RF'):
            # Make a single-figure plot for one sample (no subplots)
            x_std = X_test_standardized[idx].reshape(1, -1)
            if scaler is not None:
                x_orig = scaler.inverse_transform(x_std).flatten()
                yerr = X_err_test[idx]
            else:
                x_orig = x_std.flatten()
                yerr = X_err_test[idx]
            
            fig = plt.figure(figsize=(10, 4))
            ax = plt.gca()
            ax.errorbar(np.arange(len(x_orig)), x_orig, yerr=yerr, fmt='o', markersize=2, alpha=0.6)
            ax.axhline(np.median(x_orig), linestyle='--', linewidth=1)
            ax.set_xlabel('Time Bin')
            ax.set_ylabel('Flux')
            
            toi = metadata_test.loc[idx, 'toi_name']
            tic = metadata_test.loc[idx, 'tic']
            disp = metadata_test.loc[idx, 'disp']
            sector = metadata_test.loc[idx, 'sector']
            
            tstr = f'TOI {toi} (TIC {tic}, {disp}) - Sector {sector}'
            if proba is not None and y_true is not None and y_pred is not None:
                pred_str = 'Transit' if y_pred[idx]==1 else 'Non-Transit'
                true_str = 'Transit' if y_true[idx]==1 else 'Non-Transit'
                tstr += f'\nTrue: {true_str} | Pred: {pred_str} (p={proba[idx]:.3f})'
            
            ax.set_title(tstr)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            path = f"{save_prefix}_{idx}.png"
            plt.savefig(path, dpi=300)
            print(f"Saved: {path}")
            plt.close(fig)
    
    def plot_all_metrics(self, y_true, y_pred, proba, threshold, save_path='evaluation_grid.png'):
            # Crea una figura con una griglia 2x2 di assi
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
            # Appiattisce l'array degli assi per accedervi comodamente con un indice da 0 a 3
            ax1, ax2, ax3, ax4 = axes.flatten()
                    
            # Chiama i singoli metodi passando l'asse corrispondente
            self._plot_confusion_matrix(ax1, y_true, y_pred, threshold)
            self._plot_roc_curve(ax2, y_true, proba)
            self._plot_pr_curve(ax3, y_true, proba)
            self._plot_probability_histograms(ax4, y_true, proba)
                    
            # Sistema automaticamente gli spazi tra i sottografici
            plt.tight_layout()
                    
            # Salva l'immagine finale unica
            plt.savefig(save_path, dpi=300)
            print(f"Saved combined plot: {save_path}")
            plt.close(fig)

    def build_tree(self):
        return SklearnTREE(           ###da params.yaml
            max_depth = self.tree_params.get("max_depth",4),
            min_samples_split = self.tree_params.get("min_samples_split",10),
            min_samples_leaf = self.tree_params.get("min_samples_leaf",5),
            ccp_alpha = self.tree_params.get("ccp_alpha",0.01),
            class_weight= self.tree_params.get("class_weight",None),
            random_state=self.random_state         
            )
    
    def train_model(self):
        self.model = self.build_tree()
        self.model.fit(self.X_train,self.y_train)

    def evaluate_with_optimal_threshold(self,show,name):

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
        print("TREE MODEL EVALUATION STATISTICS")
        print("=" * 60)
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

        if(show):
            self.plot_all_metrics(self.y_test, y_pred_best, proba, self.best_threshold, save_path=f'Tree_evaluation_grid_{name}.png')
    
    def run(self,show,name):
            print(f"Loading tree data for {name}")
            self.load_data()
            print(f"Training tree model for {name}")
            self.train_model()
            print(f"Evaluating tree model for {name}") 
            self.evaluate_with_optimal_threshold(show,name)


    def evaluate_best_hyperparameters(self):
        param_grid = {
                        'max_depth': [5,10],
                        'min_samples_split': [5,10],
                        'min_samples_leaf': [1, 2, 5],
                        'ccp_alpha': [0.001, 0.01]
                     }
        self.load_data()
        self.model = self.build_tree()
        grid_search_tree = GridSearchCV(self.model, 
                              param_grid=param_grid, 
                              cv=3,
                              scoring='recall',
                              return_train_score=True)

        grid_search_tree.fit(self.X_train, self.y_train)
        print("The best hyperparameters to use for our tree model are equal to:") 
        print(grid_search_tree.best_params_)   


class CnnClassifier:

    """Placeholder for the cnn classifier class"""

    def __init__(self, config: dict):

        clf_cfg = config["classifier"]      ###goes into thios to find parameters name
        self.csv_path = clf_cfg.get("path", "data/tess_data.csv")
        self.n_bins = clf_cfg.get("n_bins", 1000)
        self.samples_per_class = clf_cfg.get("samples_per_class", 350)
        self.random_state = clf_cfg.get("random_state", 42)
        self.use_scaler = clf_cfg.get("use_scaler", False)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        self.cnn_params = clf_cfg.get("cnn", {})     ###params.yaml, distinctioin between rf and cnn

        self.scaler = None
        self.history = None
        self.model = None
        self.best_threshold = 0.5    #in the cnn case this value is rarely optimal with imbalanced data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_err_test = None  ####per i plot
        self.metadata_test = None ####per plot lightcurve

    #defining our focal loss --> only callable if specified by the user 
    def focal_loss(self):

        """Focal loss optimized for severe imbalance (binary)."""

        alpha = self.cnn_params.get("alpha",0.75)
        gamma = self.cnn_params.get("gamma",2.5)

        def focal_loss_fixed(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
            focal_weight = alpha_factor * K.pow(1 - pt, gamma)
            bce = -K.log(pt)
            return K.mean(focal_weight * bce)
        return focal_loss_fixed

    #balanced augumentation
    def create_balanced_dataset(self, X, y, samples_per_class=400):
        
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]
        
        def augment_to_target(X_orig, n_target):
            if len(X_orig) >= n_target:
                idx = np.random.choice(len(X_orig), n_target, replace=False)
                return X_orig[idx]
            
            X_result = [X_orig]
            while len(np.vstack(X_result)) < n_target:
                # number we still need (cap to avoid oversampling too big chunks)
                n_needed = n_target - len(np.vstack(X_result))
                idx = np.random.choice(len(X_orig), min(len(X_orig), n_needed))
                aug_type = np.random.rand()
                if aug_type < 0.25:
                    # Additive Gaussian noise
                    X_aug = X_orig[idx] + np.random.normal(0, 0.01, (len(idx), X_orig.shape[1]))
                elif aug_type < 0.5:
                    # Multiplicative scaling
                    scale = 1.0 + np.random.uniform(-0.03, 0.03, (len(idx), 1))
                    X_aug = X_orig[idx] * scale
                elif aug_type < 0.75:
                    # Circular shift (time shift)
                    shifts = np.random.randint(-20, 20, len(idx))
                    X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
                else:
                    # Mild combo: small scale + small noise
                    X_aug = X_orig[idx] * (1.0 + np.random.uniform(-0.02, 0.02, (len(idx), 1)))
                    X_aug += np.random.normal(0, 0.008, X_aug.shape)
                
                X_result.append(X_aug)
            
            X_final = np.vstack(X_result)
            return X_final[:n_target]
        
        X0_bal = augment_to_target(X_class0, samples_per_class)
        X1_bal = augment_to_target(X_class1, samples_per_class)
        X_balanced = np.vstack([X0_bal, X1_bal])
        y_balanced = np.concatenate([np.zeros(samples_per_class), np.ones(samples_per_class)])
        
        # Shuffle
        idx = np.arange(len(X_balanced))
        np.random.shuffle(idx)
        
        return X_balanced[idx], y_balanced[idx]
    
    def load_data(self,hyper):
        
        
        if not os.path.exists(self.csv_path):
                        raise FileNotFoundError(f"File {self.csv_path} not found.")
        
        df = pd.read_csv(self.csv_path)
        
        flux_cols = [f'flux_{i:04d}' for i in range(self.n_bins)]
        flux_err_cols = [f'flux_err_{i:04d}' for i in range(self.n_bins)]
        X = df[flux_cols].values
        X_err = df[flux_err_cols].values
        y = df['label'].values
        
        # Train/test split (keep errors aligned; stratify to preserve class ratio)
        X_train, X_test, y_train, y_test, X_err_train, X_err_test, idx_train, idx_test = train_test_split(
            X, y, X_err, np.arange(len(y)),
            test_size=0.2,
            random_state=self.random_state,
            stratify=y
        )
        
        # Balance training set
        X_train, y_train = self.create_balanced_dataset(X_train, y_train, samples_per_class=self.samples_per_class)
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Reshape for Conv1D or Conv2D (otherwise we can't use conv1d maxpooling etc because we have a wrong order)
        if(self.cnn_params.get("cnn_type","cnn1") == "cnn1" or self.cnn_params.get("cnn_type","cnn1") == "cnn3" or hyper == True):
            X_train = X_train.reshape(-1, self.n_bins, 1)
            X_test = X_test.reshape(-1, self.n_bins, 1)
        else:
            self.n_bins = int(self.n_bins)
            if not ((self.n_bins)%int(self.cnn_params.get("reshape",10)) == 0):
                raise TypeError("Expected reshape must be compatible with the number of bins")
            else:
                X_train = X_train.reshape(-1, int(self.cnn_params.get("reshape",10)) , int(self.n_bins/int(self.cnn_params.get("reshape",10))), 1)
                X_test = X_test.reshape(-1, int(self.cnn_params.get("reshape",10)) , int(self.n_bins/int(self.cnn_params.get("reshape",10))), 1)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def build_CNN1(self):

        """first cnn model implement in 1D"""

        model = models.Sequential([
            layers.Input(shape=(self.n_bins, 1)),
            
            # Feature extraction
            layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            # Classification head
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])

        if(self.cnn_params.get("loss","focal") == "focal"):
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss=self.focal_loss(),
                metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc')]
            )
        elif(self.cnn_params.get("loss","focal") == "bce"):
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss="binary_crossentropy",
                metrics=['accuracy',
                            keras.metrics.Precision(name='precision'),
                            keras.metrics.Recall(name='recall'),
                            keras.metrics.AUC(name='auc')]
                         )
        else:
            raise ValueError("Use 'focal' or 'bce'")
        
        self.model = model

    def build_CNN2(self):

        # Bidimensional architectur
        model = models.Sequential([
            layers.Input(shape=(int(self.cnn_params.get("reshape",10)) , int(self.n_bins/int(self.cnn_params.get("reshape",10))), 1)),

            # Primo blocco convoluzionale
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),

            # Secondo blocco convoluzionale
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),

            layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            # Riduzione delle feature maps
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),

            # Classificatore
            layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),        
            # Uscita binaria
            layers.Dense(1, activation="sigmoid")

        ], name="cnn2_2d")

        if(self.cnn_params.get("loss","focal") == "focal"):
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss=self.focal_loss(),
                metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc')]
                    )
        elif(self.cnn_params.get("loss","focal") == "bce"):
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss="binary_crossentropy",
                metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc')]
                    )
        else:
                raise ValueError("Use 'focal' or 'bce'")
        
        self.model = model
        
    def build_CNN3(self):

        def building_CNN3(self):
            inputs = layers.Input(shape=(self.n_bins, 1))
            x = inputs
            conv_config = [
                    (32, 7),
                    (64, 5),
                    (128, 5),
                    (128, 3),
                    (256, 3)
                ]
            
            for i, (filters, kernel_size) in enumerate(conv_config):
            
                x = layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    kernel_regularizer=regularizers.l2(1e-4)
                )(x)
            
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
            
                if i < 3:
                    x = layers.MaxPooling1D(pool_size=2)(x)
            
                if i == 2:
                    x = layers.Dropout(0.20)(x)
            
                if i == 4:
                    x = layers.Dropout(0.25)(x)
            
            x = layers.GlobalAveragePooling1D()(x)
            
            x = layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4)
            )(x)
            x = layers.Dropout(0.30)(x)
            
            x = layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4)
            )(x)
            x = layers.Dropout(0.20)(x)
            
            outputs = layers.Dense(
                1,
                activation="sigmoid"
            )(x)
            
            return models.Model(
                inputs=inputs,
                outputs=outputs,
                name="CNN3_Deep_1D"
            )
        model = building_CNN3(self)
        if(self.cnn_params.get("loss","focal") == "focal"):
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss=self.focal_loss(),
                metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc')]
                    )
        elif(self.cnn_params.get("loss","focal") == "bce"):
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss="binary_crossentropy",
                metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc')]
                    )
        else:
                raise ValueError("Use 'focal' or 'bce'")

        self.model = model
    
    def build_CNN4(self):

        def building_CNN4(self):
            inputs = layers.Input(shape=(int(self.cnn_params.get("reshape",10)) , int(self.n_bins/int(self.cnn_params.get("reshape",10))), 1))
            
            # Convolutional block 1
            x = layers.Conv2D(
                64,
                kernel_size=(3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(1e-4)
            )(inputs)
         
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
         
            # Convolutional block 2
            x = layers.Conv2D(
                128,
                kernel_size=(3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(1e-4)
            )(x)
         
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
         
            x = layers.Dropout(0.25)(x)
         
            # Classification head
            x = layers.GlobalAveragePooling2D()(x)
         
            x = layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4)
            )(x)
            x = layers.Dropout(0.30)(x)
         
            x = layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4)
            )(x)
            x = layers.Dropout(0.20)(x)
         
            outputs = layers.Dense(
                1,
                activation="sigmoid"
            )(x)
         
            return models.Model(
                 inputs=inputs,
                 outputs=outputs,
                 name="CNN4_Compact_2D"
            )
        
        model = building_CNN4(self)
        if(self.cnn_params.get("loss","focal") == "focal"):
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss=self.focal_loss(),
                metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc')]
                    )
        elif(self.cnn_params.get("loss","focal") == "bce"):
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss="binary_crossentropy",
                metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc')]
                    )
        else:
                raise ValueError("Use 'focal' or 'bce'")

        self.model = model        

    #let's now train our model
    def train_model(self):
        """Train the model with AUC-centric callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=20,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(
                'best_model_final.keras',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.cnn_params.get("epochs",200),
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

    def _plot_confusion_matrix(self, ax, y_true, y_pred, threshold):
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2),
                yticks=np.arange(2),
                xticklabels=['Non-Planet', 'Planet'],
                yticklabels=['Non-Planet', 'Planet'],
                xlabel='Predicted', ylabel='True',
                title=f'Confusion Matrix (threshold={threshold:.3f})')
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center')
        
            
    def _plot_roc_curve(self, ax, y_true, proba):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, linewidth=2)
        ax.plot([0,1], [0,1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve (AUC={auc:.4f})')
        ax.grid(alpha=0.3)

    def _plot_training_history(self, ax_left, ax_right):

        h = (
            self.history.history
            if hasattr(self.history, "history")
            else self.history
        )

        for ax, metric in zip((ax_left, ax_right), ["loss", "auc"]):
            ax.plot(h[metric], label="Training")
            ax.plot(h[f"val_{metric}"], label="Validation")
            ax.set_title(metric.upper())
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.3)

    def plot_all_metrics(self, y_true, y_pred, proba, threshold, save_path='evaluation_grid.png'):
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()

        self._plot_confusion_matrix(ax1, y_true, y_pred, threshold)
        self._plot_roc_curve(ax2, y_true, proba)

        # One method draws both bottom plots using ax3 and ax4
        self._plot_training_history(ax3, ax4)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved combined plot: {save_path}")
        plt.close(fig)


    def evaluate_with_optimal_threshold(self,show,name):
        
        y_pred_proba = self.model.predict(self.X_test, verbose=0).flatten()
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        self.best_threshold = thresholds[optimal_idx]
        
        y_pred_optimal = (y_pred_proba >= self.best_threshold).astype(int)
        
        # Metrics
        auc = roc_auc_score(self.y_test, y_pred_proba)
        acc_optimal = accuracy_score(self.y_test, y_pred_optimal)
        cm = confusion_matrix(self.y_test, y_pred_optimal)
        prec_planet = precision_score(self.y_test,y_pred_optimal, pos_label=1, zero_division=0)

        ##############OUPUT e RUN
            
        print("\n" + "=" * 60)
        print("CNN MODEL EVALUATION STATISTICS")
        print("=" * 60)
        print(f"Optimal Threshold: {self.best_threshold:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Accuracy (at optimal threshold): {acc_optimal:.4f}")
        print(f"Precision (Planet): {prec_planet:.4f}")
        print("\nConfusion Matrix:")
        print(f"  TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
        print(f"  FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")
        print("\nClassification report:")
        print(classification_report(self.y_test, y_pred_optimal,
                                                target_names=['Non-Planet', 'Planet'],
                                                digits=4,
                                                zero_division=0))
        print("=" * 60 + "\n")
        print("\nPrediction distribution (optimal threshold):")
        print(f"  Predicted 0: {(y_pred_optimal == 0).sum()}")
        print(f"  Predicted 1: {(y_pred_optimal == 1).sum()}")
        print("True distribution:")
        print(f"  True 0: {(self.y_test == 0).sum()}")
        print(f"  True 1: {(self.y_test == 1).sum()}")
        if(show):
            self.plot_all_metrics(self.y_test, y_pred_optimal, y_pred_proba, self.best_threshold, save_path=f'CNN_evaluation_grid_{name}.png' )
  
    def run(self,show,name):
        print(f"Loading CNN data for {name}")
        self.load_data(False)
        # Puliamo lo stato dei modelli precedenti
        tf.keras.backend.clear_session()
        # Fissiamo il seed per l'inizializzazione dei pesi
        tf.keras.utils.set_random_seed(self.random_state)
        print(f"Training CNN model for {name}")
        if (self.cnn_params.get("cnn_type", "cnn1") == "cnn1"):
            self.build_CNN1()
        elif(self.cnn_params.get("cnn_type", "cnn1") == "cnn2"):
            self.build_CNN2()
        elif(self.cnn_params.get("cnn_type", "cnn1") == "cnn3"):
            self.build_CNN3()
        else:
            self.build_CNN4()
        self.train_model()
        print(f"Evaluating CNN model for {name}") 
        self.evaluate_with_optimal_threshold(show,name)

    def build_CNN1_for_hyperparameters(self, loss="focal", lr=0.0005, dropout_dense=0.2, l2_reg=0.001, activationF = "relu"):

        model = models.Sequential([
            layers.Input(shape=(self.n_bins, 1)),

            # Feature extraction
            layers.Conv1D(64, kernel_size=3, padding="same", activation = activationF ),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            layers.Conv1D(128, kernel_size=3, padding="same", activation = activationF),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            layers.Conv1D(256, kernel_size=3, padding="same", activation = activationF),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),

            # Classification head
            layers.Dense(512, activation = activationF, kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(dropout_dense),
            layers.Dense(256, activation = activationF, kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(dropout_dense),
            layers.Dense(128, activation = activationF, kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(dropout_dense),

            layers.Dense(1, activation="sigmoid"),
        ])

        if loss == "focal":
            loss_fn = self.focal_loss()
        elif loss == "bce":
            loss_fn = "binary_crossentropy"

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss=loss_fn,
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )

        return model



    def evaluate_best_hyperparameters(self, hyper = True):

        self.load_data(hyper)
        keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.random_state)

        def create_keras_model(loss="focal", lr=0.0005, dropout_dense=0.2, l2_reg=0.001, activationF = "relu"):
            keras.backend.clear_session()
            return self.build_CNN1_for_hyperparameters(loss, lr, dropout_dense, l2_reg, activationF)

        callbacks = [
            EarlyStopping(
                monitor="val_auc",
                patience=8,
                restore_best_weights=True,
                mode="max",
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_auc",
                factor=0.5,
                patience=6,
                min_lr=1e-7,
                mode="max",
                verbose=0,
            ),
            ModelCheckpoint(
                'best_model_final.keras',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]

        clf = KerasClassifier(
            model=create_keras_model,
            loss="focal",
            lr=0.0005,
            dropout_dense=0.2,
            activationF = "relu",
            l2_reg=0.001,
            epochs=self.cnn_params.get("epochs",100),
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0,
        )   

        param_grid = {
            "loss": ["focal", "bce"],
            "lr": [1e-3,5e-4],
            #"dropout_dense": [0.2, 0.4],
            #"l2_reg": [0.001, 0.005],
            #"activationF": ["relu","gelu"]
        }

        grid = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=3,
            scoring="recall",
            n_jobs=1,
            refit=True,
        )

        grid.fit(self.X_train, self.y_train)

        print("Best CNN hyperparameters:", grid.best_params_)
       