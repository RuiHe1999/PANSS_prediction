# 1. packages
import os
import re
import joblib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from collections import Counter

import xgboost as xgb
from sklearn import ensemble, svm, neighbors, linear_model, gaussian_process
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, WhiteKernel, ConstantKernel

import matplotlib.pyplot as plt

# 2. constants
random_state = 42
np.random.seed(random_state)

os.makedirs('results/model_selection/', exist_ok=True)
os.makedirs('results/models/', exist_ok=True)

# 3. functions
def split_data(data):
    
    global random_state
    
    # over two items greater than 3 ?
    symptoms = ['P1', 'P2', 'P3', 'N1', 'N4', 'N6', 'G5', 'G9']
    panss_cols = [f'PANSS_{s}' for s in symptoms if f'PANSS_{s}' in data.columns]
    grp = data.groupby(['language_code', 'PAR'])[panss_cols].max(min_count=1)
    grp_thred = (grp.gt(3).sum(axis=1) >= 2).to_dict()

    def can_stratify(labels, test_size):
        """Check if each class has enough members to split into train and test (>=1 each)."""
        c = Counter(labels)
        if len(c) < 2:
            return False
        for cnt in c.values():
            n_test = np.ceil(cnt * test_size)
            n_train = cnt - n_test
            if n_test < 1 or n_train < 1:
                return False
        return True

    def safe_split(arr, labels, test_size, rs):
        """Stratify when feasible, otherwise fall back to non-stratified split."""
        strat = labels if can_stratify(labels, test_size) else None
        return train_test_split(arr, test_size=test_size, random_state=rs, stratify=strat)
            
        
    # split by participant
    train_pars, test_pars, val_pars = [], [], []

    for lang in data.language_code.unique():
        data_lang = data[data['language_code'] == lang]
        unique_par = data_lang['PAR'].unique()

        labels = [grp_thred.get((lang, p), False) for p in unique_par]  # default False is conservative
        cnt = Counter(labels)

        # If any class has only 1 participant, force that participant into TRAIN to avoid split errors
        keep_mask = [True] * len(unique_par)
        for cls_val, cls_cnt in cnt.items():
            if cls_cnt == 1:
                idx = labels.index(cls_val)
                train_pars.append(unique_par[idx])
                keep_mask[idx] = False

        # Remaining pool after forcing rare singletons into train
        pool_idx = [i for i, k in enumerate(keep_mask) if k]
        if not pool_idx:
            continue

        pool = unique_par[pool_idx]
        pool_labels = [grp_thred.get((lang, p), False) for p in pool]

        # 80/20 -> train vs temp
        train_add, temp_par = safe_split(pool, pool_labels, test_size=0.2, rs=random_state)
        train_pars.extend(train_add.tolist())

        # temp 50/50 -> val vs test
        if len(temp_par) == 1:
            val_par = temp_par
            test_par = np.array([''])
        else:
            temp_labels = [grp_thred.get((lang, p), False) for p in temp_par]
            val_par, test_par = safe_split(temp_par, temp_labels, test_size=0.5, rs=random_state)

        val_pars.extend(val_par.tolist())
        test_pars.extend(test_par.tolist())

    # label the dataset
    def assign_set(par_value):
        if par_value in train_pars:
            return 'train'
        elif par_value in test_pars:
            return 'test'
        elif par_value in val_pars:
            return 'val'
        return None 

    data['set'] = data['PAR'].apply(assign_set)

    # summarize the splitting result 
    language_stats = {}
    columns = [
        'train_par_num', 'test_par_num', 'val_par_num', 
        'train_row_num', 'test_row_num', 'val_row_num', 
        'total_par_num', 'total_row_num'
    ]

    for subset in ['train', 'test', 'val']:
        subset_data = data[data['set'] == subset]
        par_counts = subset_data.groupby('language_code')['PAR'].nunique() 
        row_counts = subset_data.groupby('language_code').size()  

        for lang in par_counts.index:
            if lang not in language_stats:
                language_stats[lang] = {col: 0 for col in columns} 
            language_stats[lang][f'{subset}_par_num'] = par_counts[lang]
            language_stats[lang][f'{subset}_row_num'] = row_counts[lang]

    total_par_counts = data.groupby('language_code')['PAR'].nunique()
    total_row_counts = data.groupby('language_code').size()
    for lang in total_par_counts.index:
        language_stats[lang]['total_par_num'] = total_par_counts[lang]
        language_stats[lang]['total_row_num'] = total_row_counts[lang]

    for lang, stats_ in language_stats.items():
        for subset in ['train', 'test', 'val']:
            stats_[f'{subset}_par_ratio'] = stats_[f'{subset}_par_num'] / stats_['total_par_num']
            
    stats_df = pd.DataFrame.from_dict(language_stats, orient='index').reset_index()
    stats_df.rename(columns={'index': 'language_code'}, inplace=True)
    
    return data, stats_df

def parallel_analysis(df, note, num_iterations=100, plot=False):
    
    global random_state
    np.random.seed(random_state)

    data = df.copy()
    
    # normalized_data = (data - data.mean()) / data.std() # for manual normalization
    # Use StandardScaler for normalization 
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    # Perform PCA on the actual data
    pca = PCA(n_components=normalized_data.shape[1] - 1)
    pca.fit(normalized_data)
    
    # Compute random eigenvalues through parallel analysis
    random_eigenvalues = np.zeros(normalized_data.shape[1] - 1)
    for i in trange(num_iterations, desc=f'PCA on {note}'):
        random_data = pd.DataFrame(np.random.normal(0, 1, [normalized_data.shape[0], normalized_data.shape[1]]))
        pca_random = PCA(n_components=normalized_data.shape[1] - 1)
        pca_random.fit(random_data)
        random_eigenvalues += pca_random.explained_variance_ratio_
    
    random_eigenvalues /= num_iterations  # Average over iterations
    
    # Decide how many components to retain
    n_components = np.sum(pca.explained_variance_ratio_ > random_eigenvalues)
    
    if plot:
        # visualize
        fig, ax = plt.subplots(figsize=(20, 20))

        ax.plot(pca.explained_variance_ratio_, '-o', color='#FFB996', label='Features')
        ax.plot(random_eigenvalues, '-x', color='#9DBC98', label='Simulated data from parallel analysis')
        ax.legend()
        ax.set_title(f'Scree plot with parallel analysis - {note}: {n_components} components')
        fig.savefig(f'results/model_selection/PCA_parallel_analysis_scree_plot_{note}.png', dpi=300, bbox_inches='tight') 
    
    return n_components, pca

def get_n_components(df, feat, note, num_iterations=100, plot=False):
    
    data = df.copy()
    
    X_train = data[data['set'] == 'train'][feat]
    n_components, pca = parallel_analysis(X_train, 
                                          note=note, 
                                          num_iterations=num_iterations, 
                                          plot=plot)
    
    return n_components, pca 

def param_to_filename(param):
    parts = []
    for k, v in param.items():

        val_str = str(v)
        val_str = re.sub(r'[^\w.-]', '_', val_str)
        parts.append(f"{k}-{val_str}")
    return "_".join(parts)

def regression_eval(model, feat_name, symp):
    
    global data, dict_feat, dict_component

    # dataframe
    df = data[data['PANSS_G5'].notna()] if 'P' not in symp else data

    # retrieve model and parameters
    reg, param_grid = dict_models[model]

    # train, val, test df
    train_df = df[df['set'] == 'train'].copy().reset_index(drop=True)
    val_df = df[df['set'] == 'val'].copy().reset_index(drop=True)
    test_df = df[df['set'] == 'test'].copy().reset_index(drop=True)

    # features & labels
    X_train = train_df[dict_feat[feat_name]]
    y_train = train_df[f'PANSS_{symp}']

    X_val = val_df[dict_feat[feat_name]]
    y_val = val_df[f'PANSS_{symp}']

    X_test = test_df[dict_feat[feat_name]]
    y_test = test_df[f'PANSS_{symp}']

    # Create PCA and regressor pipeline
    # parallel analysis for number of components
    num_components, _ = get_n_components(train_df, dict_feat[feat_name], feat_name, plot=False)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),                # Scaling
        ('pca', PCA(n_components=num_components)),   # PCA for dimensionality reduction
        ('regressor', reg)                           # Regressor
    ])  

    # parameter grid search
    params = list(ParameterGrid({f'regressor__{k}': v for k, v in param_grid.items()}))

    # iterate over all parameter combinations
    results = []
    
    for param in tqdm(params, desc='Grid search: '):
        # set model params
        pipeline.set_params(**param)
        
        # fit on train
        pipeline.fit(X_train, y_train)
        
        # predict on train/val/test
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        
        # calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results.append({
            'symptom': symp,
            'feature': feat_name,
            **param,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
        })
        
        # save the pipeline
        safe_parts = []
        for k, v in param.items():
            v_str = str(v)
            v_safe = re.sub(r'[^\w\-]+', '_', v_str) 
            safe_parts.append(f"{k}-{v_safe}")

        safe_param = "_".join(safe_parts)
        save_path = f"results/models/model-{model}_item-{symp}_feat-{feat_name}_param-{safe_param}.joblib"
        joblib.dump(pipeline, save_path)

    # save results
    results_df = pd.DataFrame(results)
    
    return results_df

# 4. commands
# 4.1 read and split data
data = pd.read_csv('panss_10lang.csv')
data, split_summary = split_data(data)
split_summary.to_csv('results/split_summary.csv', index=False)

# 4.2 regression models
dict_models = {}

# OLS linear regression 
dict_models['OLS'] = [
    linear_model.LinearRegression(), 
    {}
]

# Ridge Regression 
dict_models['Ridge'] = [
    linear_model.Ridge(),
    {
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
    }
]

# Bayesian Ridge Regression 
dict_models['BayesRidge'] = [
    linear_model.BayesianRidge(),
    {
        'alpha_init': [1e-4, 1e-2, 1, 10],
       'lambda_init': [1e-4, 1e-2, 1, 10]
    }
]

# Lasso Regression 
dict_models['Lasso'] = [
    linear_model.Lasso(max_iter=5000),
    {
        'alpha': [0.01, 0.1, 1, 10],
    }
]

dict_models['Elastic'] = [
    linear_model.ElasticNet(max_iter=5000, random_state=42),
    {
        'alpha': [0.01, 0.1, 1, 10],      
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  
    }
]

# Huber regression 
dict_models['Huber'] = [linear_model.HuberRegressor(max_iter=5000), {
        'alpha': [1e-4, 1e-3, 1e-2],
        'epsilon': [1.0, 1.35, 2.0]
    }]

# Support Vector Regression
dict_models['SVR'] = [
    svm.SVR(),
    {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2],
    }
]

# randon forest 
dict_models['RF'] = [
    ensemble.RandomForestRegressor(random_state=42),
    {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, 30],
        'max_features': [0.3, 0.5, 'sqrt', 'log2'],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [1, 4, 8],
    }
]

# extra trees 
dict_models['ET'] = [ensemble.ExtraTreesRegressor(random_state=42),
    {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, 30],
        'max_features': [0.3, 0.5, 'sqrt', 'log2'],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [1, 4, 8],
    }
]

# Gradient boosting 
dict_models['GBR'] = [
    ensemble.GradientBoostingRegressor(random_state=42),
    {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, 30],
        'max_features': [0.3, 0.5, 'sqrt', 'log2'],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [1, 4, 8],
        'learning_rate': [0.05, 0.1],
    }
]

# Adaboost
dict_models['ABR'] = [
    ensemble.AdaBoostRegressor(random_state=42),
    {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1],

    }
]

# XGBoost
dict_models['XGB'] = [
    xgb.XGBRegressor(random_state=42),
    {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 10, 20, 30],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
    }
]

# K-Nearest Neighbors 
dict_models['KNN'] = [
    neighbors.KNeighborsRegressor(),
    {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1=manhattan, 2=euclidean
    }
]

# Gaussian process
kernel_options = [
    ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed"), 
    ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed") + WhiteKernel(),
    DotProduct() + WhiteKernel(), 
    Matern(length_scale=1.0, nu=1.5) + WhiteKernel()
]

dict_models['GPR'] = [
    gaussian_process.GaussianProcessRegressor(random_state=42),
    {
        'kernel': kernel_options,
        'alpha': [1e-10, 1e-5, 1e-3, 1e-2],
    }
]

# 4.3 features with PCA components
dict_feat = dict(zip(
    ['AcouPros', 'm-HuBERT', 'Concat'], 
    [data.columns[1:120], data.columns[120:888], data.columns[1:888]]
    ))

# 4.4 predictions on eight items
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model evaluation and store results")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g., Ridge")
    parser.add_argument("--feature", type=str, required=True, help="Feature name, e.g., AcouPros")
    parser.add_argument("--symptom", type=str, required=True, help="Symptom code, e.g., G9")
    args = parser.parse_args()
    
    # check the inputs
    valid_models = dict_models.keys()
    valid_features = dict_feat.keys()
    valid_symptoms = ['P1', 'P2', 'P3', 'N1', 'N4', 'N6', 'G5', 'G9']
    
    if args.model not in valid_models:
        raise ValueError(f"Invalid model '{args.model}'. Must be one of {valid_models}")
    if args.feature not in valid_features:
        raise ValueError(f"Invalid feature '{args.feature}'. Must be one of {valid_features}")
    if args.symptom not in valid_symptoms:
        raise ValueError(f"Invalid symptom '{args.symptom}'. Must be one of {valid_symptoms}")

    # run evaluations
    results_df = regression_eval(args.model, args.feature, args.symptom)
    results_df.insert(0, 'model', args.model)

    out_path = f"results/model_selection/model-{args.model}_feature-{args.feature}_item-{args.symptom}.csv"
    results_df.to_csv(out_path, index=False)
















    





