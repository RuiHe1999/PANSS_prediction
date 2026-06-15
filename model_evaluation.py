# 1. packages
import os
import torch
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, learning_curve

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# 2. constants
random_state = 42
np.random.seed(random_state)
langs = ['cs', 'de', 'en', 'es', 'escl', 'fr', 'gsw', 'nl', 'tr', 'zh']
symptoms = ['P1', 'P2', 'P3', 'N1', 'N4', 'N6', 'G5', 'G9']
task_name = {
    # FreeSpeech
    ('FS', 'Conversation'): 'FreeSpeech',
    ('FS', 'conversation'): 'FreeSpeech',
    ('FS', 'clinic'): 'FreeSpeech',
    ('FS', 'past'): 'FreeSpeech',
    ('FS', 'self'): 'FreeSpeech',
    ('JOU', 'AboutYourself'): 'FreeSpeech',

    # Dream
    ('Dream', 'dream'): 'Dream',
    ('JOU', 'Dream'): 'Dream',

    # Reading
    ('PAR', 'CrowRead'): 'Reading',
    ('PAR', 'Paragraph'): 'Reading',
    ('Story', 'read'): 'Reading',

    # Recall
    ('ATT', 'ATT'): 'Recall',
    ('Story', 'recall'): 'Recall',
    ('Story', 'retell'): 'Recall',
    ('REC', 'CrowRecall'): 'Recall',

    # Picture
    ('PD', '1'): 'Picture',
    ('PD', '12M'): 'Picture',
    ('PD', '13'): 'Picture',
    ('PD', '13B'): 'Picture',
    ('PD', '13MF'): 'Picture',
    ('PD', '15'): 'Picture',
    ('PD', '18GF'): 'Picture',
    ('PD', '19'): 'Picture',
    ('PD', '2'): 'Picture',
    ('PD', '3'): 'Picture',
    ('PD', '3BM'): 'Picture',
    ('PD', '4'): 'Picture',
    ('PD', '5'): 'Picture',
    ('PD', '6GF'): 'Picture',
    ('PD', '7GF'): 'Picture',
    ('PD', '8BM'): 'Picture',
    ('PD', '8BN'): 'Picture',
    ('PD', '9BM'): 'Picture',
    ('PD', '9GF'): 'Picture',
    ('PD', 'bridge'): 'Picture',
    ('PD', 'cartoon'): 'Picture',
    ('PD', 'couple'): 'Picture',
    ('PD', 'farm'): 'Picture',
    ('PIC', 'Bridge'): 'Picture',
    ('PIC', 'Country'): 'Picture',
    ('PIC', 'Couple'): 'Picture',
    ('PIC', 'TATPicture'): 'Picture',
    ('STO', 'StoryBoard'): 'Picture',
}

os.makedirs('results/model_eval/learning_curve/', exist_ok=True)
os.makedirs('results/model_eval/test_predicts/', exist_ok=True)
os.makedirs('results/model_eval/par_predicts/', exist_ok=True)
os.makedirs('results/model_eval/task_predicts/', exist_ok=True)

# 3. functions
def safe_rmse(gold, pred):
    gold = np.array(gold)
    pred = np.array(pred)
    
    mask = ~np.isnan(gold) & ~np.isnan(pred)
    
    if not np.any(mask):
        return np.nan
    
    return np.sqrt(mean_squared_error(gold[mask], pred[mask]))

def safe_mae(gold, pred):
    gold = np.array(gold)
    pred = np.array(pred)

    mask = ~np.isnan(gold) & ~np.isnan(pred)

    if not np.any(mask):
        return np.nan

    return mean_absolute_error(gold[mask], pred[mask])

def split_data(data):
    
    global random_state
    
    # over two items greater than 3 ?
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

def split_grid_kfold(df, n_splits=5):
    
    global random_state
    
    rng = np.random.default_rng(random_state)
    splits = []

    lang_par_splits = {}
    for lang in df['language_code'].unique():
        df_lang = df[df['language_code'] == lang]
        unique_pars = np.array(df_lang['PAR'].unique())
        rng.shuffle(unique_pars)  

        kf = KFold(n_splits=n_splits, shuffle=False)
        lang_par_splits[lang] = [unique_pars[idx] for _, idx in kf.split(unique_pars)]

    for fold in range(n_splits):
        val_pars = []
        for lang in lang_par_splits:
            val_pars.extend(lang_par_splits[lang][fold])  

        train_idx = df[~df['PAR'].isin(val_pars)].index.to_numpy()
        val_idx   = df[df['PAR'].isin(val_pars)].index.to_numpy()
        splits.append((train_idx, val_idx))

    return splits
    
# 4. commands
# 4.1 read and split data
data = pd.read_csv('panss_10lang.csv')
data, split_summary = split_data(data)
data_g =  data[data['PANSS_G5'].notna()]
split_summary.to_csv('results/split_summary.csv', index=False)

# 4.2 features with PCA components
dict_feat = dict(zip(
    ['AcouPros', 'm-HuBERT', 'Concat'], 
    [data.columns[1:120], data.columns[120:888], data.columns[1:888]]
    ))  

# 4.3 test results and learning curves
for symp in tqdm(symptoms):
    
    # dataframe
    df = data[data['PANSS_G5'].notna()] if 'P' not in symp else data   

    # best model 
    best_models = pd.read_csv('results/model_comparison/best_model.csv')

    # training results
    model_name, feat_name, _, train_rmse, val_rmse, test_rmse, param, _, _ = best_models[best_models['symptom']==symp].values[0]
    param = eval(param)

    # train and test df
    train_df = df[df['set'] == 'train'].copy().reset_index(drop=True)
    test_df = df[df['set'] == 'test'].copy().reset_index(drop=True)

    X_train = train_df[dict_feat[feat_name]]
    y_train = train_df[f'PANSS_{symp}']
    X_test = test_df[dict_feat[feat_name]]

    # load model
    if 'MLP' not in model_name:
        # load joblib model
        model = joblib.load(f'results/model_sel/{symp}.joblib')
        
        # prediction
        test_pred = model.predict(X_test)
        
        # dataframe for test data 
        test_score = df[df['set']=='test'][
            ['PAR', 'task', 'subtask', 'seg', 'language_code', 'Age', 'Sex', 'Edu', f'PANSS_{symp}']
            ]

        test_score['TaskName'] = list(map(task_name.get, zip(test_score['task'], test_score['subtask'])))
        
        # add infromation 
        test_score['feat'] = feat_name
        test_score[f'Pred_{symp}'] = test_pred
        test_score['best_param'] = param
        
        # save the test dataframe
        test_score.to_csv(f'results/model_eval/test_predicts/{symp}.csv', index=False)
        
        # learning curve
        splits = split_grid_kfold(train_df, n_splits=5)

        # train and test learning curve 
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train,
            cv=splits,
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='neg_root_mean_squared_error',  
            n_jobs=-1
        )

        train_scores = -train_scores
        test_scores = -test_scores

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # train RMSE
        sns.lineplot(
            x=train_sizes, y=train_mean,
            label="Train RMSE", color="blue", ax=ax
        )
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")

        # test RMSE
        sns.lineplot(
            x=train_sizes, y=test_mean,
            label="Test RMSE", color="orange", ax=ax
        )
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color="orange")

        ax.set_xlabel("Training examples", fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(f"Learning Curve for {model_name} using {feat_name} to predict {symp}", fontsize=14)
        ax.legend()
        ax.set_ylim(0.5, 2)

        # save
        fig.savefig(f'results/model_eval/learning_curve/{symp}.svg', dpi=300, bbox_inches='tight')
        
        
    else:
        # load PyTorch .pth model
        model = torch.load(f'results/model_sel/{symp}.pth')
        
        # test data 
        test_pred = model['test_pred']
        
        # dataframe for test data 
        test_score = df[df['set']=='test'][
            ['PAR', 'task', 'subtask', 'seg', 'language_code', 'Age', 'Sex', 'Edu', f'PANSS_{symp}']
            ]

        test_score['TaskName'] = list(map(task_name.get, zip(test_score['task'], test_score['subtask'])))
        
        # add infromation 
        test_score['feat'] = feat_name
        test_score[f'Pred_{symp}'] = test_pred
        test_score['best_param'] = param
        
        # save the test dataframe
        test_score.to_csv(f'results/model_eval/test_predicts/{symp}.csv', index=False)
        
        # learning curve 
        train_loss = model['train_loss']
        val_loss = model['val_loss']
        
        epochs = range(1, len(train_loss) + 1)
        
        # plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # train loss
        sns.lineplot(x=epochs, y=train_loss, label="Train Loss", color="blue", ax=ax)

        # val loss 
        sns.lineplot(x=epochs, y=val_loss, label="Validation Loss", color="orange", ax=ax)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Learning Curve for {model_name} using {feat_name} to predict {symp}", fontsize=14)
        ax.legend()
        
        # save
        fig.savefig(f'results/model_eval/learning_curve/{symp}.svg', dpi=300, bbox_inches='tight')

# 4.4 RMSE and R2 at participant and segment level per language
infos = []

for symp in symptoms:
    
    # 1. read the files
    test_scores = pd.read_csv(f'results/model_eval/test_predicts/{symp}.csv')

    # 2. aggregate by subtasks and participants
    task_scores = test_scores.groupby(['PAR', 'feat', 'subtask', 'task', 'TaskName'], as_index=False).agg(
        {'language_code': 'first', 
          'Age': 'first',      
          'Sex': 'first',      
          'Edu': 'first', 
          f'PANSS_{symp}': 'median',
          f'Pred_{symp}': 'median',
        }
    )
    
    task_scores.to_csv(f'results/model_eval/task_predicts/{symp}.csv', index=False)

    par_scores = task_scores.groupby(['PAR', 'feat'], as_index=False).agg(
        {'language_code': 'first', 
          'Age': 'first',      
          'Sex': 'first',      
          'Edu': 'first', 
          f'PANSS_{symp}': 'median',
          f'Pred_{symp}': 'median',
        }
    )
    
    par_scores.to_csv(f'results/model_eval/par_predicts/{symp}.csv', index=False)

    # 3. model evaluation 
    info = {}
    info['symptom'] = symp

    # evaluate the model at the segment level 
    y_true = test_scores[f'PANSS_{symp}']
    y_pred = test_scores[f'Pred_{symp}']

    info['test_rmse_seg'] = safe_rmse(y_true, y_pred)
    info['test_mae_seg'] = safe_mae(y_true, y_pred)
    info['test_r2_seg'] = r2_score(y_true, y_pred)

    # evaluate the model at the participant level 
    y_true = par_scores[f'PANSS_{symp}']
    y_pred = par_scores[f'Pred_{symp}']

    info['test_rmse_par'] = safe_rmse(y_true, y_pred)
    info['test_mae_par'] = safe_mae(y_true, y_pred)
    info['test_r2_par'] = r2_score(y_true, y_pred)

    # 4. model evaluation per language
    for lang in langs:
        seg_true_lang = test_scores[test_scores['language_code']==lang][f'PANSS_{symp}']
        seg_pred_lang = test_scores[test_scores['language_code']==lang][f'Pred_{symp}']
        if y_true.empty or y_pred.empty:
            info[f'{lang}_rmse_seg'] = np.nan
            info[f'{lang}_mae_seg'] = np.nan
        else:
            rmse = safe_rmse(seg_true_lang, seg_pred_lang)
            info[f'{lang}_rmse_seg'] = rmse
            
            mae = safe_mae(seg_true_lang, seg_pred_lang)
            info[f'{lang}_mae_seg'] = mae
            
    for lang in langs:
        par_true_lang = par_scores[par_scores['language_code']==lang][f'PANSS_{symp}']
        par_pred_lang = par_scores[par_scores['language_code']==lang][f'Pred_{symp}']
        if y_true.empty or y_pred.empty:
            info[f'{lang}_rmse_par'] = np.nan
            info[f'{lang}_mae_par'] = np.nan
        else:
            rmse = safe_rmse(par_true_lang, par_pred_lang)
            info[f'{lang}_rmse_par'] = rmse
            mae = safe_mae(seg_true_lang, seg_pred_lang)
            info[f'{lang}_mae_par'] = mae
    
    infos.append(info)

infos_df = pd.DataFrame(infos)
infos_df.to_csv('results/model_comparison/best_model_performance.csv', index=False)















































