# 1. packages
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR

# 2. consonants
random_state = 42
torch.manual_seed(random_state)
np.random.seed(random_state)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. functions
def split_data(data):
    
    global random_state
    
    # over two items greater than 3 ?
    symptoms = ['P1', 'P2', 'P3', 'N1', 'N4', 'N6', 'G5', 'G9']
    panss_cols = [f'PANSS_{s}' for s in symptoms if f'PANSS_{s}' in data.columns]
    grp = data.groupby(['language_code', 'PAR'])[panss_cols].max(min_count=1)
    grp_thred = (grp.gt(3).sum(axis=1) >= 2).to_dict()

    def can_stratify(labels, test_size):
        '''Check if each class has enough members to split into train and test (>=1 each).'''
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
        '''Stratify when feasible, otherwise fall back to non-stratified split.'''
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

class Dataset(torch.utils.data.Dataset):
  def __init__(self, X, y):
    self.X = torch.from_numpy(X)
    self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i].to(device), self.y[i].to(device)

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps
    
    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y)+self.eps)

class MLP(nn.Module):
    def __init__(self, input_size, dim, dropout_rate=0.5):
        super().__init__()
        # Define hidden layers with configurable dimensions
        self.layer_1 = nn.Linear(in_features=input_size, out_features=dim)
        self.layer_2 = nn.Linear(in_features=dim, out_features=1)
        
        # Activation function
        self.activate = nn.ReLU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(dim)
    
    def forward(self, x):
        # Forward pass through the network
        x = self.activate(self.layer_1(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = self.layer_2(x)  # No activation for the output layer
        return x

    @staticmethod
    def _count_parameters(model):
        '''Count the number of trainable parameters in the model.'''
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def train_model(model, train_loader, val_loader, criterion, optimizer, item, 
                num_epochs=20, patience=4, min_delta=0.01, verbose=True):
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
    
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
    
    # Create random seed
    torch.manual_seed(random_state)
    # Create an empty list to store metrics for each epoch
    metrics = []

    # Variables for early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Initialize the scheduler
    scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=1000)
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)

        # Calculate average training loss for the epoch
        train_loss = running_train_loss / len(train_loader.dataset)

        # Evaluate on validation set
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs.float())
                val_loss = criterion(val_outputs, val_targets.float())
                running_val_loss += val_loss.item() * val_inputs.size(0)

        # Calculate average validation loss for the epoch
        val_loss = running_val_loss / len(val_loader.dataset)
        
        # Append epoch, training feat_nameloss, and validation loss to metrics list
        metrics.append({'Epoch': epoch+1,
                        'Train Loss': train_loss,
                        'Validation Loss': val_loss})

        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')
        
        # Update the scheduler based on validation loss
        scheduler.step()
        
        # Check if validation loss improved
        if val_loss < best_val_loss - min_delta:  # Only consider improvements larger than min_delta
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset the counter

        else:
            epochs_without_improvement += 1
            # If no improvement for 'patience' epochs, stop training
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}.')
                break

    # Convert metrics list to DataFrame
    df_metrics = pd.DataFrame(metrics)

    return df_metrics

def evaluate_model(model, data_loader, criterion):
    
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
    
    predictions = []
    actuals = []
    running_loss = 0.0

    model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.float())
            loss = criterion(outputs, targets.float())
            running_loss += loss.item() * inputs.size(0)
            
            if outputs.squeeze().size():
                predictions.extend(outputs.squeeze().tolist())
            else:
                predictions.append(outputs.squeeze().item()) 
                
            if targets.squeeze().size():
                actuals.extend(targets.squeeze().tolist())
            else:
                actuals.append(targets.squeeze().item())
                
    avg_loss = running_loss / len(data_loader.dataset)

    return actuals, predictions, avg_loss

def run_MLP(data, feat_name, sypt_name, dim, dropout_rate, lr, decay, verbose=True):
    
    print(f'>>> Using {feat_name} for {sypt_name}')
    print(f'Dim: {dim}, Dropout: {dropout_rate}, Initial learning rate: {lr}, L2 penalty: {decay}')
    
    # test results
    train_score = data[data['set'] == 'train']
    test_score = data[data['set'] == 'test']
    val_score = data[data['set'] == 'val']
    
    train_score = train_score[['PAR', 'task', 'subtask', 'seg', 'language_code', 
                               'Age', 'Sex', 'Edu', 'set', f'PANSS_{sypt_name}']]
    test_score = test_score[['PAR', 'task', 'subtask', 'seg', 'language_code', 
                             'Age', 'Sex', 'Edu', 'set', f'PANSS_{sypt_name}']]
    val_score = val_score[['PAR', 'task', 'subtask', 'seg', 'language_code', 
                           'Age', 'Sex', 'Edu', 'set', f'PANSS_{sypt_name}']]
    
    # feature dictionaries
    dict_feat = dict(zip(
        ['AcouPros', 'm-HuBERT', 'Concat'], 
        [data.columns[1:120], data.columns[120:888], data.columns[1:888]]
        ))
    
    # train data
    X_train = data[data['set'] == 'train'][dict_feat[feat_name]]
    y_train = data[data['set'] == 'train'][f'PANSS_{sypt_name}']
    X_test = data[data['set'] == 'test'][dict_feat[feat_name]]
    y_test = data[data['set'] == 'test'][f'PANSS_{sypt_name}']
    X_val = data[data['set'] == 'val'][dict_feat[feat_name]]
    y_val = data[data['set'] == 'val'][f'PANSS_{sypt_name}']

    # Prepare training, test and validation data
    train_dataset = Dataset(X=X_train.values, y=y_train.values.reshape(-1, 1))
    test_dataset = Dataset(X=X_test.values, y=y_test.values.reshape(-1, 1))
    val_dataset = Dataset(X=X_val.values, y=y_val.values.reshape(-1, 1))

    # Create train, test and validation loader
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train the model
    reg_model = MLP(input_size=X_train.shape[1], dim=dim, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.AdamW(reg_model.parameters(), lr=lr, weight_decay=decay)

    loss_function = RMSELoss()
    df_metrics = train_model(model=reg_model, 
                              train_loader=train_loader, 
                              val_loader=val_loader, 
                              criterion=loss_function, 
                              optimizer=optimizer,
                              item=sypt_name,
                              num_epochs=100,
                              patience=3, 
                              min_delta=0.01, 
                              verbose=verbose)

    # Evaluate
    actuals_train, predictions_train, avg_loss_train = evaluate_model(reg_model, train_loader, loss_function)
    actuals_test, predictions_test, avg_loss_test = evaluate_model(reg_model, test_loader, loss_function)
    actuals_val, predictions_val, avg_loss_val = evaluate_model(reg_model, val_loader, loss_function)
    
    train_score[f'Pred_{sypt_name}'] = pd.DataFrame(predictions_train, index=X_train.index, columns=[f'Pred_{sypt_name}'])
    test_score[f'Pred_{sypt_name}'] = pd.DataFrame(predictions_test, index=X_test.index, columns=[f'Pred_{sypt_name}'])
    val_score[f'Pred_{sypt_name}'] = pd.DataFrame(predictions_val, index=X_val.index, columns=[f'Pred_{sypt_name}'])

    # participant level
    train_score_par = train_score.groupby(['PAR'], as_index=False).agg(
        {f'Pred_{sypt_name}': 'median', f'PANSS_{sypt_name}': 'first'}
        )
    test_score_par = test_score.groupby(['PAR'], as_index=False).agg(
        {f'Pred_{sypt_name}': 'median', f'PANSS_{sypt_name}': 'first'}
        )
    val_score_par = val_score.groupby(['PAR'], as_index=False).agg(
        {f'Pred_{sypt_name}': 'median', f'PANSS_{sypt_name}': 'first'}
        )
    
    train_rmse_par = np.sqrt(np.nanmean((train_score_par[f'PANSS_{sypt_name}'] - train_score_par[f'Pred_{sypt_name}']) ** 2))
    test_rmse_par = np.sqrt(np.nanmean((test_score_par[f'PANSS_{sypt_name}'] - test_score_par[f'Pred_{sypt_name}']) ** 2))
    val_rmse_par = np.sqrt(np.nanmean((val_score_par[f'PANSS_{sypt_name}'] - val_score_par[f'Pred_{sypt_name}']) ** 2))

    print()
    print(f'Seg: Train Loss: {avg_loss_train:.6f}, Val Loss: {avg_loss_val:.6f}, Test Loss: {avg_loss_test:.6f}')
    print(f'PAR: Train Loss: {train_rmse_par:.6f}, Val Loss: {val_rmse_par:.6f}, Test Loss: {test_rmse_par:.6f}')
    print(f'Parameters: {MLP._count_parameters(reg_model)}')
    
    torch_dict  = {
        'model_state_dict': reg_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': df_metrics['Epoch'].max(),
        'train_loss': df_metrics['Train Loss'].tolist(),
        'val_loss': df_metrics['Validation Loss'].tolist(),
        'test_pred': predictions_test,
        'hyperparams': {
            'dim': dim,
            'dropout': dropout_rate,
            'lr': lr,
            'l2': decay,
        }}
    
    return torch_dict

   












