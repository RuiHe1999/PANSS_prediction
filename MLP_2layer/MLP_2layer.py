# 1. packages
import os
import torch
import numpy as np
import pandas as pd
import argparse
from util_2layer import split_data, run_MLP

import warnings
warnings.filterwarnings("ignore")

# 2. consonants
random_state = 42
columns = ['Feat', 'Regressor', 'PANSS', 'K', 'Round', 'R2_test', 'R2_val', 'RMSE_test', 'RMSE_val']

torch.manual_seed(random_state)
np.random.seed(random_state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs('results/', exist_ok=True)
os.makedirs('results/models/', exist_ok=True)

# 3. functions
def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP model with custom dimensions for layers.")
    parser.add_argument('--item', type=str, choices=['P1', 'P2', 'P3', 'N1', 'N4', 'N6', 'G5', 'G9'], required=True, 
                        help='The name of PANSS items (P1, P2, P3, N1, N4, N6, G5, G9)')
    parser.add_argument('--feat', type=str, choices=['AcouPros', 'm-HuBERT', 'Concat'], required=True, 
                        help='The name of feature set (AcouPros, m-HuBERT, Concat)')
    parser.add_argument('--dim', type=int, default=64, help='Number of neurons in the hidden layer')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--decay', type=float, default=1e-3, help='L2 penalty')
    parser.add_argument('--save', action='store_true', help='Whether to save the model (default: True)')  
    parser.add_argument('--verbose', action='store_true', help='Whether to print the training details (default: False)')  
    
    return parser.parse_args()

# 4. commands
if __name__ == "__main__":
    # prepare the data
    data = pd.read_csv('panss_10lang.csv')
    data, split_summary = split_data(data)
    split_summary.to_csv('results/split_summary.csv', index=False)
    data_g =  data[data['PANSS_G5'].notna()]

    # run the model 
    args = parse_args()
    df = data_g if 'P' not in args.item else data
    torch_dict = run_MLP(df, sypt_name=args.item, feat_name=args.feat, 
                          dim=args.dim, dropout_rate=args.dropout, lr=args.lr, 
                          decay=args.decay, verbose=args.verbose)
    
    if args.save:
        # save the model 
        save_path = (
            f"results/models/model-MLP2_item-{args.item}_feat-{args.feat}_param-"
            f"{{dim-{args.dim}_dropout-{args.dropout}_lr-{args.lr}_l2-{args.decay}}}.pth"
        )


        # save model state_dict with extra info
        torch.save(torch_dict, save_path)
            
    
    print('----------------------------------------------------------------')




# # prepare the data
# data = pd.read_csv('panss_10lang.csv')
# data, split_summary = split_data(data)
# split_summary.to_csv('results/split_summary.csv', index=False)
# data_g =  data[data['PANSS_G5'].notna()]

# # run the model 
# df = data_g if 'P' not in 'G5' else data
# torch_dict = run_MLP(df, sypt_name='G5', feat_name='AcouPros', 
#               dim=10, dropout_rate=0.5, lr=1e-3, decay=5e-4, verbose=True)

# # save the model 
# save_path = "results/models/MLP2_item-P1_feat-m-HuBERT_param-{dim1-20_dim2-5_dropout-0.4_lr-0.001_l2-0.001}.pth"

# # save model state_dict with extra info
# torch.save(torch_dict, save_path)






















