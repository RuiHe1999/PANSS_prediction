# 1. packages
import os
import math
import numpy as np
import pandas as pd
from collections import Counter

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
    
# 4. commands
# 4.1 read and split data
data = pd.read_csv('panss_10lang.csv')
data_g =  data[data['PANSS_G5'].notna()]

sns.set(style='whitegrid')

cols = [
    'PANSS_P1', 'PANSS_P2', 'PANSS_P3',
    'PANSS_N1', 'PANSS_N4', 'PANSS_N6',
    'PANSS_G5', 'PANSS_G9'
]

cols_exist = [c for c in cols if c in data.columns]

n = len(cols_exist)
ncol = 3
nrow = math.ceil(n / ncol)

fig, axes = plt.subplots(nrow, ncol, figsize=(4.5*ncol, 3.8*nrow))
axes = axes.flatten()

for i, c in enumerate(cols_exist):
    ax = axes[i]
    x = data[c].dropna()

    sns.histplot(
        x=x,
        bins=np.arange(0.5, 7.6, 1),
        kde=False,
        stat='count',
        shrink=0.85,
        ax=ax
    )

    ax.set_title(c)
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.set_xlim(0.5, 7.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])

for j in range(len(cols_exist), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

















