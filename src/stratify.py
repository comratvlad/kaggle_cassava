import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

info = pd.read_csv('/home/vladislav/data/Cassava/train.csv')
y = info['label']
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=38)

for i, split in enumerate(skf.split(np.zeros(len(y)), y)):
    _, dev_index = split
    info['protocol_{}'.format(i+1)] = 'train'
    info['protocol_{}'.format(i+1)][dev_index] = 'dev'

info.to_csv('/home/vladislav/data/Cassava/train.csv', index=False)
