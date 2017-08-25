# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#

import hyperopt as hy

space = {'rnn1': 'LSTM',
         'rnn1_layers': 1,
         'penultimate_layers': 0,
         'units1': hy.hp.choice('units1', [64,128,256,512]),
         'units2': hy.hp.choice('units2', [64,128,256]),
         'dropout1': hy.hp.uniform('dropout1', .0,.75),
         "position_emb": 5,
         "window_size": hy.hp.choice('window_size', [1,2,3,4,5]),
         "memory_merge": "concat",
         'optimizer': 'adam',
         'gpu': True
         }
