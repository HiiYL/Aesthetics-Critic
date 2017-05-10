import pandas as pd
from pandas import HDFStore

store = HDFStore('data/labels.h5')

ava_train = store['labels_train']
ava_test  = store['labels_test']

ava_train.comments