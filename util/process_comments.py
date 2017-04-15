from pandas import HDFStore
import codecs

store = HDFStore('labels.h5')

ava_train = store['labels_train']

sLength = ava_train.shape[0]
series = pd.Series(np.random.randn(sLength), index=ava_train.index)

for i in ava_train.index:
    with codecs.open("AVA-Comments/{}.txt".format(i), "r",encoding='utf-8', errors='ignore') as f:
        content = [ line.strip() for line in  f.readlines() ]
        concat_content = ' [END] '.join(content)
        series[i] = concat_content
ava_train.loc[:, 'comments'] = series

ava_test = store['labels_test']
sLength = ava_test.shape[0]
series = pd.Series(np.random.randn(sLength), index=ava_test.index)

for i in ava_test.index:
    with codecs.open("AVA-Comments/{}.txt".format(i), "r",encoding='utf-8', errors='ignore') as f:
        content = [ line.strip() for line in  f.readlines() ]
        concat_content = ' [END] '.join(content)
        series[i] = concat_content
ava_test.loc[:, 'comments'] = series


store = HDFStore('labels_new.h5')
store['labels_train'] = ava_train
store['labels_test'] = ava_test