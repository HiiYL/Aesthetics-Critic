import nltk
import pickle
from build_vocab import Vocabulary
from pandas import HDFStore

with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

store = HDFStore('data/labels.h5')

ava_train = store['labels_train']

comments = ava_train.ix[:, 'comments'].as_matrix()

tokenized_comments = []
total_length = len(comments)
for iteration, image_comments in enumerate(comments):
    image_tokens = []
    for comment in image_comments:
        image_tokens.append(nltk.tokenize.word_tokenize(str(comment).lower()))
    tokenized_comments.append(image_tokens)

    if iteration % 100 == 0:
        print("[{}/{}] Processing Comments - Tokenizing ...".format(iteration, total_length))


tokenized_comments_indexed = []
for image_comments in tokenized_comments:
    image_tokens = []
    for comment in image_comments:
        image_tokens.append(vocab('<start>'))
        image_tokens.extend([vocab(token) for token in comment])
        image_tokens.append(vocab('<end>'))
    tokenized_comments_indexed.append(image_tokens)
    if iteration % 100 == 0:
        print("[{}/{}] Processing Comments - Converting to Index ...".format(iteration, total_length))