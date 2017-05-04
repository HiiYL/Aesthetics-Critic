import pickle
from build_vocab import Vocabulary

with open("data/vocab.pkl", "rb") as f:
    w = pickle.load(f)

pickle.dump(w, open("data/vocab-py2.pkl","wb"), protocol=2)