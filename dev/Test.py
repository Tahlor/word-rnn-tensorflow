import pickle
import codecs

path = r"D:\PyCharm Projects\word-rnn-tensorflow\data\vocab.pkl"
path2 = r"D:\PyCharm Projects\word-rnn-tensorflow\data\poems_large.txt"

if False:
    with open(path, "rb") as f:
        l = pickle.load(f)
    print(l[0:50])
    print("\n" in l)

    with open(path2, "rb") as f:
        l = f.read()
        print(l[0:50])

with codecs.open(path2, "rb", encoding=None) as f:
    l = f.read()
    print(l[0:500])
