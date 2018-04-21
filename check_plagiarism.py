from nltk import ngrams
from nltk.probability import LidstoneProbDist

derived_source = r"D:\PyCharm Projects\word-rnn-tensorflow\save\MASTER\sample.txt"
primary_source = r"D:\PyCharm Projects\word-rnn-tensorflow\data\final\input.txt"

def read_file(path):
    with open(path, "rb") as f:
        return f.read()

source_text = read_file(primary_source)
derived_text = read_file(derived_source)

for n in [4, 5]:
    in_grams = ngrams(source_text.split(), n)
    out_grams = ngrams(derived_text.split(), n)
    total_grams = ngrams(derived_text.split(), n)

    total_out = sum(1 for _ in set(total_grams))

    intersection = len(list(set(out_grams).intersection(in_grams)))
    print(intersection)
    print 1.0*intersection/total_out

"""
    estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2) 
    lm = ngrams(n, train, estimator=estimator)
    x = lm.perplexity(derived_text)
    print(x)
"""
