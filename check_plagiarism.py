from nltk import ngrams

derived_source = "POETRY OUTPUT.txt"
primary_source = "poems_large.txt"

def read_file(path):
    with open(path, "rb") as f:
        return f.read()

source_text = read_file(primary_source)
derived_text = read_file(derived_source)
n = 5

in_grams = ngrams(source_text.split(), n)
out_grams = ngrams(derived_text.split(), n)
total_grams = ngrams(derived_text.split(), n)

total_out = sum(1 for _ in set(total_grams))

intersection = len(list(set(out_grams).intersection(in_grams)))
print(intersection)
print 1.0*intersection/total_out


from nltk.probability import LidstoneProbDist

estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2) 
lm = ngrams(5, train, estimator=estimator)
lm.perplexity(derived_text)
