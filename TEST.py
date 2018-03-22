from gensim.summarization.summarizer import summarize

text = """
Some think to lose him 
By having him confined; 
And some do suppose him, 
Poor thing, to be blind; 
But if ne'er so close ye wall him, 
Do the best that you may, 
Blind love, if so ye call him, 
Will find out his way. 

You may train the eagle 
To stoop to your fist; 
Or you may inveigle 
The phoenix of the east; 
The lioness, ye may move her 
To give o'er her prey; 
But you'll ne'er stop a lover: 
He will find out his way."""


if False:
    # This chooses best sentences
    x = summarize(text, word_count=5, split=False)
    print(x)


    # This chooses words
    from gensim.summarization import keywords
    x = keywords(text).split('\n')
    print(x)


# TF-IDF
#????
>>> import nltk.corpus
>>> from nltk.text import TextCollection
>>> print('hack'); from nltk.book import text1, text2, text3
hack...
>>> gutenberg = TextCollection(nltk.corpus.gutenberg)
>>> mytexts = TextCollection([text1, text2, text3])


# RAKE
#python -c "import nltk; nltk.download('stopwords')"
#pip install rake-nltk
from rake_nltk import Rake
r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
r = Rake("English") # To use it in a specific language supported by nltk.
print r.extract_keywords_from_text(text)
print r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.

# Each word has at least 5 characters
# Each phrase has at most 3 words
# Each keyword appears in the text at least 4 times

