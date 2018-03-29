import nltk.tokenize.punkt as pkt

# Experimental
class CustomLanguageVars(pkt.PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \n*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            \s+(?P<next_tok>\S+)     # or whitespace and some other token
        ))"""

#custom_tknzr = pkt.PunktSentenceTokenizer(lang_vars=CustomLanguageVars())
#print(custom_tknzr.tokenize(text))


import re

text = "This is a test |||| or this||||YEAH"
nl = re.compile("\s*\|\|\|\|\s*")
text = nl.sub(r"\n", text)
print(text)
