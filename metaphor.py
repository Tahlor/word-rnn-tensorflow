import requests
import sys
from lxml import etree
import lxml
import urllib
import collections
import json

def get_metaphor(phrase):
    if sys.version_info[0] < 3:
        phrase = urllib.quote_plus(phrase)
    else:
        phrase = urllib.parse.quote_plus(phrase)
        
    url = r"http://ngrams.ucd.ie/metaphor-magnet-acl/q?kw={}&xml=true".format(phrase)
    response = requests.get(url, verify = False)
    return parse_metaphor(lxml.etree.XML(response.text), response.text)

def parse_metaphor(tree, return_targets = True):
    # {word : [{score: , metaphor: , attribute: }, ...]
    master_dict = collections.defaultdict(list)

    target_source = {}
    for el in tree:
        typ = el.tag.strip()
        name = el.attrib["Name"].strip()
        target_source[typ] = name
        
        attribute, metaphor = el[0].text.strip().split(":")
        score = int(el[1].text.strip())

        new_dict = {"score":score, "attribute":attribute, "metaphor":metaphor}
        master_dict[name].append(new_dict)

    for item in master_dict:
        master_dict[item] = sorted(master_dict[item], key=lambda k: -k['score']) 
    print(target_source)
    typ = "Target" if return_targets else "Source"
    return master_dict[target_source[typ]]


def get_rhyme(rhyme, relation):
    url = r"http://api.datamuse.com/words?ml={}&rel_rhy={}&max=1000".format(relation, rhyme)
    response = requests.get(url, verify = False)
    rhymes = json.loads(response.text)
    return rhymes

if __name__ == '__main__':
    print(get_rhyme("grape", "breakfast"))
    master_dict = get_metaphor("Marriage as death")
    print(master_dict)
