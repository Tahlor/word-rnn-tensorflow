from gutenberg.query import list_supported_metadatas
from gutenberg.query import get_etexts
from gutenberg.query import get_metadata
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers

import requests
import json
import pickle
import os
import zipfile, StringIO
import traceback

OUTPUT = r".\data\gutenberg"
SAVE_DATA = ".\save\download_list"

def populate_cache():
    from gutenberg.acquire import get_metadata_cache
    cache = get_metadata_cache()
    cache.populate()

#populate_cache()
# delete lines that begin with []

def download_ids():
    text = ""
    results = []
    for i in range(1,100):
        print("Downloading page {}".format(i))
        url = "http://gutendex.com/books/?page={}&topic=poetry".format(i)
        x = download(url, file_type = "json")
        if u'detail' in x.keys() and x['detail'] == u'Invalid page.':
            print("All done!")
            return results
        results += x["results"]
    return results


def download(url, file_type = "json"):
    if file_type == "json":
        response = requests.get(url)
        return json.loads(response.text)
    elif file_type == "text":
        return requests.get(url).text
    elif file_type == "zip":
        return requests.get(url)

def save_file(text, file_path):
    with open(file_path, "wb") as f:
        f.write(text)

def save_list(l, path):
    with open(path, "wb") as f:
        pickle.dump(l, f)
    
def load_list(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def clean_path(path):
    for symbol in ['"']:
        path=path.replace(symbol,"_")
    return "".join([c for c in path if c.isalnum() or c in ['_', ' ', ',', "'", ';'] ]).rstrip()[:128]


# Input a list of things to find before first string or just first string
def parse(text, begin, end, exclude_targets = True):

    # Position 1
    pos1 = 0
    if begin == None:
        pos1 = 0
    elif type(begin) == type([]):
        for i in begin:
            pos1 = text[pos1:].find(i)
    else:
        pos1 = text.find(begin)

    # Position 2
    if end == None:
        pos2 = None
    else:
        pos2 = text[pos1:].find(end)

    # Include search text
    if exclude_targets:
        pos1 = pos1 + len(begin)
    else:
        pos2 = pos2 - len(end)

    return text[pos1:pos2]


# Download Moby Dick
#text = strip_headers(load_etext(2701)).strip()
#print(text)  # prints 'MOBY 

#print(list_supported_metadatas())

#print(get_metadata('formaturi', 2701)) # prints frozenset([u'Melville, Hermann'])

# Main
if __name__ == '__main__':

    # Download metadata
    #master_list = download_ids()
    #save_list(master_list, SAVE_DATA)
    master_list = load_list(SAVE_DATA)


    # Download everything in poetry:
    bad_items = []
    for item in master_list:
        print(item["title"])
        for i in item["formats"].keys():
            if "text/plain" in i:
                # If file does not exist
                file_name = clean_path(item["title"])+".txt"
                if not os.path.exists(os.path.join(OUTPUT,  file_name)):
                    print("Downloading {}, {}...").format(i,item["formats"][i])
                    url =item["formats"][i]
                    ext = item["formats"][i][-4:]
                    text = download(url, file_type = "zip" if ext == ".zip" else "text")
                    encoding=parse(i, "charset=", None)
                    # print(encoding)
                    # decode(encoding)
                    try:
                        if ext != ".zip":
                            text = text.encode('utf-8')
                            text = parse(text, ["*** START OF THIS PROJECT GUTENBERG", "***"], "*** END OF THIS PROJECT GUTENBERG")    
                            save_file(text, os.path.join(OUTPUT, file_name))
                        else:
                            with zipfile.ZipFile(StringIO.StringIO(text.content)) as z:
                                #z.extractall()
                                print("Unzipping...")
                                print(z.namelist())
                                for f in z.namelist():
                                    z.extract(f, OUTPUT)
                                    os.rename(os.path.join(OUTPUT, f), os.path.join(OUTPUT, file_name))
                                    print(file_name, OUTPUT)
                    except:
                        print("Problem with " + file_name)
                        traceback.print_exc()
                        print(os.path.join(OUTPUT, f))
                        bad_items.append(item)
                        raw_input("OK?")
                    break
save_list(bad_items, os.path.join(OUTPUT, "bad_items"))


# problems -- filter out Gutenberg stuff
# bad extensions sometimes
