import poetrytools

x = [["spring"]]

syllables = poetrytools.scanscion(x)

#y = "spring"
#print(poetrytools.stress(y))

def this():
    for y in ["spring", "project", "attribute", "insult"]:
        print(poetrytools.stress(y, "min"))

m = poetrytools.stress("nightingale", "all")
print m

words = "the nightingales thy coming each where sing:".split(" ")
m = ''.join([poetrytools.stress(x, "min") for x in words])
print(m)
print(len(m))

