import random

def write_out(text, f):        
    # Write out
    with open(f, "w") as fobj:
        fobj.write(text)

string = ""
for x in range(1,5000):
    string += "a " * random.randint(1, 40)
    string += "\n"

write_out(string, "./input.txt")
