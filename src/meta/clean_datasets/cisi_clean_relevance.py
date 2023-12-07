import re

results = []
with open("../../../datasets/cisi-raw/CISI.REL") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = re.sub('\t', ' ', line)
        line = re.sub(' +', ' ', line)
        result = line.split(' ')
        results.append([result[0], result[1]])

with open("../../../datasets/cisi-raw/relevance.txt", "a") as f:
    for result in results:
        f.write(result[0] + " " + result[1] + " " + "1" + "\n")

