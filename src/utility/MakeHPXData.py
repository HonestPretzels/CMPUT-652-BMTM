import csv
import numpy as np
import sys, string
import nltk
from numpy.lib.function_base import append
from PreProcesssHPData import vectorize
nltk.download('averaged_perceptron_tagger')


def main():
    wordsFile = sys.argv[1]
    output = sys.argv[2]
    vocabFile = sys.argv[3]
    timeFile = sys.argv[4]

    vocab = []
    with open(vocabFile, 'r', encoding="utf8") as vf:
        for line in vf:
            vocab.append(line.strip())

    timeToWordDict = {}
    with open(wordsFile, 'r') as wf:
        reader = csv.reader(wf)
        for row in reader:
            timeToWordDict[float(row[1])] = row[0]

    TRs = []
    with open(timeFile, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            TRs.append(row[0])
    
    lines = []
    for i in range(len(TRs)):
        currTR = [float(tr) for tr in TRs[i-4:i]]
        wordIndexes = []
        for tr in currTR:
            wordIndexes.extend([tr, tr+0.5, tr+1, tr+1.5])
        line = []
        for idx in wordIndexes:
            if idx in timeToWordDict.keys():
                line.append(timeToWordDict[idx].lower())
            else:
                line.append('NIL')
        line = ' '.join(line)
        line = line.translate(str.maketrans('', '', string.punctuation))
        currTokens = nltk.pos_tag(nltk.wordpunct_tokenize(line))
        line = [word for (word,_) in currTokens]
        if len(line) > 16:
            line = line[:17]
        lines.append(line)
    x = vectorize(lines, vocab, 16)
    print(len(x))

    x = np.array(x)
    np.save(output, x)

if __name__ == "__main__":
    main()