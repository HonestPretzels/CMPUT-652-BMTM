import csv
import numpy as np
import sys, string
import nltk
from PreProcesssHPData import vectorize
nltk.download('averaged_perceptron_tagger')


def main():
    wordsFile = sys.argv[1]
    output = sys.argv[2]
    vocabFile = sys.argv[3]

    vocab = []
    with open(vocabFile, 'r') as vf:
        for line in vf:
            vocab.append(line.strip())

    lines = []
    with open(wordsFile, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            lines.append(row)

    x = []
    for i in range(len(lines)-16):
        currX = ' '.join([(line[0]) for line in lines[i:i+16]]).lower()
        currX = currX.translate(str.maketrans('', '', string.punctuation))
        currTokens = nltk.pos_tag(nltk.wordpunct_tokenize(currX))
        words = [word for (word,_) in currTokens]
        if len(words) > 16:
            words = words[:17]
        x.append(vectorize(words, vocab, 16))

    x = np.array(x)
    np.save(output, x)

if __name__ == "__main__":
    main()