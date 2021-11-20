import sys, os
import numpy as np
from os import path
import nltk
nltk.download('averaged_perceptron_tagger')
from keras.preprocessing.sequence import pad_sequences



def getAllFiles(dir):
    return [f for f in os.listdir(dir) if path.isfile(path.join(dir, f))]

def vectorize(sequence, vocabulary, maxLen):
    '''
    Taking a sequence of words or tags, return a N*d np array of one-hot vectors
    where N is the number of items in the list and d is the length of the
    vector space
    '''
    intSequences = []
    print(vocabulary[0])
    
    for i in range(len(sequence)):
        intSequences.append([vocabulary.index(item) for item in sequence[i]])
    return pad_sequences(intSequences, maxlen=maxLen, padding="post")

def main():
    inputDir = sys.argv[1]
    vocabJSON = sys.argv[2]
    posJSON = sys.argv[3]
    outputDir = sys.argv[4]
    
    tokens = []
    sentences = []
    tags = []
    files = [path.join(inputDir, f) for f in getAllFiles(inputDir)]
    # Parse into sentences
    for fileName in files:
        with open(fileName, 'r', encoding="utf8") as f:
            lines = f.read().lower()
            currTokens = nltk.pos_tag(nltk.wordpunct_tokenize(lines))
            tokens.extend(currTokens)
            currSentence = []
            currTags = []
            for word, pos in currTokens:
                if pos == ".":
                    currSentence.append(word)
                    currTags.append(pos)
                    sentences.append(currSentence)
                    tags.append(currTags)
                    currSentence = []
                    currTags = []
                else:
                    currSentence.append(word)
                    currTags.append(pos)
                    
    vocab = ['NIL']
    vocab.extend([k for k in nltk.FreqDist(word for (word,_) in tokens).keys()])
    tagsVocab = ['NIL']
    tagsVocab.extend([k for k in nltk.FreqDist(tag for (_,tag) in tokens).keys()])
    
    # Create and save the POS data set
    # x = []
    # y = []
    # for i in range(len(sentences)):
    #     currSentence = sentences[i]
    #     currTags = tags[i]
    #     if len(currSentence) < 4:
    #         x.append(currSentence)
    #         y.append(currTags)
    #     else:
    #         for j in range(len(currSentence)-4):
    #             x.append(currSentence[j:j+4])
    #             y.append(currTags[j:j+4])
    # x = vectorize(x, vocab, 4)
    # y = vectorize(y, tagsVocab, 4)
    # x = np.array(x)
    # y = np.array(y)
    
    # outX = path.join(outputDir, 'PosX.npy')
    # outY = path.join(outputDir, 'Posy.npy')
    # np.save(outX, x)
    # np.save(outY, y)
    
    # Create and save the 4 -> 1 LM Data set
    x = []
    y = []
    for i in range(len(sentences)):
        currSentence = sentences[i]
        if len(currSentence) < 5:
            x.append(currSentence)
            y.append([])
        else:
            for j in range(len(currSentence)-4):
                x.append(currSentence[j:j+4])
                y.append([currSentence[j+4]])
    x = vectorize(x, vocab, 4)
    y = vectorize(y, vocab, 1)
    x = np.array(x)
    y = np.array(y)
    
    outX = path.join(outputDir, 'Lm4to1X.npy')
    outY = path.join(outputDir, 'Lm4to1Y.npy')
    np.save(outX, x)
    np.save(outY, y)
    
    # Create and save the 1 -> 1 LM Data set
    x = []
    y = []
    for i in range(len(sentences)):
        currSentence = sentences[i]
        if len(currSentence) < 2:
            x.append(currSentence)
            y.append([])
        else:
            for j in range(len(currSentence)-1):
                x.append([currSentence[j]])
                y.append([currSentence[j+1]])
    x = vectorize(x, vocab, 1)
    y = vectorize(y, vocab, 1)
    x = np.array(x)
    y = np.array(y)
    
    outX = path.join(outputDir, 'Lm1to1X.npy')
    outY = path.join(outputDir, 'Lm1to1Y.npy')
    np.save(outX, x)
    np.save(outY, y)
        
        
    
    # Get the vocabulary setup
    with open(vocabJSON, 'w', encoding="utf8") as vj:
        vj.writelines('\n'.join(vocab))
    with open(posJSON, 'w', encoding="utf8") as pj:
        pj.writelines('\n'.join(tagsVocab))
        
if __name__ == "__main__":
    main()