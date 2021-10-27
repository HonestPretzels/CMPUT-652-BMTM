import scipy.io
import sys
import csv
import numpy as np
import re

def oneHotToPOS(mapping, oneHot):
    '''
    Get the POS value of a oneHot encoded word. Note in the current implementation
    there are some words with multiple tags. We should take a look at why and decide
    what to do. I think it tends to be ones with punctuation. There could also be an issue
    if it is all zeros
    '''
    # TODO: Fix the above issues
    idx = np.argwhere(np.array(oneHot) == 1).flatten()
    if len(idx) > 1:
        idx = max(idx)
    elif len(idx) == 0:
        idx = -1
    else:
        idx = idx[0]
    return mapping[idx]

def main():
    inputfile = sys.argv[1]
    wordsfile = sys.argv[2]
    outputfile = sys.argv[3]
    mat = scipy.io.loadmat(inputfile)
    POS_types_mat = mat["features"][0][8][1][0]
    POS_types = []
    for pos in POS_types_mat:
        POS_types.append(pos[0])
    
    POS_types.append('NIL')
    features = mat["features"][0][8][2]
    out = []

    with open('./Data/HarryPotter/BrainData/StoryFeatures/Processed/label_dict.txt', 'w') as f:
        for i in range(len(POS_types)):
            f.write('%s: %d\n'%(POS_types[i], i))

    word_counts = {}

    with open(wordsfile, 'r') as wf:
        reader = csv.reader(wf)
        currLine = 0
        for line in reader:
            oneHot = features[currLine]
            posTag = oneHotToPOS(POS_types, oneHot)
            word = re.sub(r'[^\w\s]', '', line[0])
            if word in word_counts.keys():
                word_counts[word] += 1
            else:
                word_counts[word] = 1
            out.append((word, posTag))
            currLine += 1

    with open('./Data/HarryPotter/BrainData/StoryFeatures/Processed/word_dict.txt', 'w') as f:
        for word, count in word_counts.items():
            f.write('%s: %d\n'%(word, count))

    with open(outputfile, 'w', newline='') as of:
        writer = csv.writer(of)
        writer.writerows(out)

if __name__ == "__main__":
    main()