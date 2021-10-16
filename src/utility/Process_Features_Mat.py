import scipy.io
import sys
import os.path as path
import csv
import numpy as np

def oneHotToPOS(mapping, oneHot):
    '''
    Get the POS value of a oneHot encoded word. Note in the current implementation
    there are some words with multiple tags. We should take a look at why and decide
    what to do. I think it tends to be ones with punctuation. There could also be an issue
    if it is all zeros
    '''
    # TODO: Fix the above issues
    idx = np.argmax(np.array(oneHot))
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
    
    features = mat["features"][0][8][2]

    out = []

    with open(wordsfile, 'r') as wf:
        reader = csv.reader(wf)
        currLine = 0
        for line in reader:
            oneHot = features[currLine]
            posTag = oneHotToPOS(POS_types, oneHot)
            word = line[0]
            out.append((word, posTag))
            currLine += 1

    with open(outputfile, 'w', newline='') as of:
        writer = csv.writer(of)
        writer.writerows(out)

if __name__ == "__main__":
    main()