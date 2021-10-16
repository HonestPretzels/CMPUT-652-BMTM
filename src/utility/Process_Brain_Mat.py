import scipy.io
import sys
import os.path as path
import numpy as np
import csv

def processWords(mat_words):
    '''
    Somewhat hacky way to do this. Converts the weird matlab nested arrays to a simple 2D array
    which can later be saved as a csv
    '''
    out = []
    words = mat_words[0]
    for word_block in words:
        word = word_block[0][0][0][0]
        start = word_block[1][0][0]
        duration = word_block[2][0][0]
        row = (word, start, duration)
        out.append(row)
    return out       

# def process_meta(mat_meta):
#     print(mat_meta)

def write_csv(rows, fname):
    with open(fname, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def main():
    inputfile = sys.argv[1]
    outputDir = sys.argv[2]
    mat = scipy.io.loadmat(inputfile)
    # Words are weirdly nested so need to be preprocessed
    words = processWords(mat["words"])
    write_csv(words, path.join(outputDir, path.basename(inputfile).split('.')[0]+"_words.csv"))
    # Time is a nice simple 2D array so it is good to be written
    write_csv(mat["time"], path.join(outputDir, path.basename(inputfile).split('.')[0]+"_time.csv"))
    # Data is the same as time
    np.save(path.join(outputDir, path.basename(inputfile).split('.')[0]+"_data.npy"), np.array(mat["data"]))
    # Ignoring meta data for now since it is relatively unimportant for the time being
    # meta = process_meta(mat["meta"])
    



if __name__ == "__main__":
    main()