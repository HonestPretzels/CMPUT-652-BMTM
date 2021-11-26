import numpy as np
import sys


def main():
    fPath = sys.argv[1]
    scores = np.load(fPath)
    
    count = 0
    for s in scores:
        if s == 1.0:
            count += 1
        
    print(scores.shape, max(scores), min(scores), np.mean(scores), np.median(scores), count)
    
if __name__ == "__main__":
    main()