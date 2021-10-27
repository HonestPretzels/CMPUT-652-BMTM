import re

full = {}

with open('.\Data\HarryPotter\BrainData\StoryFeatures\Processed\word_dict.txt', 'r') as f:
    for line in f:
        s = line.split(':')
        s[0] = re.sub(r'[^\w\s]', '', s[0])
        full[s[0]] = int(s[1])
with open('.\Data\Penn\word_dict.txt', 'r') as f:
    for line in f:
        s = line.split(':')
        s[0] = re.sub(r'[^\w\s]', '', s[0])
        if s[0] in full.keys():
            full[s[0]] += int(s[1])
        else:
            full[s[0]] = int(s[1])

with open('./Data/combined_word.txt', 'w') as f:
    for key, val in full.items():
        f.write('%s: %d\n'%(key, val))