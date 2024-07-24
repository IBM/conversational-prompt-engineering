import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd

rootdir = "/Users/avishaigretz/Downloads/no_wavs/trs.txt"

train_topics = ['nuclear-power', 'olympic-games', 'casinos', 'free-market', 'ban-video-games']
test_topics = ['racial-profiling', 'compulsory-voting', 'ban-fastfood', 'sex-selection', 'student-loans',
               'electric-cars', 'organic-food', 'biofuels']

topics_to_speeches = defaultdict(list)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith("pro.trs.txt"):
            with open(os.path.join(rootdir, file), "r") as f:
                text = "\n".join([t.strip("\n") for t in f.readlines()])
                topics_to_speeches[file.split("_")[-2]].append(text)
speeches = [x for y in topics_to_speeches.values() for x in y]
lengths = [len(s.split()) for s in speeches]
print(f"total: {len(lengths)}")
average = np.mean(lengths)
print(f"average: {np.mean(lengths)}")
print(f"minimum: {np.min(lengths)}")
print(f"maximum: {np.max(lengths)}")
print(f"std err: {np.std(lengths)}")

random.seed(1)

train_speeches = []
for t in train_topics:
    speeches = [s for s in topics_to_speeches[t] if len(s.split()) > average]
    train_speeches.extend(random.sample(speeches, 2))
random.shuffle(train_speeches)

test_speeches = []
for t in test_topics:
    speeches = [s for s in topics_to_speeches[t] if len(s.split()) > average]
    test_speeches.extend(random.sample(speeches, 1))
random.shuffle(test_speeches)

test_full_speeches = []
for t in topics_to_speeches.keys():
    test_full_speeches.extend([s for s in topics_to_speeches[t] if s not in train_speeches and len(s.split()) > average])
random.shuffle(test_full_speeches)

outdir = 'public/debate_speeches'
os.makedirs(outdir, exist_ok=True)
pd.DataFrame({"text": train_speeches}).to_csv(os.path.join(outdir, "train.csv"), index=False)
pd.DataFrame({"text": test_speeches}).to_csv(os.path.join(outdir, "test.csv"), index=False)
pd.DataFrame({"text": test_full_speeches}).to_csv(os.path.join(outdir, "test_full.csv"), index=False)


