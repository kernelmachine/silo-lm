import io
import os
import json
import gzip
import zstandard as zstd
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict

DCTX = zstd.ZstdDecompressor(max_window_size=2**31)

DATA_DIR = "data"

def load_pile(split, subset):
    data = []
    if subset in ["amazon", "imdb", "subj", "cc-news", "new-imdb", "new-imdb-raw", "new-amazon", "MIMIC_III"]:
        with open(os.path.join(DATA_DIR, "{}/{}.jsonl".format(subset, split))) as f:
            for line in f:
                dp = json.loads(line)
                data.append(dp["text"].strip())
    elif split=="train":
        data = load_pile_train(subset)
    else:
        fn = os.path.join(DATA_DIR, "the-pile/{}.jsonl.zst".format(split))
        assert os.path.exists(fn), fn
        for dp in map(json.loads, read_lines_from_zst_file(fn)):
            if subset==dp["meta"]["pile_set_name"].replace(" ", "_").replace("-", "_"):
                data.append(dp["text"].strip())
    return data

def load_pile_train(subset, limit=1000000000):
    data = []
    base_dir = os.path.join(DATA_DIR, "the-pile/train-gz")
    n_tokens = []
    np.random.random(2023)
    for fn in sorted(os.listdir(base_dir)):
        if fn.endswith(".json.gz") and fn.split("-")[0]==subset:
            with gzip.open(os.path.join(base_dir, fn), "r") as f:
                for line in f:
                    dp = json.loads(line.decode())
                    data.append(dp["text"].strip())
                    n_tokens.append(len(dp["text"].strip().split()))

    n_tot_tokens = np.sum(n_tokens)

    if limit and n_tot_tokens > limit:
        # When the data is too large, it doesn't fit into RAM during tokenization
        np.random.seed(2023)
        indices = np.random.permutation(range(len(data)))
        new_data = []
        tot = 0

        for i in indices:
            new_data.append(data[i])
            tot += n_tokens[i]
            if tot >= limit:
                break

        print ("Sampled %.2fM->%.2fM sequences (%dM->%dM tokens) for %s" % (
                len(data)/1000000,
                len(new_data)/1000000,
                n_tot_tokens/1000000,
                tot/1000000,
                subset))
        data = new_data
    else:
        print ("Load %.2fM sequences (%dM tokens) for %s" % (len(data)/1000000, n_tot_tokens/1000000, subset))

    return data

def read_lines_from_zst_file(zstd_file_path:Path):
    with zstd.open(zstd_file_path, mode='rb', dctx=DCTX) as zfh:
        with io.TextIOWrapper(zfh) as iofh:
            for line in iofh:
                yield line
