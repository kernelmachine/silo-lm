import os
from index import DataStore, load_embeds
import argparse
import numpy as np
import time
import pickle as pkl
import shutil
import math
import json
import subprocess

from tqdm import tqdm
from collections import defaultdict
from data import load_pile
from model import LanguageModel

from pyserini.search.lucene import LuceneSearcher

class BM25Index(object):

    def __init__(self, index_dir, data_dir):

        if not os.path.exists(index_dir):
            print ("Start building index for %s at %s" % (data_dir, index_dir))
            command = """python -m pyserini.index.lucene \
            --collection JsonCollection \
            --input '%s' \
            --index '%s' \
            --generator DefaultLuceneDocumentGenerator \
            --storeRaw --threads 1""" % (data_dir, index_dir)
            ret_code = subprocess.run([command],
                                    shell=True,
                                    #stdout=subprocess.DEVNULL,
                                    #stderr=subprocess.STDOUT
                                    )
            if ret_code.returncode != 0:
                print("Failed to build the index")
                exit()
            else:
                print("Successfully built the index")

        self.searcher = LuceneSearcher(index_dir)

    def search(self, query, k, continuation=False, shift=False):
        hits = self.searcher.search(query, k=k)
        docs = []
        for hit in hits:
            docid = hit.docid

            if shift:
                docid = str(int(hit.docid)+1)

            raw = self.searcher.doc(docid).raw()
            input_ids = json.loads(raw)["input_ids"]

            if continuation:
                next_item = self.searcher.doc(str(int(hit.docid)+1))
                if next_item is not None:
                    next_raw = next_item.raw()
                    input_ids += json.loads(raw)["input_ids"]
                else:
                    print ("The last block retrieved, so skipping continuation...")

            docs.append(input_ids)
        return docs

def main(args):
    args.output_dir = os.path.join(
            args.output_dir,
            args.lm,
            "{}".format(args.split))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.subset is not None:
        subsets = args.subset.split(",")
    else:
        subsets = set()
        for fn in os.listdir("/gscratch/zlab/sewon/data/the-pile/train-gz"):
            subsets.add(fn.split("-")[0])

    lm = LanguageModel(args.lm)
    assert args.max_n_sequences is None or args.split.startswith("train")

    start_time = time.time()
    for i, subset in enumerate(subsets):
        process(args, lm, subset)

def process(args, lm, subset):

    tokenized_dir = args.output_dir if args.lm=="neo-1.3B" else args.output_dir.replace(args.lm, "neoX")
    if not os.path.exists(tokenized_dir):
        os.makedirs(tokenized_dir)
    tokenized_path = os.path.join(tokenized_dir, "{}-tokenized.pkl".format(subset))
    postfix = str(args.max_retrieval_seq_length)

    if args.max_n_sequences is not None:
        start = 0 #args.max_n_sequences * args.embed_idx
        end = args.max_n_sequences * (1 + args.embed_idx)
        postfix += "-[{}K-{}K]".format(start, end)

    bm25_index_path = os.path.join(tokenized_dir, "{}-{}.bm25_index".format(subset, postfix))
    bm25_data_path = os.path.join(tokenized_dir, "{}-{}.bm25_index.data".format(subset, postfix))

    val_postfix = "{}-{}".format(args.max_seq_length, args.stride if args.merge else "none")
    val_tokenized_dir = tokenized_dir.replace(args.split, args.val_split)
    val_output_dir = args.output_dir.replace(args.split, args.val_split)

    val_tokenized_path = os.path.join(val_tokenized_dir, "{}-tokenized.pkl".format(subset))
    val_replug_tokenized_path = os.path.join(
        val_tokenized_dir, "{}-replug{}-from-{}-{}-{}-tokenized.pkl".format(
            subset,
            "-{}".format(args.ensemble_k) if args.ensemble_k>1 else "",
            args.split,
            postfix,
            val_postfix))
    val_loss_path = os.path.join(
        val_output_dir, "{}-replug{}-from-{}-{}-{}-loss.npy".format(
            subset,
            "-{}".format(args.ensemble_k) if args.ensemble_k>1 else "",
            args.split,
            postfix,
            val_postfix))

    assert os.path.exists(tokenized_path), tokenized_path
    assert os.path.exists(val_tokenized_path), val_tokenized_path

    def get_input_ids_and_targets(tokenized_path,
                                  max_seq_length,
                                  max_n_sequences=None,
                                  stride=None):
        with open(tokenized_path, "rb") as f:
            input_ids = pkl.load(f)
        if "MIMIC" in subset:
            assert args.merge
            input_ids, masks = input_ids
            flatten_input_ids = np.array([_id for ids in input_ids for _id in ids])
            flatten_masks = np.array([_id for ids in masks for _id in ids])
            all_input_ids, all_targets = lm.batch_merged(
                flatten_input_ids,
                flatten_masks=flatten_masks,
                max_seq_length=max_seq_length,
                stride=args.stride)
        elif args.merge:
            flatten_input_ids = np.array([_id for ids in input_ids for _id in ids])
            all_input_ids, all_targets = lm.batch_merged(flatten_input_ids, max_seq_length=max_seq_length, stride=args.stride)
        else:
            all_input_ids, all_targets = lm.batch(input_ids, max_seq_length=max_seq_length, allow_merge=False)

        if max_n_sequences is not None:
            assert len(all_input_ids) > max_n_sequences * 1000, "No need to use max_n_sequences"
            np.random.seed(2023)
            # 1M tokens ~ 1000K sequences
            indices = np.random.permutation(len(all_input_ids))[start*1000:end*1000]
            all_input_ids = all_input_ids[indices]
            all_targets = all_targets[indices]

        return all_input_ids, all_targets

    if os.path.exists(val_loss_path):
        losses = np.load(val_loss_path)

        if args.ensemble_k > 1:
            assert len(losses) % args.ensemble_k == 0
            # convert negative log likelihood to likelihood
            probabilities = np.exp(-losses)
            # reshape it to be (n_tokens, ensemble_k)
            probabilities = probabilities.reshape(args.ensemble_k, -1).transpose()
            # average across ensemble_k, convert it back to loss
            losses = -np.log(np.mean(probabilities, -1))

        if args.do_subset:
            np.random.seed(2023)
            indices = np.random.permutation(range(len(losses)))
            batch_size = 128
            losses = losses[indices[:batch_size * 10000]]

        print ("%s\t# tokens: %d\tPPL: %.3f" % (subset, losses.shape[0], np.exp(np.mean(losses))))
        return

    assert not os.path.exists(val_loss_path)

    if not os.path.exists(val_replug_tokenized_path):

        print (f"{val_replug_tokenized_path} not found. creating...")

        if not os.path.exists(bm25_data_path):
            # build bm25_index
            train_input_ids, _ = get_input_ids_and_targets(
                tokenized_path,
                max_seq_length=args.max_retrieval_seq_length,
                max_n_sequences=args.max_n_sequences,
                stride=args.max_retrieval_seq_length)

            os.mkdir(bm25_data_path)
            offset = 0
            with open(os.path.join(bm25_data_path, "data.jsonl"), "w") as f:
                for input_ids in train_input_ids:
                    assert len(input_ids) <= args.max_retrieval_seq_length
                    text = lm.tokenizer.decode(input_ids)
                    f.write(json.dumps({
                        "id": str(offset),
                        "contents": text,
                        "input_ids": input_ids.tolist()
                    })+"\n")
                    offset += 1

            print ("Finish saving %d docs" % offset)

        searcher = BM25Index(bm25_index_path, bm25_data_path)

        all_input_ids, all_targets = get_input_ids_and_targets(
            val_tokenized_path,
            max_seq_length=args.max_seq_length,
            max_n_sequences=None,
            stride=args.stride)

        assert (2048 - args.max_seq_length) % args.max_retrieval_seq_length == 0
        concat_k = (2048 - args.max_seq_length) // args.max_retrieval_seq_length
        ensemble_k = args.ensemble_k

        all_input_ids = all_input_ids.tolist()
        all_targets = all_targets.tolist()

        new_all_input_ids = [[] for _ in range(ensemble_k)]
        new_all_targets = [[] for _ in range(ensemble_k)]

        def append_to_new(i, new_input_ids, new_targets):
            assert len(new_input_ids)==len(new_targets)==2048
            new_all_input_ids[i].append(new_input_ids)
            new_all_targets[i].append(new_targets)

        for i, (input_ids, targets) in enumerate(zip(tqdm(all_input_ids), all_targets)):
            query_ids = [_id for _id, t in zip(input_ids, targets) if t==0]
            if len(query_ids)>0:
                query = lm.tokenizer.decode(query_ids)
                docs = searcher.search(query, k=concat_k * ensemble_k)

                new_input_ids = input_ids.copy()

                if len(docs)==0:
                    new_input_ids = input_ids.copy() + [0] * (2048 - args.max_seq_length)
                    new_targets = targets.copy() + [0] * (2048 - args.max_seq_length)
                    for i in range(ensemble_k):
                        append_to_new(i, new_input_ids, new_targets)
                else:
                    offset = 0
                    for doc_idx, retrieved_input_ids in enumerate(docs):
                        new_input_ids = retrieved_input_ids + new_input_ids

                        if ensemble_k>1 or (doc_idx + 1) % concat_k == 0:
                            assert len(new_input_ids) <= 2048, len(new_input_ids)
                            if len(new_input_ids) < 2048:
                                new_input_ids = [0] * (2048 - len(new_input_ids)) + new_input_ids
                            new_targets = [0] * (2048 - args.max_seq_length) + targets
                            append_to_new(offset, new_input_ids, new_targets)
                            new_input_ids = input_ids.copy()
                            offset += 1

                    assert offset==ensemble_k
            else:
                new_input_ids = input_ids.copy() + [0] * (2048 - args.max_seq_length)
                new_targets = targets.copy() + [0] * (2048 - args.max_seq_length)
                for i in range(ensemble_k):
                    append_to_new(i, new_input_ids, new_targets)

        assert len(new_all_input_ids) == len(new_all_targets) == ensemble_k
        assert np.all([len(ids)==len(all_input_ids) for ids in new_all_input_ids])
        assert np.all([len(ids)==len(all_input_ids) for ids in new_all_targets])

        all_input_ids = [_id for ids in new_all_input_ids for _id in ids]
        all_targets = [_id for ids in new_all_targets for _id in ids]

        with open(val_replug_tokenized_path, "wb") as f:
            pkl.dump({"all_input_ids": all_input_ids, "all_targets": all_targets}, f)

    else:
        print (f"{val_replug_tokenized_path} found. loading...")
        with open(val_replug_tokenized_path, "rb") as f:
            tokenized = pkl.load(f)
        all_input_ids = tokenized["all_input_ids"]
        all_targets = tokenized["all_targets"]

    losses = lm.encode(all_input_ids,
                       all_targets,
                       batch_size=args.batch_size,
                       verbose=True)

    np.save(val_loss_path, losses)

    if args.ensemble_k > 1:
        assert len(losses) % args.ensemble_k == 0
        # convert negative log likelihood to likelihood
        probabilities = np.exp(-losses)
        # reshape it to be (n_tokens, ensemble_k)
        probabilities = probabilities.reshape(args.ensemble_k, -1).transpose()
        # average across ensemble_k, convert it back to loss
        losses = -np.log(np.mean(probabilities, -1))
    print ("%s\t# tokens: %d\tPPL: %.3f" % (subset, losses.shape[0], np.exp(np.mean(losses))))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, default="pythia-1.4B")
    parser.add_argument("--output_dir", type=str, default="out")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--do_subset", action="store_true")

    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--val_split", type=str, default="val", choices=["val", "test"])

    # encoder related
    parser.add_argument("--merge", action="store_true", default=True)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--max_retrieval_seq_length", default=1024, type=int)
    parser.add_argument("--ensemble_k", default=1, type=int)

    parser.add_argument("--stride", default=512, type=int)
    parser.add_argument("--batch_size", default=8, type=int)

    parser.add_argument("--max_n_sequences", default=None, type=int, help="for training (unit: K)")
    parser.add_argument("--embed_idx", default=0, type=int, help="for training")

    args = parser.parse_args()
    print (args)

    main(args)
