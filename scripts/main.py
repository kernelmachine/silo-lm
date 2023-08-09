import os
try:
    from index import DataStore, load_embeds
except Exception:
    pass
import argparse
import numpy as np
import time
import pickle as pkl
import math
import json

from tqdm import tqdm
from collections import defaultdict
from data import load_pile
from model import LanguageModel

DIMENSION = 2048

def softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

def main(args):
    args.output_dir = os.path.join(args.output_dir, args.lm.split("/")[-1], args.split)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert args.subset is not None
    subsets = args.subset.split(",")

    lm = None if args.task=="index" else LanguageModel(args.lm)
    assert args.max_n_sequences is None or args.split.startswith("train")

    for i, subset in enumerate(subsets):
        for task in args.task.split(","):
            process(args, lm, task, subset)

def process(args, lm, task, subset):

    tokenized_dir = args.output_dir if args.lm=="neo-1.3B" else args.output_dir.replace(args.lm.split("/")[-1], "neoX")
    if not os.path.exists(tokenized_dir):
        os.makedirs(tokenized_dir)
    tokenized_path = os.path.join(tokenized_dir, "{}-tokenized.pkl".format(subset))

    if task=="tokenize":
        if os.path.exists(tokenized_path):
            with open(tokenized_path, "rb") as f:
                input_ids = pkl.load(f)
            if "MIMIC" in subset:
                input_ids, masks = input_ids
        else:
            data = load_pile(args.split, subset=subset)
            assert len(data)>0
            if "MIMIC" in subset:
                input_ids, masks = lm.tokenize(data, handle_pii=True)
                with open(tokenized_path, "wb") as f:
                    pkl.dump([input_ids, masks], f)
            else:
                input_ids = lm.tokenize(data, handle_pii=False)
                with open(tokenized_path, "wb") as f:
                    pkl.dump(input_ids, f)

        n_tokens = np.sum([len(ids) for ids in input_ids])
        if "MIMIC" in subset:
            n_valid_tokens = np.sum([np.sum(mask) for ids, mask in zip(input_ids, masks)])
            print ("Saved %.1fM tokens (%.1fM valid tokens) for %s" % (n_tokens/1000000, n_valid_tokens/1000000, subset))
        else:
            print ("Saved %.1fM tokens for %s" % (n_tokens/1000000, subset))
        return

    ################ define various paths

    postfix = "{}-{}".format(args.max_seq_length, args.stride if args.merge else "none")
    s_postfix_prev_embeds = None

    if args.max_n_sequences is not None:
        start = args.max_n_sequences * args.embed_idx
        end = args.max_n_sequences * (1 + args.embed_idx)
        s_postfix = "-[{}K-{}K]".format(start, end)

        s_postfix_index = "-[0K-{}K]".format(end)
        s_postfix_prev_index = "-[0K-{}K]".format(start) if start>0 else None

        if args.embed_idx > 0:
            s_postfix_prev_embeds = ["-[{}K-{}K]".format(args.max_n_sequences*i, args.max_n_sequences*(i+1)) for i in range(args.embed_idx)]
    else:
        s_postfix = ""
        s_postfix_index = ""
        s_postfix_prev_index = None
        s_postfix_prev_embeds = None

    embed_path = os.path.join(
            args.output_dir,
            "{}-{}-embed.float16.npy".format(subset, postfix + s_postfix))
    loss_path = os.path.join(
            args.output_dir,
            "{}-{}-loss.npy".format(subset, postfix + s_postfix))
    index_path = os.path.join(
            args.output_dir,
            "{}-{}.index".format(subset, postfix + s_postfix_index))
    if s_postfix_prev_index is not None:
        prev_index_path = os.path.join(args.output_dir, "{}-{}.index".format(
            subset, postfix + s_postfix_prev_index))
    else:
        prev_index_path = None
    if s_postfix_prev_embeds is not None:
        prev_embed_paths = [
            os.path.join(args.output_dir, "{}-{}-embed.float16.npy".format(subset, postfix + sp))
            for sp in s_postfix_prev_embeds]
    else:
        prev_embed_paths = None

    trained_index_path = os.path.join(
        "/".join(args.output_dir.split("/")[:-1]),
        "{}-{}.trained.index".format(subset, postfix))

    postfix += s_postfix_index
    assert os.path.exists(tokenized_path), tokenized_path

    def get_input_ids_and_targets(tokenized_path, split):
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
                max_seq_length=args.max_seq_length,
                stride=args.stride)
        elif args.merge:
            flatten_input_ids = np.array([_id for ids in input_ids for _id in ids])
            all_input_ids, all_targets = lm.batch_merged(
                    flatten_input_ids,
                    max_seq_length=args.max_seq_length,
                    stride=args.stride)
        else:
            all_input_ids, all_targets = lm.batch(input_ids, max_seq_length=args.max_seq_length,
                    stride=args.stride)

        if split.startswith("train") and args.max_n_sequences is not None:
            assert len(all_input_ids) > args.max_n_sequences * 1000, "No need to use max_n_sequences"
            assert len(all_input_ids) > start * 1000, "This embed_idx is unnecessary because the total # of sequences is {}".format(len(all_input_ids))
            np.random.seed(2023)
            indices = np.random.permutation(len(all_input_ids))[start*1000:end*1000]
            all_input_ids = all_input_ids[indices]
            all_targets = all_targets[indices]

        return all_input_ids, all_targets

    def get_flatten_targets(tokenized_path, split, for_search=False):
        def _load_flatten_targets(flatten_target_path):
            if os.path.exists(flatten_target_path):
                flatten_targets = np.load(flatten_target_path)
            else:
                _, all_targets = get_input_ids_and_targets(tokenized_path, split=split)
                flatten_targets = []
                for targets in all_targets.tolist():
                    flatten_targets += [tgt for tgt in targets if tgt!=lm.pad_token]
                flatten_targets = np.array(flatten_targets)
                np.save(flatten_target_path, flatten_targets)
            return flatten_targets

        if for_search and split.startswith("train") and args.max_n_sequences is not None and args.embed_idx > 0:
            all_flatten_targets = []
            for i in range(args.embed_idx+1):
                flatten_target_path = tokenized_path.replace(
                    ".pkl",
                    "_[{}K-{}K]_flatten.npy".format(args.max_n_sequences*i, args.max_n_sequences*(i+1)))
                assert os.path.exists(flatten_target_path)
                all_flatten_targets.append(_load_flatten_targets(flatten_target_path))
            flatten_targets = np.concatenate(all_flatten_targets)

        else:
            flatten_target_path = tokenized_path.replace(
                ".pkl",
                "{}_flatten.npy".format("" if args.max_n_sequences is None or not split.startswith("train") else "_[{}K-{}K]".format(start, end)))
            flatten_targets = _load_flatten_targets(flatten_target_path)

        return flatten_targets

    def get_val_data(batch_size, do_load_embed):
        val_tokenized_dir = tokenized_dir.replace(args.split, args.val_split)
        val_output_dir = args.output_dir.replace(args.split, args.val_split)

        if args.val_subset is not None:
            val_tokenized_dir = val_tokenized_dir.replace(subset, args.val_subset)
            val_output_dir = val_output_dir.replace(subset, args.val_subset)

        val_subset = args.val_subset if args.val_subset else subset

        val_tokenized_path = os.path.join(val_tokenized_dir, "{}-tokenized.pkl".format(val_subset))
        postfix = "{}-{}".format(args.max_seq_length, args.stride if args.merge else "none")
        val_embed_path = os.path.join(val_output_dir, "{}-{}-embed.float16.npy".format(val_subset, postfix))
        val_loss_path = os.path.join(val_output_dir, "{}-{}-loss.npy".format(val_subset, postfix))

        val_targets = get_flatten_targets(val_tokenized_path, args.val_split)
        val_losses = np.load(val_loss_path)
        assert len(val_targets)==len(val_losses)
        if do_load_embed:
            val_embeds = load_embeds(val_embed_path, len(val_targets), DIMENSION, np.float16)
            assert len(val_targets)==len(val_losses)==len(val_embeds), (len(val_targets), len(val_losses), len(val_embeds))

        if args.do_subset:
            np.random.seed(2023)
            indices = np.random.permutation(range(len(val_targets)))
            indices = indices[:batch_size * 10000]

            val_targets = np.array(val_targets)[indices]
            val_losses = val_losses[indices]
            if do_load_embed:
                val_embeds = np.array(val_embeds)
                val_embeds = val_embeds[indices]

        if do_load_embed:
            return val_targets, val_losses, val_embeds

        return val_targets, val_losses

    if task=="encode":
        if os.path.exists(loss_path) and os.path.exists(embed_path):
            losses = np.load(loss_path)
            if args.do_subset:
                np.random.seed(2023)
                indices = np.random.permutation(range(len(losses)))
                batch_size = 128
                losses = losses[indices[:batch_size * 10000]]

            print ("%s\t# tokens: %d\tPPL: %.1f" % (subset, losses.shape[0], np.exp(np.mean(losses))))
            return

        assert not os.path.exists(embed_path), embed_path

        all_input_ids, all_targets = get_input_ids_and_targets(tokenized_path, split=args.split)
        dstore_size = np.sum(all_targets!=lm.pad_token)
        losses = lm.encode(all_input_ids,
                           all_targets,
                           batch_size=args.batch_size,
                           verbose=True,
                           embed_path=None if args.skip_embed else embed_path)

        np.save(loss_path, losses)
        print ("%s\t# tokens: %d\tPPL: %.3f" % (subset, losses.shape[0], np.exp(np.mean(losses))))

    elif task=="inference":
        assert args.split.startswith("train")

        result_path = os.path.join(args.output_dir, "{}-{}{}-{}-k={}-p={}{}.jsonl".format(
                subset if args.val_subset is None else "{}-from-{}".format(args.val_subset, subset),
                args.val_split,
                "_subset" if args.do_subset else "",
                postfix,
                args.k,
                args.probe,
                "_approx" if args.approximate else ""
        ))

        if not os.path.exists(result_path):
            dstore_targets = get_flatten_targets(tokenized_path, split=args.split)
            dstore_size = len(dstore_targets)
            dstore = DataStore(embed_path=embed_path,
                              index_path=index_path,
                              trained_index_path=trained_index_path,
                              prev_index_path=prev_index_path,
                              prev_embed_paths=prev_embed_paths,
                              dstore_size=dstore_size,
                              dimension=DIMENSION,
                              dtype=np.float16,
                              ncentroids=4096,
                              code_size=64,
                              probe=args.probe)

            if args.remove_embed:
                os.remove(embed_path)

            if prev_index_path and args.remove_prev_index:
                os.remove(prev_index_path)

            if args.skip_eval:
                return

            batch_size = 128
            val_targets, val_losses, val_embeds = get_val_data(batch_size=batch_size, do_load_embed=True)

            lambdas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
            losses = defaultdict(list)
            results = []

            if args.embed_idx > 0:
                dstore_targets = get_flatten_targets(tokenized_path, split=args.split, for_search=True)
            batch_indices = range(math.ceil(len(val_embeds) / batch_size))

            for batch_idx in tqdm(batch_indices):
                start = batch_size * batch_idx
                end = batch_size * (batch_idx+1)
                curr_val_embeds = val_embeds[start:end].astype(np.float32)
                # curr_val_targets = val_targets[start:end]
                # curr_val_losses = val_losses[start:end]

                all_scores, all_indices = dstore.search(curr_val_embeds, k=args.k)

                if args.approximate:
                    all_knn_scores = -all_scores
                else:
                    all_knn_scores = dstore.get_knn_scores(curr_val_embeds, all_indices)

                for (indices, knn_scores) in zip(all_indices, all_knn_scores):
                    results.append({"knn_scores": knn_scores.tolist(), "knn_indices": indices.tolist()})

                if (batch_idx+1) % 100 == 0:
                    with open(result_path, "a+") as f:
                        for r in results:
                            f.write(json.dumps(r) + "\n")
                        results = []

            if len(results) > 0:
                with open(result_path, "a+") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")
                    results = []

        ##### Now, compute perplexity
        ks = [128, 256, 512, 1024, 2048, 4096]
        while ks[-1] < args.k:
            ks.append(ks[-1] * 2)
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
        if not args.approximate:
            temperatures = [0.5, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
        elif "pythia" in args.lm:
            temperatures = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        else:
            temperatures = [1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

        losses = defaultdict(list)
        dstore_targets = get_flatten_targets(tokenized_path, split=args.split, for_search=True)

        print ("Reading %s" % result_path)
        val_targets, val_losses = get_val_data(batch_size=128, do_load_embed=False)
        assert len(val_targets)==len(val_losses)
        offset = 0

        with open(result_path, "r") as f:
            for line in tqdm(f, total=len(val_targets)):
                target = val_targets[offset]
                p_lm = np.exp(-val_losses[offset])
                dp = json.loads(line)
                knn_scores = np.array(dp["knn_scores"])
                indices = np.array(dp["knn_indices"])
                offset += 1
                is_target = dstore_targets[indices]==target

                for k in ks:
                    for temperature in temperatures:
                        knn_probs = softmax(knn_scores[:k] / temperature)
                        p_knn = np.sum(is_target[:k] * knn_probs)
                        for lambda_ in lambdas:
                            if type(lambda_)==float and lambda_==0 and (k!=ks[0] or temperature!=temperatures[0]):
                                continue
                            loss = -np.log(p_lm * (1-lambda_) + p_knn * lambda_)
                            losses[(k, temperature, lambda_)].append(loss)

        aggregated_losses = {k: np.exp(np.mean(v)) for k, v in losses.items()}
        for key, value in sorted(aggregated_losses.items(), key=lambda x: x[1])[:10]:
            print ("%s\tPPL=%.3f" % (key, value))
        print ("-"*20)
        key = (ks[0], temperatures[0], 0)
        print ("%s\tPPL=%.3f" % (key, aggregated_losses[key]))

    else:
        raise NotImplementedError(task)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="encode", choices=["tokenize", "encode", "inference"])
    parser.add_argument("--lm", type=str, default="pythia-1.4B")
    parser.add_argument("--output_dir", type=str, default="out")

    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--val_subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--val_split", type=str, default="val", choices=["val", "test"])

    # encoder related
    parser.add_argument("--merge", action="store_true", default=True)
    parser.add_argument("--max_seq_length", default=1024, type=int)
    parser.add_argument("--stride", default=512, type=int)
    parser.add_argument("--batch_size", default=8, type=int)

    parser.add_argument("--max_n_sequences", default=None, type=int, help="for training (unit: K)")
    parser.add_argument("--embed_idx", default=0, type=int, help="for training")

    # index related
    parser.add_argument("--k", default=4096, type=int)
    parser.add_argument("--probe", default=8, type=int)
    parser.add_argument("--approximate", action="store_true")

    # misc configs
    parser.add_argument("--do_subset", action="store_true")
    parser.add_argument("--skip_embed", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--remove_embed", action="store_true")
    parser.add_argument("--remove_prev_index", action="store_true")


    args = parser.parse_args()
    print (args)
    main(args)
