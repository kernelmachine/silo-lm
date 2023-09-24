import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
import time
import numpy as np
import os
from collections import defaultdict
from utils import get_key, gpt3, cross_entropy_list_gpt3, log_softmax

Acc = []
'''
test demo
'''
all_scores = []

class EvaluatingWrapper():
    def __init__(self, model, encoder, knn_model, knn_tokenizer, examples, knn_dstore, args):
        self.model = model
        self.encoder = encoder
        self.knn_model = knn_model
        self.knn_tokenizer = knn_tokenizer
        self.examples = examples
        self.label2synonym = examples[0]["label2synonym"]
        self.label2synonym_id = {}
        self.init_label2synonym_id()

        self.labels = examples[0]["label_list"]
        self.labels_id = self.encoder(self.labels)["input_ids"]
        # print("labels_id: ", self.labels_id)
        # print("label2synonym_id: ", self.label2synonym_id)

        # assert len(self.label2synonym[0]) == len(self.label2synonym_id[0])
        # assert len(self.label2synonym[1]) == len(self.label2synonym_id[1])

        self.knn_dstore = knn_dstore
        self.args = args
        self.hist_path = None
        self.max_len = 0

    def init_label2synonym_id(self):
        for k, v in self.label2synonym.items():
            synonym_id = []
            for word in v:
                if len(self.encoder(word)["input_ids"]) == 1:
                    synonym_id.append(self.encoder(word)["input_ids"])
            self.label2synonym_id[k] = torch.LongTensor(synonym_id)

        # self.label2synonym_id = {k: torch.LongTensor(self.encoder(v)["input_ids"]).cuda() for k, v in self.label2synonym.items()}



    def load_cache_data(self):
        if "/" in self.args.model:
            model_name = self.args.model.rsplit("/", 1)[1]
        else:
            model_name = self.args.model
        self.hist_path = f'{self.args.output_dir}/{model_name}-{self.args.split}.hist'
        if (not os.path.exists(self.hist_path)) or self.args.load_cache is False:
            cache = {}
            with open(self.hist_path, 'w') as f:
                f.write(json.dumps(cache))
        else:
            MB = os.path.getsize(self.hist_path) / 1000000
            print('=' * 50)
            print('Loading existing cache, size {} MB'.format(MB))
            print('=' * 50)

        with open(self.hist_path, 'r') as f:
            cache = json.loads(f.read())
        cache = defaultdict(list, cache)
        return cache

    def save_cache(self, cache):
        print('=' * 50)
        print('saving cache to {}'.format(self.hist_path))
        print('=' * 50)
        with open(self.hist_path, 'w') as f:
            f.write(json.dumps(cache))

    def save_score(self, klambda2result, klambda2predictions_list):
        # save scores
        results_path = f'{self.args.output_dir}/{self.args.split}.accs'
        with open(results_path, 'w') as out:
            out.write(json.dumps(klambda2result))

        # save predicted labels
        preds_path = f'{self.args.output_dir}/{self.args.split}.preds'
        with open(preds_path, 'w') as out:
            out.write(json.dumps(klambda2predictions_list))

    def print_overview(self):
        # print the first example to make sure the format is ok
        print('=' * 50)
        print('MAKE SURE TOKENIZATION AND FORMATTING LOOKS OK')
        print('\nprint example 0 of {}:'.format(len(self.examples)))
        ex = self.examples[0]
        options = ex['options']
        opt = options[0]
        print('CONDITIONAL:')
        print(self.encoder.decode(self.encoder.encode(opt['premise'])) + '<BREAK>' + self.encoder.decode(
            self.encoder.encode(opt['hypothesis'])))
        print('UNCONDITIONAL:')
        print(self.encoder.decode(self.encoder.encode(opt['uncond_premise'])) + '<BREAK>' + self.encoder.decode(
            self.encoder.encode(opt['uncond_hypothesis'])))
        print('=' * 50)

    # def combine_knn_and_vocab_probs(self, knn_p, vocab_p, knn_lambda):
    #     combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    #     coeffs = torch.ones_like(combine_probs)
    #     coeffs[0] = np.log(1 - knn_lambda)
    #     coeffs[1] = np.log(knn_lambda)
    #     curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
    #     return curr_prob

    def combine_knn_and_vocab_probs(self, knn_p, vocab_p, knn_lambda):
        # knn_p, vocab_p = -knn_p, -vocab_p
        # print("knn_p, vocab_p: ", knn_p, vocab_p)
        # vocab_p = torch.softmax(vocab_p/15, dim=0)
        # knn_p = torch.softmax(knn_p, dim=0)
        # print("knn_p, vocab_p: ", knn_p, vocab_p)
        combine_probs = (1 - knn_lambda)*vocab_p + knn_lambda*knn_p
        return combine_probs

    def get_knn_scores(self, outputs, all_lens, targets, gold_labels=None, sources=None):
        def dist_func(d, k, q):
            # Default behavior for L2 metric is to recompute distances, Default behavior for IP metric is to return faiss distances.
            qsize = q.shape
            if self.args.sim_func == 'l2':
                start = time.time()
                knns_vecs = torch.from_numpy(self.knn_dstore.keys[k]).cuda().view(qsize[0], self.args.k, -1)
                if self.half:
                    knns_vecs = knns_vecs.half()
                query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
                return -1 * l2 # negative distance, higher better
            elif self.args.sim_func == 'do_not_recomp_l2':
                return -1 * d
            elif self.args.sim_func == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.knn_dstore.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

        all_lens = torch.LongTensor(all_lens).cuda()
        keys = outputs['hidden_states'][-1]
        queries = keys[torch.arange(keys.shape[0]), all_lens - 1]

        dists, knns = self.knn_dstore.get_knns(queries) # smaller --> better

        knn_ids = self.knn_dstore.vals[knns].squeeze()

        dists = torch.from_numpy(dists).cuda()
        # dists = dist_func(dists, knns, queries) # negative, dists: larger -> better
        dists.div_(self.args.knn_temp) # control the peakness

        probs = log_softmax(dists, dim=-1)

        # extract token id
        indices = torch.from_numpy(self.knn_dstore.vals[knns]).long().cuda()
        indices = indices.view(queries.shape[0], self.args.k)

        unique_indices, mapping = torch.unique(indices, return_inverse=True)
        knn_scores_by_index = torch.ones([indices.shape[0], indices.shape[1], len(unique_indices)]).cuda()

        knn_scores_by_index[:] = 0  # -math.inf
        knn_vals_by_index = torch.ones([indices.shape[0], indices.shape[1], len(unique_indices)]).long().cuda()
        knn_vals_by_index[:] = 0

        indices = indices.unsqueeze(2)
        probs = probs.unsqueeze(2)
        mapping = mapping.unsqueeze(2)
        knn_scores_by_index.scatter_(dim=2, index=mapping, src=probs) # batch x k x unique_index
        knn_vals_by_index.scatter_(dim=2, index=mapping, src=indices)
        # (Bxbeam)xn
        knn_scores_by_index = knn_scores_by_index.sum(dim=1) # after sum: batch x unique_index
        knn_vals_by_index = knn_vals_by_index.max(dim=1)[0] # batch x unique_index: val in vocab if mentioned, otherwise -1
        full_knn_scores = torch.ones([queries.shape[0], self.encoder.vocab_size]).cuda() # batch x vocab
        full_knn_scores[:] = 0  # -math.inf
        full_knn_scores.scatter_(dim=1, index=knn_vals_by_index, src=knn_scores_by_index)
        # full_knn_scores = torch.softmax(full_knn_scores, dim=-1) # swj: add softmax
        # change back to index 0 values to -10000 full_knn_scores[0, :] = -10000

        # extract score for labels
        targets = torch.LongTensor(targets)
        assert targets.shape[1] == 1 # one token for label
        label_knn_scores = full_knn_scores[torch.arange(full_knn_scores.shape[0]), targets.squeeze()] # score for label word

        # extract score for synonyms
        synonym_knn_scores = []
        for i, t in enumerate(targets):
            '''
            debug
            '''
            # print("knn_ids[i, :]: ", knn_ids[i, :])
            # decode_output = self.encoder.decode(knn_ids[i, :])
            # decode_list = decode_output.split()
            # label_num_list = {}
            # for k, v in self.label2synonym.items():
            #     label_num = 0
            #     for l in v:
            #         label_num += decode_list.count(l.strip())
            #     label_num_list[k] = label_num
            # print(f"label: {gold_labels[i]} | ", f"{label_num_list} | {decode_output}")
            # print("debug t: ", t)
            # print("debug self.labels_id: ",  self.labels_id)
            g_index = self.labels_id.index(t.cpu().tolist())
            synonym_tensor = self.label2synonym_id[g_index]
            synonym_score = torch.mean(full_knn_scores[i, synonym_tensor]) # do not use sum for that
            # synonym_score = bad_num if g_index is 0 else good_num
            synonym_knn_scores.append(synonym_score)

        synonym_knn_scores = torch.FloatTensor(synonym_knn_scores)
        # label_knn_scores = -synonym_knn_scores # score for synonyms
        label_knn_scores = synonym_knn_scores
        return full_knn_scores, label_knn_scores


    def cal_result(self, cond_ce, options, uncond_ce, domain_cond_ce):
        '''
        prediction
        '''
        ## get average CE by token
        avg_cond_ce = [ce / len(opt['hypothesis']) for ce, opt in zip(cond_ce, options)]

        # calculate dcpmi
        dcpmi = [ce_0-ce_1 for ce_0, ce_1 in zip(domain_cond_ce, cond_ce)]
        pmi = [ce_0-ce_1 for ce_0, ce_1 in zip(uncond_ce, cond_ce)]

        ## make predictions based on different scores
        lm_pred = cond_ce.index(min(cond_ce))
        lm_avg_pred = avg_cond_ce.index(min(avg_cond_ce))
        lm_domain_cond_pred = domain_cond_ce.index(min(domain_cond_ce))
        dcpmi_pred = dcpmi.index(max(dcpmi))
        pmi_pred = pmi.index(max(pmi))
        pred = {
            'lm': lm_pred,
            'tok_mean': lm_avg_pred,
            'dcpmi': dcpmi_pred,
            'pmi': pmi_pred,
            'domain_cond': lm_domain_cond_pred,
        }
        return pred

    def cross_entropy_list(self, sources, targets, gold_labels, cache=None, batch=False, calculate=True):
        '''
        Gets a list of CE values, where the ith item is a list of cross-entropies
        for targets[i] with sources[i] as contexts
        targets and sources are lists of lists of tokens (integers)
        model is a language model
        batch is the batch size to break things up into, batch=False means don't
        break things up into batches, do them all in one go.

        CACHING:

        cache is a dictionary for single source/target pairs
          accessed by cache[get_key(source,target)]
          it has fields source, target, result

        calculate decides whether to immediates calculate for batch of input
          sources/targets or just log them as todo in the cache. To efficiently
          batch, we can first log many todo calculations by calling cross_entropy_list
          multiple times with calculate=False and the same input cache
          Then finally calling it with calculate=True which will then catch up on all
          todo calculations, caching them together efficiently

        '''

        def prepare_data(sources, targets):
            '''
            actual calculations
            '''
            max_len = max([len(s + t) for s, t in zip(sources, targets)])
            input_ids = torch.zeros((n_seqs, max_len)).long()
            # -100 is the padding token, which is ignored by F.cross_entropy below
            labels = -100 * torch.ones((n_seqs, max_len)).long()

            all_lens = []
            # for each source, target pair, set values in the input tensors
            for i, (source, target) in enumerate(zip(sources, targets)):
                s = torch.tensor(source).long()
                t = torch.tensor(target).long()
                input_ids[i, :len(s)] = s
                input_ids[i, len(s):len(s) + len(t)] = t
                # ignore all predictions except in the target span
                labels[i, len(s):len(s) + len(t)] = t
                all_lens.append(len(s))
            return input_ids, labels, all_lens, max_len

        ###############################
        # This block handles caching of results (LAZY EVALUATION)
        # this is useful for efficient batching. First, add all todo
        # calculations to the cache with calculate = False (won't do them yet)
        # then run with calculate=True to work through all cached calculations
        # in efficient batches
        if cache is not None:

            # log calculations we have not done yet
            for source, target, gold_label in zip(sources, targets, gold_labels):
                if get_key(source, target) not in cache:
                    cache[get_key(source, target)] = {'source': source, 'target': target, 'gold_label': gold_label, 'result': None}

            # if not calculating, return dummy values
            if not calculate:
                return None

            # if caching and calculating, we calculate for all examples
            # that have been cached but not calculated
            cache_todo = [(v['source'], v['target'], v['gold_label']) for v in cache.values() if "result_0.0" not in v]

            ## if there are calculations to do, do them
            if len(cache_todo) > 0:
                sources_todo = list(zip(*cache_todo))[0]
                targets_todo = list(zip(*cache_todo))[1]
                labels_todo = list(zip(*cache_todo))[2]
                cache_results = self.cross_entropy_list(sources_todo, targets_todo, labels_todo, cache=None, batch=batch)
                for k, v in cache_results.items():
                    for source, target, result in zip(sources_todo, targets_todo, v):
                        cache[get_key(source, target)][f'result_{k}'] = result
            klambda2results = defaultdict(list)
            for knn_lambda in np.linspace(0, 1, 21):
                ## return results for this example
                for source, target in zip(sources, targets):
                    klambda2results[knn_lambda].append(cache[get_key(source, target)][f'result_{knn_lambda}'])
            return klambda2results

        ###############################

        assert (len(sources) == len(targets))
        n_seqs = len(sources)

        torch.cuda.empty_cache()
        device = self.model.transformer.wte.weight.device

        # if batching, break it up into smaller pieces
        if batch:
            klambda2ce_list = defaultdict(list)

            n_batches = math.ceil(len(sources) / batch)
            list_fun = (lambda v: tqdm(list(v))) if cache is not None else list

            for i in tqdm(list(range(n_batches))):
                klambda2prob = self.cross_entropy_list(sources[i * batch:(i + 1) * batch], targets[i * batch:(i + 1) * batch], gold_labels[i * batch:(i + 1) * batch], batch=False)
                for k, v in klambda2prob.items():
                    klambda2ce_list[k] += v
            return klambda2ce_list

            # initialize input tensors
        input_ids, labels, all_lens, max_len = prepare_data(sources, targets)
        # print("all_lens, labels, input_ids: ", all_lens, labels, input_ids)
        # get logits from the model
        with torch.no_grad():
            input_ids = input_ids.to(device)
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            if self.args.model == self.args.knn_model:
                knn_outputs = outputs
            else:
                knn_outputs = self.knn_model(input_ids=input_ids, output_hidden_states=True)
            # knn score: not efficient for multichoice, only support one token label
            full_knn_scores, label_knn_scores = self.get_knn_scores(knn_outputs, all_lens, targets, gold_labels=gold_labels, sources=sources) # 10xvocab size

            # get LM prob
            logits = outputs.logits.cpu()[:, :-1].contiguous()  # remove last token

        # all_scores.append(sum_score)
        # get cross-entropies given the logits
        logit_shape = logits.shape
        logits = logits.view(-1, logit_shape[-1])
        ce_list = F.cross_entropy(logits, labels[:, 1:].contiguous().view(-1), reduction='none')
        ce_list = ce_list.view(n_seqs, max_len - 1).sum(dim=1, keepdim=True) # cross entropy list: the lower the better
        # label_knn_scores = label_knn_scores * self.args.knn_score_factor # after devide 2000, no much difference
        klambda2prob = {}
        for knn_lambda in np.linspace(0, 1, 21):
            curr_prob = self.combine_knn_and_vocab_probs(knn_p=label_knn_scores.cpu().unsqueeze(dim=1), vocab_p=ce_list, knn_lambda=knn_lambda)
            curr_prob = curr_prob.squeeze().tolist()
            klambda2prob[knn_lambda] = curr_prob
        return klambda2prob

    def inference_autobatch(self, example, prelog=False, cache=None):
        '''

               if prelog is true, then we're just logging calculations to do in one big batch calculate
               (used for caching)
               if we are just prelogging cross entropy calculations to do later,
               we will set caclulate=False for cross_entropy_list and it will output
               a dummy value for now and just log calculations to do. Then the output
               of inference_autobatch will not be correct, calling it in this case is
               just to log calculations to do in big batches
            '''
        def setup(prelog, cache):
            calculate = False if prelog and (cache is not None) else True
            #####
            ## input data handling
            #####
            # i.e. if we're using GPT-3 through the OpenAI API
            if type(self.model) == str:
                self.max_len = 2048
                gpt3 = True
            else:
                self.max_len = 1024
                gpt3 = False
            return gpt3, calculate

        def encode_options():
            options = []
            for opt_raw in example['options']:
                label = example["label"]
                if gpt3:
                    options.append(opt_raw)
                else:
                    # first, encode the option
                    opt = {key: self.encoder.encode(opt_raw[key]) for key in opt_raw.keys()}
                    opt["label"] = label
                    ## trim the option to the max length for gpt2
                    opt['premise'] = opt['premise'][-(self.max_len - len(opt['hypothesis'])):]
                    assert (len(opt['premise'] + opt['hypothesis']) <= self.max_len)
                    # then add the encoded, trimmed option
                    options.append(opt)
            return options

        gpt3, calculate = setup(prelog, cache)
        options = encode_options()

        #####
        ## cross-entropy calculation
        #####
        ## get conditional CEs
        cond_ce = self.cross_entropy_list([opt['premise'] for opt in options],
                                     [opt['hypothesis'] for opt in options],
                                     [opt['label'] for opt in options],
                                      cache=cache, batch=self.args.batch_size, calculate=calculate)
        ## get domain conditional CEs
        domain_cond_ce = self.cross_entropy_list([opt['uncond_premise'] for opt in options],
                                            [opt['uncond_hypothesis'] for opt in options],
                                            [opt['label'] for opt in options],
                                            cache=cache, batch=self.args.batch_size, calculate=calculate)

        ## get unconditional CEs
        uncond_ce = self.cross_entropy_list([[25] for opt in options],
                                       [opt['uncond_hypothesis'] for opt in options],
                                       [opt['label'] for opt in options],
                                       cache=cache, batch=self.args.batch_size, calculate=calculate)

        klambda2pred = {}
        for knn_lambda in np.linspace(0, 1, 21):
            if cond_ce is None:
                pred = None
            else:
                pred = self.cal_result(cond_ce[knn_lambda], options, uncond_ce[knn_lambda], domain_cond_ce[knn_lambda])
            klambda2pred[knn_lambda] = pred
        return klambda2pred

    def cal_score(self, labels, predictions_list):
        # get predictions into list by scoring key
        predictions_dict = {key: list(map(lambda v: v[key], predictions_list)) for key in
                            predictions_list[0].keys()}

        # calculate accuracies
        results = {key: round((sum(list(map(lambda v: v[0] == v[1], zip(predictions_dict[key], labels)))) / len(labels)) * 100, 2) for
                   key
                   in
                   predictions_dict.keys()}

        # save labels for later
        predictions_dict['labels'] = labels
        return results, predictions_dict

    def fwd(self, cache=None):
        '''
        This is designed for gpt2-style language models

        Inputs: (any you don't know)
            model - a HuggingFace Transformers gpt-2 model
            encoder - a HuggingFace Transformers tokenizer
            examples = [ex1, ex2, ...]
                where ex = [opt1, opt2, ...] (multiple choice options)
                where opt = (premise, hypothesis)

            batch: is the max allowed batch size (set to 1 for no batching)
        '''
        predictions_list = []
        ## in this loop, prelog is set to true so we are just logging cross_entropy_list calculations
        ## but not doing them yet
        if cache is not None:
            # print('logging examples')
            for example in tqdm(self.examples):
                _ = self.inference_autobatch(example, prelog=True, cache=cache)
        ## in this loop, we actually do the calculations from above in efficient batches, storing results
        ## in the cache and calculating actual predictions
        # print('actually calculating')
        klambda2predictions_list = defaultdict(list)
        for example in tqdm(self.examples):
            klambda2pred = self.inference_autobatch(example, prelog=False, cache=cache)
            for k, v in klambda2pred.items():
                klambda2predictions_list[k].append(v)

        labels = [ex['label'] for ex in self.examples]

        klambda2result = {}
        klambda2predictions_dict = {}
        for k, v in klambda2predictions_list.items():
            results, predictions_dict = self.cal_score(labels, v)
            klambda2result[k] = results
            klambda2predictions_dict[k] = predictions_dict
        return klambda2result, klambda2predictions_dict

    def score(self):
        cache = self.load_cache_data()
        klambda2result, klambda2predictions_list = self.fwd(cache)
        for k, v in klambda2result.items():
            print("{:.2f}".format(k) , v)

        self.save_cache(cache)
        self.save_score(klambda2result, klambda2predictions_list)



