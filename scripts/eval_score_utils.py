import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import math
import time
import numpy as np
# import statistics
import os
from collections import defaultdict
from pdb import set_trace as bp

Acc = []
'''
test demo
'''
all_scores = []

class EvaluatingWrapper():
    def __init__(self, model, encoder, knn_model, knn_tokenizer, examples, knn_dstore, dstore_targets, args):
        self.model = model
        self.encoder = encoder
        self.vocab = 50432 
        self.knn_model = knn_model
        self.knn_tokenizer = knn_tokenizer
        self.examples = examples
        self.label2synonym = examples[0]["label2synonym"]
        self.label2synonym_id = self.init_label2word_id(self.label2synonym)

        self.labels = examples[0]["label_list"]
        self.labels_id = self.encoder(self.labels)["input_ids"]

        self.label2word = {i: [v] for i, v in enumerate(examples[0]["label_list"])}
        self.label2word_id = self.init_label2word_id(self.label2word)

        self.knn_dstore = knn_dstore
        self.args = args
        self.hist_path = None
        self.max_len = 0
        self.dstore_targets = dstore_targets
        self.inter_lambda = args.inter_lambda
        if self.inter_lambda == 0:
            print("Warning: Using interpolation lamda 0")


    def init_label2word_id(self, label2synonym):
        label2synonym_id = {}
        for k, v in label2synonym.items():
            synonym_id = []
            for word in v:
                if len(self.encoder(word)["input_ids"]) == 1: # check single word
                    synonym_id.append(self.encoder(word)["input_ids"]) # change later

            label2synonym_id[k] = torch.LongTensor(synonym_id).cuda()
        return label2synonym_id


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


    def combine_knn_and_vocab_probs(self, knn_p, vocab_p, knn_lambda):
        # print(self.args.scoring)
        if self.args.scoring.startswith("log_softmax"):
            combine_probs = torch.stack([vocab_p, knn_p], dim=0)
            coeffs = torch.ones_like(combine_probs)
            coeffs[0] = np.log(1 - knn_lambda)
            coeffs[1] = np.log(knn_lambda)
            curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
        else:
            curr_prob = knn_lambda*knn_p + (1-knn_lambda)*vocab_p
        return curr_prob

    def logmeanexp(self, x, dim=0):
        mmax, _ = torch.max(x, dim=dim, keepdim=True)
        return (torch.squeeze(mmax, dim=dim) +
                torch.log(torch.mean(torch.exp((x - mmax)), dim=dim)))
 
    def get_knn_scores(self, outputs, knn_temp):
        queries = outputs['hidden_states'][-1][:, -1, :].cpu().numpy()
        all_scores, all_indices = self.knn_dstore.search(queries, k=self.args.k)
        dists = torch.from_numpy(-all_scores).cuda()  # check this
        knn_ids = np.array(self.dstore_targets[all_indices])

        probs = torch.softmax(dists / knn_temp, dim=-1)
        probs = probs.detach().cpu().numpy()

        probs = probs.squeeze()
        knn_ids = knn_ids.squeeze()
        full_knn_scores = defaultdict(int)
        for vocab, p in zip(knn_ids, probs):
            full_knn_scores[vocab.item()] += p

        label2knn_prob_np = np.zeros((self.vocab, ))
        for label, prob in full_knn_scores.items(): 
            label2knn_prob_np[label] = prob
        return label2knn_prob_np

    def cal_result(self, cond_ce, options, uncond_ce, domain_cond_ce):
        '''
        prediction
        '''
        ## get average CE by token
        avg_cond_ce = [ce / len(opt['hypothesis']) for ce, opt in zip(cond_ce, options)]

        # calculate dcpmi
        dcpmi = [ce_0/ce_1 for ce_0, ce_1 in zip(domain_cond_ce, cond_ce)]
        pmi = [ce_0/ce_1 for ce_0, ce_1 in zip(uncond_ce, cond_ce)]

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


    def cal_score(self, labels, predictions_list):
        # get predictions into list by scoring key
        predictions_dict = {key: list(map(lambda v: v[key], predictions_list)) for key in
                            predictions_list[0].keys()}

        # calculate accuracies
        results = {key: round((sum(list(map(lambda v: v[0] == v[1], zip(predictions_dict[key], labels)))) / len(labels)) * 100, 2) for key in predictions_dict.keys()}

        # save labels for later
        predictions_dict['labels'] = labels
        return results, predictions_dict

    def compute_LM_prob4tokens(self, outputs):
        # compuate softmax
        logits = outputs.logits[:, :].contiguous()
        if self.args.scoring.startswith("logsoftmax_"):
            last_token_softmax = torch.log_softmax(logits[:, -1, :], dim=-1)
        else:
            last_token_softmax = torch.softmax(logits[:, -1, :], dim=-1)
        last_token_softmax = last_token_softmax.squeeze()
        return last_token_softmax


    def eval_one_ex(self, input_texts, knn_input_texts, knn_temp=None, knn_only=False):
        if knn_temp is None:
            knn_temp = self.args.knn_temp

        knn_inputs = self.encoder.encode_plus(knn_input_texts, return_tensors="pt").to("cuda")  # bc assumes same tokenizer?
        knn_input_ids = knn_inputs["input_ids"]

        if len(knn_input_ids[0]) > 1024:
            knn_input_ids = knn_input_ids[0][-1024:]
            knn_input_ids = knn_input_ids.unsqueeze(0)

        with torch.no_grad():
            # get kNN outputs
            knn_outputs = self.knn_model(knn_input_ids, output_hidden_states=True)
            label2knn_prob = self.get_knn_scores(knn_outputs, knn_temp)  # vocab, 1
        
        if not knn_only:
            inputs = self.encoder.encode_plus(input_texts, return_tensors="pt").to("cuda")
            input_ids = inputs["input_ids"]

            # hard-coded to be within 1024 context length
            if len(input_ids[0]) > 1024:  
                input_ids = input_ids[0][-1024:]
                input_ids = input_ids.unsqueeze(0)
            
            with torch.no_grad():
                # get LM outputs
                outputs = self.model(input_ids, output_hidden_states=True)
                label2LM_prob = self.compute_LM_prob4tokens(outputs) # vocab, 1
                label2LM_prob = label2LM_prob.cpu().numpy()
            return label2LM_prob, label2knn_prob
        return None, label2knn_prob
 



    def vocab2label(self, final_prob, label2synonym_id):
        label2knn_prob = np.zeros((len(label2synonym_id), ))
        for label, s_ids in label2synonym_id.items():
            s_ids = s_ids.squeeze(dim=-1)
            for s_id in s_ids:
                prob = final_prob[s_id.item()]
                label2knn_prob[label] += prob
        label2knn_prob = torch.FloatTensor(label2knn_prob)
        return label2knn_prob


    def score(self):
        all_label = []
        all_pred_parametric = []
        all_pred_silo = []

        # compute domain prior
        domain_text = self.examples[0]["options"][0]["uncond_premise"]
        print(f"domain text: {domain_text}")
        domain_label2LM_prob, domain_label2knn_prob = self.eval_one_ex(domain_text, domain_text)
        print(f"domain_label2LM_prob: {domain_label2LM_prob.shape} shape \n {domain_label2LM_prob}")
        print(f"domain_label2knn_prob: {domain_label2knn_prob.shape} shape \n {domain_label2knn_prob}")
        print(f"Using interpolation of {self.inter_lambda}")
        for ex in tqdm(self.examples):
            all_label.append(ex["label"])
            input_text = ex["options"][0]["premise"]
            knn_input_text = ex["options"][0]["knn_premise"]
            label2LM_prob, label2knn_prob = self.eval_one_ex(input_text, knn_input_text) 

            final_prob_pmi = np.log(label2LM_prob+1e-10) - np.log(domain_label2LM_prob+1e-10)
            label2prob_pmi = self.vocab2label(final_prob_pmi, self.label2word_id)
            pred_parametric = torch.argmax(label2prob_pmi).item()
            all_pred_parametric.append(pred_parametric)

            final_prob = self.combine_knn_and_vocab_probs(label2knn_prob, label2LM_prob, self.inter_lambda)
            final_prob_domain = self.combine_knn_and_vocab_probs(domain_label2knn_prob, domain_label2LM_prob, self.inter_lambda)
            final_prob_pmi = np.log(final_prob+1e-10) - np.log(final_prob_domain+1e-10)
            label2prob_pmi = self.vocab2label(final_prob_pmi, self.label2word_id) #self.label2synonym_id)  # ablate fuzzy verbalizer
            pred_silo = torch.argmax(label2prob_pmi).item()
            all_pred_silo.append(pred_silo)


        print("=============Parametric Only========================")
        acc = self.compute_accuracy(all_pred_parametric, all_label)
        print("acc: ", acc)

        print("=============SILO=====================")
        # knnlm

        acc = self.compute_accuracy(all_pred_silo, all_label)
        print("acc: ", acc)
    

    def optimal_config_score(self):
        # define search space
        lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        temperatures = [1.0, 10.0, 20.0, 25.0, 30.0, 40.0]
        
        all_pred_silo = defaultdict(list)      
        all_label = []
        all_pred_parametric = []
        for knn_temp in temperatures:
            # compute domain prior
            domain_text = self.examples[0]["options"][0]["uncond_premise"]
            domain_label2LM_prob, domain_label2knn_prob = self.eval_one_ex(domain_text, domain_text, knn_temp)
            for ex in tqdm(self.examples):
                input_text = ex["options"][0]["premise"]
                knn_input_text = ex["options"][0]["knn_premise"]
                
                if knn_temp != temperatures[0]:
                    # avoid repetitive LM inference
                    assert label2LM_prob is not None
                    _, label2knn_prob = self.eval_one_ex(input_text, knn_input_text, knn_temp, knn_only=True) 
                else:
                    all_label.append(ex["label"])
                    
                    label2LM_prob, label2knn_prob = self.eval_one_ex(input_text, knn_input_text, knn_temp, knn_only=False) 
                    
                    final_prob_pmi = np.log(label2LM_prob+1e-10) - np.log(domain_label2LM_prob+1e-10)
                    label2prob_pmi = self.vocab2label(final_prob_pmi, self.label2word_id)
                    pred_parametric = torch.argmax(label2prob_pmi).item()
                    all_pred_parametric.append(pred_parametric)

                for _lambda in lambda_list:
                    if _lambda == 0.0:
                        continue
                    final_prob = self.combine_knn_and_vocab_probs(label2knn_prob, label2LM_prob, _lambda)
                    final_prob_domain = self.combine_knn_and_vocab_probs(domain_label2knn_prob, domain_label2LM_prob, _lambda)
                    final_prob_pmi = np.log(final_prob+1e-10) - np.log(final_prob_domain+1e-10)
                    label2prob_pmi = self.vocab2label(final_prob_pmi, self.label2word_id) #self.label2synonym_id)  # ablate fuzzy verbalizer
                    pred_silo = torch.argmax(label2prob_pmi).item()
                    all_pred_silo[(_lambda, knn_temp)].append(pred_silo)


        print("=============Parametric Only========================")
        acc = self.compute_accuracy(all_pred_parametric, all_label)
        print("acc: ", acc)

        print("=============SILO=====================")
        # knnlm
        silo_accs = {k: self.compute_accuracy(v, all_label) for k, v in all_pred_silo.items()}
        for key, value in sorted(silo_accs.items(), key=lambda x: x[1])[-10:]:
            print ("%s\tACC=%.3f" % (key, value))
    

    def compute_accuracy(self, all_pred, all_label):
        return round(sum(1 for x, y in zip(all_label, all_pred) if x==y)/len(all_label), 4)
