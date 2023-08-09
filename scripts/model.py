import numpy as np
import torch
import time
import re
import os

from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GPTNeoXTokenizerFast

from utils.transformers.model import OpenLMforCausalLM

def batch_merged(flatten_input_ids, max_seq_length, stride, pad_token, flatten_masks=None):
    all_input_ids = []
    all_targets = []
    prev_end_loc = 0

    for begin_loc in range(0, len(flatten_input_ids)-1, stride):
        end_loc = min(begin_loc + max_seq_length, len(flatten_input_ids)-1)
        trg_len = end_loc - prev_end_loc

        # we feed begin_loc ~ prev_end_log ~ end_log
        # but calculcate loss only for prev_end_log ~ end_log
        input_ids = flatten_input_ids[begin_loc:end_loc].copy()
        target_ids = flatten_input_ids[begin_loc+1:end_loc+1].copy()

        if flatten_masks is not None:
            for i, m in enumerate(flatten_masks[begin_loc+1:end_loc+1]):
                if not m:
                    target_ids[i] = pad_token

        target_ids[:-trg_len] = pad_token
        assert input_ids.shape==target_ids.shape

        if end_loc == len(flatten_input_ids)-1 and len(input_ids)==len(target_ids)<max_seq_length:
            pads = np.array([pad_token for _ in range(max_seq_length-len(input_ids))])
            input_ids = np.concatenate([input_ids, pads])
            target_ids = np.concatenate([target_ids, pads])

        assert len(input_ids)==len(target_ids)==max_seq_length, (begin_loc, end_loc, len(flatten_input_ids))

        all_input_ids.append(input_ids)
        all_targets.append(target_ids)

        prev_end_loc = end_loc

        if end_loc == len(flatten_input_ids)-1:
            break

    assert np.all([len(input_ids)==max_seq_length for input_ids in all_input_ids])
    assert np.all([len(input_ids)==max_seq_length for input_ids in all_targets])
    return np.stack(all_input_ids), np.stack(all_targets)

def get_masks(all_input_ids, texts, tokenizer):

    def replace_text_with_placeholder(input_string):
        # Define the pattern to match
        pattern = r"\[\*\*(.*?)\*\*\]"

        # Use regular expression to find all matches
        matches = re.findall(pattern, input_string)

        # Replace each match with <PLACEHOLDER>
        for match in matches:
            input_string = input_string.replace('[**{}**]'.format(match), tokenizer.eos_token)

        return input_string

    all_masked_input_ids =  tokenizer([replace_text_with_placeholder(text) for text in texts])["input_ids"]

    masks, mask_ratio = [], []
    for input_ids, masked_input_ids, text in zip(tqdm(all_input_ids), all_masked_input_ids, texts):

        mask = []
        offset = 0
        i = 0
        while i < len(input_ids):
            _id = input_ids[i]
            if _id == masked_input_ids[offset]:
                assert _id > 0
                mask.append(1)
                i += 1
                offset += 1
            else:
                # masked_input_ids[offset] should be 0, but sometimes it's there later
                while masked_input_ids[offset] != 0:
                    offset += 1
                    if offset+1 >= len(masked_input_ids):
                        # its highly likely it has something messed up
                        # just break
                        while i < len(input_ids):
                            mask.append(0)
                            i += 1
                        break

                if offset+1>=len(masked_input_ids):
                    # this is the end of the sequence
                    while i < len(input_ids):
                        mask.append(0)
                        i += 1
                    break

                while input_ids[i] != masked_input_ids[offset+1]:
                    mask.append(0)
                    i += 1
                    if i==len(input_ids):
                        break

                if i==len(input_ids):
                    break

                mask.append(1)
                i += 1
                offset += 2

            if offset == len(masked_input_ids):
                while i < len(input_ids):
                    mask.append(0)
                    i += 1
                break

        assert len(mask)==len(input_ids), (len(mask), len(input_ids))
        masks.append(mask)
        mask_ratio.append(np.sum(mask) / len(mask))

    print ("Masking ratio = %.1f%% (token-level) %.1f%% (sequence-level)" % (
        100*np.mean(mask_ratio),
        100*np.mean([np.sum(mask)>0 for mask in masks])
    ))
    # from IPython import embed; embed(); exit()
    return masks

class LanguageModel(object):

    def __init__(self, name):
        self.name = name
        self.model = None

        if self.name == "neo-1.3B":
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            self.pad_token = self.tokenizer.eos_token_id
        elif self.name == "pythia-1.4B":
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped-v0")
            self.pad_token = self.tokenizer.eos_token_id
        elif self.name == "pythia-6.9B":
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped-v0")
            self.pad_token = self.tokenizer.eos_token_id
        else:
            self.tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
            self.pad_token = self.tokenizer.eos_token_id

    def load_model(self, cuda=True):

        if cuda and not torch.cuda.is_available():
            raise NotImplementedError("No CUDA found.")

        if self.name == "neo-1.3B":
            self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
            self.dimension = self.model.config.hidden_size
        elif self.name == "pythia-1.4B":
            self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b-deduped-v0")
            self.dimension = self.model.config.hidden_size
        elif self.name == "pythia-6.9B":
            self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-6.9b-deduped-v0")
            self.dimension = self.model.config.hidden_size
        elif self.name.startswith("silo-"):
            self.model = OpenLMforCausalLM.from_pretrained("kernelmachine/" + self.name)
            self.dimension = self.model.config.hidden_dim
        else:
            model_dir = "ckpt/" + self.name
            assert os.path.isdir(model_dir), model_dir
            self.model = AutoModelforCausalLM.from_pretrained(model_dir)
            self.dimension = self.model.config.hidden_dim

        if cuda:
            self.model.cuda()
        self.model.eval()
        print ("Loaded %s with %.1fB parameters" % (
                self.name, np.sum([p.numel() for p in self.model.parameters()])/1000000000))

    def output_prob_from_embed(self, h):
        if self.name == "pythia-1.4B":
            return self.model.embed_out(h)
        else:
            return self.model.model.output(h)

    def tokenize(self, texts, print_statistics=False, handle_pii=False):
        all_input_ids = self.tokenizer(texts)["input_ids"]

        if handle_pii:
            masks = get_masks(all_input_ids, texts, self.tokenizer)

            final_input_ids, final_masks = [], []
            for input_ids, mask in zip(all_input_ids, masks):
                if np.mean(mask) >= 0.5:
                    final_input_ids.append(input_ids)
                    final_masks.append(mask)

            print ("%d -> %d sequences (%d%%) with >=50%% mask ratio" % (
                len(all_input_ids), len(final_input_ids), 100*len(final_input_ids)/len(all_input_ids)
            ))

            return final_input_ids, final_masks

        if print_statistics:
            n_tokens = [len(_ids) for _ids in all_input_ids]
            print ("Avg %d tokens, Median %d tokens" % (np.mean(n_tokens), np.median(n_tokens)))

        return all_input_ids

    def batch(self, input_ids, max_seq_length,stride):
        all_input_ids, all_targets = [], []
        for _input_ids in input_ids:
            _all_input_ids, _all_targets = batch_merged(_input_ids, max_seq_length, stride, self.pad_token)
            all_input_ids.append(_all_input_ids)
            all_targets.append(_all_targets)
        return np.concatenate(all_input_ids, 0), np.concatenate(all_targets, 0)

    def batch_merged(self, flatten_input_ids, max_seq_length, stride, flatten_masks=None):
        return batch_merged(flatten_input_ids, max_seq_length, stride, self.pad_token, flatten_masks=flatten_masks)

    def encode(self,
               all_input_ids,
               all_targets,
               batch_size,
               verbose=False,
               embed_path=None):

        if self.model is None:
            self.load_model()

        dataloader = get_dataloader(all_input_ids, all_targets, batch_size=batch_size)
        if verbose:
            print ("Encoding %d sequences with %d batches" % (len(all_input_ids), len(dataloader)))
            dataloader = tqdm(dataloader)

        m = torch.nn.LogSoftmax(dim=-1)
        nll = torch.nn.NLLLoss(reduction='none')

        skip_embed = embed_path is None

        if embed_path is not None:
            dstore_size = np.sum(all_targets!=self.pad_token)
            dimension = self.dimension
            all_embeds_memmap = np.memmap(embed_path, dtype=np.float16, mode="w+", shape=(dstore_size, dimension))

        if not skip_embed:
            all_embeds = []

        all_losses = []

        offset = 0
        for input_ids, targets in dataloader:
            with torch.no_grad():
                input_ids = input_ids.cuda()
                targets = targets.cuda()
                outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)

                logits = outputs.logits # [batch_size, max_seq_length, n_vocabs]
                hidden_states = outputs.hidden_states[-1] # [batch_size, max_seq_length, hidden_size]

                logits = logits.reshape(-1, logits.shape[-1])
                # targets = input_ids[:, 1:].reshape(-1)
                targets = targets.reshape(-1)

                losses = nll(m(logits), targets)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

                assert len(hidden_states)==len(losses)==len(targets)
                targets = targets.detach().cpu().numpy()
                hidden_states = hidden_states.detach().cpu().numpy()
                losses = losses.detach().cpu().numpy()

                for target, hidden, loss in zip(targets, hidden_states, losses):
                    if target!=self.pad_token:
                        all_losses.append(loss)

                        if not skip_embed:
                            all_embeds.append(hidden)

                        if embed_path is not None and len(all_embeds) == 1000000:
                            all_embeds_memmap[offset:offset+len(all_embeds), :] = all_embeds
                            offset += len(all_embeds)
                            all_embeds = []

        if skip_embed:
            return np.array(all_losses)

        if embed_path is not None:
            all_embeds_memmap[offset:offset+len(all_embeds), :] = all_embeds
            offset += len(all_embeds)

        if embed_path is not None:
            assert offset==dstore_size
            return np.array(all_losses)

        raise NotImplementedError()

def get_dataloader(all_input_ids, all_targets, batch_size):
    all_input_ids = torch.LongTensor(all_input_ids)
    all_targets = torch.LongTensor(all_targets)
    dataset = TensorDataset(all_input_ids, all_targets)
    sampler=SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
