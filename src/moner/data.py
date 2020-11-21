from os import pardir
from torch import is_storage
from torch.utils.data import Dataset, DataLoader
from itertools import chain
import torch
import numpy as np
from dotmap import DotMap

def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item

DEFAULT_TOKENIZER = DotMap({
    'tokenize': lambda x: x.split()
    }
)

class NERMultiOutput(Dataset):
    """NER Multi Output Dataset."""

    def __init__(self, 
                filepath, 
                tokenizer=DEFAULT_TOKENIZER,
                sep='\t', 
                num_labels=1, 
                label_scheme='BILUO', 
                attention_to_start_end=True, 
                max_length=128
        ):
        """
        Args:
            filepath (string): Path to CONLL2003 formatted sequence tagging dataset file.
            sep (string, optional): Character separating tokens and labels
            n_labels (int, optional): Number of labels in the dataset 
            label_scheme (string, optional): Sequential labeling scheme
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.filepath = filepath
        self.sep = sep
        self.max_length = max_length
        self.num_labels = num_labels
        self.label_scheme = label_scheme
        self.OUTSIDE = self.label_scheme[-1]
        self.tokenizer = tokenizer
        self.start_end_attn_val = int(attention_to_start_end)

        self.tokens, self.labels = self._parse_conll()

        self._get_label_indexes()
        self._encode_label_indexes()
        self.all_outside = self._get_all_outside()

        #self.tokens, self.token_starts, self.token_lengths = zip(*[self._get_token_starts(t) for t in self.tokens])
        #_, self.token_starts, self.token_lengths = zip(*[self._get_token_starts(t) for t in self.tokens])

        #self._get_label_indexes()
        #self._encode_label_indexes()
        #self.all_outside = self._get_all_outside()

        #self.labels_aligned = [self._align_tokens_labels(labels, lengths) for labels, lengths in zip(self.labels_encoded, self.token_lengths)]
        #self.labels_start_end = [[self.all_outside] + l + [self.all_outside] for l in self.labels_aligned]

        #self.token_input = [self.tokenizer([tok], is_split_into_words=True, padding='max_length', max_length=self.max_length) for tok in self.tokens]

        #self.subword_attention_mask = [[self.start_end_attn_val] + i + [self.start_end_attn_val] for i in self.token_starts]

        #for i, mask, labels in zip(self.token_input, self.subword_attention_mask, self.labels_start_end):
        #    i['attention_mask'] = mask
        #    i['label'] = labels

        #self.token_input = [
        #    {
        #        k: torch.LongTensor(v).to(self.device).unsqueeze(0)
        #        for k, v in i.items()
        #    } 
        #    for i in self.token_input
        #]


    def _parse_conll(self):
        """
        Parse the raw text of a CONLL2003 formatted seq dataset
        """
        raw = open(self.filepath).read().split('\n\n')
        #raw = [i.strip('"') for i in raw]
        lines = [[i.split(self.sep) for i in r.split('\n') if i] for r in raw]
        lines = [[i for i in j if len(i[1:]) == self.num_labels] for j in lines]
        lines = [[ll for ll in l if len(ll[0]) > 0] for l in lines]
        tokens = [[i[0] for i in j] for j in lines]
        labels = [[i[1:] for i in j] for j in lines]

        tokens, labels = zip(*[(t, l) for t, l in zip(tokens, labels)])

        return tokens, labels

    def _get_all_outside(self):
        all_outside = []
        for idx, lkp in self.label2idx.items():
            all_outside.append(lkp[self.OUTSIDE])
        return all_outside

    def _get_label_indexes(self):

        self.label2idx = {}

        complete_labels = list(zip(*chain.from_iterable(self.labels)))
        for it, channel in enumerate(complete_labels):
            # Dumb fix for null and '-' showing up in labels
            channel = (i for i in set(channel) if i not in ['', '-'])
            self.label2idx[it] = {label: i for i, label in enumerate(channel)}

        self.idx2label = {
                    k: {vv:kk for kk, vv in v.items()}
                    for k, v in self.label2idx.items()
                }

    def _encode_label_indexes(self):

        self.labels_encoded = []

        for l in self.labels:
            l_out = []
            channel_labels = zip(*l)
            for channel_it, ll in enumerate(channel_labels):
                lkp = self.label2idx[channel_it]
                ll = [lll if lll in lkp.keys() else self.OUTSIDE for lll in ll]
                ll = [lkp[lll] for lll in ll]

                l_out.append(ll)
            l_out = list(zip(*l_out))

            self.labels_encoded.append(l_out)

    def _get_token_starts(self, tokens):

        subword_tokenized = [self.tokenizer.tokenize(w) for w in tokens]
        #subword_tokenized = [sw for sw in subword_tokenized if sw]
        flattened_subword_tokenized = list(chain.from_iterable(subword_tokenized))
        token_starts = list(chain.from_iterable([[1] + [0]*(len(sw)-1) for sw in subword_tokenized]))
        token_lengths = [len(sw) for sw in subword_tokenized]
        return flattened_subword_tokenized, subword_tokenized, token_starts, token_lengths

        #subword_tokenized = list(map(self.tokenizer.tokenize, tokens))
        #token_lengths = list(map(len, subword_tokenized))
        #flattened_subword_tokenized = list(flatten(subword_tokenized))
        #token_start_idxs = 1 + np.cumsum([0] + token_lengths[:-1])
        #token_starts = [1 if it in token_start_idxs else 0 for it, _ in range(len(flattened_subword_tokenized))]
        #return flattened_subword_tokenized, subword_tokenized, token_starts, token_lengths



    def _align_tokens_labels(self, labels_encoded, token_lengths):
        return list(chain.from_iterable([[lab] * l for lab, l in zip(labels_encoded, token_lengths)]))

    def __len__(self):
        
        return len(self.tokens)
    
    def __getitem__(self, idx):

        #return {
        #    k: torch.LongTensor(v).unsqueeze(0)
        #    for k, v in self.token_input[idx].items()
        #}
        tokens = self.tokens[idx]
        tokens_, subword_tokenized, token_starts, token_lengths = self._get_token_starts(tokens)
        
        labels = self.labels_encoded[idx]

        labels_aligned = self._align_tokens_labels(labels, token_lengths)
        labels_start_end = [self.all_outside] + labels_aligned[:self.max_length-2] + [self.all_outside]
        labels_start_end_padded = labels_start_end + (self.max_length - len(labels_start_end)) * [self.all_outside] 

        #subword_attention_mask = [self.start_end_attn_val] + token_starts + [self.start_end_attn_val]
        subword_attention_mask = [self.start_end_attn_val] + token_starts[:self.max_length-2] + [self.start_end_attn_val]
        subword_attention_mask_padded = subword_attention_mask + (self.max_length - len(labels_start_end)) * [0] 
    
        input_encoded = self.tokenizer(
                                [tokens], 
                                is_split_into_words=True, 
                                padding='max_length', 
                                truncation=True,
                                max_length=self.max_length)

        input_encoded['attention_mask'] = [subword_attention_mask_padded]
        input_encoded['labels'] = [labels_start_end_padded]

        return {
            k: torch.LongTensor(v).squeeze(0)
            for k, v in input_encoded.items()
        }

        return {
            'tokens_': tokens_, 
            'tokens': self.tokens[idx],
            'token_starts': token_starts, 
            'token_lengths': token_lengths,
            'subword_tokenized': subword_tokenized, 
            'labels': self.labels[idx], 
            'labels_aligned': labels_aligned, 
            'labels_start_end': labels_start_end
        }, a
        #return self.token_input[idx]
        

