from transformers import (
    AutoTokenizer, AutoModel, 
    AutoModelWithLMHead, AutoModelForCausalLM, AutoModelForMaskedLM
    )
import torch
import torch.nn as nn
from masked_bert import MaskedWordLevelBert
from masked_linear import MaskedLinear, L0Mask
from bert import unpack
from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np
import itertools
import logging
from pathlib import Path
from util import use_cuda, from_numpy


LOGPATH = Path("../log")
if not LOGPATH.exists():
    LOGPATH.parent.mkdir(parents=True, exist_ok=True)
    LOGPATH.touch()
logging.basicConfig(filename=LOGPATH.as_posix())
LOGGER = logging.getLogger()


__accepted_models__ = {
    "bert": ["bert-base-cased", "bert-base-uncased"],
    "bart": ["facebook/bart-base"],
    "t5": ["t5-base", "t5-small"],
    # "gpt": ["gpt2", "gpt2-medium", "gpt2-large"],
    "roberta": ["roberta-base", "roberta-large"]} # TODO - add support for other models

_reverse_model_lookup = dict(itertools.chain(
    *([zip(modelnames, [modelcat for i in modelnames]) 
    for modelcat, modelnames in __accepted_models__.items()])
        )
    )


class WordLevelModel:
    """generalized word level models"""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if _reverse_model_lookup.get(model_name) in ["bart", "gpt", "roberta"]:
                model = AutoModel.from_pretrained(model_name)
            elif _reverse_model_lookup.get(model_name) in ["bert"]:
                model = AutoModelForMaskedLM.from_pretrained(model_name)
            elif _reverse_model_lookup.get(model_name) in ["t5"]:
                model = AutoModelWithLMHead.from_pretrained(model_name)
            else: raise NotImplementedError(f"model {model_name} is not supported!")
            
            self.mode_name = model_name
            self.model_family = _reverse_model_lookup.get(model_name)
            self.register_buffer(model_name, self.model_family)
            self.model = model
            self.dim = self.model.pooler.dense.in_features
            self.max_len = self.model.embeddings.position_embeddings.num_embeddings
            if use_cuda:
                self.cuda()
        except Exception as exp:
            LOGGER.exception(exp)
    
    def forward(self, sentences, padded = True, include_clssep = True):
        return self.annotate(sentences, output_hidden = False, padded = padded, include_clssep = include_clssep)[0]

    def annotate(self, sentences, output_hidden = False, padded = True, include_clssep = True):
        """
        Input: sentences, which is a list of sentences
            Each sentence is a list of words.
            Each word is a string.
        Output: an array with dimensions (batch, pad_len, dim).
        """
        if output_hidden:
            self.model.config.output_hidden_states = True
            self.model.config.output_attentions = True
        else:
            self.model.config.output_hidden_states = False
            self.model.config.output_attentions = False

        sent_lengths = [len(s) + 2 for s in sentences]
        
        # Each row is the token ids for a sentence, padded with zeros.
        all_input_ids = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for real tokens and 0 for padding.
        all_input_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        # Mask with 1 for the last subword for each word.
        all_end_mask = np.zeros((len(sentences), self.max_len), dtype = int)
        max_sent = 0
        for s_num, sentence in enumerate(sentences):
            tokens = []
            end_mask = []
            tokens.append("[CLS]")
            end_mask.append(int(include_clssep))
            for word in sentence:
                word_tokens = self.bert_tokenizer.tokenize(word)
                assert len(word_tokens) > 0, \
                    "Unknown word: {} in {}".format(word, sentence)
                for _ in range(len(word_tokens)):
                    end_mask.append(0)
                end_mask[-1] = 1
                tokens.extend(word_tokens)
            tokens.append("[SEP]")
            end_mask.append(int(include_clssep))
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            
            all_input_ids[s_num, :len(input_ids)] = input_ids
            all_input_mask[s_num, :len(input_ids)] = 1
            all_end_mask[s_num, :len(end_mask)] = end_mask
            max_sent = max(max_sent, len(input_ids))
        all_input_ids = all_input_ids[:, :max_sent]
        all_input_ids = from_numpy(np.ascontiguousarray(all_input_ids))
        all_input_mask = all_input_mask[:, :max_sent]
        all_input_mask = from_numpy(np.ascontiguousarray(all_input_mask))
        all_end_mask = all_end_mask[:, :max_sent]
        all_end_mask = from_numpy(np.ascontiguousarray(all_end_mask))
        
        # all_input_ids: num_sentences x max_sentence_len
        if output_hidden:
            features, _, hidden, attention = self.bert(all_input_ids, attention_mask = all_input_mask)
        else:
            features, _ = self.bert(all_input_ids, attention_mask = all_input_mask)
            hidden, attention = None, None
        del _
        # for each word, only keep last encoded token.
        all_end_mask = all_end_mask.to(torch.uint8).unsqueeze(-1)
        features_packed = features.masked_select(all_end_mask)
        # packed_len x dim
        features_packed = features_packed.reshape(-1, features.shape[-1])

        if padded:
            features = unpack(features_packed, sent_lengths)
        else:
            features = (features_packed, None)
        return features, all_input_ids, (hidden, attention)
    
    def reset_weights(self, encoder_only = True):
        for name, module in self.named_modules():
            if hasattr(module, 'reset_parameters') and ('encoder' in name or not encoder_only):
                module.reset_parameters()

    def freeze_model(self, freeze = True):
        for name, param in self.named_parameters():
            if 'mask_scores' not in name:
                param.requires_grad = not freeze

    def unfreeze_model(self):
        self.freeze_model(freeze = False)

