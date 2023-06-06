from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from hypernymysuite.evaluation import all_evaluations
from hypernymysuite.base import HypernymySuiteModel
import os
import pandas as pd
import numpy as np
import copy


def transform_mask(raw_mask):
    c_true_mask = raw_mask.copy()
    c_true_mask[0] = 0
    c_true_mask[c_true_mask.sum()] = 0
    c_true_mask = c_true_mask.astype(bool)
    
    return c_true_mask


class HFLMScorer():
    def __init__(self, model_name, device='cpu'):
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def score_batch(self, batch):
        input = self.tokenizer.batch_encode_plus([self.tokenizer.eos_token + s + self.tokenizer.eos_token for s in batch], padding=True, return_tensors='pt')
        ids_np = input['input_ids'].detach().numpy()
        ids = input['input_ids'].to(self.model.device)
        mask = input['attention_mask'].numpy()
        with torch.no_grad():
            r = self.model(ids)[0]
            r = torch.nn.LogSoftmax(dim=-1)(r).cpu().detach().numpy()

        scores = []
        for ci in range(r.shape[0]):
            c_true_mask = transform_mask(mask[ci])
            score = r[ci, range(c_true_mask.sum()), ids_np[ci][c_true_mask]].sum()
            scores.append(score)

        return scores

    def score_sentences(self, sentences, split_size=32):
        batch_count = len(sentences) // split_size + int(len(sentences) % split_size != 0)
        scores = []
        for i in tqdm(range(batch_count)):
            scores += self.score_batch(sentences[i * split_size: (i + 1) * split_size])
        return scores

class GPTHypernymySuiteModel(HypernymySuiteModel):
    def __init__(self, model, patterns, vocab):
        #super(GPTHypernymySuiteModel, self).__init__()
        self.model = model
        self.patterns = patterns
        self.vocab = vocab
        self.word2cohypos = {}

    def set_word2cohypos(self, word2cohypos):
        self.word2cohypos = word2cohypos

    def predict(self, hypo, hyper):
        all_res = []
        cohypos = self.word2cohypos.get(hypo, [])
        for pattern in self.patterns:
            res = self.model.score_sentences([self.generate_sentence(pattern, hypo, hyper, cohypos)])
            all_res.append(res[0])
        return np.mean(all_res)

    def predict_many(self, hypos, hypers):
        all_res = []
        cohypos = []
        for hypo in hypos:
            cohypos.append(self.word2cohypos.get(hypo, []))

        for pattern in self.patterns:
            sentences = []
            for x, y, z in zip(hypos, hypers, cohypos):
                sentences.append(self.generate_sentence(pattern, x, y, z))

            res = np.array(self.model.score_sentences(sentences))
            all_res.append(res)
        #print(all_res)
        return np.mean(all_res, axis=0)

    def generate_sentence(self, pattern, hypo, hyper, cohypos=[]):
        sent = pattern.replace('<hypo>', hypo).replace('<hyper>', hyper)
        for i, cohypo in enumerate(cohypos):
            sent = sent.replace(f'<cohypo{i}>', cohypo)
        return sent
