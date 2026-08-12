"""
NER Tagger Suite implementing:
1. sklearn-crfsuite CRF Tagger
2. PyTorch GPU BiLSTM-CRF Tagger with Viterbi Decoding (Section 3.3 Item 23 & Task 3.1)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any

try:
    import sklearn_crfsuite
    HAS_CRFSUITE = True
except ImportError:
    HAS_CRFSUITE = False


class SklearnCRFTagger:
    """CRF Sequence Tagger using sklearn-crfsuite."""
    
    def __init__(self, c1: float = 0.1, c2: float = 0.1, max_iterations: int = 100):
        if not HAS_CRFSUITE:
            raise ImportError("sklearn-crfsuite is not installed.")
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True
        )
        
    def _sent2features(self, sent: List[str]) -> List[Dict[str, Any]]:
        features = []
        for i, word in enumerate(sent):
            feat = {
                "bias": 1.0,
                "word": word,
                "word.lower()": word.lower(),
                "word.isupper()": word.isupper(),
                "word.isdigit()": word.isdigit(),
                "BOS": i == 0,
                "EOS": i == len(sent) - 1
            }
            if i > 0:
                feat["-1:word"] = sent[i - 1]
            if i < len(sent) - 1:
                feat["+1:word"] = sent[i + 1]
            features.append(feat)
        return features
        
    def fit(self, X_sents: List[List[str]], y_tags: List[List[str]]):
        X_feats = [self._sent2features(s) for s in X_sents]
        self.crf.fit(X_feats, y_tags)
        return self
        
    def predict(self, X_sents: List[List[str]]) -> List[List[str]]:
        X_feats = [self._sent2features(s) for s in X_sents]
        return self.crf.predict(X_feats)


def argmax(vec):
    return torch.argmax(vec).item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size(1))
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF_PyTorch(nn.Module):
    """PyTorch BiLSTM-CRF Architecture supporting GPU Acceleration (CUDA)."""
    
    def __init__(self, vocab_size: int, tag_to_ix: Dict[str, int], embedding_dim: int = 100, hidden_dim: int = 128):
        super(BiLSTM_CRF_PyTorch, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            num_layers=1, bidirectional=True, batch_first=True
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # Matrix of transition parameters. Entry (i, j) is score of transitioning TO i FROM j.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000.0
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000.0
        
    def _get_lstm_emissions(self, sentence):
        embeds = self.word_embeds(sentence).view(1, len(sentence), -1)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
        
    def _forward_alg(self, feats):
        init_vvars = torch.full((1, self.tagset_size), -10000.0, device=feats.device)
        init_vvars[0][self.tag_to_ix["<START>"]] = 0.0
        
        forward_var = init_vvars
        
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
            
        terminal_vars = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
        alpha = log_sum_exp(terminal_vars)
        return alpha
        
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1, device=feats.device)
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long, device=feats.device), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score
        
    def _viterbi_decode(self, feats):
        backpointers = []
        
        init_vvars = torch.full((1, self.tagset_size), -10000.0, device=feats.device)
        init_vvars[0][self.tag_to_ix["<START>"]] = 0.0
        
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
            
        terminal_vars = forward_var + self.transitions[self.tag_to_ix["<STOP>"]]
        best_tag_id = argmax(terminal_vars)
        path_score = terminal_vars[0][best_tag_id]
        
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
            
        start = best_path.pop()
        assert start == self.tag_to_ix["<START>"]
        best_path.reverse()
        return path_score, best_path
        
    def forward(self, sentence):
        lstm_feats = self._get_lstm_emissions(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
        
    def loss(self, sentence, tags):
        feats = self._get_lstm_emissions(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def extract_sentence_embedding(self, sentence: torch.Tensor) -> np.ndarray:
        """
        Extracts pooled BiLSTM hidden representations (Method 3.2c/3.3c)
        to feed into downstream GBDT Classifiers (XGBoost/LightGBM/CatBoost).
        """
        with torch.no_grad():
            embeds = self.word_embeds(sentence).view(1, len(sentence), -1)
            lstm_out, _ = self.lstm(embeds)
            pooled = torch.mean(lstm_out, dim=1).squeeze(0)
            return pooled.cpu().numpy()

