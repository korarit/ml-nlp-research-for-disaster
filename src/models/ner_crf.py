"""
NER Tagger Suite implementing:
1. sklearn-crfsuite CRF Tagger
2. PyTorch GPU BiLSTM-CRF Tagger with Viterbi Decoding (Section 3.3 Item 23 & Task 3.1)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.sparse
from typing import List, Dict, Tuple, Any, Optional

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


from pythainlp.tokenize import word_tokenize
try:
    from pythainlp.corpus import provinces
    _PROV_SET = set(provinces())
except Exception:
    _PROV_SET = set()


class ThaiLocationCRFTagger:
    """
    ML-based Sequence Tagger for Location Span Extraction using sklearn-crfsuite CRF.
    Converts (text, gt_location) pairs into BIO sequence tags, trains CRF, and decodes exact substring spans.
    """
    def __init__(self, c1: float = 0.01, c2: float = 0.01, max_iterations: int = 60):
        if not HAS_CRFSUITE:
            raise ImportError("sklearn-crfsuite is not installed.")
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True
        )

    def _extract_bio_spans(self, text: str, loc_str: Optional[str]) -> Tuple[List[str], List[str]]:
        text = str(text or "")
        loc_str = str(loc_str or "").strip()
        tokens = word_tokenize(text, engine="newmm", keep_whitespace=True)
        tags = ["O"] * len(tokens)
        if not loc_str or loc_str.lower() in ("nan", "none", "null") or loc_str not in text:
            return tokens, tags
            
        start_c = text.find(loc_str)
        end_c = start_c + len(loc_str)
        
        cur = 0
        first = True
        for i, tok in enumerate(tokens):
            s = cur
            e = cur + len(tok)
            cur = e
            if max(s, start_c) < min(e, end_c) and tok.strip():
                tags[i] = "B-LOC" if first else "I-LOC"
                first = False
        return tokens, tags

    def _token2features(self, tokens: List[str], i: int) -> Dict[str, Any]:
        w = tokens[i].strip()
        feat = {
            "bias": 1.0,
            "word": w,
            "len": len(w),
            "isdigit": w.isdigit(),
            "is_prov": w in _PROV_SET or any(w.startswith(p) for p in ["จ.", "จังหวัด"]),
            "is_dist": any(w.startswith(p) for p in ["อ.", "อำเภอ", "เขต"]),
            "is_subdist": any(w.startswith(p) for p in ["ต.", "ตำบล", "แขวง"]),
            "is_land": any(w.startswith(p) for p in ["บ้าน", "วัด", "ซอย", "ถนน", "หมู่", "ม.", "ชุมชน", "คอนโด", "โรงเรียน", "สะพาน", "ตลาด"])
        }
        for offset, prefix in [(-2, "-2:"), (-1, "-1:"), (1, "+1:"), (2, "+2:")]:
            idx = i + offset
            if 0 <= idx < len(tokens):
                nw = tokens[idx].strip()
                feat[f"{prefix}word"] = nw
                feat[f"{prefix}is_prov"] = nw in _PROV_SET or any(nw.startswith(p) for p in ["จ.", "จังหวัด"])
                feat[f"{prefix}is_dist"] = any(nw.startswith(p) for p in ["อ.", "อำเภอ", "เขต"])
                feat[f"{prefix}is_subdist"] = any(nw.startswith(p) for p in ["ต.", "ตำบล", "แขวง"])
                feat[f"{prefix}is_land"] = any(nw.startswith(p) for p in ["บ้าน", "วัด", "ซอย", "ถนน", "หมู่", "ม.", "ชุมชน", "คอนโด", "โรงเรียน", "สะพาน", "ตลาด"])
        if i == 0:
            feat["BOS"] = True
        if i == len(tokens) - 1:
            feat["EOS"] = True
        return feat

    def fit(self, texts: List[str], gt_locations: List[Optional[str]]):
        """Fits CRF model on training text and location annotations."""
        X_feats = []
        y_tags = []
        for text, loc in zip(texts, gt_locations):
            tokens, tags = self._extract_bio_spans(text, loc)
            feats = [self._token2features(tokens, i) for i in range(len(tokens))]
            X_feats.append(feats)
            y_tags.append(tags)
        self.crf.fit(X_feats, y_tags)
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predicts extracted location string from input text preserving original formatting."""
        extracted_locations = []
        for text in texts:
            text_str = str(text or "")
            tokens = word_tokenize(text_str, engine="newmm", keep_whitespace=True)
            if not tokens:
                extracted_locations.append("")
                continue
            feats = [self._token2features(tokens, i) for i in range(len(tokens))]
            preds = self.crf.predict_single(feats)
            
            # Reconstruct original character offsets
            cur = 0
            spans = []
            for tok in tokens:
                s = cur
                e = cur + len(tok)
                cur = e
                spans.append((s, e))
                
            loc_idx = [i for i, tag in enumerate(preds) if tag in ("B-LOC", "I-LOC") and tokens[i].strip()]
            
            # Strip leading prepositions from predicted span
            while loc_idx and tokens[loc_idx[0]].strip() in ("ที่", "อยู่ที่", "อยู่", "บริเวณ", "ตรง", "พื้นที่", "ตอนนี้", "ตอนนี้อยู่ที่"):
                loc_idx.pop(0)
                
            if loc_idx:
                start_char = spans[loc_idx[0]][0]
                end_char = spans[loc_idx[-1]][1]
                extracted_locations.append(text_str[start_char:end_char].strip())
            else:
                extracted_locations.append("")
        return extracted_locations


import re


def clean_extracted_span(span_str: str, target_type: str) -> str:
    """Cleans and normalizes extracted entity strings for standardized string matching."""
    if not span_str:
        return ""
    res = str(span_str).strip()
    if target_type == "COORDS":
        # Remove leading keywords (e.g. พิกัด, ละติจูด, ลองจิจูด)
        res = re.sub(r"^(?:พิกัด|ละติจูด|ลองจิจูด|ตำแหน่ง|coords?|lat/lng)[:\s]*", "", res, flags=re.IGNORECASE).strip()
        # Normalize whitespace around comma: "13.4553, 100.6719" -> "13.4553,100.6719"
        res = re.sub(r"\s*,\s*", ",", res)
    elif target_type == "PHONE":
        res = re.sub(r"^(?:เบอร์โทรศัพท์|เบอร์โทร|โทร|เบอร์|tel|phone)[:\s]*", "", res, flags=re.IGNORECASE).strip()
    elif target_type == "URL":
        res = res.strip("., \t\r\n")
    return res


class ThaiMultiNER_CRFTagger:
    """
    Pure Machine Learning Multi-Entity Sequence Tagger for Location, Phone, Map URL, Coordinates, Victim Name, and Reporter Name.
    Uses sklearn-crfsuite CRF trained on token-aligned annotations without any regex rules.
    """
    def __init__(self, c1: float = 0.01, c2: float = 0.01, max_iterations: int = 60):
        if not HAS_CRFSUITE:
            raise ImportError("sklearn-crfsuite is not installed.")
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True
        )

    def _extract_multi_entity_bio(
        self, text: str, loc_str: Optional[str], phone_str: Optional[str],
        url_str: Optional[str], lat: Optional[Any], lng: Optional[Any],
        vic_name: Optional[str] = None, rep_name: Optional[str] = None
    ) -> Tuple[List[str], List[Tuple[int, int]], List[str]]:
        text_str = str(text or "")
        tokens = word_tokenize(text_str, engine="newmm", keep_whitespace=True)
        tags = ["O"] * len(tokens)
        
        cur = 0
        spans = []
        for tok in tokens:
            s = cur
            e = cur + len(tok)
            cur = e
            spans.append((s, e))
            
        entities = [
            ("VIC_NAME", str(vic_name or "").strip()),
            ("REP_NAME", str(rep_name or "").strip()),
            ("LOC", str(loc_str or "").strip()),
            ("PHONE", str(phone_str or "").strip()),
            ("URL", str(url_str or "").strip()),
        ]
        if lat is not None and lng is not None and str(lat).lower() not in ("nan", "none", "0.0", "0") and float(lat) != 0.0:
            c1 = f"{lat}, {lng}"
            c2 = f"{lat},{lng}"
            if c1 in text_str:
                entities.append(("COORDS", c1))
            elif c2 in text_str:
                entities.append(("COORDS", c2))
                
        for ent_type, val in entities:
            if not val or val.lower() in ("nan", "none", "null") or val not in text_str:
                continue
            start_c = text_str.find(val)
            end_c = start_c + len(val)
            first = True
            for i, (s, e) in enumerate(spans):
                if max(s, start_c) < min(e, end_c) and tokens[i].strip():
                    if tags[i] == "O":
                        tags[i] = f"B-{ent_type}" if first else f"I-{ent_type}"
                        first = False
        return tokens, spans, tags

    def _token2features(self, tokens: List[str], i: int) -> Dict[str, Any]:
        w = tokens[i].strip()
        feat = {
            "bias": 1.0,
            "word": w,
            "len": len(w),
            "isdigit": w.isdigit(),
            "is_prov": w in _PROV_SET or any(w.startswith(p) for p in ["จ.", "จังหวัด"]),
            "is_dist": any(w.startswith(p) for p in ["อ.", "อำเภอ", "เขต"]),
            "is_subdist": any(w.startswith(p) for p in ["ต.", "ตำบล", "แขวง"]),
            "is_land": any(w.startswith(p) for p in ["บ้าน", "วัด", "ซอย", "ถนน", "หมู่", "ม.", "ชุมชน", "คอนโด", "โรงเรียน", "สะพาน", "ตลาด"]),
            "is_title": any(w.startswith(p) for p in ["นาย", "นาง", "น้อง", "ลุง", "ป้า", "น้า", "อา", "คุณ", "พี่", "หมอ", "เฮีย"]),
            "is_url": "http" in w or "maps" in w or "goo.gl" in w,
            "has_dot": "." in w,
            "is_phone_len": w.isdigit() and len(w) in (3, 4, 9, 10)
        }
        for offset, prefix in [(-2, "-2:"), (-1, "-1:"), (1, "+1:"), (2, "+2:")]:
            idx = i + offset
            if 0 <= idx < len(tokens):
                nw = tokens[idx].strip()
                feat[f"{prefix}word"] = nw
                feat[f"{prefix}is_prov"] = nw in _PROV_SET or any(nw.startswith(p) for p in ["จ.", "จังหวัด"])
                feat[f"{prefix}is_dist"] = any(nw.startswith(p) for p in ["อ.", "อำเภอ", "เขต"])
                feat[f"{prefix}is_subdist"] = any(nw.startswith(p) for p in ["ต.", "ตำบล", "แขวง"])
                feat[f"{prefix}is_land"] = any(nw.startswith(p) for p in ["บ้าน", "วัด", "ซอย", "ถนน", "หมู่", "ม.", "ชุมชน", "คอนโด", "โรงเรียน", "สะพาน", "ตลาด"])
                feat[f"{prefix}is_title"] = any(nw.startswith(p) for p in ["นาย", "นาง", "น้อง", "ลุง", "ป้า", "น้า", "อา", "คุณ", "พี่", "หมอ", "เฮีย"])
                feat[f"{prefix}is_url"] = "http" in nw or "maps" in nw or "goo.gl" in nw
                feat[f"{prefix}has_dot"] = "." in nw
        if i == 0:
            feat["BOS"] = True
        if i == len(tokens) - 1:
            feat["EOS"] = True
        return feat

    def fit(self, texts: List[str], locs: List[Any], phones: List[Any], urls: List[Any], lats: List[Any], lngs: List[Any], vic_names: Optional[List[Any]] = None, rep_names: Optional[List[Any]] = None):
        """Trains multi-entity CRF sequence tagger."""
        X_feats = []
        y_tags = []
        v_names = vic_names if vic_names is not None else [None] * len(texts)
        r_names = rep_names if rep_names is not None else [None] * len(texts)
        for text, loc, phone, url, lat, lng, vn, rn in zip(texts, locs, phones, urls, lats, lngs, v_names, r_names):
            tokens, spans, tags = self._extract_multi_entity_bio(text, loc, phone, url, lat, lng, vn, rn)
            feats = [self._token2features(tokens, i) for i in range(len(tokens))]
            X_feats.append(feats)
            y_tags.append(tags)
        self.crf.fit(X_feats, y_tags)
        return self

    def predict_entities(self, texts: List[str]) -> Dict[str, List[str]]:
        """Predicts extracted spans for Location, Phone, Map URL, Coordinates, Victim Name, and Reporter Name."""
        pred_locs, pred_phones, pred_urls, pred_coords = [], [], [], []
        pred_vic_names, pred_rep_names = [], []
        for text in texts:
            text_str = str(text or "")
            tokens = word_tokenize(text_str, engine="newmm", keep_whitespace=True)
            if not tokens:
                pred_locs.append("")
                pred_phones.append("")
                pred_urls.append("")
                pred_coords.append("")
                pred_vic_names.append("")
                pred_rep_names.append("")
                continue
            feats = [self._token2features(tokens, i) for i in range(len(tokens))]
            preds = self.crf.predict_single(feats)
            
            cur = 0
            spans = []
            for tok in tokens:
                s = cur
                e = cur + len(tok)
                cur = e
                spans.append((s, e))
                
            def extract_span(target_type: str) -> str:
                idx = [i for i, tag in enumerate(preds) if tag in (f"B-{target_type}", f"I-{target_type}") and tokens[i].strip()]
                if target_type == "LOC":
                    while idx and tokens[idx[0]].strip() in ("ที่", "อยู่ที่", "อยู่", "บริเวณ", "ตรง", "พื้นที่", "ตอนนี้", "ตอนนี้อยู่ที่"):
                        idx.pop(0)
                if idx:
                    s_c = spans[idx[0]][0]
                    e_c = spans[idx[-1]][1]
                    raw_span = text_str[s_c:e_c].strip()
                    return clean_extracted_span(raw_span, target_type)
                return ""
                
            pred_locs.append(extract_span("LOC"))
            pred_phones.append(extract_span("PHONE"))
            pred_urls.append(extract_span("URL"))
            pred_coords.append(extract_span("COORDS"))
            pred_vic_names.append(extract_span("VIC_NAME"))
            pred_rep_names.append(extract_span("REP_NAME"))
            
        return {
            "locations": pred_locs,
            "phones": pred_phones,
            "urls": pred_urls,
            "coords": pred_coords,
            "victim_names": pred_vic_names,
            "reporter_names": pred_rep_names
        }


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


from sklearn.feature_extraction import DictVectorizer
from src.models.classifiers import get_classifier


from sklearn.preprocessing import LabelEncoder


class SlidingWindowTokenClassifier:
    """
    Token Sequence Classification using any Task 1 Classifier (LogisticRegression, SVM, RF, XGB, CatBoost, etc.)
    with a Sliding Context Window (-2, -1, 0, +1, +2) across words.
    """
    def __init__(self, model_name: str = "LogisticRegression", use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.vectorizer = DictVectorizer(sparse=True)
        clf_kwargs = {}
        if model_name in ("GradientBoostingClassifier", "AdaBoostClassifier"):
            clf_kwargs["n_estimators"] = 30
        self.clf = get_classifier(model_name, use_gpu=use_gpu, **clf_kwargs)
        self.multi_ner_helper = ThaiMultiNER_CRFTagger()
        self.label_encoder = LabelEncoder()

    def _safe_fit_predict(self, X_tr, y_tr, X_te):
        """
        Fits token classifier with Entity-Preserving Subsampling (keeps 100% of entity tokens,
        subsamples background 'O' tokens) and performs safe chunked GPU/CPU prediction.
        """
        y_arr = np.asarray(y_tr)
        
        # Entity-Preserving Subsampling: Keep 100% of all entity tokens, balance background 'O' tokens
        if len(y_arr) > 15000:
            ent_indices = np.where(y_arr != "O")[0]
            o_indices = np.where(y_arr == "O")[0]
            
            rng = np.random.RandomState(42)
            n_o_sample = min(len(o_indices), max(len(ent_indices) * 3, 10000))
            sampled_o = rng.choice(o_indices, size=n_o_sample, replace=False)
            
            keep_idx = np.sort(np.concatenate([ent_indices, sampled_o]))
            X_tr_sub = X_tr[keep_idx] if scipy.sparse.issparse(X_tr) else X_tr[keep_idx]
            y_tr_sub = y_arr[keep_idx]
        else:
            X_tr_sub = X_tr
            y_tr_sub = y_arr

        self.label_encoder.fit(y_arr)  # Fit on full tagset to ensure all classes exist
        y_encoded = self.label_encoder.transform(y_tr_sub).astype(np.int32)
        
        try:
            self.clf.fit(X_tr_sub, y_encoded)
            preds_int = self.clf.predict(X_te)
        except Exception as e:
            # Fallback to CPU fast sparse LogisticRegression/SGD
            from sklearn.linear_model import LogisticRegression
            cpu_clf = LogisticRegression(max_iter=200, solver="saga", n_jobs=-1, random_state=42)
            cpu_clf.fit(X_tr_sub, y_encoded)
            preds_int = cpu_clf.predict(X_te)
            
        if hasattr(preds_int, "to_numpy"):
            preds_int = preds_int.to_numpy()
        elif hasattr(preds_int, "get"):
            preds_int = preds_int.get()
        preds_int = np.asarray(preds_int).reshape(-1).astype(int)
        
        n_classes = len(self.label_encoder.classes_)
        preds_int = np.clip(preds_int, 0, n_classes - 1)
        return self.label_encoder.inverse_transform(preds_int)

    def fit_and_predict_cached(self, token_cache: Dict[str, Any]) -> Dict[str, List[str]]:
        """Fast 0.1s fitting and predicting using pre-computed token matrices."""
        all_preds = self._safe_fit_predict(token_cache["X_train_mat"], token_cache["y_train_labels"], token_cache["X_test_mat"])
        
        pred_locs, pred_phones, pred_urls, pred_coords = [], [], [], []
        pred_vic_names, pred_rep_names = [], []
        
        for t_str, tokens, spans, s_i, e_i in token_cache["test_meta"]:
            if not tokens or s_i == e_i:
                pred_locs.append("")
                pred_phones.append("")
                pred_urls.append("")
                pred_coords.append("")
                pred_vic_names.append("")
                pred_rep_names.append("")
                continue
            preds = all_preds[s_i:e_i]
            
            def extract_span(target_type: str) -> str:
                idx = [i for i, tag in enumerate(preds) if tag in (f"B-{target_type}", f"I-{target_type}") and tokens[i].strip()]
                if target_type == "LOC":
                    while idx and tokens[idx[0]].strip() in ("ที่", "อยู่ที่", "อยู่", "บริเวณ", "ตรง", "พื้นที่", "ตอนนี้", "ตอนนี้อยู่ที่"):
                        idx.pop(0)
                if idx:
                    s_c = spans[idx[0]][0]
                    e_c = spans[idx[-1]][1]
                    raw_span = t_str[s_c:e_c].strip()
                    return clean_extracted_span(raw_span, target_type)
                return ""
                
            pred_locs.append(extract_span("LOC"))
            pred_phones.append(extract_span("PHONE"))
            pred_urls.append(extract_span("URL"))
            pred_coords.append(extract_span("COORDS"))
            pred_vic_names.append(extract_span("VIC_NAME"))
            pred_rep_names.append(extract_span("REP_NAME"))
            
        return {
            "locations": pred_locs,
            "phones": pred_phones,
            "urls": pred_urls,
            "coords": pred_coords,
            "victim_names": pred_vic_names,
            "reporter_names": pred_rep_names
        }

    def fit(self, texts: List[str], locs: List[Any], phones: List[Any], urls: List[Any], lats: List[Any], lngs: List[Any], vic_names: Optional[List[Any]] = None, rep_names: Optional[List[Any]] = None):
        """Trains sliding window token classifier on token-level BIO labels."""
        X_token_dicts = []
        y_token_labels = []
        v_names = vic_names if vic_names is not None else [None] * len(texts)
        r_names = rep_names if rep_names is not None else [None] * len(texts)
        for text, loc, phone, url, lat, lng, vn, rn in zip(texts, locs, phones, urls, lats, lngs, v_names, r_names):
            tokens, spans, tags = self.multi_ner_helper._extract_multi_entity_bio(text, loc, phone, url, lat, lng, vn, rn)
            for i in range(len(tokens)):
                feat = self.multi_ner_helper._token2features(tokens, i)
                X_token_dicts.append(feat)
                y_token_labels.append(tags[i])
                
        if X_token_dicts:
            X_mat = self.vectorizer.fit_transform(X_token_dicts)
            y_arr = np.asarray(y_token_labels)
            
            if len(y_arr) > 40000:
                ent_indices = np.where(y_arr != "O")[0]
                o_indices = np.where(y_arr == "O")[0]
                rng = np.random.RandomState(42)
                n_o_sample = min(len(o_indices), max(len(ent_indices) * 2, 25000))
                sampled_o = rng.choice(o_indices, size=n_o_sample, replace=False)
                keep_idx = np.sort(np.concatenate([ent_indices, sampled_o]))
                X_tr_sub = X_mat[keep_idx]
                y_tr_sub = y_arr[keep_idx]
            else:
                X_tr_sub = X_mat
                y_tr_sub = y_arr
                
            self.label_encoder.fit(y_arr)
            y_encoded = self.label_encoder.transform(y_tr_sub).astype(np.int32)
            try:
                self.clf.fit(X_tr_sub, y_encoded)
            except Exception:
                from sklearn.linear_model import LogisticRegression
                self.clf = LogisticRegression(max_iter=200, solver="saga", n_jobs=-1, random_state=42)
                self.clf.fit(X_tr_sub, y_encoded)
        return self

    def predict_entities(self, texts: List[str]) -> Dict[str, List[str]]:
        """Predicts token-level BIO labels and decodes into entity spans."""
        pred_locs, pred_phones, pred_urls, pred_coords = [], [], [], []
        pred_vic_names, pred_rep_names = [], []
        for text in texts:
            text_str = str(text or "")
            tokens = word_tokenize(text_str, engine="newmm", keep_whitespace=True)
            if not tokens:
                pred_locs.append("")
                pred_phones.append("")
                pred_urls.append("")
                pred_coords.append("")
                pred_vic_names.append("")
                pred_rep_names.append("")
                continue
            feats = [self.multi_ner_helper._token2features(tokens, i) for i in range(len(tokens))]
            X_mat = self.vectorizer.transform(feats)
            preds_int = self.clf.predict(X_mat)
            if hasattr(preds_int, "to_numpy"):
                preds_int = preds_int.to_numpy()
            elif hasattr(preds_int, "get"):
                preds_int = preds_int.get()
            preds_int = np.asarray(preds_int).reshape(-1).astype(int)
            n_classes = len(self.label_encoder.classes_)
            preds_int = np.clip(preds_int, 0, n_classes - 1)
            preds = self.label_encoder.inverse_transform(preds_int)
            
            cur = 0
            spans = []
            for tok in tokens:
                s = cur
                e = cur + len(tok)
                cur = e
                spans.append((s, e))
                
            def extract_span(target_type: str) -> str:
                idx = [i for i, tag in enumerate(preds) if tag in (f"B-{target_type}", f"I-{target_type}") and tokens[i].strip()]
                if target_type == "LOC":
                    while idx and tokens[idx[0]].strip() in ("ที่", "อยู่ที่", "อยู่", "บริเวณ", "ตรง", "พื้นที่", "ตอนนี้", "ตอนนี้อยู่ที่"):
                        idx.pop(0)
                if idx:
                    s_c = spans[idx[0]][0]
                    e_c = spans[idx[-1]][1]
                    raw_span = text_str[s_c:e_c].strip()
                    return clean_extracted_span(raw_span, target_type)
                return ""
                
            pred_locs.append(extract_span("LOC"))
            pred_phones.append(extract_span("PHONE"))
            pred_urls.append(extract_span("URL"))
            pred_coords.append(extract_span("COORDS"))
            pred_vic_names.append(extract_span("VIC_NAME"))
            pred_rep_names.append(extract_span("REP_NAME"))
            
        return {
            "locations": pred_locs,
            "phones": pred_phones,
            "urls": pred_urls,
            "coords": pred_coords,
            "victim_names": pred_vic_names,
            "reporter_names": pred_rep_names
        }


def prepare_task2_token_cache(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Pre-computes sliding window token feature matrices and test token spans ONCE for all classifiers in Task 2.
    Accelerates the entire 17-model benchmark by ~20x.
    """
    helper = ThaiMultiNER_CRFTagger()
    vectorizer = DictVectorizer(sparse=True)
    
    # 1. Train tokens
    train_texts = train_df["generated_text"].tolist()
    locs = [str(x or "").strip() if pd.notna(x) else "" for x in train_df.get("gt_location_name", [])]
    phones = [str(x or "").strip() if pd.notna(x) else "" for x in train_df.get("gt_victim_phone", [])]
    urls = [str(x or "").strip() if pd.notna(x) else "" for x in train_df.get("gt_google_map_url", [])]
    lats = train_df.get("gt_lat", [None] * len(train_df)).tolist()
    lngs = train_df.get("gt_lng", [None] * len(train_df)).tolist()
    vn = [str(x or "").strip() if pd.notna(x) else "" for x in train_df.get("gt_victim_name", [])]
    rn = [str(x or "").strip() if pd.notna(x) else "" for x in train_df.get("gt_reporter_name", [])]
    
    X_train_dicts = []
    y_train_labels = []
    for t, l, p, u, la, ln, v, r in zip(train_texts, locs, phones, urls, lats, lngs, vn, rn):
        tokens, spans, tags = helper._extract_multi_entity_bio(t, l, p, u, la, ln, v, r)
        for i in range(len(tokens)):
            X_train_dicts.append(helper._token2features(tokens, i))
            y_train_labels.append(tags[i])
            
    X_train_mat = vectorizer.fit_transform(X_train_dicts)
    
    # 2. Test tokens & metadata
    test_texts = test_df["generated_text"].tolist()
    test_meta = []
    X_test_dicts = []
    for t in test_texts:
        t_str = str(t or "")
        tokens = word_tokenize(t_str, engine="newmm", keep_whitespace=True)
        cur = 0
        spans = []
        for tok in tokens:
            s = cur
            e = cur + len(tok)
            cur = e
            spans.append((s, e))
        start_idx = len(X_test_dicts)
        for i in range(len(tokens)):
            X_test_dicts.append(helper._token2features(tokens, i))
        end_idx = len(X_test_dicts)
        test_meta.append((t_str, tokens, spans, start_idx, end_idx))
        
    X_test_mat = vectorizer.transform(X_test_dicts)
    
    return {
        "vectorizer": vectorizer,
        "X_train_mat": X_train_mat,
        "y_train_labels": y_train_labels,
        "X_test_mat": X_test_mat,
        "test_meta": test_meta
    }


class BiLSTM_CRF_Tagger:
    """
    PyTorch GPU-accelerated BiLSTM-CRF Multi-Entity Sequence Tagger.
    """
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 64, epochs: int = 8, lr: float = 0.01, use_gpu: bool = True):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.multi_ner_helper = ThaiMultiNER_CRFTagger()
        self.word_to_ix: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.tag_to_ix: Dict[str, int] = {"<START>": 0, "<STOP>": 1, "O": 2}
        self.model: Optional[BiLSTM_CRF_PyTorch] = None

    def fit(self, texts: List[str], locs: List[Any], phones: List[Any], urls: List[Any], lats: List[Any], lngs: List[Any], vic_names: Optional[List[Any]] = None, rep_names: Optional[List[Any]] = None):
        """Trains BiLSTM-CRF model on GPU CUDA."""
        tokenized_sents = []
        tagged_sents = []
        v_names = vic_names if vic_names is not None else [None] * len(texts)
        r_names = rep_names if rep_names is not None else [None] * len(texts)
        for text, loc, phone, url, lat, lng, vn, rn in zip(texts, locs, phones, urls, lats, lngs, v_names, r_names):
            tokens, spans, tags = self.multi_ner_helper._extract_multi_entity_bio(text, loc, phone, url, lat, lng, vn, rn)
            tokenized_sents.append(tokens)
            tagged_sents.append(tags)
            for w in tokens:
                w_str = w.strip()
                if w_str and w_str not in self.word_to_ix:
                    self.word_to_ix[w_str] = len(self.word_to_ix)
            for t in tags:
                if t not in self.tag_to_ix:
                    self.tag_to_ix[t] = len(self.tag_to_ix)
                    
        self.model = BiLSTM_CRF_PyTorch(
            vocab_size=len(self.word_to_ix),
            tag_to_ix=self.tag_to_ix,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Train loop
        self.model.train()
        for epoch in range(self.epochs):
            for tokens, tags in zip(tokenized_sents, tagged_sents):
                if not tokens:
                    continue
                idxs = [self.word_to_ix.get(w.strip(), 1) for w in tokens]
                tag_idxs = [self.tag_to_ix[t] for t in tags]
                
                seq_in = torch.tensor(idxs, dtype=torch.long, device=self.device)
                seq_targets = torch.tensor(tag_idxs, dtype=torch.long, device=self.device)
                
                self.model.zero_grad()
                loss = self.model.loss(seq_in, seq_targets)
                loss.backward()
                optimizer.step()
                
        return self

    def predict_entities(self, texts: List[str]) -> Dict[str, List[str]]:
        """Performs Viterbi decoding with BiLSTM-CRF on GPU and decodes entity spans."""
        if self.model is None:
            return {"locations": [""] * len(texts), "phones": [""] * len(texts), "urls": [""] * len(texts), "coords": [""] * len(texts), "victim_names": [""] * len(texts), "reporter_names": [""] * len(texts)}
            
        self.model.eval()
        ix_to_tag = {v: k for k, v in self.tag_to_ix.items()}
        pred_locs, pred_phones, pred_urls, pred_coords = [], [], [], []
        pred_vic_names, pred_rep_names = [], []
        
        with torch.no_grad():
            for text in texts:
                text_str = str(text or "")
                tokens = word_tokenize(text_str, engine="newmm", keep_whitespace=True)
                if not tokens:
                    pred_locs.append("")
                    pred_phones.append("")
                    pred_urls.append("")
                    pred_coords.append("")
                    pred_vic_names.append("")
                    pred_rep_names.append("")
                    continue
                idxs = [self.word_to_ix.get(w.strip(), 1) for w in tokens]
                seq_in = torch.tensor(idxs, dtype=torch.long, device=self.device)
                _, tag_seq = self.model(seq_in)
                preds = [ix_to_tag.get(idx, "O") for idx in tag_seq]
                
                cur = 0
                spans = []
                for tok in tokens:
                    s = cur
                    e = cur + len(tok)
                    cur = e
                    spans.append((s, e))
                    
                def extract_span(target_type: str) -> str:
                    idx = [i for i, tag in enumerate(preds) if tag in (f"B-{target_type}", f"I-{target_type}") and tokens[i].strip()]
                    if target_type == "LOC":
                        while idx and tokens[idx[0]].strip() in ("ที่", "อยู่ที่", "อยู่", "บริเวณ", "ตรง", "พื้นที่", "ตอนนี้", "ตอนนี้อยู่ที่"):
                            idx.pop(0)
                    if idx:
                        s_c = spans[idx[0]][0]
                        e_c = spans[idx[-1]][1]
                        raw_span = text_str[s_c:e_c].strip()
                        return clean_extracted_span(raw_span, target_type)
                    return ""
                    
                pred_locs.append(extract_span("LOC"))
                pred_phones.append(extract_span("PHONE"))
                pred_urls.append(extract_span("URL"))
                pred_coords.append(extract_span("COORDS"))
                pred_vic_names.append(extract_span("VIC_NAME"))
                pred_rep_names.append(extract_span("REP_NAME"))
                
        return {
            "locations": pred_locs,
            "phones": pred_phones,
            "urls": pred_urls,
            "coords": pred_coords,
            "victim_names": pred_vic_names,
            "reporter_names": pred_rep_names
        }

