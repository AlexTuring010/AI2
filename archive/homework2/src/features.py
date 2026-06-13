"""Numeric HW1-style features for Q/A pairs.

These features stay outside the transformer text. They are standardized on the
train split and fed through a small side branch before concatenation with the
pooled transformer representation.
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "multiple_questions_flag",
    "affirmative_questions_flag",
    "inaudible_flag",
    "question_order",
    "q_word_len",
    "a_word_len",
    "a_to_q_len_ratio",
    "abs_len_diff",
    "q_char_len",
    "a_char_len",
    "shared_unique_count",
    "unique_jaccard",
    "question_coverage",
    "answer_coverage",
    "shared_fraction_shorter",
    "question_unanswered_ratio",
    "shared_bigram_count",
    "question_bigram_coverage",
    "answer_bigram_coverage",
    "answer_unique_ratio",
    "answer_question_mark_count",
    "question_question_mark_count",
    "answer_exclamation_count",
    "answer_ellipsis_count",
    "answer_quote_count",
    "question_sentence_count",
    "answer_sentence_count",
    "answer_avg_sentence_len",
    "answer_digit_count",
    "hedge_count",
    "negation_count",
    "nonreply_count",
    "deflection_count",
    "contrast_count",
    "causal_count",
    "numeric_evidence_count",
    "temporal_evidence_count",
    "person_evidence_count",
    "location_evidence_count",
    "answer_starts_yes",
    "answer_starts_no",
    "answer_starts_uncertain",
    "answer_starts_deflect",
    "answer_starts_answer_to_question",
    "answer_mentions_second_question",
    "answer_has_i",
    "answer_has_we",
    "answer_has_you",
    "question_has_i",
    "question_has_we",
    "question_has_you",
    "target_shift_you_to_we",
    "target_shift_you_to_i",
    "question_is_yesno",
    "question_is_why",
    "question_is_when",
    "question_is_where",
    "question_is_who",
    "question_is_how",
    "question_is_how_many",
    "question_is_how_much",
    "question_is_what",
    "qa_yesno_polarity_match",
    "qa_why_causal_match",
    "qa_when_temporal_match",
    "qa_where_location_match",
    "qa_who_person_match",
    "qa_how_many_numeric_match",
    "idf_shared_weight_sum",
    "idf_question_coverage",
    "idf_answer_coverage",
    "idf_cosine",
    "rare_question_token_unanswered_ratio",
    "answer_is_long",
    "answer_is_very_long",
    "answer_many_sentences",
    "long_low_overlap",
    "long_with_evidence",
    "long_with_uncertainty",
    "long_with_deflection",
    "evidence_minus_hedge_deflect",
    "evidence_after_deflection",
    "starts_deflect_but_has_evidence",
    "evidence_and_uncertainty",
    "evidence_density",
]

BASE_FEATURE_NAMES = [
    "q_word_len",
    "a_word_len",
    "a_to_q_len_ratio",
    "abs_len_diff",
    "q_char_len",
    "a_char_len",
    "shared_unique_count",
    "unique_jaccard",
    "question_coverage",
    "answer_coverage",
    "shared_fraction_shorter",
    "question_unanswered_ratio",
    "shared_bigram_count",
    "question_bigram_coverage",
    "answer_bigram_coverage",
    "answer_unique_ratio",
    "answer_question_mark_count",
    "answer_exclamation_count",
    "answer_ellipsis_count",
    "answer_quote_count",
    "answer_digit_count",
    "hedge_count",
    "negation_count",
    "nonreply_count",
    "deflection_count",
    "contrast_count",
    "causal_count",
    "numeric_evidence_count",
    "temporal_evidence_count",
    "person_evidence_count",
    "location_evidence_count",
    "answer_starts_yes",
    "answer_starts_no",
    "answer_starts_uncertain",
    "answer_starts_deflect",
    "answer_has_i",
    "answer_has_we",
    "answer_has_you",
    "question_has_i",
    "question_has_we",
    "question_has_you",
    "target_shift_you_to_we",
    "target_shift_you_to_i",
    "question_is_yesno",
    "question_is_why",
    "question_is_when",
    "question_is_where",
    "question_is_who",
    "question_is_how",
    "question_is_how_many",
    "question_is_how_much",
    "question_is_what",
    "qa_yesno_polarity_match",
    "qa_why_causal_match",
    "qa_when_temporal_match",
    "qa_where_location_match",
    "qa_who_person_match",
    "qa_how_many_numeric_match",
]

COVERAGE_EVIDENCE_EXTRA_FEATURE_NAMES = [
    "question_question_mark_count",
    "question_sentence_count",
    "answer_sentence_count",
    "answer_avg_sentence_len",
    "answer_starts_answer_to_question",
    "answer_mentions_second_question",
    "idf_shared_weight_sum",
    "idf_question_coverage",
    "idf_answer_coverage",
    "idf_cosine",
    "rare_question_token_unanswered_ratio",
]

LONG_MIXED_EXTRA_FEATURE_NAMES = COVERAGE_EVIDENCE_EXTRA_FEATURE_NAMES + [
    "answer_is_long",
    "answer_is_very_long",
    "answer_many_sentences",
    "long_low_overlap",
    "long_with_evidence",
    "long_with_uncertainty",
    "long_with_deflection",
    "evidence_minus_hedge_deflect",
    "evidence_after_deflection",
    "starts_deflect_but_has_evidence",
    "evidence_and_uncertainty",
    "evidence_density",
]

FEATURE_PROFILES = {
    "base": BASE_FEATURE_NAMES,
    "coverage_evidence": BASE_FEATURE_NAMES + COVERAGE_EVIDENCE_EXTRA_FEATURE_NAMES,
    "long_mixed": BASE_FEATURE_NAMES + LONG_MIXED_EXTRA_FEATURE_NAMES,
    "rich": FEATURE_NAMES,
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "do", "does", "did", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "there", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _content_words(text: str) -> List[str]:
    return [w for w in _words(text) if len(w) > 1 and w not in STOPWORDS]


def _bigrams(tokens: List[str]) -> set:
    return set(zip(tokens, tokens[1:]))


def _count_patterns(text: str, patterns: List[str]) -> int:
    text = str(text).lower()
    return int(sum(len(re.findall(p, text)) for p in patterns))


def _has_pattern(text: str, pattern: str) -> int:
    return int(re.search(pattern, str(text).lower()) is not None)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _fit_idf(df: pd.DataFrame) -> Dict[str, float]:
    docs = []
    for _, row in df.iterrows():
        docs.append(set(_content_words(str(row.get("question", "")))))
        docs.append(set(_content_words(str(row.get("answer", "")))))
    n_docs = max(len(docs), 1)
    counts = {}
    for doc in docs:
        for tok in doc:
            counts[tok] = counts.get(tok, 0) + 1
    return {
        tok: float(math.log((1 + n_docs) / (1 + count)) + 1.0)
        for tok, count in counts.items()
    }


def _tfidf_cosine(q_tokens: List[str], a_tokens: List[str], idf: Dict[str, float]) -> float:
    q_counts = {}
    a_counts = {}
    for tok in q_tokens:
        q_counts[tok] = q_counts.get(tok, 0) + 1
    for tok in a_tokens:
        a_counts[tok] = a_counts.get(tok, 0) + 1
    keys = set(q_counts) | set(a_counts)
    dot = q_norm = a_norm = 0.0
    for tok in keys:
        weight = idf.get(tok, 1.0)
        qv = q_counts.get(tok, 0) * weight
        av = a_counts.get(tok, 0) * weight
        dot += qv * av
        q_norm += qv * qv
        a_norm += av * av
    return _safe_div(dot, math.sqrt(q_norm) * math.sqrt(a_norm))


def compute_numeric_feature_row(row, idf: Dict[str, float] = None) -> List[float]:
    idf = idf or {}
    q = str(row.get("question", ""))
    a = str(row.get("answer", ""))
    q_l = q.lower().strip()
    a_l = a.lower().strip()
    a_open = " ".join(a_l.split()[:12])

    q_words = _words(q)
    a_words = _words(a)
    q_tok = _content_words(q)
    a_tok = _content_words(a)
    q_set = set(q_tok)
    a_set = set(a_tok)
    inter = q_set & a_set
    union = q_set | a_set
    q_bi = _bigrams(q_tok)
    a_bi = _bigrams(a_tok)
    bi_inter = q_bi & a_bi
    q_idf_sum = sum(idf.get(tok, 1.0) for tok in q_set)
    a_idf_sum = sum(idf.get(tok, 1.0) for tok in a_set)
    shared_idf_sum = sum(idf.get(tok, 1.0) for tok in inter)
    rare_q = {tok for tok in q_set if idf.get(tok, 1.0) >= 3.0}
    rare_q_unanswered = rare_q - a_set

    hedge = _count_patterns(
        a_l,
        [r"\bmaybe\b", r"\bperhaps\b", r"\bprobably\b", r"\bpossibly\b",
         r"\bi think\b", r"\bi believe\b", r"\bit seems\b", r"\bit depends\b"],
    )
    negation = _count_patterns(
        a_l,
        [r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bnone\b", r"\bcannot\b",
         r"\bcan't\b", r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b", r"\bwon't\b"],
    )
    nonreply = _count_patterns(
        a_l,
        [r"\bi (do not|don't|cannot|can't) (know|say|comment|answer)\b",
         r"\bno comment\b", r"\bnot going to answer\b", r"\bcan't answer\b",
         r"\bdecline to\b"],
    )
    deflection = _count_patterns(
        a_l,
        [r"\blet me\b", r"\bthe real\b", r"\bfirst of all\b", r"\blook\b",
         r"\bwell\b", r"\bwhat i can tell you\b"],
    )
    contrast = _count_patterns(a_l, [r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bon the other hand\b"])
    causal = _count_patterns(a_l, [r"\bbecause\b", r"\btherefore\b", r"\bas a result\b", r"\bso\b"])
    numeric = _count_patterns(a_l, [r"\b\d+([.,]\d+)?%?\b", r"\b(million|billion|trillion|percent|dollars?)\b"])
    temporal = _count_patterns(
        a_l,
        [r"\b(today|tomorrow|yesterday|week|month|year|years|months|days|hour|hours)\b",
         r"\b(19|20)\d{2}\b"],
    )
    person = _count_patterns(a_l, [r"\b(president|minister|secretary|senator|governor|mr|mrs|ms)\b"])
    location = _count_patterns(a_l, [r"\b(united states|america|china|russia|europe|country|countries|city|state)\b"])

    answer_starts_yes = _has_pattern(a_open, r"^(yes|yeah|absolutely|certainly|sure)\b")
    answer_starts_no = _has_pattern(a_open, r"^(no|not really|never)\b")
    answer_starts_uncertain = _has_pattern(a_open, r"^(maybe|perhaps|i don't know|i do not know|it depends)\b")
    answer_starts_deflect = _has_pattern(a_open, r"^(well|look|let me|first of all)\b")
    answer_starts_answer_to_question = _has_pattern(a_open, r"^(the answer|answering|to your question|on your question)\b")
    answer_mentions_second_question = _has_pattern(a_l, r"\b(second question|second part|first question|first part)\b")

    answer_has_i = _has_pattern(a_l, r"\b(i|me|my)\b")
    answer_has_we = _has_pattern(a_l, r"\b(we|our|us)\b")
    answer_has_you = _has_pattern(a_l, r"\b(you|your)\b")
    question_has_i = _has_pattern(q_l, r"\b(i|me|my)\b")
    question_has_we = _has_pattern(q_l, r"\b(we|our|us)\b")
    question_has_you = _has_pattern(q_l, r"\b(you|your)\b")

    q_yesno = _has_pattern(q_l, r"^(do|does|did|is|are|was|were|will|would|can|could|should|has|have|had)\b")
    q_why = _has_pattern(q_l, r"\bwhy\b")
    q_when = _has_pattern(q_l, r"\bwhen\b")
    q_where = _has_pattern(q_l, r"\bwhere\b")
    q_who = _has_pattern(q_l, r"\bwho\b")
    q_how = _has_pattern(q_l, r"\bhow\b")
    q_how_many = _has_pattern(q_l, r"\bhow many\b")
    q_how_much = _has_pattern(q_l, r"\bhow much\b")
    q_what = _has_pattern(q_l, r"\bwhat\b")
    polarity = answer_starts_yes or answer_starts_no or _has_pattern(a_l, r"\b(yes|no)\b")
    q_sentence_count = max(1, len(re.findall(r"[.!?]+", q)))
    a_sentence_count = max(1, len(re.findall(r"[.!?]+", a)))
    evidence_count = numeric + temporal + person + location + causal
    hedge_deflect_count = hedge + deflection + contrast
    answer_is_long = int(len(a_words) >= 160)
    answer_is_very_long = int(len(a_words) >= 260)
    answer_many_sentences = int(a_sentence_count >= 6)
    long_low_overlap = int(answer_is_long and _safe_div(len(inter), len(q_set)) < 0.18)
    long_with_evidence = int(answer_is_long and evidence_count > 0)
    long_with_uncertainty = int(answer_is_long and hedge > 0)
    long_with_deflection = int(answer_is_long and deflection > 0)
    first_deflect = re.search(r"\b(well|look|first of all|the real|let me)\b", a_l)
    first_evidence = re.search(
        r"\b(\d+|year|years|month|months|percent|million|billion|"
        r"dollars?|because|therefore|president|minister|united states|china|russia)\b",
        a_l,
    )
    evidence_after_deflection = int(
        deflection > 0
        and evidence_count > 0
        and first_deflect is not None
        and first_evidence is not None
        and first_deflect.start() < first_evidence.start()
    )
    starts_deflect_but_has_evidence = int(answer_starts_deflect and evidence_count > 0)
    evidence_and_uncertainty = int(evidence_count > 0 and hedge > 0)

    vals = [
        int(bool(row.get("multiple_questions", False))),
        int(bool(row.get("affirmative_questions", False))),
        int(bool(row.get("inaudible", False))),
        float(row.get("question_order", 0) if pd.notna(row.get("question_order", 0)) else 0),
        len(q_words),
        len(a_words),
        _safe_div(len(a_words), len(q_words)),
        abs(len(a_words) - len(q_words)),
        len(q),
        len(a),
        len(inter),
        _safe_div(len(inter), len(union)),
        _safe_div(len(inter), len(q_set)),
        _safe_div(len(inter), len(a_set)),
        _safe_div(len(inter), min(len(q_set), len(a_set))),
        1.0 - _safe_div(len(inter), len(q_set)),
        len(bi_inter),
        _safe_div(len(bi_inter), len(q_bi)),
        _safe_div(len(bi_inter), len(a_bi)),
        _safe_div(len(a_set), len(a_tok)),
        a.count("?"),
        q.count("?"),
        a.count("!"),
        a.count("...") + a.count("…"),
        a.count('"') + a.count("'"),
        q_sentence_count,
        a_sentence_count,
        _safe_div(len(a_words), a_sentence_count),
        sum(ch.isdigit() for ch in a),
        hedge,
        negation,
        nonreply,
        deflection,
        contrast,
        causal,
        numeric,
        temporal,
        person,
        location,
        answer_starts_yes,
        answer_starts_no,
        answer_starts_uncertain,
        answer_starts_deflect,
        answer_starts_answer_to_question,
        answer_mentions_second_question,
        answer_has_i,
        answer_has_we,
        answer_has_you,
        question_has_i,
        question_has_we,
        question_has_you,
        int(question_has_you and answer_has_we),
        int(question_has_you and answer_has_i),
        q_yesno,
        q_why,
        q_when,
        q_where,
        q_who,
        q_how,
        q_how_many,
        q_how_much,
        q_what,
        int(q_yesno and polarity),
        int(q_why and causal > 0),
        int(q_when and temporal > 0),
        int(q_where and location > 0),
        int(q_who and person > 0),
        int((q_how_many or q_how_much) and numeric > 0),
        shared_idf_sum,
        _safe_div(shared_idf_sum, q_idf_sum),
        _safe_div(shared_idf_sum, a_idf_sum),
        _tfidf_cosine(q_tok, a_tok, idf),
        _safe_div(len(rare_q_unanswered), len(rare_q)),
        answer_is_long,
        answer_is_very_long,
        answer_many_sentences,
        long_low_overlap,
        long_with_evidence,
        long_with_uncertainty,
        long_with_deflection,
        evidence_count - hedge_deflect_count,
        evidence_after_deflection,
        starts_deflect_but_has_evidence,
        evidence_and_uncertainty,
        _safe_div(evidence_count, len(a_words)),
    ]
    return [float(v) for v in vals]


def compute_numeric_features(df: pd.DataFrame, stats: Dict[str, list] = None) -> np.ndarray:
    idf = (stats or {}).get("idf", {})
    feats = np.asarray(
        [compute_numeric_feature_row(row, idf=idf) for _, row in df.iterrows()],
        dtype=np.float32,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _select_profile(raw: np.ndarray, profile: str) -> Tuple[np.ndarray, List[str]]:
    names = FEATURE_PROFILES.get(profile, FEATURE_PROFILES["base"])
    idx = [FEATURE_NAMES.index(name) for name in names]
    return raw[:, idx], names


def fit_numeric_feature_transform(
    df: pd.DataFrame, profile: str = "base"
) -> Tuple[np.ndarray, Dict[str, list]]:
    stats = {"feature_names": FEATURE_NAMES, "idf": _fit_idf(df)}
    raw = compute_numeric_features(df, stats=stats)
    raw, selected_names = _select_profile(raw, profile)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    stats["feature_names"] = selected_names
    stats["feature_profile"] = profile
    stats["mean"] = mean.tolist()
    stats["std"] = std.tolist()
    return ((raw - mean) / std).astype(np.float32), stats


def transform_numeric_features(df: pd.DataFrame, stats: Dict[str, list]) -> np.ndarray:
    raw = compute_numeric_features(df, stats=stats)
    profile = stats.get("feature_profile", "base")
    raw, _ = _select_profile(raw, profile)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((raw - mean) / std).astype(np.float32)


def save_feature_stats(stats: Dict[str, list], path) -> None:
    Path(path).write_text(json.dumps(stats, indent=2))


def load_feature_stats(path) -> Dict[str, list]:
    return json.loads(Path(path).read_text())
