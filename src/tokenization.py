import re

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizerBase

QUESTION_SPECIAL_TOKENS = ["[AFFIRM_Q]", "[MULTI_Q]"]
NEGATION_SPECIAL_TOKEN = "[NEG]"
ENGINEERED_CUE_TOKENS = [
    "[Q_YESNO]",
    "[Q_WHY]",
    "[Q_WHEN]",
    "[Q_WHERE]",
    "[Q_WHO]",
    "[Q_HOW]",
    "[Q_WHAT]",
    "[A_YES_OPEN]",
    "[A_NO_OPEN]",
    "[A_UNCERTAIN_OPEN]",
    "[A_DEFLECT_OPEN]",
    "[A_NONREPLY]",
    "[A_HEDGE]",
    "[A_CONTRAST]",
    "[A_CAUSAL]",
    "[A_NUMERIC]",
    "[A_TEMPORAL]",
    "[OV_LOW]",
    "[OV_MID]",
    "[OV_HIGH]",
    "[A_SHORT]",
    "[A_LONG]",
]

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "do", "does", "did", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "there", "they",
    "this", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
}


def ensure_special_tokens(tokens: list, tokenizer, model) -> int:
    """Προσθέτω tokens στο tokenizer vocab και κάνω resize τα model embeddings.
    Επιστρέφω πόσα νέα tokens προστέθηκαν.
    """
    added = tokenizer.add_special_tokens({"additional_special_tokens": list(tokens)})
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return added


def add_question_special_tokens(tokenizer, model) -> int:
    return ensure_special_tokens(QUESTION_SPECIAL_TOKENS, tokenizer, model)


def build_question_prefix(row) -> str:
    """Φτιάχνω το prefix string από τα dataset booleans. Κενό string αν καμία flag."""
    parts = []
    if bool(row.get("multiple_questions", False)):
        parts.append("[MULTI_Q]")
    if bool(row.get("affirmative_questions", False)):
        parts.append("[AFFIRM_Q]")
    return " ".join(parts)


def apply_question_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Επιστρέφω αντίγραφο του df με prefix στο question column."""
    df = df.copy()
    prefixes = df.apply(build_question_prefix, axis=1)
    df["question"] = (prefixes + " " + df["question"].astype(str)).str.strip()
    return df


def _simple_tokens(text: str) -> list:
    return [
        t for t in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]


def _has_any(text: str, patterns: list) -> bool:
    text = str(text).lower()
    return any(re.search(p, text) for p in patterns)


def build_engineered_cue_prefix(row) -> str:
    """Build compact HW1-inspired cue tokens for a question-answer pair.

    The point is not to replace the transformer text, but to expose high-level
    signals that worked well in the previous linear system: question intent,
    answer evidence/style, and lexical alignment between question and answer.
    """
    q = str(row.get("question", ""))
    a = str(row.get("answer", ""))
    q_l = q.lower().strip()
    a_l = a.lower().strip()
    a_open = " ".join(a_l.split()[:10])
    parts = []

    if re.match(r"^(do|does|did|is|are|was|were|will|would|can|could|should|has|have|had)\b", q_l):
        parts.append("[Q_YESNO]")
    if re.search(r"\bwhy\b", q_l):
        parts.append("[Q_WHY]")
    if re.search(r"\bwhen\b", q_l):
        parts.append("[Q_WHEN]")
    if re.search(r"\bwhere\b", q_l):
        parts.append("[Q_WHERE]")
    if re.search(r"\bwho\b", q_l):
        parts.append("[Q_WHO]")
    if re.search(r"\bhow\b", q_l):
        parts.append("[Q_HOW]")
    if re.search(r"\bwhat\b", q_l):
        parts.append("[Q_WHAT]")

    if re.match(r"^(yes|yeah|absolutely|certainly|sure)\b", a_open):
        parts.append("[A_YES_OPEN]")
    if re.match(r"^(no|not really|never)\b", a_open):
        parts.append("[A_NO_OPEN]")
    if _has_any(a_open, [r"\bi don'?t know\b", r"\bi'?m not sure\b", r"\bit depends\b", r"\bmaybe\b"]):
        parts.append("[A_UNCERTAIN_OPEN]")
    if _has_any(a_open, [r"\bwell\b", r"\blet me\b", r"\blook\b", r"\bthe real\b", r"\bfirst of all\b"]):
        parts.append("[A_DEFLECT_OPEN]")

    if _has_any(a_l, [r"\bi (do not|don't|cannot|can't) (know|say|comment|answer)\b", r"\bno comment\b", r"\bnot going to\b", r"\bcan't answer\b"]):
        parts.append("[A_NONREPLY]")
    if _has_any(a_l, [r"\bmaybe\b", r"\bperhaps\b", r"\bprobably\b", r"\bpossibly\b", r"\bi think\b", r"\bi believe\b", r"\bit seems\b"]):
        parts.append("[A_HEDGE]")
    if _has_any(a_l, [r"\bbut\b", r"\bhowever\b", r"\balthough\b", r"\bon the other hand\b"]):
        parts.append("[A_CONTRAST]")
    if _has_any(a_l, [r"\bbecause\b", r"\btherefore\b", r"\bso\b", r"\bas a result\b"]):
        parts.append("[A_CAUSAL]")
    if re.search(r"\b\d+([.,]\d+)?%?\b|\b(million|billion|trillion|percent)\b", a_l):
        parts.append("[A_NUMERIC]")
    if _has_any(a_l, [r"\b(today|tomorrow|yesterday|week|month|year|years|months|days)\b", r"\b(19|20)\d{2}\b"]):
        parts.append("[A_TEMPORAL]")

    q_tokens = set(_simple_tokens(q))
    a_tokens = set(_simple_tokens(a))
    coverage = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
    if coverage < 0.12:
        parts.append("[OV_LOW]")
    elif coverage < 0.34:
        parts.append("[OV_MID]")
    else:
        parts.append("[OV_HIGH]")

    a_len = len(a.split())
    if a_len < 35:
        parts.append("[A_SHORT]")
    elif a_len > 180:
        parts.append("[A_LONG]")

    return " ".join(dict.fromkeys(parts))


def apply_engineered_cue_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix answer text with deterministic feature tokens."""
    df = df.copy()
    prefixes = df.apply(build_engineered_cue_prefix, axis=1)
    df["answer"] = (prefixes + " " + df["answer"].astype(str)).str.strip()
    return df


_MONTH_RE = (
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|"
    r"aug\.?|sep\.?|sept\.?|oct\.?|nov\.?|dec\.?)\b"
)


def normalize_surface_words(text: str) -> str:
    """Aggressively replace surface evidence with ordinary English words.

    This is the DeBERTa-friendly version of the HW1 canonical-token idea:
    we avoid private symbols like [A_YEAR] and use words already seen in
    pretraining. The goal is to preserve the kind of evidence, not its value.
    """
    text = "" if text is None else str(text)
    text = re.sub(r"\b(can|could|do|does|did|is|are|was|were|will|would|should|has|have|had)n't\b", r"\1 not", text, flags=re.I)
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.I)
    text = re.sub(r"\bcan't\b", "can not", text, flags=re.I)

    # Longer / more specific patterns first.
    text = re.sub(r"[$€£]\s?\d+(?:[,.]\d+)*(?:\.\d+)?\s?(?:million|billion|trillion|m|bn|b)?", " money amount ", text, flags=re.I)
    text = re.sub(r"\b\d+(?:[,.]\d+)*(?:\.\d+)?\s?(?:dollars?|euros?|pounds?|usd|eur|gbp)\b", " money amount ", text, flags=re.I)
    text = re.sub(r"\b\d+(?:\.\d+)?\s?(?:%|percent|percentage points?)\b", " percent ", text, flags=re.I)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:-|–|—|to)\s*\d+(?:\.\d+)?\b", " number range ", text, flags=re.I)
    text = re.sub(rf"{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?", " date ", text, flags=re.I)
    text = re.sub(rf"\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_RE}(?:\s+\d{{4}})?", " date ", text, flags=re.I)
    text = re.sub(rf"{_MONTH_RE}", " month ", text, flags=re.I)
    text = re.sub(r"\b(?:18|19|20)\d{2}\b", " year ", text)
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " ordinal number ", text, flags=re.I)
    text = re.sub(r"\b\d+(?:[,.]\d+)*(?:\.\d+)?\b", " number ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_surface_word_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize question and answer with known English words."""
    df = df.copy()
    df["question"] = df["question"].astype(str).map(normalize_surface_words)
    df["answer"] = df["answer"].astype(str).map(normalize_surface_words)
    return df


def _content_token_set(text: str) -> set:
    return set(_simple_tokens(text))


def _sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    return [p.strip() for p in parts if p and p.strip()]


def build_focused_answer(row, edge_words: int = 80, max_overlap_sentences: int = 3) -> str:
    """Shorten long answers to opening + question-relevant sentences + closing.

    This targets the observed validation errors: long answers with mixed
    evidence, hedging, and deflection. Short answers are left unchanged.
    """
    question = str(row.get("question", ""))
    answer = str(row.get("answer", ""))
    words = answer.split()
    if len(words) <= edge_words * 2:
        return answer

    q_tokens = _content_token_set(question)
    sentences = _sentences(answer)
    ranked = []
    for pos, sent in enumerate(sentences):
        s_tokens = _content_token_set(sent)
        overlap = len(q_tokens & s_tokens)
        coverage = overlap / max(len(q_tokens), 1)
        evidence = int(bool(re.search(
            r"\b(year|month|date|money amount|percent|number|number range|because|therefore|reason)\b",
            sent.lower(),
        )))
        ranked.append((coverage, overlap, evidence, -pos, sent))

    selected = []
    for _, _, _, _, sent in sorted(ranked, reverse=True):
        if sent not in selected:
            selected.append(sent)
        if len(selected) >= max_overlap_sentences:
            break

    opening = " ".join(words[:edge_words])
    closing = " ".join(words[-edge_words:])
    middle = " ".join(selected)
    focused = f"{opening} {middle} {closing}"
    return re.sub(r"\s+", " ", focused).strip()


def apply_focused_answer_view(df: pd.DataFrame) -> pd.DataFrame:
    """Replace long answers with an opening/relevant-sentences/closing view."""
    df = df.copy()
    df["answer"] = df.apply(build_focused_answer, axis=1)
    return df


def build_evidence_word_prefix(row) -> str:
    """Natural-language cue prefix using only ordinary words."""
    q = str(row.get("question", ""))
    a = str(row.get("answer", ""))
    q_l = q.lower()
    a_l = a.lower()
    a_open = " ".join(a_l.split()[:12])

    evidence = []
    if re.search(r"\bmoney amount\b|\b(dollars?|euros?|pounds?)\b", a_l):
        evidence.append("money")
    if re.search(r"\bpercent\b", a_l):
        evidence.append("percent")
    if re.search(r"\byear\b|\bmonth\b|\bdate\b|\b(today|tomorrow|yesterday)\b", a_l):
        evidence.append("time")
    if re.search(r"\bnumber\b|\bnumber range\b", a_l):
        evidence.append("number")
    if re.search(r"\b(because|therefore|reason|as a result)\b", a_l):
        evidence.append("reason")
    if re.search(r"\b(president|minister|secretary|senator|governor|mr|mrs|ms)\b", a_l):
        evidence.append("person")
    if re.search(r"\b(united states|america|china|russia|europe|country|countries|city|state)\b", a_l):
        evidence.append("place")

    style = []
    if re.search(r"^(yes|yeah|absolutely|certainly|sure)\b", a_open):
        style.append("yes")
    if re.search(r"^(no|not really|never)\b", a_open):
        style.append("no")
    if re.search(r"\b(maybe|perhaps|probably|possibly|i think|i believe|it depends|not sure)\b", a_l):
        style.append("uncertain")
    if re.search(r"\b(no comment|not going to answer|can not answer|do not know|decline to)\b", a_l):
        style.append("non reply")
    if re.search(r"\b(well|look|first of all|the real question)\b", a_open):
        style.append("deflection")

    qtype = []
    if re.match(r"^(do|does|did|is|are|was|were|will|would|can|could|should|has|have|had)\b", q_l):
        qtype.append("yes no question")
    for word in ["why", "when", "where", "who", "how", "what"]:
        if re.search(rf"\b{word}\b", q_l):
            qtype.append(f"{word} question")

    evidence_text = " ".join(dict.fromkeys(evidence)) or "none"
    style_text = " ".join(dict.fromkeys(style)) or "plain"
    qtype_text = " ".join(dict.fromkeys(qtype)) or "general question"
    return f"question type {qtype_text}. answer evidence {evidence_text}. answer style {style_text}."


def apply_evidence_word_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Prepend a compact natural-language evidence/style summary to answer."""
    df = df.copy()
    prefixes = df.apply(build_evidence_word_prefix, axis=1)
    df["answer"] = (prefixes + " " + df["answer"].astype(str)).str.strip()
    return df


_spacy_nlp = None


def _get_spacy():
    """Lazy-load en_core_web_sm. Κάνω import spacy μόνο όταν χρειάζεται."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' δεν είναι εγκατεστημένο. "
                "Εκτέλεσε: python -m spacy download en_core_web_sm"
            ) from e
    return _spacy_nlp


def mark_negations(text: str) -> str:
    """Προσθέτω [NEG] token πριν από verbs που έχουν negation dependent.
    Χρησιμοποιώ spaCy dependency parse. Αν το spacy model λείπει, πετάει error.
    """
    if not text:
        return text
    nlp = _get_spacy()
    doc = nlp(text)
    # indices των heads που έχουν neg dependent
    neg_heads = {tok.head.i for tok in doc if tok.dep_ == "neg"}
    if not neg_heads:
        return text
    pieces = []
    for tok in doc:
        if tok.i in neg_heads:
            pieces.append(f"{NEGATION_SPECIAL_TOKEN} {tok.text_with_ws}")
        else:
            pieces.append(tok.text_with_ws)
    return "".join(pieces)


def apply_negation_markers(df: pd.DataFrame, answer_col: str = "answer") -> pd.DataFrame:
    """Επιστρέφω αντίγραφο του df με [NEG] markers στα answers."""
    df = df.copy()
    df[answer_col] = df[answer_col].astype(str).map(mark_negations)
    return df


def tokenize_pairs(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    input_fmt: str = "two_segment",
    label_col: str = "label_id",
) -> TensorDataset:
    """Tokenize Q+A pairs. Returns TensorDataset of (input_ids, attention_mask, labels).

    Το `label_col` επιλέγει τί στόχο βάζω στα labels: "label_id" για clarity
    3-class, "evasion_id" για evasion 9-class training.
    """
    questions = df["question"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()

    if input_fmt == "two_segment":
        enc = tokenizer(
            questions,
            answers,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    elif input_fmt == "concat_sep":
        sep = tokenizer.sep_token or "[SEP]"
        texts = [f"{q} {sep} {a}" for q, a in zip(questions, answers)]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        raise ValueError(f"Unknown input_fmt: {input_fmt!r}")

    labels = torch.tensor(df[label_col].values, dtype=torch.long)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], labels)


def tokenize_pairs_with_features(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    input_fmt: str,
    features,
    label_col: str = "label_id",
) -> TensorDataset:
    """Tokenize Q+A pairs and attach standardized numeric features."""
    base = tokenize_pairs(df, tokenizer, max_length, input_fmt, label_col=label_col)
    input_ids, attention_mask, labels = base.tensors
    feature_tensor = torch.tensor(features, dtype=torch.float)
    return TensorDataset(input_ids, attention_mask, labels, feature_tensor)


def tokenize_dual_view_pairs_with_features(
    df_full: pd.DataFrame,
    df_second: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    input_fmt: str,
    features,
    label_col: str = "label_id",
) -> TensorDataset:
    """Tokenize two Q/A views and attach one shared numeric feature vector."""
    base_a = tokenize_pairs(df_full, tokenizer, max_length, input_fmt, label_col=label_col)
    base_b = tokenize_pairs(df_second, tokenizer, max_length, input_fmt, label_col=label_col)
    input_ids_a, attention_mask_a, labels = base_a.tensors
    input_ids_b, attention_mask_b, _ = base_b.tensors
    feature_tensor = torch.tensor(features, dtype=torch.float)
    return TensorDataset(
        input_ids_a,
        attention_mask_a,
        input_ids_b,
        attention_mask_b,
        labels,
        feature_tensor,
    )


def tokenize_pairs_multitask(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    input_fmt: str = "two_segment",
) -> TensorDataset:
    """Like tokenize_pairs αλλά επιστρέφει (input_ids, attention_mask, clarity_id, evasion_id).
    Για multi-task training με αυξιλιαρικό evasion head.
    """
    questions = df["question"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()

    if input_fmt == "two_segment":
        enc = tokenizer(
            questions, answers, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
    elif input_fmt == "concat_sep":
        sep = tokenizer.sep_token or "[SEP]"
        texts = [f"{q} {sep} {a}" for q, a in zip(questions, answers)]
        enc = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
    else:
        raise ValueError(f"Unknown input_fmt: {input_fmt!r}")

    clarity = torch.tensor(df["label_id"].values, dtype=torch.long)
    evasion = torch.tensor(df["evasion_id"].values, dtype=torch.long)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], clarity, evasion)
