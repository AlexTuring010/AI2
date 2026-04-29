import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizerBase

QUESTION_SPECIAL_TOKENS = ["[AFFIRM_Q]", "[MULTI_Q]"]
NEGATION_SPECIAL_TOKEN = "[NEG]"


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
