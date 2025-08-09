# backend/utils.py
from rapidfuzz import process, fuzz

def fuzzy_topic_match(query: str, candidates: list, limit=5):
    """
    candidates: list of strings (titles)
    Returns top matches with score
    """
    results = process.extract(query, candidates, scorer=fuzz.WRatio, limit=limit)
    return results
