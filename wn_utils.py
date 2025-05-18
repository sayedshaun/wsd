import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
nltk.download('wordnet', quiet=True)
from typing import Optional, Dict, Any, List, Union


def get_all_senses(row: pd.Series, pos_map: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Return up to `max_senses` for this row's lemma+POS.
    If you have a sense_key and want to enforce it exactly,
    you could insert it first and then fill up to `max_senses`
    with other candidates.
    """
    wn_pos = pos_map.get(row['target_pos'], None)
    if wn_pos:
        syns = wn.synsets(str(row['target_lemma']), pos=wn_pos)
    elif pd.notna(row['target_lemma']):
        syns = wn.synsets(str(row['target_lemma']))
    else:
        word = row['sentence'].split()[row['target_index_start']]
        syns = wn.synsets(str(word))

    return [s.definition() for s in syns]


def get_correct_sense(row: pd.Series) -> Union[str, None]:
    """
    Return the correct sense for this row's lemma+POS.
    """
    if 'sense_key' in row and pd.notna(row['sense_key']):
        lem = wn.lemma_from_key(row['sense_key'])
        syn = lem.synset()
        return syn.definition()


def fill_empty_sense(row: pd.Series) -> pd.Series:
    """
    Fill empty sense_list with the first sense of the word.
    """
    if not row['sense_list']:
        word = row['sentence'].split()[row['target_index_start']]
        syns = wn.synsets(str(word))
        row['sense_list'] = [s.definition() for s in syns]
    return row