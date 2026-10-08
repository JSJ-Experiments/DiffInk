"""Nested dataset/capacity gate for the standalone direct text research mapper."""
from .ocr_pool import normalized_text

DATA = 'checkpoints/iam_generation_coverage/20261008-045818'
SOURCE_H5_SHA = 'd513ec4e81330685055d006b5527043bac7e07d16d688251d41621af43648c71'
DATASET_SHA = 'b27631cc38d6f8d4a8b38c7160959d2c3bbcf406fae3e8ac9fb7e74c703e39f4'
WHITENING_SHA = 'cf7075533ca56e6ef1e4eaaa345590776eef86c0c198453c12857f1d4dffbadc'
ARMS = ('small128', 'small256', 'larger256')


def validate_budget(steps):
    if type(steps) is not int or not 2000 <= steps <= 24000:
        raise ValueError('bounded 2000–24000 updates per arm required')


def select_capacity(data, sizes=(128, 256)):
    a, b = sizes
    if not 41 < a < b <= 1024:
        raise ValueError('nested intermediate sizes required')
    broad = data['training_ids']['broad1024']; held = data['splits']['unseen_prompt']
    records = data['records']; retained = data['training_ids']['small32']
    same = data['training_ids']['writer_all']
    if len(broad) < b or len(set(broad)) != len(broad):
        raise ValueError('unique sufficient source scope required')
    small, large = broad[:a], broad[:b]
    if not set(retained) <= set(same) <= set(small) or set(large) & set(held):
        raise ValueError('original same-writer scope must be retained, held excluded')
    forms = {records[i]['prompt_family'] for i in held}
    texts = {normalized_text(records[i]['text']) for i in held}
    if any(records[i]['prompt_family'] in forms or normalized_text(records[i]['text']) in texts for i in large):
        raise ValueError('global held form/text leakage')
    return dict(training_ids=dict(small128=small, small256=large, larger256=list(large)),
                splits=dict(retained_train=list(retained), added_same_writer=[i for i in same if i not in retained],
                            new_train128=[i for i in small if i not in same], expansion256=large[a:],
                            unseen_prompt=list(held), all_train128=small, all_train256=large))


def model_config(vocab_size, writer_count, larger=False):
    return dict(channels=384, vocab_size=vocab_size, writer_count=writer_count,
                width=256 if larger else 128, depth=6 if larger else 4, heads=4)
