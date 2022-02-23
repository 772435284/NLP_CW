

import os
os.system('python -m pip install nlpaug')
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nltk
import pandas as pd
from back_translation import translate_text

nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def augment_substitute_with_context(text):
    aug = naw.ContextualWordEmbsAug(action='substitute')
    return  aug.augment(text)

def augment_back_translation(text):
    aug = naw.BackTranslationAug()
    return aug.augment(text)

def augment_random_delete(text):
    aug = naw.RandomWordAug()
    return aug.augment(text)


def augment_back_translation(text):
    return translate_text(text)



