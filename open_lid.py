#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path
import fasttext

# suppress warning:
# Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
fasttext.FastText.eprint = lambda x: None

class LIDModel:

    # mapping from open-lid lang codes to IETF lang codes
    # https://github.com/laurieburchell/open-lid-dataset/blob/main/languages.md
    long_to_short = {
        'spa_Latn': 'es', # 3850
        'eng_Latn': 'en', # 3627
        'rus_Cyrl': 'ru', # 754
        'deu_Latn': 'de', # 351
        'zho_Hans': 'zh', # 330
        'fra_Latn': 'fr', # 259
        'cat_Latn': 'ca', # 250
        'tha_Thai': 'th', # 167
        'por_Latn': 'pt', # 164
        'ita_Latn': 'it', # 113
        'isl_Latn': 'is',
        'hin_Deva': 'hi',
        'ell_Grek': 'el',
        'slv_Latn': 'sl',
        'jpn_Jpan': 'ja',
        'vie_Latn': 'vi',
        'kor_Hang': 'ko',
        'bul_Cyrl': 'bg',
        'swe_Latn': 'sv',
        'hun_Latn': 'hu',
        'fin_Latn': 'fi',
    }

    short_to_long = {v: k for k, v in long_to_short.items()}

    def __init__(self, model_path: str = 'resources/lid/lid201-model.bin'):

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = fasttext.load_model(model_path)

    def _predict(self, text):
        return self.model.predict(text, k=1)
    
    def predict(self, text):

        text = self.preprocess(text)

        pred = self._predict(text)

        label = pred[0][0].split("__")[-1]

        confidence = pred[1][0]

        return label, confidence

    def get_short_tag(self, lang):
        return self.long_to_short.get(lang, lang)

    def get_long_tag(self, lang):
        return self.short_to_long.get(lang, lang)

    def preprocess(self, text):
        text = re.sub('\n', ' ', text)
        text = re.sub('(### Human:| ### Assistant:)', ' ', text).strip()
        return text

if __name__ == "__main__":
    
    model = LIDModel('resources/lid/lid201-model.bin')
    
    texts = [
        '### Human: Hola, ¿cómo estás?### Assistant:',
        '### Human: Hello, how are you?### Assistant:',
        '### Human: 你好吗？### Assistant:',
        '### Human: Hallo, wie geht es dir?### Assistant:',
        '### Human: Привет, как дела?### Assistant:',
        '### Human: Bonjour, comment allez-vous?### Assistant:',
    ]
    
    for text in texts:
        print(model.predict(text))