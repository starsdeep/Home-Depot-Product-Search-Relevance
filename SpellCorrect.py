import requests
import re
import time
from random import randint
import json
import enchant
from enchant.checker import SpellChecker

"""
pyenchant 提供了基本的拼写检查，但是做拼写检查的时候，没有考虑到句子的context信息，所以效果并不好

SpellCheckGoogleOnline, SpellCheckGoogleOffline 来自于 https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos/notebook

"""
import config.project as project

class SpellCheckEnchant():

    def __init__(self):
        self.chkr = SpellChecker("en_US")

    def spell_correct(self, sentence):
        self.chkr.set_text(sentence)
        for err in self.chkr:
            suggest_words = self.chkr.suggest(err.word)
            if suggest_words:
                err.replace(suggest_words[0])
        return err.get_text()

class SpellCheckGoogleOffline():
    """
    先从https://www.kaggle.com/steubk/home-depot-product-search-relevance/fixing-typos/notebook  构造一个dict, 放到./input下，
    并命名为 google_spell_check_dict.json
    """
    def __init__(self):
        with open('%s/input/google_spell_check_dict.json' % project.project_path) as infile:
            self.spell_correct_dict = json.load(infile)
        print("load google spell check dict done. number of item: %d." % len(self.spell_correct_dict))

    def spell_correct(self, sentence):
        return self.spell_correct_dict[sentence] if sentence in self.spell_correct_dict else sentence


class SpellCheckGoogleOnline():
    START_SPELL_CHECK = "<span class=\"spell\">Showing results for</span>"
    END_SPELL_CHECK = "<br><span class=\"spell_orig\">Search instead for"

    HTML_Codes = (
        ("'", '&#39;'),
        ('"', '&quot;'),
        ('>', '&gt;'),
        ('<', '&lt;'),
        ('&', '&amp;'),
    )

    def __init__(self):
        pass

    def spell_correct(self, sentence):
        q = '+'.join(sentence.split())
        time.sleep(randint(0, 2))  # relax and don't let google be angry
        r = requests.get("https://www.google.co.uk/search?q=" + q)
        content = r.text
        start = content.find(self.START_SPELL_CHECK)
        if ( start > -1 ):
            start = start + len(self.START_SPELL_CHECK)
            end = content.find(self.END_SPELL_CHECK)
            search = content[start:end]
            search = re.sub(r'<[^>]+>', '', search)
            for code in self.HTML_Codes:
                search = search.replace(code[1], code[0])
            search = search[1:]
        else:
            search = sentence
        return search


if __name__ == '__main__':

    chkr1 = SpellCheckEnchant()
    chkr2 = SpellCheckGoogleOffline()
    chkr3 = SpellCheckGoogleOnline()

    text = "This is sme text with a fw speling errors in it."
    print(text)

    print("SpellCheckEnchant: %s" % chkr1.spell_correct(text))
    print("SpellCheckGoogleOffline: %s" % chkr2.spell_correct(text))
    # print("SpellCheckGoogleOnline: %s" % chkr3.spell_correct(text))



