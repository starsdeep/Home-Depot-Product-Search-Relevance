#encoding=utf8
import re
from nltk.stem.porter import *
from nltk.metrics import edit_distance
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus.reader import wordnet

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def lemmatize(token, tag):
    try:
        morphy_tag = {'NN':wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}[tag[:2]]
        return lemmatizer.lemmatize(token, morphy_tag)
    except:
        return token

stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}


def str_stem(s, by_pos_tag=False):
    """
    :param s:
    :return: stemmed s

    stem sentence

    Example:
    >>> str_stem("I have a gathering")
    'i have a gathering'
    >>> str_stem("I am gathering")
    'i be gather'

    """
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = re.sub(r"([0-9]),([0-9])", r"\1\2", s)
        s = s.replace(","," ") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])

        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air condition")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        if by_pos_tag:
            tagged_corpus = pos_tag(s.split())
            words = [lemmatize(token, tag) for token, tag in tagged_corpus]
        else:
            words = [stemmer.stem(z) for z in s.split()]
        return " ".join(words)
    else:
        return ""


def len_of_str(str):
    return len(str.split())

def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r


def num_common_word(str1, str2):
    """
    number of words in str1 that also in str2
    :param str1:
    :param str2:
    :return: cnt
    """
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
        # count 0.5 for words unfound but edit distance<2
        if cnt==0 and len(word)>3:
            s1 = [z for z in list(set(str2.split(" "))) if abs(len(z)-len(word))<2]
            t1 = sum([1 for z in s1 if edit_distance(z, word)<2])
            if t1 > 1:
                cnt+=0.5
    return cnt

def num_whole_word(word, str):
    """
    number of times that word appears in str
    :param word:
    :param str:
    :return: cnt
    """
    cnt = 0
    i = 0
    if len(word.split())==0:
        return 0
    while i < len(str):
        i = str.find(word, i)
        if i == -1:
            return cnt
        else:
            cnt += 1
            i += len(word)
    return cnt


def num_size_word(word, str):
    """
    number of times that words fo number and size appears in str
    :param word:
    :param str:
    :return: cnt
    """
    size = ['cm','m','foot','inch']
    wordlist = word.split(' ')
    num_size_list = []
    for i in range(1,len(wordlist)):
        if wordlist[i] in size and (isinstance(wordlist[i-1],int) or isinstance(wordlist[i-1],float)):
            num_size_list.add(wordlist[i-1]+' '+wordlist[i]);
    
    cnt = 0
    for word in num_size_list:
        i = 0
        while i < len(str):
            i = str.find(word, i)
            if i == -1:
                return cnt
            else:
                cnt += 1
                i += len(word)
    return cnt

if __name__=='__main__':
    import doctest
    doctest.testmod()
