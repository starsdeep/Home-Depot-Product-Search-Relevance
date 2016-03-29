#encoding=utf8
import re
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.metrics import edit_distance
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus.reader import wordnet
from sklearn.feature_extraction.text import CountVectorizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stop_w = {'for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'} #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
stopwords = set(stopwords.words('english')) | stop_w

bigram_vectorizer = CountVectorizer(ngram_range=(2, 2),token_pattern=r'\b\w+\b')
bigram_analyzer = bigram_vectorizer.build_analyzer()

def lemmatize(token, tag):
    try:
        morphy_tag = {'NN':wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}[tag[:2]]
        return lemmatizer.lemmatize(token, morphy_tag)
    except:
        return token

def main_title_extract(title):
    title = re.sub(r'\([^\)]*\)', '', title) # remove brackets
    title = re.sub(r'-DISCONTINUED$', '', title)
    title = re.sub(r' [w|W]ith .* [a|A]nd .*$', '', title) # remove str 'with ... and ...'
    title = re.sub(r' [f|F]or .*$', '', title) # remove str 'for ...'
    title = re.sub(r' [i|I]ncludes .*$', '', title) # remove str 'includes ...'
    title = re.sub(r' [w|W]ithout .*$', '', title) # remove str 'includes ...'
    title = re.sub(r' - .* [n|N]ot [i|I]ncluded\s*$', '', title) # remove str ' - ... not included'
    title = re.sub(r'  ', ' ', title) # remove spaces introduced

    prepositions=[(' - \S+\s*\S*\s*\S*\s*$', 0), (' - \D', 0), ('(\S+), ', 1), (' [u|U]p [t|T]o ', 0), (',* [w|W]ith ', 0), ('([^\d\s]+) [i|I]n ', 1), (',* [w|W]ith ', 0)] # sorted by occurences and confidence
   # if str behind a 'preposition' is shorter than str before, it may be not important
    for regex, preceding in prepositions: 
        m = list(re.finditer(regex, title) or [])
        while len(m)>0:
            str_left = title[:m[-1].start()]
            for i in range(1, preceding + 1):
                str_left += m[-1].group(i)
            if len(str_left.split()) * 2 + 1 >= len(title.split()):
                title = str_left
            else:
                break
            if regex[-1]=='$':
                m = list(re.finditer(regex, title) or [])
            else:
                m = m[:-1]

    kick_words = ['L', 'W', 'O\.D\.', 'OD', 'O\.C\.', 'OC', 'in\.', 'lb\.', 'lbs\.', 'x', '\d+[/\d]*', 'ft\.\.', 'ft\.', 'Lumens', 'CRI', 'Thick', '\+', 'AWG', 'oz.', 'Gauge', 'and', '\S+-\S+']
    for i in range(len(kick_words)):
        kick_words[i] = '^'+kick_words[i]+'$' # require full match
    title = title.split()
    while len(title)>0:
        matched = False
        for word in kick_words:
            if re.match(word, title[-1]):
                matched = True
                break
        if matched:
            title = title[:-1]
        else:
            break
    title = " ".join(title)

    title = re.sub(r'\s+$', '', title) # remove endings spaces
    return title

def model_number_extract(title):
    regex = "(?<=\s)#*[A-Z]+\d*-?\d+-*[A-Z]?(?=\s|$)"
    return " ".join(re.findall(regex, title))

def str_remove_stopwords(s):
    word_list = s.split()
    return ' '.join([word for word in word_list if word not in stop_w])


def str_is_meaningful(s):
    for t in s.split():
        is_number = re.match(r'^[0-9]+$', t)
        if (len(t)>2) and (not is_number) and (t not in stop_w):
            return True
    return False

def str_stem(s, by_lemmatizer=False):
    """
    :param s:
    :return: stemmed s

    transform words in sentence into basic forms,
    1. by cropping tails of words via stemmer
    or 
    2. by transforming words into different basic forms considering context via lemmatizer
    To understand their difference, see 4th answer in http://stackoverflow.com/questions/1787110/what-is-the-true-difference-between-lemmatization-vs-stemming

    Example:
    >>> str_stem("I have a gathering")
    'i have a gathering'
    >>> str_stem("I am gathering")
    'i be gather'

    """
    if not isinstance(s, str):
        return ""

    s = s.lower()
    s = s.replace("  "," ")

    # remove punctuations
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
    s = re.sub(r"^(\.|/)", r"", s)
    s = re.sub(r"(\.|/)$", r"", s)

    # remove seperators
    s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
    s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
    s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
    s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
    s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
    s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
    s = s.replace(" . "," ")

    # eliminate single number
    strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
    s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])

    # transform prepositions and measurements
    s = s.replace(" x "," xbi ")
    s = s.replace("*"," xbi ")
    s = s.replace(" by "," xbi ")
    s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?(?=\s|$)", r"\1in. ", s)
    s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?(?=\s|$)", r"\1ft. ", s)
    s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?(?=\s|$)", r"\1lb. ", s)
    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?(?=\s|$)", r"\1sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?(?=\s|$)", r"\1cu.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?(?=\s|$)", r"\1gal. ", s)
    s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?(?=\s|$)", r"\1oz. ", s)
    s = re.sub(r"([0-9]+)( *)(fl)\.? ?(oz)\.?(?=\s|$)", r"\1fl.oz. ", s)
    s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?(?=\s|$)", r"\1cm. ", s)
    s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?(?=\s|$)", r"\1mm. ", s)
    s = s.replace("°"," degrees(?=\s|$)")
    s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?(?=\s|$)", r"\1deg. ", s)
    s = re.sub(r"([0-9]+)( *)(volts|volt|v)\.?(?=\s|$)", r"\1volt. ", s)
    s = re.sub(r"([0-9]+)( *)(watts|watt|w)\.?(?=\s|$)", r"\1watt. ", s)
    s = re.sub(r"([0-9]+)( *)(yd|yds)\.?(?=\s|$)", r"\1yd. ", s)
    s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?(?=\s|$)", r"\1amp. ", s)
    measures=['hour','year','gauge','gpm','psi','hp','kw','qt','cfm','cc','vdc','btu','gpf','grit','ton','seer','tpi','tvl','awg','swe','mph','cri','lumens']#pvc, od, oc
    for m in measures:
        regex1 = "([0-9]+)( *)" + m + "\.?(?=\s|$)"
        regex2 = r"\1" + m + ". "
        s = re.sub(regex1, regex2, s)

    s = s.replace("  "," ") # consequent space may be added by above rules
    #s = (" ").join([z for z in s.split(" ") if z not in stop_w])

    # fix typos
    s = s.replace("&amp;", "&") # most '&' in title are turned to "&amp;"
    s = s.replace("&#39;", "'") # escaped in title
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

    # use lemmatizer or stemmer to stem words
    if by_lemmatizer:
        tagged_corpus = pos_tag(s.split())
        words = [lemmatize(token, tag) for token, tag in tagged_corpus]
    else:
        words = [stemmer.stem(z) for z in s.split()]
    return " ".join(words)

def words_of_str(str):
    return len(str.split())

def noun_of_str(tags):
    cnt = 0
    for key, tag in tags:
        if tag=='NN':
            cnt += 1
    return cnt

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


def num_common_word(str1, str2, ngram=1):
    """
    number of words in str1 that also in str2
    :param str1:
    :param str2:
    :return: cnt
    """
    words = []
    if ngram == 1:
        words = str1.split()
    elif ngram == 2:
        words = bigram_analyzer(str1)
    else:
        print(str(ngram) + " not supported yet")

    cnt = 0
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

def num_common_word_ordered(str1, str2):
    """
    number of words in str1 that also in str2, occuring in same order
    :param str1:
    :param str2:
    :return: cnt
    """
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
            str2 = str2[str2.find(word)+len(word):]
    return cnt

def match_last_k_noun(s, tags, k):
    """
    total number of noun(s) which is both in s and last k noun(s) of str2(tags is pos_tag of str2)
    :param str1:
    :param tags:
    :return: cnt
    """
    nouns, cnt = [], 0
    for key, tag in reversed(tags):
        if tag=='NN':
            nouns.append(key)
            if len(nouns)>=k:
                break
    for noun in nouns:
        if s.find(noun)>=0:
            cnt += 1
    return cnt

def num_common_noun(s, tags):
    """
    number of nouns in s that also in str2(tags is pos_tag of str2)
    :param str1:
    :param tags:
    :return: cnt
    """
    words, cnt = s.split(), .0
    for word in words:
        for key, tag in tags:
            if tag=='NN':
                if word==key:
                    cnt += 1
                    break
                if edit_distance(word, key)<2:
                    cnt += 0.5
                    break
    return cnt

def num_common_noun_ordered(s, tags):
    """
    number of nouns in s that also in str2(tags is pos_tag of str2)
    :param str1:
    :param tags:
    :return: cnt
    """
    words, cnt, idx = s.split(), .0, 0
    for word in words:
        for i in range(idx, len(tags)):
            if tags[i][1]=='NN' and word==tags[i][0]:
                cnt += 1
                idx = i+1
                break
    return cnt

def num_whole_word(word, str):
    """
    number of times that word(sentence) appears in str
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
    number of times that words of number and size appears in str
    :param word:
    :param str:
    :return: cnt
    """
    #size = ['cm.','in.','ft.','watt.','qt.','gal.','oz.','sq.ft.','cu.ft.','mm.','lb.','volt.','amp.']
    #reObj can match three types of strings, such as 2.3in. |  2.5x3sq.ft.  |   23x23
    reObj = re.compile('(([0-9]+(\.|\/)?[0-9]+)(cm\.|ft\.|in\.|watt.|qt.|gal.|oz.|sq.ft.|cu.ft.|mm.|lb.|volt.|amp.)|\
                        ([0-9]+(\.|\/)?[0-9]+)x([0-9]+(\.|\/)?[0-9]+)(cm\.|ft\.|in\.|watt.|qt.|gal.|oz.|sq.ft.|cu.ft.|mm.|lb.|volt.|amp.)|\
                        ([0-9]+(\.|\/)?[0-9]+)x([0-9]+(\.|\/)?[0-9]+))')

    num_size_list = reObj.findall(word)  

    cnt = 0
    for num_size in num_size_list:
        i = 0
        #print num_size[0]
        while i < len(str):
            i = str.find(num_size[0],i)
            if i == -1:
                break
            else:
                cnt += 1
                i += len(num_size[0])
    return cnt

def words_of_numsize_str(str):
    """
    number of times that number and size appears in str
    :param str:
    :return: number
    """
    #size = ['cm.','in.','ft.','watt.','qt.','gal.','oz.','sq.ft.','cu.ft.','mm.','lb.','volt.','amp.']
    #reObj can match three types of strings, such as 2.3in. |  2.5x3sq.ft.  |   23x23
    reObj = re.compile('(([0-9]+(\.|\/)?[0-9]+)(cm\.|ft\.|in\.|watt.|qt.|gal.|oz.|sq.ft.|cu.ft.|mm.|lb.|volt.|amp.)|\
                        ([0-9]+(\.|\/)?[0-9]+)x([0-9]+(\.|\/)?[0-9]+)(cm\.|ft\.|in\.|watt.|qt.|gal.|oz.|sq.ft.|cu.ft.|mm.|lb.|volt.|amp.)|\
                        ([0-9]+(\.|\/)?[0-9]+)x([0-9]+(\.|\/)?[0-9]+))')

    num_size_list = reObj.findall(str)  

    return len(num_size_list)

def count_er_word_in_(x):
    """
    function to count word which end with er x

    Example:

    >>> count_er_word_in_("col1 qfeqer 12 er adfer a")
    3
    >>> count_er_word_in_("col2 afdas 12")
    0
    """
    count = 0
    for word in x.split():
        if word[-2:] == "er":
            count += 1
    return count

def find_er_position(query, title):
    position = -1
    first_er_in_title = ""
    for word in query.split():
        if word[-2:] == "er":
            first_er_in_title = word
            break
    if first_er_in_title == "":
        return position

    for index, word in enumerate(title.split()):
        if word == first_er_in_title:
            position = index
            break
    return position

if __name__=='__main__':
#    import doctest
#    doctest.testmod()
    word = 'asdasdzxc 0.2in.  asdzxcz21x32qt. asdasdasd 3/4ft.asdasdaszxc 23watt.'
    str = 'asdasdzczxczxc 0.2in.  asdzxccxcz21x32qt. adasd 3/4ft.asd3/4ft.aasczxc3/4ft.'
    cnt = num_size_word(word, str)
    #print cnt
