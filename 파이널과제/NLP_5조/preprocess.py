# Requirements
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re


SOS_token = 0
EOS_token = 1

##############################################################################
# CLASS Lang
# 아래 두 개의 dict를 사용해 단어마다 고유의 index를 저장하고 사용할 수 있게 하는 클래스
## 1. word -> index(word2index)
## 2. index -> word(index2word)
# rare words : word2count dict를 이용해서 replace 한다.
class Lang:
    # initialize
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence): # sentence to word -> add
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index: # append to dictionaries
            self.word2index[word] = self.n_words
            self.word2count[word] = 1 # count # of words
            self.index2word[self.n_words] = word
            self.n_words += 1
        else: # if new to dict
            self.word2count[word] += 1


##############################################################################
# input 정제하는 함수 : unicodeToAscii, normalizeString

# 유니코드 input을 컴퓨터가 잘 이해할 수 있는 Ascii 언어로 바꿔주는 함수
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


##############################################################################
# input을 읽어들이는 함수 : readLangs

# reverse
## True : English -> Other language (translation)
## False : Other language -> English (translation)
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


##############################################################################
# 빠르게 학습시키기 위해서 data trimming 한다.

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

##############################################################################
# 위의 함수들을 이용해서 데이터를 준비하는 함수! (prepareData)
# 1. 텍스트 파일을 읽고 line으로 나눈 후, 그 line을 또 pair로 나눈다.
# 2. 텍스트를 normalize하고 length와 content로 filter한다.
# 3. 문장들로부터 단어 리스트를 만든다.

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse) # 1번
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs) # 2번
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs: # 3번
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
# print(random.choice(pairs))
