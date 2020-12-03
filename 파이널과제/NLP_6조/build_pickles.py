import os
import pickle
import argparse

import pandas as pd
from pathlib import Path
from utils import convert_to_dataset

from torchtext import data as ttd
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer


def build_tokenizer():
    """
    입력되는 한국어 문장을 tokenize 할 soynlp tokenizer를 학습한다
    """
    print(f'Now building soy-nlp tokenizer . . .')

    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'corpus.csv')
    """
    학습되는 데이터가 있는 경로 지정 후 파일을 불러온다
    """
    df = pd.read_csv(train_file, encoding='utf-8')
    """
    text인 행만 분석한다
    """
    kor_lines = [row.korean
                 for _, row in df.iterrows() if type(row.korean) == str]
    """
    soynlp 모듈에서 가져온 WordExtractor 함수로 branching entropy, accessor variety, cohesion score의 단어 score 도출한다
    이 단어 score들은 각각 다른 방법으로 token의 경계를 찾는 값이다
    그 중 cohesion score(단어를 구성하는 글자들이 얼마나 같이 나오는지에 대한 값)만 추출한다.
    자세한 단어 score의 식과 코드는 https://github.com/lovit/soynlp/blob/master/tutorials/wordextractor_lecture.ipynb 에 자세히 나와있다.
    """
    word_extractor = WordExtractor(min_frequency=5)
    word_extractor.train(kor_lines)
    
    word_scores = word_extractor.extract()
    cohesion_scores = {word: score.cohesion_forward
                       for word, score in word_scores.items()}
    """
    pickle로 저장한다
    """
    with open('pickles/tokenizer.pickle', 'wb') as pickle_out:
        pickle.dump(cohesion_scores, pickle_out)


def build_vocab(config):
    """
    위에서 얻은 score를 이용하여 vocab을 만든다.
    """
    pickle_tokenizer = open('pickles/tokenizer.pickle', 'rb')
    cohesion_scores = pickle.load(pickle_tokenizer)
    tokenizer = LTokenizer(scores=cohesion_scores)
    """
    tokenizer 중 cohesion score를 기준으로 단어를 구분하는 LTokenizer을 사용한다.
    한국어 어절을 '명사/동사/형용사/부사'(L part) + '조사 등'(R part)으로 보고, 의미가 핵심적인 L part의 점수를 도출한다.
    """
    # Field를 통해 단어를 tokenize하고 tensor로 바꾼다.
    # Field에 대한 다양한 parameter에 대한 정보는 https://torchtext.readthedocs.io/en/latest/data.html 에서 얻을 수 있다.
    kor = ttd.Field(tokenize=tokenizer.tokenize,
                    lower=True,
                    batch_first=True)
    
    # 영어를 tokenize하는 함수는 spacy이다. 이후 항상 첫 token은 <sos>, 마지막 token은 <eos>로 지정한다.
    eng = ttd.Field(tokenize='spacy',
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True)

    data_dir = Path().cwd() / 'data'
    train_file = os.path.join(data_dir, 'train.csv')
    train_data = pd.read_csv(train_file, encoding='utf-8')
    train_data = convert_to_dataset(train_data, kor, eng)

    print(f'Build vocabulary using torchtext . . .')
    
    # 읽어 온 data를 한국어는 한국어 토큰으로, 영어는 영어 토큰으로 나누어 저장한다.
    kor.build_vocab(train_data, max_size=config.kor_vocab)
    eng.build_vocab(train_data, max_size=config.eng_vocab)
    
    # unique token 개수를 출력한다.
    print(f'Unique tokens in Korean vocabulary: {len(kor.vocab)}')
    print(f'Unique tokens in English vocabulary: {len(eng.vocab)}')
    
    # 가장 많이 쓰인 한국어/영어 단어를 출력한다.
    print(f'Most commonly used Korean words are as follows:')
    print(kor.vocab.freqs.most_common(20))
    print(f'Most commonly used English words are as follows:')
    print(eng.vocab.freqs.most_common(20))

    # 생성된 한국어/영어 vocab을 pickle로 저장한다
    with open('pickles/kor.pickle', 'wb') as kor_file:
        pickle.dump(kor, kor_file)

    with open('pickles/eng.pickle', 'wb') as eng_file:
        pickle.dump(eng, eng_file)

# name space에 kor_vocab와 eng_vocab으로 저장
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle Builder')

    parser.add_argument('--kor_vocab', type=int, default=55000)
    parser.add_argument('--eng_vocab', type=int, default=30000)

    config = parser.parse_args()

    build_tokenizer()
    build_vocab(config)
