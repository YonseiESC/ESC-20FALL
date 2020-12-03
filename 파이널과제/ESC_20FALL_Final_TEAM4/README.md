ESC_20FALL_Final_TEAM4
=======================
# CNN을 이용한 MR(Movie Review) 데이터 감성 분석
### ESC NLP 4조 최정윤, 신예진, 김민규, 오다건

Pytorch re-implementation of Convolutional Neural Networks for Sentence Classification.
* Original github url : https://github.com/yuna-102/ESC_20FALL_Final_TEAM4.git

## Requirements

~~~
  nltk==3.2.5.
  torch==1.7.0
~~~

## 1. Dataset
> MR Dataset Sentiment Polarity Dataset Version 2.0  http://www.cs.cornell.edu/people/pabo/movie-review-data/
> |text|label|
> |------|----|
> |simplistic , silly and tedious .	|negative|
> |it's so laddish and juvenile , only teenage boys could possibly find it funny . |negative|
> |the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .|positive|
> |a masterpiece four years in the making .|positive|
> |together , tok and o orchestrate a buoyant , darkly funny dance of death . in the process , they demonstrate that there's still a lot of life in hong kong cinema . |positive|
> |interesting , but not compelling . |negative|
> |the action clich�s just pile up .|negative|
> |[a] rare , beautiful film .|positive|


## 2. Preprocessing
> 한글, 영문, 숫자, 괄호, 쉼표, 느낌표, 물음표, 작음따옴표, 역따옴표 제외한 나머지 모두 찾아서 공백(" ")으로 바꾸기
> |Preprocessing 전|Preprocessing 후|
> |------|----|
> |[a] rare , beautiful film . | a rare , beautiful film . |


## 3. Implementation
~~~
python main.py --mode rand
python main.py --mode static
python main.py --mode non-static
~~~
or
~~~
python main.py --help
~~~
You will get:

    usage: main.py [-h] [--batch-size BATCH_SIZE] [--dropout DROPOUT] 
                    [--epoch EPOCH] [--learning-rate LEARNING_RATE]
                    [----predict PREDICT] [--mode MODE]
                    [--num-feature-maps NUM_FEATURE_MAPS]
                    [--pretrained-word-vectors PRETRAINED_WORD_VECTORS]
                    [--save-word-vectors SAVE_WORD_VECTORS]

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE
      --dropout DROPOUT
      --epoch EPOCH
      --learning-rate LEARNING_RATE
      --mode MODE           available models: rand, static, non-static,
      --num-feature-maps NUM_FEATURE_MAPS
      --pretrained-word-vectors           available models: fasttext, Word2Vec
      --save-word-vectors SAVE_WORD_VECTORS           default:False
      
> 또는 cnn_for_sentence_classification.ipynb를 통해 colab으로 연결하여 바로 실행 해볼 수 있습니다.
* 실행하기 : https://colab.research.google.com/github/yuna-102/ESC_20FALL_Final_TEAM4/blob/main/cnn_for_sentence_classification.ipynb

## 4. Results
Baseline from the paper

> | Model | MR | 
> | ----- | -- | 
> | random | 76.1 | 
> | static | 81.0 | 
> | non-static | 81.5 | 


Re-implementation with Word2Vec and fasttext

> | Model | MR (Word2Vec) | MR (fasttext) |
> | ----- | -- | -- | 
> | random | *73.11 | *73.11 | 
> | static | 81.30 | 82.56 | 
> | non-static | 81.75| 82.65 |

  *cnn-rand의 경우 pre-trained word vector를 사용하지 않음


## 5. References

[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

[Kim's implementation of the model in Theano](https://github.com/yoonkim/CNN_sentence)

[Shawn's implementation of the model in Theano](https://github.com/Shawn1993/cnn-text-classification-pytorch)

[Chriskhanhtran's implementation of the model in Pytorch](https://chriskhanhtran.github.io/posts/cnn-sentence-classification/)


