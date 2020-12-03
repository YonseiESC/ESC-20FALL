# ESC 2020 FALL Final / 논문구현 Project
NLP TEAM5 (강동인, 김서영, 김수연, 손지우, 조민주)

# Nueral Machine Translation By Jointly Learing to Align And Translate
[참고사이트](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

#### 구성 및 사용법
1. main.py
2. preprocess.py
3. model.py
4. train.py
- Terminal에서 python main.py만 실행하면 된다.

## 1-1. Train Dataset (eng-fra.txt)
[다운로드 링크](https://download.pytorch.org/tutorial/data.zip)
|영어|프랑스어|
|------|----|
|Jump.|Saute. |
|No way! | En aucun cas. |
|I saw it. | Je l'ai vu. |
|I'm tall. | Je suis grande. |
|You're not tired, are you? | Vous n'êtes pas fatigué, si ? |
|I really appreciate you meeting with me. | J'apprécie vraiment que tu me rencontres. |
|He lived on crackers and water for three days. | Il a vécu de crackers et d'eau pendant trois jours. |
|She promised that she would meet him after school. | Elle lui promit qu'elle le rencontrerait après l'école. |

## 1-2. Test Dataset 
- 출처: TED (The Neurons that shaped Civilization by V.S.Ramachandran) [링크](https://www.ted.com/talks/vilayanur_ramachandran_the_neurons_that_shaped_civilization?language=ko)
- Selenium, Beautifulsoup 등을 활용한 크롤링

|예시|
|----|
|cela est reellement renversant .|
|l evolution darwinienne est lente elle prend des centaines de milliers d annees .|

## 2. Preprocessing
- 2-1. 유니코드 input을 컴퓨터가 잘 이해할 수 있는 Ascii 언어로 바꿔주기
- 2-2. Lowercase, trim, and remove non-letter characters
- 2-3(optional). 빠르게 학습시키기 위한 data trimming (ex. 10개 단어 이하 문장만 & 특정 문구로 시작하는 문장만 뽑기)

|전처리 전|전처리 후|
|-----|-----|
| Et c'est véritablement la chose la plus étonnante au monde. | et c est veritablement la chose la plus etonnante au monde . |
| C'est le résultat de notre compréhension de la neuroscience de base. | c est le resultat de notre comprehension de la neuroscience de base . | 

## 3. Model
한 줄 요약 : Seq2Seq with Attention (Encoder & Attention Decoder)

|model |description|
|----|----|
|Encoder| RNN that outputs some value for every word from the input sentence. For every input word the encoder outputs a vector and a hidden state, and uses the hidden state for the next input word.|
|Decoder| another RNN that takes the encoder output vector or vectors and outputs a sequence of words to create the translation.|
|Attention Decoder| let decoder to focus on a different part of the encoder's outputs.  |
| | 1. calculate a set of attention weights <- multiplied by the encoder output vectors |
| | 2. result(attn_applied) : info about the specific part of the input sequence |

## 4. Train
- Step 1. Timer 시작(시간 측정)
- Step 2. Optimizer(SGD), criterion 초기화
- Step 3. Trainding Data 준비하기 (get indexes from sentence / get tensor from sentence / get tensors from pair)
- Step 4. run the input sentence thru the encoder + keep track of every output and the latest hidden state
- Step 5. decoder : first input - <SOS> token, first hidden state - encoder's last hidden state (option: Teacher forcing)
- Step 6. Evaluation: evaluation은 target이 없음 -> simply feed the decoder's predictions back to itself for each step

## 5. Visualization (4. Train의 전체적인 Process 이후)
- Step 6. 시각화를 위한 Empty losses array 채우기
- Step 7. train 과정 출력
- 구체적인 시각화 결과는 발표자료를 통해 참고하기
