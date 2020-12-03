import pickle
import numpy as np
import torch
import torch.nn as nn

pickle_eng = open('pickles/eng.pickle', 'rb')
eng = pickle.load(pickle_eng)
pad_idx = eng.vocab.stoi['<pad>']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_subsequent_mask(target):
    """
    if target length is 5 and diagonal is 1, this function returns
        [[0, 1, 1, 1, 1],
         [0, 0, 1, 1, 1],
         [0, 0, 0, 1, 1],
         [0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0]]
    :param target: [batch size, target length]
    
    ## look ahead mask: 자기 자신보다 미래에 있는 단어들에 대한 마스킹 적용
    """
    batch_size, target_length = target.size()

    # torch.triu: 상삼각행렬 with diagonal=사용자 지정
    subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().to(device)
    # subsequent_mask = [target length, target length]

    # subsequent_mask를 batch size만큼 반복 (batch에 있는 데이터를 다 다루기 위해)
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    # subsequent_mask = [batch size, target length, target length]

    return subsequent_mask


def create_source_mask(source):
    """
    create masking tensor for encoder's self attention
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :return: source mask
    
    #padding mask: 실질적인 의미를 갖지 않는 <pad> 토큰을 마스킹
    boolean tensor에 대해서는 true를 가진 position들이 입력으로 취급하지 않음.
    즉, <pad>만 true
    
    """
    source_length = source.shape[1]

    # <pad> 토큰을 마스킹할 boolean tensors 생성 (source & target 문장 모두에 대해서)
    source_mask = (source == pad_idx)
    # source_mask = [batch size, source length]

    # 문장길이만큼 반복
    source_mask = source_mask.unsqueeze(1).repeat(1, source_length, 1)
    # source_mask = [batch size, source length, source length]

    return source_mask


def create_target_mask(source, target):
    """
    create masking tensor for decoder's self attention and decoder's attention on the output of encoder
    if sentence is [2, 193, 9, 27, 10003, 1, 1, 1, 3] and 2 denotes <sos>, 3 denotes <eos> and 1 denotes <pad>
    masking tensor will be [False, False, False, False, False, True, True, True, False]
    :param source: [batch size, source length]
    :param target: [batch size, target length]
    :return:
    """
    target_length = target.shape[1]

    subsequent_mask = create_subsequent_mask(target)
    # subsequent_mask = [batch size, target length, target length]

    source_mask = (source == pad_idx)
    target_mask = (target == pad_idx)
    # target_mask    = [batch size, target length]

    # 문장길이만큼 반복
    dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1)
    target_mask = target_mask.unsqueeze(1).repeat(1, target_length, 1)

    # decoder의 self attention: padding mask O + look ahead mask O ==> 결합 (attention mask)
    target_mask = target_mask | subsequent_mask
    # target_mask = [batch size, target length, target length]
    return target_mask, dec_enc_mask


def create_position_vector(sentence):
    """
    create position vector which contains positional information
    0th position is used for pad index
    :param sentence: [batch size, sentence length]
    :return: [batch size, sentence length]
    """
    # sentence = [batch size, sentence length]
    batch_size, _ = sentence.size()
    pos_vec = np.array([(pos+1) if word != pad_idx else 0
                        for row in range(batch_size) for pos, word in enumerate(sentence[row])])
    pos_vec = pos_vec.reshape(batch_size, -1)
    pos_vec = torch.LongTensor(pos_vec).to(device)
    return pos_vec


def create_positional_encoding(max_len, hidden_dim):
    """
    position encoding값 구하기: 단어의 상대적 위치에 대한 정보 제공
    PE(pos, 2i)     = sin(pos/10000 ** (2*i / hidden_dim)) : 짝수
    PE(pos, 2i + 1) = cos(pos/10000 ** (2*i / hidden_dim)) : 홀수 
    """
    sinusoid_table = np.array([pos / np.power(10000, 2 * i / hidden_dim)
                               for pos in range(max_len) for i in range(hidden_dim)])
    # sinusoid_table = [max len * hidden dim]

    sinusoid_table = sinusoid_table.reshape(max_len, -1)
    # sinusoid_table = [max len, hidden dim]

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 짝수 차원
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 홀수 차원

    # numpy에서 torch.tensor로 변환 후에 'batch size' 만큼 반복
    sinusoid_table = torch.FloatTensor(sinusoid_table).to(device)
    sinusoid_table[0] = 0.

    return sinusoid_table


def init_weight(layer):
    nn.init.xavier_uniform_(layer.weight)  # Xavier Initialization 사용
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
