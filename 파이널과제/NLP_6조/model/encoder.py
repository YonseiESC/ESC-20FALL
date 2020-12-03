import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector


# multihead attention, feed forward netword
class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6) #layer normalization 진행하는 부분
        self.self_attention = MultiHeadAttention(params) # MultiheadAttention 진행하는 부분
        self.position_wise_ffn = PositionWiseFeedForward(params) # PositionWiseFeedForward 네트워크 통과 

    def forward(self, source, source_mask):
        # source          = [batch size, 문장길이, 512]
        # source_mask     = [batch size, 문장길이, 문장길이]
        
        #1. Multi-head attention
        output = source + self.self_attention(source, source, source, source_mask)[0]
        # Multihead attention + residual connection
        # Multi-head Attention에 넣은(F(X)) + 인풋값(x)  = F(x) + x

        normalized_output = self.layer_norm(output) 
        # layer normalization 
        # 정리하면, normalized_output은 multihead attention, add&norm까지 완료함. FFNN의 인풋값이 된다 
        
        #2. Feed forward
        output = normalized_output + self.position_wise_ffn(normalized_output) 
        # FFNN + residual connectoin 
        # FFNN의 아웃풋(F(x))과 인풋값 x를 더해준다.
        
        output = self.layer_norm(output)
        # layer normalization
        # output = [batch size, 문장길이, 512]

        return output


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx) 
        # 임베딩 테이블 만드는 함수 
        # params.input_dim=단어 집합의 수 params.hidden_dim = 단어 임베딩 차원 = 512 padding_idx= 패딩을 하는 경우 패딩 인덱스를 알려줘야 함
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        # 가우시안 분포로부터 난수를 발생하여 weight matrix 초기화 
        self.embedding_scale = params.hidden_dim ** 0.5
        # 임베딩 벡터에 scaling해주는 부분(중요해보이진 않음)
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True) #포지셔널 인코딩 하는 부분 

        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)]) 
        # EncoderLayer는 multihead attention, feed forward netword sublayers로 구성됨(앞부분에서 정의)
        # 이를 n_layer 수만큼 반복해준다 --> 여기서 n_layer = 6이다 
        
        self.dropout = nn.Dropout(params.dropout) #dropout --> 0.1


    def forward(self, source):
        # source = [batch size, 문장 길이] embedding matrix로 만들기 이전 상태 
        source_mask = create_source_mask(source)      # [batch size, 문장 길이, 문장 길이]
        source_pos = create_position_vector(source)   # [batch size, 문장 길이] positional encoding의 인풋값으로 source와 동일한 차원 
        
        # 1. 임베딩 행렬 구성 
        source = self.token_embedding(source) * self.embedding_scale 
        # [batch size, 문장 길이] 인풋값을 self.token_embedding(source) 통해서 임베딩 테이블 구축
        # self.embedding_scale를 통해 임베딩 벡터 scaling 
        # [batch size, 문장 길이] --> [batch size, 문장길이, 512] 
        
        # 2. positional encoding
        source = self.dropout(source + self.pos_embedding(source_pos))
        # 임베딩 행렬에 positional encoding을 더해주어 위치정보를 포함시킴
        # [batch size, 문장길이, 512] 

        # 3. encoding layer
        for encoder_layer in self.encoder_layers:
            source = encoder_layer(source, source_mask)
        # 인코더 layer에 인풋값을 넣는다. 인코더는 6개의 layer로 구성되어있기 때문에 반복해서 연산 수행
        # [batch size, 문장길이, 512] 
        
        return source
