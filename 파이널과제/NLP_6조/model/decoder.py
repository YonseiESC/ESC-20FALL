import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_target_mask, create_position_vector

# MultiHeadAttention 두번 PositionWiseFeedForward 한번 진행되는 부분 ==> 디코더는 3개의 sublayers로 구성 
class DecoderLayer(nn.Module):
    
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(params) # masked multi head attention 부분 
        self.encoder_attention = MultiHeadAttention(params) # multi-head attention 부분
        self.position_wise_ffn = PositionWiseFeedForward(params) #FFNN sublayer

    def forward(self, target, encoder_output, target_mask, dec_enc_mask):
        # target          = [batch size, target 문장 길이, 512]
        # encoder_output  = [batch size, source 문장 길이, 512]
        # target_mask     = [batch size, target 문장 길이, 512]
        # dec_enc_mask    = [batch size, target 문장 길이, 512]

        # 1. masked multi-head attention
        output = target + self.self_attention(target, target, target, target_mask)[0]
        # masked multi-head attention + residual connection
        # masked multi-head attention 넣은 (F(X)) + 인풋값(x)  = F(x) + x
        norm_output = self.layer_norm(output)
        # layer normalization
        # 정리하면, norm_output은 masked multi-head attention, add&norm까지 완료함. 두번째 sublayer인 multi-head attention의 인풋값이 된다 
        
        
   
        # 2. Multi-head attention
        sub_layer, attn_map = self.encoder_attention(norm_output, encoder_output, encoder_output, dec_enc_mask)
        # decoder의 두번째 sublayer는 쿼리를 구성하기 위해 decoder 첫번째 sublayer output이 사용되며, key value를 구성하기 위해 인코더 아웃풋이 사용된다.
        output = output + sub_layer
        # Multi-head attention + residual connection
        # F(x) + x
        norm_output = self.layer_norm(output)
        # layer normalization
        # 정리하면, norm_output은 multi-head attention + add$norm까지 완료. FFNN 네트워크의 인풋값이 된다 
        
        # 3. Feed forward
        output = output + self.position_wise_ffn(norm_output)
        # FFNN + residual connectoin 
        # FFNN의 아웃풋(F(x))과 인풋값 x를 더해준다.
        output = self.layer_norm(output)
        # layer normalization
        # output = [batch size, 문장길이, 512]

        return output, attn_map


class Decoder(nn.Module):
    
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(params.output_dim, params.hidden_dim, padding_idx=params.pad_idx)
        # 임베딩 테이블 만드는 함수 
        # params.input_dim=단어 집합의 수 params.hidden_dim = 단어 임베딩 차원 = 512 padding_idx= 패딩을 하는 경우 패딩 인덱스를 알려줘야 함
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)
        # 가우시안 분포로부터 난수를 발생하여 weight matrix 초기화 
        self.embedding_scale = params.hidden_dim ** 0.5
        # 임베딩 벡터에 scaling해주는 부분(중요해보이진 않음)
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)
        #포지셔널 인코딩 하는 부분 

        self.decoder_layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.n_layer)])
        # decoderLayer는 masked multihead attention, multi-head attention, ffnn으로 구성되어 있다
        # 이를 n_layers 수만큼 반복 --> 여기서 n_layer = 6이다 
        self.dropout = nn.Dropout(params.dropout) # dropout --> 0.1

    def forward(self, target, source, encoder_output):
        # target              = [batch size, target 문장 길이]
        # source              = [batch size, source 문장 길이]
        # encoder_output      = [batch size, source 문장 길이, 512]
        
        # 1. 임베딩 행렬 구성 
        target_mask, dec_enc_mask = create_target_mask(source, target)
        # target_mask는 zero padding masking과 미래시점 참조 못하도록 하는 masking 포함
        # dec_enc_mask는 zero padding masking
        target_pos = create_position_vector(target)  # [batch size, target 문장 길이] positional encoding의 인풋값으로 target과 동일한 차원
        target = self.token_embedding(target) * self.embedding_scale
        # [batch size, target 문장 길이] 인풋값을 self.token_embedding(target) 통해서 임베딩 테이블 구축
        # self.embedding_scale를 통해 임베딩 벡터 scaling 
        # [batch size, target 문장 길이] --> [batch size, target 문장 길이, 512] 
        
        # 2. positional encoding
        target = self.dropout(target + self.pos_embedding(target_pos))
        # 임베딩 행렬에 positional encoding을 더해주어 위치정보를 포함시킴
        # [batch size, target 문장 길이, 512] 
        
        
        # 3. decoder layer
        for decoder_layer in self.decoder_layers:
            target, attention_map = decoder_layer(target, encoder_output, target_mask, dec_enc_mask)
        # 디코더 layer에 인풋값(target,encoder_output)을 넣는다. 인코더는 6개의 layer로 구성되어있기 때문에 반복해서 연산 수행
        # [batch size, 문장길이, 512] 

        output = torch.matmul(target, self.token_embedding.weight.transpose(0, 1))
        # dense layer 부분 [batch size, target 문장 길이, output dimension = corpus 단어 수]
        
        return output, attention_map
