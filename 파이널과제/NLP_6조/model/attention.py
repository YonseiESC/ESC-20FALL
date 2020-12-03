import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        assert params.hidden_dim % params.n_head == 0
        """
        self_attentions : self-attention을 num_head번 반복하도록 선언
        """
        self.attentions = nn.ModuleList([SelfAttention(params) for _ in range(params.n_head)])

        """
        self.o_w : 가중치 행렬 선언 및 초기화
        """
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        init_weight(self.o_w)
        """
        self.dropout : Dropout 선언
        """
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]
        # self_attentions = [batch size, sentence length, attention dim] * num head
        """
        weighted_vs : 어텐션 값 행렬 (어텐션 헤드)
        attentions : 어텐션 스코어 행렬
        """
        weighted_vs = [weighted_v[0] for weighted_v in self_attentions]
        attentions = [weighted_v[1] for weighted_v in self_attentions]

        """
        weighted_v : 어텐션 헤드들을 concatenate한 것
        """
        weighted_v = torch.cat(weighted_vs, dim=-1)
        # weighted_v = [batch size, sentence length, hidden dim]

        """
        어텐션 헤드들을 concatenate한 것에 가중치 행렬을 곱하여 최종 output
        """
        output = self.dropout(self.o_w(weighted_v))
        # output = [batch size, sentence length, hidden dim]

        return output, attentions


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.hidden_dim // params.n_head

        """
        self.q_w, self.k_w, self.v_w : Q, K, V에 대한 가중치 행렬 선언 및 초기화
        """
        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        init_weight(self.q_w)
        init_weight(self.k_w)
        init_weight(self.v_w)

        """
        self.dropout : Dropout 선언
        """
        self.dropout = nn.Dropout(params.dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device)

    def forward(self, query, key, value, mask=None):
        # query, key, value = [batch size, sentence length, hidden dim]

        # create Q, K, V matrices using identical input sentence to calculate self-attention score
        """
        각 가중치 행렬을 곱하여 Q, K, V 행렬 구하기
        q, k, v : Q, K, V 행렬
        """
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)
        # q, k, v = [batch size, sentence length, attention dim]

        """
        어텐션 함수(scaled dot product attention)를 사용하여 어텐션 스코어 구하기
        self_attention : 어텐션 스코어 행렬
        """
        self_attention = torch.bmm(q, k.permute(0, 2, 1))
        self_attention = self_attention / self.scale_factor
        # self_attention = [batch size, sentence length, sentence length]

        """
        마스킹
        """
        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf)

        # normalize self attention score by applying soft max function on each row
        """
        소프트맥스 함수를 사용하여 어텐션 분포 구하기
        norm_attention_score : 어텐션 분포 + Dropout
        """
        attention_score = F.softmax(self_attention, dim=-1)
        norm_attention_score = self.dropout(attention_score)
        # attention_score = [batch size, sentence length, sentence length]

        # compute "weighted" value matrix using self attention score and V matrix
        """
        어텐션 분포와 V 행렬을 곱하여 어텐션 값 행렬 구하기
        weighted_v : 어텐션 값 행렬
        """
        weighted_v = torch.bmm(norm_attention_score, v)
        # weighted_v = [batch size, sentence length, attention dim]

        return self.dropout(weighted_v), attention_score
