import torch.nn as nn
import torch.nn.functional as F

from model.ops import init_weight


class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super(PositionWiseFeedForward, self).__init__()
        # nn.Conv1d input = (batch size , # of channels)
	"""
        Multi Head Attention에서 각 head가 만들어낸 self-attention을 치우치치 않게 균등하게 섞는 역할
	선형변환이 position마다 동일하게 적용이 되지만, layer마다 다른 파라미터를 사용한다. 이를 kernel size가 1d인 2 개의 convolution들로 나타냄
	"""
	self.conv1 = nn.Conv1d(params.hidden_dim, params.feed_forward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(params.feed_forward_dim, params.hidden_dim, kernel_size=1)
        init_weight(self.conv1)
        init_weight(self.conv2)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        # x = [batch size, sentence length, hidden dim]

	# nn.Conv1d에 적용하기 위해 x의 차원 재구성 (transpose는 2D에만 사용가능하기 때문에 permute )
        x = x.permute(0, 2, 1)                        # x = [batch size, hidden dim, sentence length]
        
	"""
	f1 = xW1 + b1 (linear)
	f2 = max(0,f1) (activation relu)   #논문대로 relu를 사용했지만, gelu가 성능이 더 좋다고 함.
	논문에 구체적으로 언급되어있지 않지만 transformer에 2개의 dropout이 더 추가됨.
	position-wise feed-forward network에서 ReLU 이후 (1) attention에서 SoftMax 이후 (2)
	"""
	
        output = self.conv2(output)                   # output = [batch size, hidden dim, sentence length)
        output = self.dropout(F.relu(self.conv1(x)))  # output = [batch size, feed forward dim, sentence length)
        
	"""
	f3 = f2W2 + b2  (linear)
	"""
        # 원래대로 원상복구
        output = output.permute(0, 2, 1)              # output = [batch size, sentence length, hidden dim]
        return self.dropout(output)
