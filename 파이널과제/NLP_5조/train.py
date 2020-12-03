# Requirements
from __future__ import unicode_literals, print_function, division
from torch import optim
import time
import math
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.switch_backend('agg')

##############################################################################
# PREPARE the Training Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10

# each pair : need input tensor (indexes of the words in the input sentence)
#                   + target tensor (indexes of the words in the target sentence)

# get indexes from sentence
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

# get tensor from sentence
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token) # append EOS token to both sequences
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# get tensors from pair
def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


##############################################################################
# Train the Model
# Step 1. run the input sentence thru the encoder
# + keep track of every output and the latest hidden state
# Step 2. decoder : first input - <SOS> token, first hidden state - encoder's last hidden state
# Step 3.

# the higher, the more teacher_forcing happens
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):

    # Initialize the values and states
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # Step 1 (run the encoder)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # Step 2 (decoder receives value from the encoder)
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Teacher Forcing : concept of using the "real" target outputs as each next input
    # instead of using decoder's guess as the next input
    # -> makes the model to converge faster
    # -> but may exhibit instability

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


##############################################################################
# 해당 모델이 얼마나 시간이 걸렸는지 + 얼마나 걸릴지 예상시간 출력해주는 함수들

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


##############################################################################
# 전체적인 Traingin Process
# 1. Timer 시작 (시간 측정)
# 2. Optimizer, criterion 초기화
# 3. Set of Training pairs 생성
# 4. 시각화를 위한 Empty losses array 채우기
# 5. train 과정 출력


def trainIters(encoder, decoder, metadata, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    input_lang, output_lang, pairs = metadata

    # Step 1
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Step 2
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # Step 3
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]
    # (Step 2 - continued)
    criterion = nn.NLLLoss()

    # Step 4 + Step 5
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


##############################################################################
# 시각화를 위한 함수

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


##############################################################################
# Evaluation
# - training과 거의 동일 - evaluation은 target이 없음
# -> simply feed the decoder's predictions back to itself for each step

def evaluate(encoder, decoder, metadata, sentence, max_length=MAX_LENGTH):
    input_lang, output_lang, _ = metadata
    with torch.no_grad():
        # Initialize the values and states
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # Step 1 (run the encoder)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        # Step 2 (decoder receives value from the encoder)
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)


        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            # every time it predicts a word, we add it to the output string
            # if a predicted word is EOS token -> stop there
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            # else, just append to the string
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

# to make some subjective quality judgements
def evaluateRandomly(encoder, decoder, metadata, n=10):
    _, _, pairs = metadata
    for i in range(n):
        # evaluate random sentences from the training set
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, metadata, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

##############################################################################
# For Better Visualization

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with color bar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(encoder, decoder, matadata, input_sentence):
    output_words, attentions = evaluate(encoder, decoder, matadata, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)
