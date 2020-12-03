# Requirements
from __future__ import unicode_literals, print_function, division
from preprocess import prepareData
from model import EncoderRNN, AttnDecoderRNN
from train import trainIters, evaluateRandomly, evaluate, evaluateAndShowAttention
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device :", device, "\n")
    # Preprocess data
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print("Finished Preprocessing\n")
    # Seq2Seq Model
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
    metadata = (input_lang, output_lang, pairs)
    trainIters(encoder1, attn_decoder1, metadata, n_iters=500, print_every=100)  # 원래는 n_iters=75000, print_every=5000

    # Check
    evaluateRandomly(encoder1, attn_decoder1, metadata)

    # Evaluate and Visualize
    output_words, attentions = evaluate(encoder1, attn_decoder1, metadata, "je suis trop froid .")
    plt.matshow(attentions.numpy())

    evaluateAndShowAttention(encoder1, attn_decoder1, metadata, "elle a cinq ans de moins que moi .")
    evaluateAndShowAttention(encoder1, attn_decoder1, metadata, "elle est trop petit .")
    evaluateAndShowAttention(encoder1, attn_decoder1, metadata, "je ne crains pas de mourir .")
    evaluateAndShowAttention(encoder1, attn_decoder1, metadata, "c est un jeune directeur plein de talent .")


if __name__ == "__main__":
    # parsing
    run()
