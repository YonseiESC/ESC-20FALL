import os
import argparse
import datetime
import torch
import model
import train
import dataset
import pretrained_vectors
from sklearn.model_selection import train_test_split
import save_embeddings

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', default=50, type=int)
  parser.add_argument('--dropout', default=0.5, type=float)
  parser.add_argument('--epoch', default=20, type=int)
  parser.add_argument('--learning-rate', default=0.1, type=float)
  parser.add_argument("--mode", default="non-static", help="available models: rand, static, non-static")
  parser.add_argument('--num-feature-maps', default=100, type=int) 
  parser.add_argument("--pretrained-word-vectors", default="fasttext", help="available models: fasttext, Word2Vec")
  parser.add_argument("--save-word-vectors", action='store_true', default=False, help='save trained word vectors')
  parser.add_argument("--predict", action='store_true', default=False, help='classify your sentence')
  args = parser.parse_args()
  
  # load data
  print("Load data...\n")
  texts, labels = dataset.load_data()

  print("Tokenizing...\n")
  tokenized_texts, word2idx, max_len = dataset.tokenize(texts)
  input_ids = dataset.encode(tokenized_texts, word2idx, max_len)
  train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.1, random_state=42)

  print("Creating Dataloader...\n")
  train_dataloader, val_dataloader = dataset.data_loader(train_inputs, val_inputs, train_labels, val_labels, batch_size = args.batch_size)
  


  if args.mode == 'rand':
    # CNN-rand: Word vectors are randomly initialized.
    train.set_seed(42)
    cnn_model, optimizer = model.initilize_model(vocab_size=len(word2idx),
                                          embed_dim=300,
                                          learning_rate=args.learning_rate,
                                          dropout=args.dropout)
    train.train(cnn_model, optimizer, train_dataloader, val_dataloader, epochs=args.epoch)


  elif args.mode == 'static':
    # CNN-static: fastText pretrained word vectors are used and freezed during training.
    train.set_seed(42)
    embeddings = pretrained_vectors.get_embeddings(word2idx, args.pretrained_word_vectors )
    cnn_model, optimizer = model.initilize_model(pretrained_embedding=embeddings,
                                            freeze_embedding=True,
                                            learning_rate=args.learning_rate,
                                            dropout=args.dropout)
    train.train(cnn_model, optimizer, train_dataloader, val_dataloader, epochs=args.epoch)

  else:
    # CNN-non-static: fastText pretrained word vectors are fine-tuned during training.
    train.set_seed(42)
    embeddings = pretrained_vectors.get_embeddings(word2idx, args.pretrained_word_vectors )
    cnn_model, optimizer = model.initilize_model(pretrained_embedding=embeddings,
                                                freeze_embedding=False,
                                                learning_rate=args.learning_rate,
                                                dropout=args.dropout)
    train.train(cnn_model, optimizer, train_dataloader, val_dataloader, epochs=args.epoch)


  if args.save_word_vectors == True:
    save_embeddings.write_embeddings('trained_embeddings_{}.txt'.format(args.mode),
    cnn_model.embedding.weight.data,
    word2idx)

  if args.predict == True:
    x = input('영어 텍스트를 입력하세요! : ')
    x = str(x)
    train.predict(x, cnn_model, word2idx)


  

if __name__ == '__main__':
	main()