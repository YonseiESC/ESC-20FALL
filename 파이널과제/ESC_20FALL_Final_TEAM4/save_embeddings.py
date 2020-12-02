from tqdm import tqdm

def write_embeddings(path, embeddings, vocab):
    
    with open(path, 'w') as f:
        for i, embedding in enumerate(tqdm(embeddings)):
            vocab = {v: k for k, v in vocab.items()}
            word = vocab.get(i)
            #skip words with unicode symbols
            if word is not None:
              vector = ' '.join([str(i) for i in embedding.tolist()])
              f.write(f'{word} {vector}\n')

