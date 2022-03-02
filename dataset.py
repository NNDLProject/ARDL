# Make necessary imports
random_state = 0
import random
import numpy as np
random.seed(random_state)
np.random.seed(random_state)

import torch
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_20newsgroups 
import re



def get_train_test():
  # Training subset
  newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
  # Testing subset
  newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

  return newsgroups_train, newsgroups_test

categories = ['alt.atheism',
              'comp.graphics', 
              'comp.os.ms-windows.misc',
              'comp.sys.ibm.pc.hardware',  
              'comp.sys.mac.hardware',
              'comp.windows.x', 
              'misc.forsale', 
              'rec.autos', 
              'rec.motorcycles', 
              'rec.sport.baseball', 
              'rec.sport.hockey', 
              'sci.crypt', 
              'sci.electronics', 
              'sci.med', 
              'sci.space', 
              'soc.religion.christian', 
              'talk.politics.guns', 
              'talk.politics.mideast', 
              'talk.politics.misc', 
              'talk.religion.misc']



# Get index for each word
def get_word_2_index(vocab):
      word2index = {}
      for i, word in enumerate(vocab):
          word2index[word.lower()] = i
      return word2index

def get_embeddings_using_bow(newsgroups_train, newsgroups_test):
  """ Function to get vocabulary and indices for dataset """
  # Build a vocabulary
  vocab = Counter()

  # Iterate through training samples
  for text in newsgroups_train.data:
      for word in text.split(' '):
          vocab[word.lower()]+=1
  # Iterate through testing samples
  for text in newsgroups_test.data:
      for word in text.split(' '):
          vocab[word.lower()]+=1
  # Vocabulary size
  total_words = len(vocab)
  print("Vocabulary size [Bag-of-words]: ", total_words)

  word2index = get_word_2_index(vocab)

  return vocab, word2index

# Dictionary for merging similar classes together
dict_categories = {0: 0,
                   1: 1, 
                   2: 1,
                   3: 1,  
                   4: 1,
                   5: 1,
                   6: 2, 
                   7: 3, 
                   8: 3, 
                   9: 3, 
                   10: 3,
                   11: 4, 
                   12: 4, 
                   13: 4, 
                   14: 4,
                   15: 5, 
                   16: 6,
                   17: 6, 
                   18: 6, 
                   19: 6}

def clean(text):
  """ Function to clean the text """
  text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
  texter = re.sub(r"<br />", " ", text)
  texter = re.sub(r"&quot;", "\"",texter)
  texter = re.sub('&#39;', "\"", texter)
  texter = re.sub('\n', " ", texter)
  texter = re.sub(' u '," you ", texter)
  texter = re.sub('`',"", texter)
  texter = re.sub(' +', ' ', texter)
  texter = re.sub(r"(!)\1+", r"!", texter)
  texter = re.sub(r"(\?)\1+", r"?", texter)
  texter = re.sub('&amp;', 'and', texter)
  texter = re.sub('\r', ' ',texter)
  # Remove numbers from string
  texter = re.sub(pattern=r"[+-]?\d+(?:\.\d+)?", repl="", string=texter, count=0, flags=0)
  texter = texter.replace("  ", " ")
  clean = re.compile('<.*?>')
  texter = texter.encode('ascii', 'ignore').decode('ascii')
  texter = re.sub(clean, '', texter)
  if texter == "":
    texter = ""
  return texter

def get_word_2_index(vocab):
  """ Function to get index for each word """
  word2index = {}
  for i, word in enumerate(vocab):
    word2index[word.lower()] = i
  return word2index

def get_vocab_using_bow(newsgroups_train, newsgroups_test):
  """ Function to get vocabulary and indices for dataset """
  # Build a vocabulary
  vocab = Counter()

  # Iterate through training samples
  for text in newsgroups_train.data:
    text = clean(text)
    for word in text.split(' '):
      vocab[word.lower()]+=1
  # Iterate through testing samples
  for text in newsgroups_test.data:
    text = clean(text)
    for word in text.split(' '):
      vocab[word.lower()]+=1

  word2index = get_word_2_index(vocab)
  print("Vocabulary size [Bag-of-words]: ", len(vocab))
  return vocab, word2index

def get_vocab_using_glove(dim):
  """ Function to get vocabulary using GloVe Embeddings
  dim can take a value from 50, 100, 200, 300"""

  vocab = {}
  with open("/content/drive/MyDrive/Github/glove.6B.{}d.txt".format(dim), 'r') as f:
    for line in f:
      values = line.split()
      word = values[0]
      vector = np.asarray(values[1:], "float32")
      vocab[word] = vector

  print("Vocabulary size [GloVe]: ", len(vocab))
  return vocab, None

def get_batch(df, i, batch_size, vocab, emb, word2index):
  """ Function to convert text into embeddings for a batch of data 
  emb can take values "bow", "gloveGEN" and "gloveLSTM" correspoding 
  to Bag of words model, GloVe embeddings (one vector per paragraph) and
  GloVe embeddings (a matrix corresponding per paragraph) respectively.
  """
  batches = []
  results = []

  texts = df.data[i*batch_size : i*batch_size+batch_size]
  categories = df.target[i*batch_size : i*batch_size+batch_size]
  max_size = 1000
  for text in texts:
    text = clean(text)
    if emb == "bow":
      layer = np.zeros(len(vocab), dtype=float)
    elif emb == "gloveGEN":
      layer = np.zeros(len(vocab["a"]), dtype=float)
    elif emb == "gloveLSTM":
      layer = [[0 for j in range(300)]]*max_size
      i = 0

    for word in text.split(' '):
      # Computation for bag of words model
      if emb == "bow":
        layer[word2index[word.lower()]] += 1
      # Computation for GloVe embedding - general
      elif emb == "gloveGEN":
        if word in vocab:
          layer += vocab[word]
      # Computation for GloVe embedding - LSTM
      elif emb == "gloveLSTM":
        if word in vocab:
          # print(len(vocab[word]))
          layer[i] = np.array(vocab[word])

          # layer.append(vocab[word])
      else:
        print("### Invalid embedding type ###")
      
    # if isinstance(layer, list):
    #   layer = np.array(layer)

    batches.append(layer)

  for category in categories:
    index_y = dict_categories[category]
    results.append(index_y)
  
  return np.array(batches),np.array(results)
