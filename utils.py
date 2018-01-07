import numpy as np
import os
from nltk.tokenize import word_tokenize

def read_data(data_path):
  vocab = ["<EOT>", "<PAD>"] # EOT = end of text
  texts = []
  text_encode = []
  
  for file in os.listdir(data_path):
    with open(os.path.join(data_path, file)) as f:
      lines = f.readlines()
      lines = " ".join(lines[1:]) # line 1 for title of text
      if len(lines) < 1: continue
      temp_text = []
      temp_encode_text = []
      for word in word_tokenize(lines.strip()):
        if word not in vocab:
          vocab.append(word)
        temp_text.append(word)
        temp_encode_text.append(vocab.index(word))
      temp_encode_text.append(0)
      temp_text.append("<EOT>")
      
      texts.append(temp_text)
      text_encode.append(temp_encode_text)
  return vocab, text_encode, texts

def batch(inputs, max_sequence_length=None):
  """
  Args:
      inputs:
          list of sentences (integer lists)
      max_sequence_length:
          integer specifying how large should `max_time` dimension be.
          If None, maximum sequence length would be used
    
  Outputs:
      inputs_time_major:
          input sentences transformed into time-major matrix 
          (shape [max_time, batch_size]) padded with 0s
      sequence_lengths:
          batch-sized list of integers specifying amount of active 
          time steps in each input sequence
  """
    
  sequence_lengths = [len(seq) for seq in inputs]
  batch_size = len(inputs)
    
  if max_sequence_length is None:
    max_sequence_length = max(sequence_lengths)
    
  inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
  for i, seq in enumerate(inputs):
    for j, element in enumerate(seq):
      inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
  inputs_time_major = inputs_batch_major.swapaxes(0, 1)

  return inputs_time_major, sequence_lengths


def batch_iter(full_data, batch_size):
  """
  Args:
      full_data:
          list of sentences (integer lists)
      batch_size:
          size of one batch
    
  Outputs:
      sentences:
          list of sentences in one batch
      sentences_length:
          length of each sentence in batch
  """
  total_batch = len(full_data) // batch_size
  for i in range(total_batch + 1):
    batch_sentences = full_data[i:i+batch_size]
    sentences, sentences_length = batch(batch_sentences)
    yield sentences, sentences_length
