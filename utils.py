import numpy as np

def read_data(data_path):
  vocab = ["<EOS>", "<PAD>"]
  sentences = []
  sentence_encode = []
  with open(data_path) as f:
    for line in f:
      if len(line) < 1: continue
      temp_sentence = []
      temp_encode_sentence = []
      for word in line.strip().split():
        if word not in vocab:
          vocab.append(word)
        temp_sentence.append(word)
        temp_encode_sentence.append(vocab.index(word))
      temp_encode_sentence.append(0)
      temp_sentence.append("<EOS>")
      
      sentences.append(temp_sentence)
      sentence_encode.append(temp_encode_sentence)
  return vocab, sentence_encode, sentences

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
