import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

class LSTMAutoEncoder():
  def __init__(self, vocab_size, encoder_hidden_units, input_embedding_size, learning_rate):
    self.vocab_size = vocab_size
    self.encoder_hidden_units = encoder_hidden_units
    self.decoder_hidden_units = encoder_hidden_units
    self.input_embedding_size = input_embedding_size
    self.learning_rate = learning_rate
    self._build_graph()

  def _create_placeholder(self):
    with tf.variable_scope("placeholder"):
      self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
      self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
    
  def _get_embedding(self):
    with tf.variable_scope("embedding"):
      self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1.0, 1.0), 
                                    dtype=tf.float32)
      self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
    
  def _encoder(self):
    with tf.variable_scope("encoder"):
      encoder_cell = LSTMCell(self.encoder_hidden_units)
      encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
        encoder_cell, self.encoder_inputs_embedded, self.encoder_inputs_length,
        dtype=tf.float32, time_major=True
      )
      return encoder_outputs, encoder_final_state
      
  def _decoder(self, encoder_final_state):
    with tf.variable_scope("decoder"):
      decoder_cell = LSTMCell(self.decoder_hidden_units)
      encoder_max_time, batch_size = tf.unstack(tf.shape(self.encoder_inputs))
      decoder_length = self.encoder_inputs_length

      W = tf.Variable(tf.random_uniform([self.decoder_hidden_units, self.vocab_size], -1, 1), dtype=tf.float32)
      b = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32)
      
      eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
      pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

      eos_step_embedded = tf.nn.embedding_lookup(self.embeddings, eos_time_slice)
      pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, pad_time_slice)
      
      def loop_fn_initial():
        initial_elements_finished = (0 >= decoder_length )  # all False at the initial step
        initial_input = eos_step_embedded
        initial_cell_state = encoder_final_state
        initial_cell_output = None
        initial_loop_state = None  # we don't need to pass any additional information
        return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)

      def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

        def get_next_input():
          output_logits = tf.add(tf.matmul(previous_output, W), b)
          prediction = tf.argmax(output_logits, axis=1)
          next_input = tf.nn.embedding_lookup(self.embeddings, prediction)
          return next_input
        elements_finished = (time >= decoder_length) # this operation produces boolean tensor of [batch_size]
        # defining if corresponding sequence has ended
        finished = tf.reduce_all(elements_finished) # -> boolean scalar
        input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
        state = previous_state
        output = previous_output
        loop_state = None
        return (elements_finished, 
                input,
                state,
                output,
                loop_state)

      def loop_fn(time, previous_output, previous_state, previous_loop_state):
        if previous_state is None:    # time == 0
          assert previous_output is None and previous_state is None
          return loop_fn_initial()
        else:
          return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
      
      decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
      decoder_outputs = decoder_outputs_ta.stack()
      
      decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
      decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
      decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
      decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.vocab_size))
      
      decoder_prediction = tf.argmax(decoder_logits, 2)
      
      return decoder_logits, decoder_prediction

  def _add_optimize(self):
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=tf.one_hot(self.encoder_inputs, depth=self.vocab_size, dtype=tf.float32),
      logits = self.decoder_logits
    )
    
    loss = tf.reduce_mean(stepwise_cross_entropy)
    train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
    return loss, train_op

  def _build_graph(self):
    self._create_placeholder()
    self._get_embedding()
    _, self.encoder_final_state = self._encoder()
    #self.code = tf.concat([encoder_final_state.h. encoder_final_state.c])
    self.decoder_logits, self.decoder_prediction = self._decoder(self.encoder_final_state)
    self.loss, self.train_op = self._add_optimize()

    
