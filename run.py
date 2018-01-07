import utils
from StandardModel.Model import LSTMAutoEncoder
import tensorflow as tf
import numpy as np
import dill
import os
import datetime

tf.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.flags.DEFINE_integer("n_epochs", 100, "num epochs")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_integer("embedding_dim", 300, "word embedding dimession")
tf.flags.DEFINE_integer("hidden_dim", 300, "hidden dim of network")
tf.flags.DEFINE_string("data_path", "data", "data path")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("out_dir", "runs/", "path to save checkpoint")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS

if __name__ == "__main__":
  vocab, sentence_encode, _ = utils.read_data(FLAGS.data_path)
  vocab_size = len(vocab)
  print("size of dictionary:{} word\nsize of corpus:{} sentences".format(vocab_size, len(sentence_encode)))
  if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)
  checkpoint_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "checkpoints"))
  dill.dump(vocab,open(os.path.join(FLAGS.out_dir, "dictionary.pkl"),'wb')) 
  tf.reset_default_graph()
  session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)

  with tf.Session(config=session_conf) as sess:
    Model = LSTMAutoEncoder(vocab_size, FLAGS.hidden_dim, FLAGS.embedding_dim, FLAGS.learning_rate)
    print("model created")
    global_step = tf.Variable(0, name="global_step", trainable=False)
    increment_global_step_op = tf.assign(global_step, global_step+1)
    loss_summary = tf.summary.scalar("loss", Model.loss)
    summary_dir = os.path.join(FLAGS.out_dir, "summaries")
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    

    sess.run(tf.global_variables_initializer())

    def train_step(session,Model, X_batch, X_batch_length, epoch):
      
      feed_dict={
        Model.encoder_inputs : X_batch,
        Model.encoder_inputs_length: X_batch_length,        
      }

      _, loss, step, summary = sess.run([Model.train_op, Model.loss,  increment_global_step_op, loss_summary]
                                             , feed_dict=feed_dict)
      time_str = datetime.datetime.now().isoformat()
      print("{}: epoch:{} step {}, loss {:g}".format(time_str, epoch, step, loss))
      summary_writer.add_summary(summary, step)
    

      def encode_transform_step(session, Model, X, X_length):
        feed_dict={
          Model.encoder_inputs : X,
          Model.encoder_inputs_length: X_length,        
        }
        final = sess.run([Model.encoder_final_state], feed_dict=feed_dict)
        #return np.concatenate((final[0],final[1]),1)
        #print(len(final))
        #return final
        return np.concatenate((final[0].c, final[0].h), 1)
      
    for e in range(FLAGS.n_epochs):
      for X_batch, X_batch_length in utils.batch_iter(sentence_encode, FLAGS.batch_size):
        train_step(sess, Model, X_batch, X_batch_length, e)
      save_path = saver.save(sess, os.path.join(checkpoint_dir, "checkpoint"), epoch)
      print("Model saved in file: %s" % save_path)  
      
