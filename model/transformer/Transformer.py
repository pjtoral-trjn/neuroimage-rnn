import tensorflow as tf
from model.transformer.Encoder import Encoder
from model.transformer.Decoder import Decoder
from model.rnn.rnn import get_head_layer
from model.cnn.tcnn import tcnn
from tensorflow.keras.layers import TimeDistributed

class Transformer(tf.keras.Model):
  def __init__(self, *, args, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.args = args

    images = tf.keras.Input((self.args.sequence_length, self.args.width, self.args.height, self.args.depth, 1),
                            batch_size=self.args.batch_size)

    self.feature_extractor = TimeDistributed(self.get_feature_extractor(),input_shape=(self.args.width, self.args.height,
                                                                                       self.args.depth, 1),
                                             batch_size=self.args.batch_size)

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = get_head_layer(args)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the

    # first argument.
    x = self.feature_extractor(inputs)
    context = self.encoder(x)  # (batch_size, context_len, d_model)
    x = self.decoder(x, context)  # (batch_size, target_len, d_model)
    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits

  def get_feature_extractor(self):
      # Load Embedding Model
      tcnn_full_model = tcnn(self.args)
      tcnn_full_model.set_weights(tf.keras.models.load_model(self.args.embedding_model_pathway, compile=False).get_weights())
      embedding_model = tf.keras.models.Sequential(name="3DCNN")

      # Pop head off Embedding Model to Output Embedding and NOT HEAD
      for layer in tcnn_full_model.layers[0:-1]:
          embedding_model.add(layer)

      # Any Trainable Layers in the Embedding CNN? None currently activated
      i = 0
      trainable_layers = round(len(embedding_model.layers) * 0.0)
      for layer in embedding_model.layers:
          if i < len(embedding_model.layers) - trainable_layers:
              layer.trainable = False
          elif i >= len(embedding_model.layers) - trainable_layers:
              layer.trainable = True
          i += 1
      print("\n\n\n Embedding Model (Feature Extractor)")
      embedding_model.summary()
      return embedding_model