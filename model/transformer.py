import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from model.tcnn import tcnn
from model.rnn import get_head_layer
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    # Feed Forward Part
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(inputs.shape[-1])(ff)
    ff = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return ff

def embedding_model(args):
    images = tf.keras.Input((args.sequence_length, args.width, args.height, args.depth, 1), batch_size=args.batch_size)

    # Load Embedding Model
    tcnn_full_model = tcnn(args)
    tcnn_full_model.set_weights(tf.keras.models.load_model(args.embedding_model_pathway, compile=False).get_weights())
    embedding_model = tf.keras.models.Sequential(name="3DCNN")

    # Pop head off Embedding Model to Output Embedding and NOT HEAD
    for layer in tcnn_full_model.layers[0:-1]:
        embedding_model.add(layer)

    # Any Trainable Layers in the Embedding CNN? No trainable layers currently activated, essentially trusting the
    # pretraining task to extract task specific related features
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

def transformer(args):
    embeddings = tf.keras.Input((args.sequence_length, 9600), batch_size=args.batch_size)
    x = transformer_encoder(embeddings, head_size=64, num_heads=4, ff_dim=256, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.1)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = get_head_layer(args)(x)

    model = tf.keras.Model(inputs=embeddings, outputs=outputs, name="Transformer")
    print("\n\n\n ----- Transformer -----")
    print(model.summary())
    return model


def transformer_model(args):
    images = tf.keras.Input((args.sequence_length, args.width, args.height, args.depth, 1), batch_size=args.batch_size)
    embeddings = tf.keras.Input((args.sequence_length, 9600), batch_size=args.batch_size)

    em = embedding_model(args)
    trans = transformer(args)
    t_model = tf.keras.Sequential(name="TCNN-Transformer")
    t_model.add(images)
    t_model.add(layers.TimeDistributed(em, input_shape=(args.width, args.height, args.depth, 1)
                                  , batch_size=args.batch_size))
    t_model.add(trans)
    return t_model