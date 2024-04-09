import tensorflow as tf
from model.cnn.tcnn import tcnn
from model.transformer.Transformer import Transformer
from tensorflow.keras.layers import TimeDistributed

def tcnn_transformer(args):
    images = tf.keras.Input((args.sequence_length, args.width, args.height, args.depth, 1), batch_size=args.batch_size)

    # Load Embedding Model
    tcnn_full_model = tcnn(args)
    tcnn_full_model.set_weights(tf.keras.models.load_model(args.embedding_model_pathway, compile=False).get_weights())
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

    num_layers = 2
    d_model = 64
    dff = 256
    num_heads = 4
    dropout_rate = 0.1
    transformer = Transformer(args=args,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=9600,
        target_vocab_size=9600,
        dropout_rate=dropout_rate)

    model = tf.keras.Sequential(name="TCNN-Transformer")
    model.add(images)
    model.add(TimeDistributed(embedding_model, input_shape=(args.width, args.height, args.depth, 1)
                                  , batch_size=args.batch_size))
    model.add(transformer)
    print("\n\n\n Full Transformer")
    embedding_model.summary()
