import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Conv3D, Flatten, MaxPooling3D, Dropout, Bidirectional, GRU
from model.tcnn import tcnn

def rnn(args):
    """
    Constructor for the RNN Model. The function receives experiment parameters and outputs the associated RNN. During
    which,the function loads the pre-trained embedding model and constructs rnn
    Pre-Trained-CNN (Feature Selector) -> RNN (Sequence Processor) -> Decision Network (Classification/Regression Head)
    :param args:
    :return: Keras Model
    """
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

    # Start Creation of RNN
    rnn_model = tf.keras.Sequential(name="TCNN-"+args.rnn_selection+"-DN")
    rnn_model.add(images)
    rnn_model.add(TimeDistributed(embedding_model, input_shape=(args.width, args.height, args.depth, 1)
                                  , batch_size=args.batch_size))

    # Add RNN Backbone
    if args.rnn_selection == "gru":
        rnn_model.add(Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.025)))
    elif args.rnn_selection == "lstm":
        rnn_model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.025)))

    # Configure Head
    if args.include_decision_network:
        dn = decision_network(args)
        rnn_model.add(TimeDistributed(dn))
        print("\n\n\n Decision Network (Head)")
        dn.summary()
    elif not args.include_decision_network:
        head = get_head_layer(args)
        rnn_model.add(TimeDistributed(head))

    print("\n\n\n Full RNN Architecture")
    rnn_model.summary()
    return rnn_model

def decision_network(args):
    """
    :param args:
    :return:
    """
    decision_network_input = tf.keras.Input((args.sequence_length, 256), batch_size=args.batch_size)
    x = tf.keras.layers.Dense(units=64, activation="relu")(decision_network_input)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(units=32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(units=16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    head = get_head_layer(args)
    output = head()(x)
    return Model(inputs=decision_network_input, outputs=output, name="Decision-Network")

def get_head_layer(args):
    if args.task_selection == "binary_classification":
        return tf.keras.layers.Dense(units=1, name="Binary-Classifier", activation="sigmoid")
    elif args.task_selection == "multi_classification":
        return tf.keras.layers.Dense(units=3, name="Multi-Classifier", activation="softmax")
    elif args.task_selection == "regression":
        return tf.keras.layers.Dense(units=1, name="Regression")