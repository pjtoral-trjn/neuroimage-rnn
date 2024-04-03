import argparse

def config_parser():
    configuration_parser = argparse.ArgumentParser()
    # defaults
    configuration_parser.add_argument("-tls", "--trainable_layers", type=float, default=1, help="Trainable Layers")
    configuration_parser.add_argument("-e", "--epochs", type=int, default=50, help="Training Epochs")
    configuration_parser.add_argument("-s", "--seed", type=int, default=5, help="Randomizer Seed")
    configuration_parser.add_argument("-tc", "--target_column", type=str, default="label", help="Target Column")
    configuration_parser.add_argument("-vc", "--volume_column", type=str, default="volume", help="Volume Column")
    configuration_parser.add_argument("-o", "--optimizer", type=str, default="adamw", help="Model Optimizer")
    configuration_parser.add_argument("-smp", "--saved_model_pathway", type=str, default="", help="Saved Model Pathway")
    configuration_parser.add_argument("-w", "--width", type=int, default=91, help="Width of input")
    configuration_parser.add_argument("-hei", "--height", type=int, default=109, help="Height of input")
    configuration_parser.add_argument("-d", "--depth", type=int, default=91, help="Depth of input")
    configuration_parser.add_argument("-sel", "--sequence_length", type=int, default=None
                                      , help="Sequence Length of input")
    configuration_parser.add_argument("-eo", "--evaluate_only", type=bool, default=False
                                      , help="Evaluation Only")
    configuration_parser.add_argument("-idn", "--include_decision_network", type=bool, default=False
                                      , help="Include Decision Network?")
    configuration_parser.add_argument("-dno", "--decision_network_dropout", type=float, default=0.25
                                      , help="Decision Network Dropout")
    configuration_parser.add_argument("-rno", "--recurrent_dropout", type=float, default=0.1
                                      , help="Recurrent Network Dropout")


    # necessary configuration
    configuration_parser.add_argument("-g", "--gpu", type=int, help="GPU ID Selection")
    configuration_parser.add_argument("-bs", "--batch_size", type=int, help="Batch Size")
    configuration_parser.add_argument("-es", "--early_stop", type=int, help="Early Stopping")
    configuration_parser.add_argument("-ma", "--model_architecture", type=str, help="Model Architecture Selection")
    configuration_parser.add_argument("-l", "--loss", type=str, help="Model Loss")
    configuration_parser.add_argument("-en", "--experiment_name", type=str, help="Experiment Name, for storage")
    configuration_parser.add_argument("-trp", "--train_pathway", default="", type=str, help="Train Pathway CSV")
    configuration_parser.add_argument("-tep", "--test_pathway", default="", type=str, help="Test Pathway CSV")
    configuration_parser.add_argument("-ilr", "--init_learning_rate", default=0.001, type=float,
                                      help="Initial Learning Rate for the optimizer")
    configuration_parser.add_argument("-wd", "--weight_decay", type=float, help="Weight Decay for ADAMW optimizer")
    configuration_parser.add_argument("-do", "--drop_out", type=float, help="Model Dropout")
    configuration_parser.add_argument("-emp", "--embedding_model_pathway", type=str, default=""
                                      , help="Embedding Model Pathway")
    configuration_parser.add_argument("-rs", "--rnn_selection", type=str, default="gru"
                                      , help="RNN Model Pathway GRU || LSTM")
    configuration_parser.add_argument("-ts", "--task_selection", type=str, default="binary_classification"
                                      , help="Task Selection binary_classification || multi_classification (3)"
                                             " || regression")

    return configuration_parser
