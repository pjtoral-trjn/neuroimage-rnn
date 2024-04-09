import tensorflow as tf

def get_head_layer(args):
    if args.task_selection == "binary_classification":
        return tf.keras.layers.Dense(units=1, name="Binary-Classifier", activation="sigmoid")
    elif args.task_selection == "multi_classification":
        return tf.keras.layers.Dense(units=3, name="Multi-Classifier", activation="softmax")
    elif args.task_selection == "regression":
        return tf.keras.layers.Dense(units=1, name="Regression")