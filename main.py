from utils.argparser import config_parser
from pipeline import Pipeline
import random
import tensorflow as tf
import numpy as np

def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    print("----- Experiment Begin -----")
    print("Running the following configuration:")
    print(args)
    set_seed(args.seed)
    pipeline = Pipeline(args)
    if args.evaluate_only:
        print("----- Evaluation Start -----")
        pipeline.evaluate_only()
        print("----- Evaluation End -----")
    else:
        pipeline.run_pipeline()
    print("----- Experiment Complete -----")