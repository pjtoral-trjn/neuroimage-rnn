from utils.argparser import config_parser
from pipeline import Pipeline

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    print("----- Experiment Begin -----")
    print("Running the following configuration:")
    print(args)
    pipeline = Pipeline(args)
    if args.evaluate_only:
        print("----- Evaluation Start -----")
        pipeline.evaluate_only()
        print("----- Evaluation End -----")
    else:
        pipeline.run_pipeline()
    print("----- Experiment Complete -----")