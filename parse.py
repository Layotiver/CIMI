import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    data_args = parser.add_argument_group('Dataset options')
    data_args.add_argument('--dataset', type=str, default='clickbait')

    trainingArgs = parser.add_argument_group('Training options')
    trainingArgs.add_argument('--device', type=int, default=0, help='cuda device')
    trainingArgs.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')

    return parser.parse_args()
