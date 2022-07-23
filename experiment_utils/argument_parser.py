from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('dataset')
args = parser.parse_args()
dataset = args.dataset.strip()
