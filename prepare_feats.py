import argparse
from graph_dataset.csv2pt import *





def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str,help="graph or MAT or ...")
    parser.add_argument("--dataset_path", type=str)
    return parser.parse_args()


def main():
    args = parse()
    if args.dataset_type == "graph":
        preparer = Graph2feats(args.dataset_path)
        preparer.process()


if __name__ == "__main__":
    main()