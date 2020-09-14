import argparse
from rexnet import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='rexnet',
        description='Pytorch Training code of Clova ReXNet'
    )
    subparsers = parser.add_subparsers(dest='subcommands')

    train_model.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)
