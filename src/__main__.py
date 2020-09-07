import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='rexnet',
        description='Pytorch Training code of Clova ReXNet'
    )

    args = parser.parse_args()
    args.func(args)