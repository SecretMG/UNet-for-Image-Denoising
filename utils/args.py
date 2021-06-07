import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    'description',
    help='本次训练的特点'
)   # UDP端口1
args = parser.parse_args()