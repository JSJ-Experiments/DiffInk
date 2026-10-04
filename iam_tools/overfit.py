"""Prepare 8×24 train and 8×4 line-disjoint validation samples, without training."""
import argparse
import json
from .build import build

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',default='data/raw/iam')
    p.add_argument('--canonical',default='data/canonical/iam/overfit')
    p.add_argument('--out',default='data/diffink/iam_overfit')
    p.add_argument('--overfit-writers',type=int,default=8)
    p.add_argument('--lines-per-writer',type=int,default=24)
    p.add_argument('--val-lines-per-writer',type=int,default=4)
    args=vars(p.parse_args());print(json.dumps(build(**args),indent=2))
if __name__=='__main__':main()
