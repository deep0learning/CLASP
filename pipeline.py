from mycore import detection as dt
from mycore import classification as cl
from mycore import tracking as tk
from mycore import extract_track_features as ex
from mycore import association as asc
from mycore import reidentification as rid
import sys
import argparse


def run(args):
    # dt.detection(exp=args.exp, dtype=args.dtype, startf=args.startf, vis=args.vis)
    # dt.detection(exp=args.exp, dtype='hand', startf=args.startf, vis=False)
    # tk.tracking(args.exp, 'person', args.vis=="True")
    # tk.tracking(args.exp, 'bin', args.vis=="True")
    # ex.extract_features(args.exp)
    

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Pipeline")
    parser.add_argument(
        "--exp", help="Name of video file",
        default=None, required=True)
    parser.add_argument(
	    "--startf", help="Start frame",
	    default=0, required=False)
    parser.add_argument(
	    "--dtype", help="perbin/hand for detection",
	    default="perbin", required=False)
    parser.add_argument(
	    "--vis", help="Start frame",
	    default=True, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)   

