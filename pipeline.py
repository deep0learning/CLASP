from mycore import detection as dt
from mycore import classification as cl
from mycore import tracking as tk
from mycore import extract_track_features as ex
from mycore import association as asc
import sys


def run(exp, startf, fps=20.0):
    dt.detection(exp, startf=startf, fps=fps)
    tk.tracking(exp, 'person')
    tk.tracking(exp, 'bin')
    ex.extract_features(exp, startf=startf, fps=fps)
    asc.associate(exp, startf=startf, fps=35)

if __name__ == "__main__":
    e = sys.argv[1]
    run(exp=e, startf=0)
