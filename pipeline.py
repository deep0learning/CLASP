from mycore import detection as dt
from mycore import classification as cl
from mycore import tracking as tk
from mycore import extract_track_features as ex
from mycore import association as asc
from mycore import reidentification as rid
import sys


def run(exp, startf, fps=20.0):
    #dt.detection(exp, startf=startf, fps=fps, vis=False)
    #tk.tracking(exp, 'person')
    #tk.tracking(exp, 'bin')
    asc.associate(exp, startf=startf, fps=45)
    #ex.extract_features(exp, startf=startf, fps=fps)

if __name__ == "__main__":
    e = sys.argv[1]
    run(exp=e, startf=0)
