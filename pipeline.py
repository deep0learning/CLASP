from mycore import detection as dt
from mycore import classification as cl
import sys


def run(exp, startf, d_flag, c_flag, fps=20.0):
    if d_flag == 'True':
        dt.detection(exp, startf=startf, fps=fps)
    if c_flag == 'True':
        cl.classification(exp, startf=startf, fps=fps)

if __name__ == "__main__":
    e = sys.argv[1]
    s = int(sys.argv[2])
    d = sys.argv[3]
    c = sys.argv[4]
    run(exp=e, startf=s, d_flag=d, c_flag=c)
