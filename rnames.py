from __future__ import print_function

import os
import re

import sys


def rnames(path, p="_all_"):
    l = list()
    for fn in os.listdir(path):
        pattern = re.compile(p)
        if pattern.search(fn):
            try:
                with open(path + fn, 'r') as fd:
                    print("Reading " + fn)
                    templ = eval(fd.readline().strip())
                    l.append(templ[0])
            except:
                raise
    with open(os.path.join(os.path.dirname(path), "name"), "w") as fd:
        fd.write(str(sorted(l, key=str.lower)) + "\n")
def main(args):
    rnames(args[0])

if __name__ == '__main__':
    main(sys.argv[1:])