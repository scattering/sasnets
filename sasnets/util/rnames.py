from __future__ import print_function

import os
import re

import sys


def rnames(path, p="_all_"):
    '''
    Method to collect all filenames of models from a directory.

    rnames reads all files contained in path that match the regex p, and outputs
    the model name to a new file named name. If name exists, it is overwritten.

    :param path: A string representing a filepath.
    :param p: A regex to match files to.
    :return: None
    '''
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