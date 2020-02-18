import os
import re
import ast

def rnames(path, p="_all_"):
    """
    Method to collect all filenames of models from a directory.

    rnames reads all files contained in path that match the regex p, and outputs
    the model name to a new file named name. If name exists, it is overwritten.

    The input files should be csv with quoted strings, with the first column
    of the first line containing the model name.  The output file will contain
    a single line with names case insensitive alphabetical order::

        ['name1', 'name2', ...]

    :param path: A string representing a filepath.
    :param p: A regex to match files to.
    :return: None
    """
    pattern = re.compile(p)
    l = list()
    for fn in os.listdir(path):
        if pattern.search(fn):
            with open(path + fn, 'r') as fd:
                print("Reading " + fn)
                templ = ast.literal_eval(fd.readline().strip())
                l.append(templ[0])
    with open(os.path.join(os.path.dirname(path), "name"), "w") as fd:
        fd.write(str(sorted(l, key=str.lower)) + "\n")

def main(args):
    rnames(args[0])

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
