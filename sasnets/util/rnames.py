import os
import argparse

from .. import sas_io

parser = argparse.ArgumentParser()
parser.add_argument(
    "--tag", type=str, default="train",
    help="Tag for the generated data: train or test.")
def rnames(opts):
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
    models = set()
    ifiles = sas_io.iread_1d(
        opts.path, tag=opts.tag, format=opts.format, verbose=opts.verbose)
    for _, model in ifiles:
        models.add(model)
    with open(os.path.join(os.path.dirname(opts.path), "name"), "w") as fd:
        fd.write(str(sorted(models, key=str.lower)) + "\n")

def main(args):
    opts = parser.parse_args(args)
    rnames(opts)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
