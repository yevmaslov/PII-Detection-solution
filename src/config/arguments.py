from argparse import ArgumentParser
import argparse
from types import SimpleNamespace


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

def get_input_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--debug',
        type=str2bool,
        default=True
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )
    parser.add_argument(
        '--exp_name',
        type=str,
    )
    parser.add_argument(
        '--pseudo_from',
        type=str,
    )
    parser.add_argument(
        '--state_from',
        type=str,
    )
    args = parser.parse_args()
    return vars(args)


def get_default_args():
    args = SimpleNamespace()
    args.debug = False
    args.fold = 0
    args.exp_name = 'test'
    return vars(args)


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
    

def get_args():
    its_notebook = is_notebook()
    if its_notebook:
        args = get_default_args()
    else:
        args = get_input_args()
        
    return args