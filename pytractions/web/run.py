import os
from streamlit.web import cli as stcli
import sys

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'welcome.py')


def main(args):
    sys.exit(stcli.main_run([filename], standalone_mode=False))


def make_parsers(subparsers):
    p_web = subparsers.add_parser('web', help='Run web interface')
    p_web.set_defaults(command=main)
    return p_web

