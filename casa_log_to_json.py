import sys
import os
import re
import json
import argparse
import datetime
import casatb

parser = argparse.ArgumentParser(description='Generate CASA JSON stat file from log')
parser.add_argument('--casa_log',   help='CASA log file', required=True)
args = parser.parse_args()

casatb.write_json(args.casa_log)
