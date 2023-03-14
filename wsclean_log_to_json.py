import sys
import os
import re
import json
import argparse
import datetime
import wscleantb

parser = argparse.ArgumentParser(description='Generate WSClean JSON stat file from log')
parser.add_argument('--wsc_log',   help='WSClean log file', required=True)
args = parser.parse_args()

wscleantb.write_json(args.wsc_log)
