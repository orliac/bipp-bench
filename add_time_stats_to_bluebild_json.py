import sys
import os
import re
import json
import argparse
import datetime
import wscleantb

parser = argparse.ArgumentParser(description='Augment Bluebild JSON stat file with times from builtin time')
parser.add_argument('--bb_json',     help='Bluebild json file', required=True)
parser.add_argument('--bb_time_log', help='Bluebild Bash builtin time log file', required=True)
args = parser.parse_args()

print(f"-I- Will add Bash time information from: {args.bb_time_log} to: {args.bb_json}")

time_stats = {}
with open(args.bb_time_log) as fp:
    for line in fp:
        key, val = line.split()
        time_stats[key] = float(val)

with open(args.bb_json) as fp:
  stats = json.load(fp)
  stats.update({"time": time_stats})

with open(args.bb_json, 'w') as fp:
    fp.write(json.dumps(stats, indent=4))
