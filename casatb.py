import sys
import os
import re
import json
import argparse
import datetime


def get_casa_info_from_log(log_file):
    lines = open(log_file, "r").readlines()
    for line in lines:
        #print(line)
        line = line.strip()
        patt = "NRows selected :\s*"
        if re.search(patt, line):
            casa_totvis = re.split(patt, line)[-1]
        patt = "Task tclean complete.\s*"
        if re.search(patt, line):
            casa_t = re.split(patt, line)[-1]
            casa_ts, casa_te = re.split("\s*Start time:\s*|\s*End time:\s*", casa_t)[-2:]
            #print(casa_ts, casa_te)
            casa_ts = datetime.datetime.strptime(casa_ts, '%Y-%m-%d %H:%M:%S.%f')
            casa_te = datetime.datetime.strptime(casa_te, '%Y-%m-%d %H:%M:%S.%f')
            #print('casa_ts', casa_ts)
            #print('casa_te', casa_te)
    #print("casa_totvis:", casa_totvis)
    #print("casa_t_inv :", (casa_te - casa_ts).total_seconds())
    return casa_totvis, (casa_te - casa_ts).total_seconds()
    sys.exit(0)


def write_json(casa_log):

    if not re.search(".log$", casa_log):
        raise Exception("CASA log file expected to have .log extension")

    output_directory = os.path.dirname(casa_log)
    output_basename  = os.path.basename(casa_log).replace('.log', '.json')
    output_json =  os.path.join(output_directory, output_basename)

    vis_tot, t_tclean = get_casa_info_from_log(casa_log)

    stats = { 
        "timings": {
            't_tclean': t_tclean,
        },
        "visibilities": {
            'tot':  vis_tot,
        },
    }
    
    with open(output_json, 'w') as outfile:
        outfile.write(json.dumps(stats, indent=4))

    print("-I- wrote: ", os.path.join(output_directory, output_basename))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate WSClean JSON stat file from log')
    parser.add_argument('--wsc_log',   help='WSClean log file', required=True)
    args = parser.parse_args()

    write_json(args.wsc_log)
