import sys
import os
import re
import json
import argparse
import datetime


def get_wsclean_info_from_log(wsclean_log):
    lines = open(wsclean_log, "r").readlines()
    for line in lines:
        line = line.strip()
        patt = "Total nr. of visibilities to be gridded:\s*"
        if re.search(patt, line):
            wsc_totvis = re.split(patt, line)[-1]

        # If natural weighting
        patt = "effective count after weighting:\s*"
        if re.search(patt, line):
            print("###", line)
            print("###", re.split(patt, line))
            wsc_gridvis = re.split(patt, line)[-1]

        # If uniform gridding
        #patt = "Gridded visibility count:\s*"
        #if re.search(patt, line):
        #    print("#-#", line)
        #    wsc_gridvis = re.split(patt, line)[-1]

        if re.search("Inversion:", line):
            wsc_t_inv, wsc_t_pred, wsc_t_deconv = re.split("\s*Inversion:\s*|\s*,\s*prediction:\s*|\s*,\s*deconvolution:\s*", line)[-3:]
            t_inv    = datetime.datetime.strptime(wsc_t_inv,   '%H:%M:%S.%f') - datetime.datetime(1900,1,1)
            t_pred   = datetime.datetime.strptime(wsc_t_pred,  '%H:%M:%S') - datetime.datetime(1900,1,1)
            t_deconv = datetime.datetime.strptime(wsc_t_deconv,'%H:%M:%S') - datetime.datetime(1900,1,1)


    wsc_info  = f"package      = WSClean (wsc)\n"
    wsc_info += f"wsc totvis   = {wsc_totvis}\n"
    wsc_info += f"wsc gridvis  = {wsc_gridvis}\n"
    wsc_info += f"wsc t_inv    = {t_inv.total_seconds():.3f}\n"
    wsc_info += f"wsc t_pred   = {t_pred.total_seconds():.3f}\n"
    wsc_info += f"wsc t_deconv = {t_deconv.total_seconds():.3f}\n"

    print(json.dumps(wsc_info))

    return wsc_totvis, wsc_gridvis, t_inv.total_seconds(), t_pred.total_seconds(), t_deconv.total_seconds(), wsc_info


def write_json(wsc_log):

    if not re.search(".log$", wsc_log):
        raise Exception("WSClean log file expected to have .log extension")

    output_directory = os.path.dirname(wsc_log)
    output_basename  = os.path.basename(wsc_log).replace('.log', '.json')
    output_json =  os.path.join(output_directory, output_basename)

    vis_tot, vis_grid, t_inv, t_pred, t_deconv, _ = get_wsclean_info_from_log(wsc_log)

    stats = { 
        "timings": {
            't_inv':    t_inv,
            't_pred':   t_pred,
            't_deconv': t_deconv,
        },
        "visibilities": {
            'tot':  vis_tot,
            'grid': vis_grid
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
