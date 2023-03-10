import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits


def get_wsclean_info_from_log(wsclean_log):
    lines = open(wsclean_log, "r").readlines()
    for line in lines:
        line = line.strip()
        patt = "Total nr. of visibilities to be gridded:"
        if re.search(patt, line):
            wsc_totvis = line.split(patt)[-1]
        patt = "effective count after weighting:"
        if re.search(patt, line):
            wsc_gridvis = line.split(patt)[-1]
        if re.search("Inversion:", line):
            wsc_t_inv, wsc_t_pred, wsc_t_deconv = re.split("\s*Inversion:\s*|\s*,\s*prediction:\s*|\s*,\s*deconvolution:\s*", line)[-3:]
            t_inv    = datetime.datetime.strptime(wsc_t_inv,   '%H:%M:%S.%f') - datetime.datetime(1900,1,1)
            t_pred   = datetime.datetime.strptime(wsc_t_pred,  '%H:%M:%S') - datetime.datetime(1900,1,1)
            t_deconv = datetime.datetime.strptime(wsc_t_deconv,'%H:%M:%S') - datetime.datetime(1900,1,1)

    return wsc_totvis, wsc_gridvis, t_inv.total_seconds(), t_pred.total_seconds(), t_deconv.total_seconds()


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


def read_fits_file(fits_file):
    hdul = fits.open(fits_file)
    print("-I- FITS hdul.info()\n", hdul.info())
    header = hdul[0].header
    data   = hdul[0].data
    print("-I- FITS header", header)
    print("-I- FITS data.shape:", data.shape)
    return header, data


def plot_bluebild_casa(bipp_grid_npy, bipp_data_npy, bipp_json, fits_file, log_file):
    print("==========================================================")
    print(" Plotting Bluebild vs CASA")
    print("==========================================================")

    header, data = read_fits_file(fits_file)
    totvis, t_inv = get_casa_info_from_log(log_file)

    title  = f"CASA visibilities: total {totvis}\n"
    title += f"CASA times: inv {t_inv:.2f}"

    plot_bluebild_vs_fits(bipp_grid_npy, bipp_data_npy, bipp_json, 'CASA', data, title)


def plot_bluebild_wsclean(bipp_grid_npy, bipp_data_npy, bipp_json, fits_file, log_file):
    print("==========================================================")
    print(" Plotting Bluebild vs WSClean")
    print("==========================================================")

    header, data = read_fits_file(fits_file)
    totvis, gridvis, t_inv, t_pred, t_deconv = get_wsclean_info_from_log(log_file)

    title = f"WSClean visibilities: total {totvis}, effective after weighting {gridvis}\n"
    title = f"WSClean times: inv {t_inv:.2f}, pred {t_pred:.2f}, deconv {t_deconv:.2f}"

    plot_bluebild_vs_fits(bipp_grid_npy, bipp_data_npy, bipp_json, 'WSClean', data, title)


def plot_wsclean_casa(wsc_fits, wsc_log, casa_fits, casa_log):
    print("==========================================================")
    print(" Plotting WSClean vs CASA")
    print("==========================================================")

    casa_header, casa_data = read_fits_file(casa_fits)
    casa_totvis, casa_t_inv = get_casa_info_from_log(casa_log)

    wsc_header, wsc_data = read_fits_file(wsc_fits)
    wsc_totvis, wsc_gridvis, wsc_t_inv, wsc_t_pred, wsc_t_deconv = get_wsclean_info_from_log(wsc_log)

    title = f"WSClean / CASA visibilities: {wsc_totvis} / {casa_totvis}"


    fig, ax = plt.subplots(ncols=3, figsize=(25,12))
    plt.suptitle(title, fontsize=22)

    print(f"-I- wsclean min, max: {np.min(wsc_data[0,0,:,:]):.3f}, {np.max(wsc_data[0,0,:,:]):.3f}")
    wsc_normed  = wsc_data[0,0,:,:]
    #print(f"min, max fits : {np.min(normed_fits)}, {np.max(normed_fits)}")
    im1 = ax[0].imshow(wsc_normed)
    ax[0].set_title(f"WSClean dirty", fontsize=20)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)

    print(f"-I- casa min, max: {np.min(casa_data[0,0,:,:]):.3f}, {np.max(casa_data[0,0,:,:]):.3f}")
    casa_normed  = casa_data[0,0,:,:]
    #print(f"min, max fits : {np.min(normed_fits)}, {np.max(normed_fits)}")
    im1 = ax[1].imshow(casa_normed)
    ax[1].set_title(f"CASA dirty", fontsize=20)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    diff = wsc_normed - casa_normed
    print(f"-I- (WSClean - CASA) min, max: {np.min(diff):.3f}, {np.max(diff):.3f}")
    im2 = ax[2].imshow(diff)
    ax[2].set_title(f"WSClean minus CASA", fontsize=20)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)

    plt.tight_layout()
    fig.savefig("WSClean_CASA.png")


    # Shifted + Normalized
    fig.clf()
    fig, ax = plt.subplots(ncols=3, figsize=(25,12))
    plt.suptitle(title, fontsize=22)
    wsc_normed  = wsc_data[0,0,:,:] - np.min(wsc_data[0,0,:,:])
    wsc_normed /= np.max(wsc_normed)
    im1 = ax[0].imshow(wsc_normed)
    ax[0].set_title(f"Shifted normalized WSClean dirty", fontsize=20)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)

    casa_normed  = casa_data[0,0,:,:] - np.min(casa_data[0,0,:,:])
    casa_normed /= np.max(casa_normed)
    im1 = ax[1].imshow(casa_normed)
    ax[1].set_title(f"Shifted normalized CASA dirty", fontsize=20)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    diff = wsc_normed - casa_normed
    im2 = ax[2].imshow(diff)
    ax[2].set_title(f"WSClean minus CASA", fontsize=20)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)

    plt.tight_layout()
    fig.savefig("WSClean_CASA_shifted_normalized.png")


def plot_bluebild_vs_fits(bipp_grid_npy, bipp_data_npy, bipp_json, fits_name, fits_data, fits_title):

    # BIPP npy data
    bipp_data = np.load(bipp_data_npy)
    print("-I- bipp data:", bipp_data.dtype, bipp_data.shape)
    nlev = bipp_data.shape[0]

    # BIPP json file
    json_file = open(bipp_json)
    json_data = json.load(json_file)

    # Produce different images including additional energy levels
    #
    for nlev in range(1, bipp_data.shape[0] + 1):
        
        fig, ax = plt.subplots(ncols=3, figsize=(25,12))
        plt.suptitle(f"Bluebild energy levels from 0 to {nlev - 1}, " +
                     f"{json_data['visibilities']['ifim']} visibilities\n" +
                     fits_title, fontsize=22)

        bb_eq_cum = np.zeros([bipp_data.shape[1], bipp_data.shape[2]])
        for i in range(0, nlev):
            bb_eq_cum += bipp_data[i,:,:]
            #print(f"-I- level {i} bluebild min, max: {np.min(bb_eq_cum)}, {np.max(bb_eq_cum)}")
        print(f"-I- bluebild min, max: {np.min(bb_eq_cum):.3f}, {np.max(bb_eq_cum):.3f}")

        # Align min to 0.0 and normalize
        bb_eq_cum  = np.fliplr(bb_eq_cum)
        bb_eq_cum -= np.min(bb_eq_cum)
        bb_eq_cum /= np.max(bb_eq_cum)
        
        im0 = ax[0].imshow(bb_eq_cum)
        ax[0].set_title("Shifted normalized BB LSQ dirty", fontsize=20)
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax)
        
        print(f"-I- wsclean min, max: {np.min(fits_data[0,0,:,:]):.3f}, {np.max(fits_data[0,0,:,:]):.3f}")
        normed_fits  = fits_data[0,0,:,:] - np.min(fits_data[0,0,:,:])
        normed_fits /= np.max(normed_fits)
        #print(f"min, max fits : {np.min(normed_fits)}, {np.max(normed_fits)}")
        im1 = ax[1].imshow(normed_fits)
        ax[1].set_title(f"Shifted normalized {fits_name} dirty", fontsize=20)
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        diff = bb_eq_cum - normed_fits
        print(f"-I- (bluebild - {fits_name}) min, max: {np.min(diff):.3f}, {np.max(diff):.3f}")
        im2 = ax[2].imshow(diff)
        ax[2].set_title(f"BB minus {fits_name}", fontsize=20)
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
        
        plt.tight_layout()
        
        fig.savefig(f"Bluebild_{fits_name}_0-"+ str(nlev - 1) + '.png')

    json_file.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run general plotting utility')
    parser.add_argument('--bb_grid',   help='Bluebild grid .npy')
    parser.add_argument('--bb_data',   help='Bluebild data .npy')
    parser.add_argument('--bb_json',   help='Bluebild json .npy')
    parser.add_argument('--wsc_fits',  help='WSClean fits file')
    parser.add_argument('--wsc_log',   help='WSClean log file')
    parser.add_argument('--casa_fits', help='CASA fits file')
    parser.add_argument('--casa_log',  help='CASA log file')
    args = parser.parse_args()

    do_bb = False
    if args.bb_grid   and args.bb_data and args.bb_json: do_bb   = True
    if args.wsc_fits  and args.wsc_log:                  do_wsc  = True 
    if args.casa_fits and args.casa_log:                 do_casa = True
    print("-I- consider Bluebild?", do_bb)
    print("-I- consider WSClean? ", do_wsc)
    print("-I- consider CASA?    ", do_casa)

    if do_wsc and do_casa:
        plot_wsclean_casa(args.wsc_fits, args.wsc_log, args.casa_fits, args.casa_log)

    if do_bb and do_wsc:
        plot_bluebild_wsclean(args.bb_grid, args.bb_data, args.bb_json, args.wsc_fits, args.wsc_log)

    if do_bb and do_casa:
        plot_bluebild_casa(args.bb_grid, args.bb_data, args.bb_json, args.casa_fits, args.casa_log)
