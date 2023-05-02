import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json
import argparse
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
import wscleantb
import casatb


def read_fits_file(fits_file):
    hdul = fits.open(fits_file)
    print("-I- FITS hdul.info()\n", hdul.info())
    header = hdul[0].header
    data   = hdul[0].data
    #print("-I- FITS header", header)
    #print("-I- FITS data.shape:", data.shape)
    return header, data


def plot_bluebild_casa(bipp_grid_npy, bipp_data_npy, bipp_json, fits_file, log_file):
    print("==========================================================")
    print(" Plotting Bluebild vs CASA")
    print("==========================================================")

    header, data = read_fits_file(fits_file)
    totvis, t_inv = casatb.get_casa_info_from_log(log_file)

    title  = f"{'CASA':8s}: {int(totvis):7d} vis   runtime: {t_inv:6.2f} sec"

    plot_bluebild_vs_fits(bipp_grid_npy, bipp_data_npy, bipp_json, 'CASA', data, title)


def plot_bluebild_wsclean(bipp_grid_npy, bipp_data_npy, bipp_json, fits_file, log_file, outname, outdir):
    print("==========================================================")
    print(" Plotting Bluebild vs WSClean")
    print("==========================================================")

    header, data = read_fits_file(fits_file)
    totvis, gridvis, t_inv, t_pred, t_deconv = wscleantb.get_wsclean_info_from_log(log_file)

    title  = f"{'WSClean':8s}: {int(totvis):7d} vis   runtime: {t_inv:6.2f} sec"

    plot_bluebild_vs_fits(bipp_grid_npy, bipp_data_npy, bipp_json, 'WSClean', data, title, outname, outdir)


def plot_wsclean_casa(wsc_fits, wsc_log, casa_fits, casa_log):
    print("==========================================================")
    print(" Plotting WSClean vs CASA")
    print("==========================================================")

    casa_header, casa_data = read_fits_file(casa_fits)
    casa_totvis, casa_t_inv = casatb.get_casa_info_from_log(casa_log)

    wsc_header, wsc_data = read_fits_file(wsc_fits)
    wsc_totvis, wsc_gridvis, wsc_t_inv, wsc_t_pred, wsc_t_deconv = wscleantb.get_wsclean_info_from_log(wsc_log)

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


def plot_bluebild_vs_fits(bipp_grid_npy, bipp_data_npy, bipp_json, fits_name, fits_data, fits_title,
                          outname, outdir):

    # BIPP npy data
    bipp_data = np.load(bipp_data_npy)
    print("-I- Bluebild data:", bipp_data.dtype, bipp_data.shape)
    nlev = bipp_data.shape[0]

    # BIPP json file
    json_file = open(bipp_json)
    json_data = json.load(json_file)
    bb_vis = json_data['visibilities']['ifim']
    bb_tot = json_data['timings']['tot']

    fits_data = fits_data[0,0,:,:]
    print(f"-I- {fits_name} min, max: {np.min(fits_data):.3f}, {np.max(fits_data):.3f}")


    # Shifted + normalized intensity
    #
    fig, ax = plt.subplots(ncols=3, figsize=(25,12))
    bb_title = f"{'Bluebild':8s}: {int(bb_vis):7d} vis   runtime: {bb_tot:6.2f} sec"
        
    fp = FontProperties(family="monospace", size=22, weight="bold")
    plt.suptitle(bb_title + "\n" + fits_title, x=0.25, y=0.92, ha='left').set_fontproperties(fp)

    bb_eq_cum = np.zeros([bipp_data.shape[1], bipp_data.shape[2]])
    for i in range(0, nlev):
        bb_eq_cum += bipp_data[i,:,:]
        print(f"-I- .. Bluebild energy level {i} min, max: {np.min(bipp_data[i,:,:]):.3f}, {np.max(bipp_data[i,:,:]):.3f}")
    print(f"-I- Bluebild cum energy levels [0, {nlev-1}]: min, max: {np.min(bb_eq_cum):.3f}, {np.max(bb_eq_cum):.3f}," +
          f" vis. scaled = {np.min(bb_eq_cum)/bb_vis:.3f}, {np.max(bb_eq_cum)/bb_vis:.3f}")

    # Align min to 0.0 and normalize
    if args.flip_lr:
        bb_eq_cum  = np.fliplr(bb_eq_cum)
    bb_eq_cum -= np.min(bb_eq_cum)
    bb_eq_cum /= np.max(bb_eq_cum)
    #print(f"-I- Bluebild normalized cum energy levels [0, {nlev-1}]: min, max: {np.min(bb_eq_cum):.3f}, {np.max(bb_eq_cum):.3f}")

    im0 = ax[0].imshow(bb_eq_cum)
    ax[0].set_title("Shifted normalized BB LSQ dirty", fontsize=20)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)
    
    normed_fits  = fits_data - np.min(fits_data)
    normed_fits /= np.max(normed_fits)
    #print(f"min, max fits : {np.min(normed_fits)}, {np.max(normed_fits)}")
    im1 = ax[1].imshow(normed_fits)
    ax[1].set_title(f"Shifted normalized {fits_name} dirty", fontsize=20)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
    
    diff = bb_eq_cum - normed_fits
    #print(f"-I- Bluebild normalized minus {fits_name} min, max: {np.min(diff):.3f}, {np.max(diff):.3f}")
    im2 = ax[2].imshow(diff)
    ax[2].set_title(f"BB minus {fits_name}", fontsize=20)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"{outname}_Bluebild_{fits_name}_normalized.png"))
        

    # Raw intensity
    #
    fig, ax = plt.subplots(ncols=3, figsize=(25,12))
    bb_title = f"{'Bluebild':8s}: {int(bb_vis):7d} vis   runtime: {bb_tot:6.2f} sec"
    fp = FontProperties(family="monospace", size=22, weight="bold")
    plt.suptitle(bb_title + "\n" + fits_title, x=0.25, y=0.92, ha='left').set_fontproperties(fp)

    bb_eq_cum = np.zeros([bipp_data.shape[1], bipp_data.shape[2]])
    for i in range(0, bipp_data.shape[0]):
        bb_eq_cum += bipp_data[i,:,:]

    # Align min to 0.0 and normalize
    if args.flip_lr:
        bb_eq_cum  = np.fliplr(bb_eq_cum)
        
    im0 = ax[0].imshow(bb_eq_cum)
    ax[0].set_title("Bluebild LSQ dirty", fontsize=20)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)
        
    im1 = ax[1].imshow(fits_data)
    ax[1].set_title(f"{fits_name} dirty", fontsize=20)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
        
    diff = bb_eq_cum - fits_data
    #print(f"-I- Bluebild minus {fits_name} min, max: {np.min(diff):.3f}, {np.max(diff):.3f}")
    im2 = ax[2].imshow(diff)
    ax[2].set_title(f"Bluebild LSQ minus {fits_name}", fontsize=20)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)
    
    plt.tight_layout()
    
    fig.savefig(os.path.join(outdir, f"{outname}_Bluebild_{fits_name}.png"))


    # =================================================================
    # Bluebild / number of visibilities + ratio to fits
    # =================================================================
    fig, ax = plt.subplots(ncols=3, figsize=(25,12))
    bb_title = f"{'Bluebild':8s}: {int(bb_vis):7d} vis   runtime: {bb_tot:6.2f} sec"
        
    fp = FontProperties(family="monospace", size=22, weight="bold")
    plt.suptitle(bb_title + "\n" + fits_title, x=0.33, y=0.92, ha='left').set_fontproperties(fp)

    bb_eq_cum = np.zeros([bipp_data.shape[1], bipp_data.shape[2]])
    for i in range(0, nlev):
        bb_eq_cum += bipp_data[i,:,:]

    # Divide by number of visibilities
    bb_eq_cum /= bb_vis
    if args.flip_lr:
        bb_eq_cum  = np.fliplr(bb_eq_cum)
    if args.flip_ud:
        bb_eq_cum  = np.flipud(bb_eq_cum)

    im0 = ax[0].imshow(bb_eq_cum)
    ax[0].set_title("Bluebild LSQ dirty / nb. vis", fontsize=20)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im0, cax=cax)

    im1 = ax[1].imshow(fits_data)
    ax[1].set_title(f"{fits_name} dirty", fontsize=20)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax)
        
    diff = bb_eq_cum - fits_data
    im2 = ax[2].imshow(diff)
    ax[2].set_title(f"Bluebild minus {fits_name}", fontsize=20)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax)

    print(f"-I- Bluebild minus {fits_name} min, max: {np.min(diff):.3f}, {np.max(diff):.3f}")

    plt.tight_layout()    
    fig.savefig(os.path.join(outdir, f"{outname}_Bluebild_{fits_name}_vis_scaled.png"))
    json_file.close()


def plot_eigen_vectors(X, filename, outdir, title):

    fig, ax = plt.subplots(2, 2, figsize=(12,12))
    fig.suptitle(title, fontsize=22)

    im = ax[0,0].matshow(X.real,      cmap = 'seismic')
    ax[0,0].set_ylabel('aaaa')
    ax[0,0].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = ax[0,1].matshow(X.imag,      cmap = 'seismic')
    ax[0,1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    divider = make_axes_locatable(ax[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.savefig(os.path.join(outdir, filename + '.png'))
    plt.close()


def plot_beamweight_matrix(X, filename, outdir, title):
    plot_complex_matrix(X, filename, outdir, title, 'Station index', 'Station index')


def plot_gram_matrix(X, filename, outdir, title):
    plot_complex_matrix(X, filename, outdir, title, 'Beam index', 'Beam index')


def plot_visibility_matrix(X, filename, outdir, title):    
    plot_complex_matrix(X, filename, outdir, title, 'Station index', 'Station index')


def plot_complex_matrix(X, filename, outdir, title, xlabel, ylabel):

    fig, ax = plt.subplots(2, 2, figsize=(12,12))
    fig.suptitle(title, fontsize=22)

    im = ax[0,0].matshow(X.real,      cmap = 'seismic')
    ax[0,0].set_ylabel(ylabel)
    ax[0,0].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = ax[0,1].matshow(X.imag,      cmap = 'seismic')
    ax[0,1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    divider = make_axes_locatable(ax[0,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = ax[1,0].matshow(abs(X),      cmap = 'seismic')
    ax[1,0].set_xlabel(xlabel)
    ax[1,0].set_ylabel(ylabel)
    ax[1,0].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    divider = make_axes_locatable(ax[1,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = ax[1,1].matshow(np.angle(X), cmap = 'seismic')
    ax[1,1].set_xlabel(xlabel)
    ax[1,1].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    divider = make_axes_locatable(ax[1,1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    ax[0,0].set_title(f"Real (sum diag = {X.real.diagonal().sum():.3f})")
    ax[0,1].set_title(f"Imag (sum diag = {X.imag.diagonal().sum():.3f})")
    ax[1,0].set_title(f"Amplitude (sum = {abs(X).sum():.3f}, sum diag = {abs(X).diagonal().sum():.3f})")
    ax[1,1].set_title(f"Phase (sum diag = {np.angle(X).diagonal().sum():.3f})")

    plt.savefig(os.path.join(outdir, filename + '.png'))
    plt.close()


def plot_uvw(uvw, pixw, filename, outdir, title):

    if pixw % 2 != 0:
        raise Exception("pixw must be modulo 2!")

    uvw = np.reshape(uvw, (-1,3))

    # Remove zero baselines
    rows_to_del = []
    for i in range(0, uvw.shape[0]):
        if uvw[i,:].all() == 0:
            rows_to_del.append(i)
    #print("-W- deleting zero baselines at indices:", rows_to_del)
    uvw = np.delete(uvw, rows_to_del, axis=0)
       
    #print("uvw.shape =", uvw.shape)

    ui = np.rint(uvw[:,0])
    vi = np.rint(uvw[:,1])
    uvi_max = int(max(np.amax(ui), np.amax(vi)))
    #print("uvi_max =", uvi_max)
    scalor = pixw / 2 / uvi_max
    #print("scalor = ", scalor)
    uv_map = np.zeros((pixw+1, pixw+1))
    print(uv_map, uv_map.dtype, uv_map.shape)

    already_set = 0
    for i in range(0, len(uvw[:,0])):
        #if i < 10:
        #    print(" ... adding:", ui[i], vi[i], uvi_max, i)
        ix = int(np.rint((ui[i]+uvi_max)*scalor))
        iy = int(np.rint((vi[i]+uvi_max)*scalor))
        if uv_map[ix][iy] == 1:
            already_set += 1
        else:
            uv_map[ix][iy] = 1
    print(f"-W- {already_set} points already previously set. Visibilities = {uvw.shape[0]} - {already_set} = {uvw.shape[0] - already_set}")

    #for a in range(-1, 2):
    #    for b in range(-1, 2):
    #        print(f"Origin + {a},{b}", uv_map[int(pixw/2)+a][int(pixw/2)+b])

    fig, ax = plt.subplots(1, 3, figsize=(27, 10))
    fig.suptitle(f"PSF\n(normalized by sum of unitary weights, i.e. nb of unique \"gridded\" baselines - {uvw.shape[0] - already_set})", fontsize=22)

    im = ax[0].imshow(np.abs(np.fft.fftshift(np.fft.fft2(uv_map))) / (uvw.shape[0] - already_set))
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = ax[1].imshow(np.angle(np.fft.fftshift(np.fft.fft2(uv_map))))
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = ax[2].imshow(uv_map)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    ax[0].set_title(f"Amplitude [?]")
    ax[1].set_title(f"Phase [rad]")
    ax[2].set_title(f"uv map")

    plt.savefig(os.path.join(outdir, filename + '_PSF.png'))
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run general plotting utility')
    parser.add_argument('--bb_grid',   help='Bluebild grid .npy')
    parser.add_argument('--bb_data',   help='Bluebild data .npy')
    parser.add_argument('--bb_json',   help='Bluebild json .npy')
    parser.add_argument('--wsc_fits',  help='WSClean fits file')
    parser.add_argument('--wsc_log',   help='WSClean log file')
    parser.add_argument('--casa_fits', help='CASA fits file')
    parser.add_argument('--casa_log',  help='CASA log file')
    parser.add_argument('--outname',   help='Plots naming prefix', required=True)
    parser.add_argument('--outdir',    help='Plots output directory', required=True)
    parser.add_argument('--flip_lr',   help='Flip image left-rigth', action='store_true')
    parser.add_argument('--flip_ud',   help='Flip image up-down',    action='store_true')
    args = parser.parse_args()

    do_bb   = False
    do_wsc  = False
    do_casa = False
    if args.bb_grid   and args.bb_data and args.bb_json: do_bb   = True
    if args.wsc_fits  and args.wsc_log:                  do_wsc  = True 
    if args.casa_fits and args.casa_log:                 do_casa = True
    print("-I- consider Bluebild?", do_bb)
    print("-I- consider WSClean? ", do_wsc)
    print("-I- consider CASA?    ", do_casa)

    if do_wsc and do_casa:
        plot_wsclean_casa(args.wsc_fits, args.wsc_log, args.casa_fits, args.casa_log)

    if do_bb and do_wsc:
        plot_bluebild_wsclean(args.bb_grid, args.bb_data, args.bb_json, args.wsc_fits, args.wsc_log,
                              args.outname, args.outdir)

    if do_bb and do_casa:
        plot_bluebild_casa(args.bb_grid, args.bb_data, args.bb_json, args.casa_fits, args.casa_log)
