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
from skimage.metrics import structural_similarity as ssim
from matplotlib.patches import Circle


def read_fits_file(fits_file):
    hdul = fits.open(fits_file)
    print("-I- FITS hdul.info()\n", hdul.info())
    header = hdul[0].header
    data   = hdul[0].data
    #print("-I- FITS header", header)
    #print("-I- FITS data.shape:", data.shape)
    return header, data


def get_bipp_info_from_json(bipp_json):
    jf = open(bipp_json)
    json_data = json.load(jf)
    bipp_info  = f"sol package  = BIPP\n"
    bipp_info += f"bipp ifim vis = {json_data['visibilities']['ifim']}\n"
    bipp_info += f"bipp t_ifpe   = {json_data['timings']['ifpe']:.3f}\n"
    bipp_info += f"bipp t_ifim   = {json_data['timings']['ifim']:.3f}\n"
    bipp_info += f"bipp t_tot    = {json_data['timings']['tot']:.3f}\n"
    jf.close()
    return bipp_info


def plot_wsc_casa_bb(wsc_fits, wsc_log, casa_fits, casa_log, bb_grid, bb_data, bb_json,
                     outdir, outname):

    outname += f"_wsc_casa_bb"

    bb_info = get_bipp_info_from_json(bb_json)
    _, _, casa_info = casatb.get_casa_info_from_log(casa_log)
    _, _, _, _, _, wsc_info = wscleantb.get_wsclean_info_from_log(wsc_log)
    print(bb_info)
    print(casa_info)
    print(wsc_info)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #  WARNING! Hacking wsclean fits file to injec bipp's data
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    import shutil
    bb_fits = wsc_fits + '_bb_hack.fits'
    shutil.copyfile(wsc_fits, bb_fits)
    hdulist = fits.open(bb_fits)
    hdulist[0].data = np.sum(np.load(bb_data).transpose(1,2,0), axis=2)
    if args.flip_lr:
        hdulist[0].data = np.fliplr(hdulist[0].data)
    hdulist.writeto(bb_fits, overwrite=True)
    hdulist.close()

    from astropy.wcs import WCS

    wsc_hdu  = fits.open(wsc_fits)[0]
    casa_hdu = fits.open(casa_fits)[0]
    bb_hdu   = fits.open(bb_fits)[0]
    for hdu in wsc_hdu, casa_hdu, bb_hdu:
        #print(hdu.data.shape)
        if hdu.data.shape[0] == 1 and hdu.data.shape[1] == 1:
            hdu.data = hdu.data.reshape(hdu.data.shape[2:4])
        #print(hdu.data.shape)

    wsc_wcs  = WCS(wsc_hdu.header)
    casa_wcs = WCS(casa_hdu.header)
    bb_wcs   = WCS(bb_hdu.header)

    wcs = wsc_wcs

    wsc_data  = wsc_hdu.data
    casa_data = casa_hdu.data
    bb_data   = bb_hdu.data

    diff_casa_wsc = casa_data - wsc_data
    diff_bb_wsc   = bb_data   - wsc_data
    diff_bb_casa  = bb_data   - casa_data

    # color bar min/max
    zmin = min(np.amin(wsc_data), np.amin(casa_data), np.amin(bb_data))
    zmax = max(np.amax(wsc_data), np.amax(casa_data), np.amax(bb_data))
    print(f"-D- range = [{zmin:.3f}, {zmax:.3f}]")
    max_absz = max(abs(zmin), abs(zmax))
    dmin = min(np.amin(diff_casa_wsc), np.amin(diff_bb_wsc), np.amin(diff_bb_casa))
    dmax = max(np.amax(diff_casa_wsc), np.amax(diff_bb_wsc), np.amax(diff_bb_casa))
    print(f"-D- diff  = [{dmin:.3f}, {dmax:.3f}]")
    abs_max = max(abs(dmin), abs(dmax))
    print(f"-D- max abs diff = {abs_max:.3f}")

    rad1 = 0.50
    rad2 = 0.75
    rad3 = 1.00

    half_width_pix = bb_data.shape[0] / 2 #EO: check if that always hold!
    rad1_pix = rad1 * half_width_pix
    rad2_pix = rad2 * half_width_pix
    rad3_pix = rad3 * half_width_pix

    plt.clf()
    fig = plt.figure(figsize=(21,14)) 

    plt.rc('axes',  titlesize=20) 
    plt.rc('axes',  labelsize=16)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    plt.subplot(231, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(wsc_data, vmin=-max_absz, vmax=max_absz)
    plt.grid(color='white', ls='solid')
    plt.title('(a) WSClean', pad=14)
    plt.gca().coords[0].set_axislabel(" ")
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.gca().coords[1].set_axislabel("Declination")

    plt.subplot(232, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(casa_data, vmin=-max_absz, vmax=max_absz)
    plt.grid(color='white', ls='solid')
    plt.title('(b) CASA', pad=14)
    plt.gca().coords[0].set_axislabel(" ")
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.gca().coords[1].set_axislabel(" ")
    plt.gca().coords[1].set_ticklabel_visible(False)

    plt.subplot(233, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(bb_data, vmin=-max_absz, vmax=max_absz)
    plt.grid(color='white', ls='solid')
    plt.title('(c) Bluebild', pad=14)
    plt.gca().coords[0].set_axislabel(" ")
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.gca().coords[1].set_axislabel(" ")
    plt.gca().coords[1].set_ticklabel_visible(False)

    cb_ax = fig.add_axes([0.92, 0.535, 0.012, 0.34])
    cbar = plt.colorbar(cax=cb_ax)
    cbar.ax.tick_params(size=0)

    plt.subplot(234, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(diff_casa_wsc, vmin=-abs_max, vmax=abs_max, cmap='seismic')
    plt.grid(color='gray', ls='solid')
    plt.title('(d) CASA minus WSClean', pad=14)
    plt.gca().coords[0].set_axislabel("Right ascension")
    plt.gca().coords[1].set_axislabel("Declination")

    plt.gca().set_autoscale_on(False)

    r1 = Circle((half_width_pix, half_width_pix), rad1_pix, edgecolor='darkorange', facecolor='none')
    plt.gca().add_patch(r1)
    r2 = Circle((half_width_pix, half_width_pix), rad2_pix, edgecolor='fuchsia', facecolor='none')
    plt.gca().add_patch(r2)
    r3 = Circle((half_width_pix, half_width_pix), rad3_pix, edgecolor='black', facecolor='none')
    plt.gca().add_patch(r3)

    plt.gca().set_autoscale_on(False)
    plt.subplot(235, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(diff_bb_casa, vmin=-abs_max, vmax=abs_max, cmap='seismic')
    plt.grid(color='gray', ls='solid')
    plt.title('(e) Bluebild minus CASA', pad=14)
    plt.gca().coords[0].set_axislabel("Right ascension")
    plt.gca().coords[1].set_axislabel(" ")
    plt.gca().coords[1].set_ticklabel_visible(False)
    r1 = Circle((half_width_pix, half_width_pix), rad1_pix, edgecolor='darkorange', facecolor='none')
    plt.gca().add_patch(r1)
    r2 = Circle((half_width_pix, half_width_pix), rad2_pix, edgecolor='fuchsia', facecolor='none')
    plt.gca().add_patch(r2)
    r3 = Circle((half_width_pix, half_width_pix), rad3_pix, edgecolor='black', facecolor='none')
    plt.gca().add_patch(r3)

    plt.gca().set_autoscale_on(False)
    plt.subplot(236, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(diff_bb_wsc, vmin=-abs_max, vmax=abs_max, cmap='seismic')
    plt.grid(color='gray', ls='solid')
    plt.title('(f) Bluebild minus WSClean', pad=14)
    plt.gca().coords[0].set_axislabel("Right ascension")
    plt.gca().coords[1].set_axislabel(" ")
    plt.gca().coords[1].set_ticklabel_visible(False)
    r1 = Circle((half_width_pix, half_width_pix), rad1_pix, edgecolor='darkorange', facecolor='none')
    plt.gca().add_patch(r1)
    r2 = Circle((half_width_pix, half_width_pix), rad2_pix, edgecolor='fuchsia', facecolor='none')
    plt.gca().add_patch(r2)
    r3 = Circle((half_width_pix, half_width_pix), rad3_pix, edgecolor='black', facecolor='none')
    plt.gca().add_patch(r3)

    cb_ax = fig.add_axes([0.92, 0.115, 0.012, 0.34])
    cbar = plt.colorbar(cax=cb_ax)
    cbar.ax.tick_params(size=0)

    #plt.tight_layout()
    basename = os.path.join(outdir, outname)
    file_png  = basename + '.png'
    plt.savefig(file_png)
    print("-I-", file_png)
    sys.exit(0)

    # Write .png and associated .misc file
    basename = os.path.join(outdir, outname)
    file_png  = basename + '.png'
    file_misc = basename + '.misc'

    with open(file_misc, 'w') as f:
        f.write("WSClean\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(wsc_info)
        f.write("\nCASA\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(casa_info)
        f.write("\nBluebild\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(bb_info)
        """
        f.write(f"diff min = {np.amin(diff_data):.3f}\n")
        f.write(f"diff max = {np.amax(diff_data):.3f}\n")
        f.write(f"diff rms = {rmse(data2, data1):.3f}\n")
        f.write(f"diff rms 1s = {rmse_sig(data2, data1, 0.680):.3f}\n")
        f.write(f"diff rms 2s = {rmse_sig(data2, data1, 0.950):.3f}\n")
        f.write(f"diff rms 3s = {rmse_sig(data2, data1, 0.997):.3f}\n")
        f.write(f"diff rms check = {rmse_sig(data2, data1, 1.415):.3f}\n")
        f.write("-------------------------------------------------------------------\n")
        """

    plt.tight_layout()
    plt.savefig(file_png)
    print("-I-", file_png)
    print("-I-", file_misc)




def plot_bluebild_casa(bipp_grid_npy, bipp_data_npy, bipp_json, fits_file, log_file, outname, outdir):
    print("==========================================================")
    print(" Plotting Bluebild vs CASA")
    print("==========================================================")

    header, data = read_fits_file(fits_file)
    totvis, t_inv, casa_info = casatb.get_casa_info_from_log(log_file)

    title  = f"{'CASA':8s}: {int(totvis):7d} vis   runtime: {t_inv:6.2f} sec"

    plot_bluebild_vs_fits(bipp_grid_npy, bipp_data_npy, bipp_json, 'CASA', data, title, outname, outdir)


    # Astropy based plot
    bipp_info = get_bipp_info_from_json(bipp_json)
    outname += f"_Bluebild_vs_CASA_astro"

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #  WARNING! Hacking wsclean fits file to injec bipp's data
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    import shutil
    bipp_fits = fits_file + '_bipp_hack.fits'
    shutil.copyfile(fits_file, bipp_fits)
    hdulist = fits.open(bipp_fits)
    hdulist[0].data = np.sum(np.load(bipp_data_npy).transpose(1,2,0), axis=2)
    if args.flip_lr:
        hdulist[0].data = np.fliplr(hdulist[0].data)
    hdulist.writeto(bipp_fits, overwrite=True)
    hdulist.close()

    plot_fits_vs_fits('CASA',     fits_file, casa_info,
                      'Bluebild', bipp_fits, bipp_info,
                      outdir, outname)

def plot_bluebild_wsclean(bipp_grid_npy, bipp_data_npy, bipp_json, fits_file, log_file, outname, outdir):
    print("==========================================================")
    print(" Plotting Bluebild vs WSClean")
    print("==========================================================")

    header, data = read_fits_file(fits_file)
    #print(header)
    totvis, gridvis, t_inv, t_pred, t_deconv, wsc_info = wscleantb.get_wsclean_info_from_log(log_file)

    title  = f"{'WSClean':8s}: {int(totvis):7d} vis   runtime: {t_inv:6.2f} sec"

    plot_bluebild_vs_fits(bipp_grid_npy, bipp_data_npy, bipp_json, 'WSClean', data, title, outname, outdir)

    bipp_info = get_bipp_info_from_json(bipp_json)

    outname += f"_Bluebild_vs_WSClean_astro"

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #  WARNING! Hacking wsclean fits file to injec bipp's data
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    import shutil
    bipp_fits = fits_file + '_bipp_hack.fits'
    shutil.copyfile(fits_file, bipp_fits)
    hdulist = fits.open(bipp_fits)
    hdulist[0].data = np.sum(np.load(bipp_data_npy).transpose(1,2,0), axis=2)
    if args.flip_lr:
        hdulist[0].data = np.fliplr(hdulist[0].data)
    hdulist.writeto(bipp_fits, overwrite=True)
    hdulist.close()

    plot_fits_vs_fits('WSClean',  fits_file, wsc_info,
                      'Bluebild', bipp_fits, bipp_info,
                      outdir, outname)


def plot_wsclean_casa(wsc_fits, wsc_log, casa_fits, casa_log, outname, outdir):
    print("==========================================================")
    print(" Plotting WSClean vs CASA")
    print("==========================================================")

    casa_header, casa_data = read_fits_file(casa_fits)
    casa_totvis, casa_t_inv, casa_info = casatb.get_casa_info_from_log(casa_log)

    wsc_header, wsc_data = read_fits_file(wsc_fits)
    wsc_totvis, wsc_gridvis, wsc_t_inv, wsc_t_pred, wsc_t_deconv, wsc_info = wscleantb.get_wsclean_info_from_log(wsc_log)

    title = f"WSClean / CASA visibilities: {wsc_totvis} / {casa_totvis}"

    outname += f"_WSClean_vs_CASA_astro"
    plot_fits_vs_fits('WSClean', wsc_fits, wsc_info, 'CASA', casa_fits, casa_info, outdir, outname)


def plot_fits_vs_fits(name1, fits1, info1, name2, fits2, info2, outdir, outname):

    from astropy.wcs import WCS

    hdu1 = fits.open(fits1)[0]
    hdu2 = fits.open(fits2)[0]
    for hdu in hdu1, hdu2:
        #print(hdu.data.shape)
        if hdu.data.shape[0] == 1 and hdu.data.shape[1] == 1:
            hdu.data = hdu.data.reshape(hdu.data.shape[2:4])
        #print(hdu.data.shape)

    wcs1 = WCS(hdu1.header)
    wcs2 = WCS(hdu2.header)
    #print(wcs1)
    #print(wcs2)
    wcs = wcs1

    data1 = hdu1.data
    data2 = hdu2.data
    diff_data = data2 - data1
    print(diff_data)
    rel_diff = diff_data / data1
    print(f"rel diff = {np.amin(rel_diff):.3f}, {np.amax(rel_diff):.3f}")

    # color bar min/max
    zmin = min(np.amin(data1), np.amin(data2))
    zmax = max(np.amax(data1), np.amax(data2))
    print(f"-D- range = [{zmin:.3f}, {zmax:.3f}]")
    dmin = np.amin(diff_data)
    dmax = np.amax(diff_data)
    print(f"-D- diff  = [{dmin:.3f}, {dmax:.3f}]")
    abs_max = max(abs(dmin), abs(dmax))
    print("????????????????", abs_max)

    plt.clf()
    plt.figure(figsize=(20,7))

    # data1
    ax = plt.subplot(131, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(data1, vmin=zmin, vmax=zmax)
    plt.grid(color='white', ls='solid')
    plt.xlabel('Right ascension')
    plt.ylabel('Declination')
    plt.title(name1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.grid(False)
    plt.gca().coords[1].set_ticks_position('r')
    plt.gca().coords[1].set_ticklabel_position('r')
    plt.gca().coords[1].set_axislabel(' ')
    plt.gca().coords[1].set_axislabel_position('r')
    plt.gca().coords[1].set_ticks_visible(False)
    plt.gca().coords[0].set_ticks_visible(False)
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.colorbar(cax=cax)    

    # data2
    ax = plt.subplot(132, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(data2, vmin=zmin, vmax=zmax)
    plt.grid(color='white', ls='solid')
    plt.xlabel('Right ascension')
    plt.ylabel(' ')
    ya = plt.gca().coords[1]
    ya.set_ticks_visible(False)
    ya.set_ticklabel_visible(False)
    ya.set_axislabel('')
    plt.title(name2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.gca().coords[0].set_ticks_visible(False)
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.gca().coords[1].set_ticks_visible(False)
    plt.gca().coords[1].set_ticklabel_visible(False)
    plt.gca().axis('off')
    
    """
    cax = divider.append_axes("left", size="5%", pad=0.0)
    plt.gca().coords[0].set_ticks_visible(False)
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.gca().coords[1].set_ticks_visible(False)
    plt.gca().coords[1].set_ticklabel_visible(False)
    #plt.gca().axis('off')
    """

    # data2 minus data1
    ax = plt.subplot(133, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(diff_data, cmap='seismic', vmin=-abs_max, vmax=abs_max)
    #plt.imshow(diff_data / data1, cmap='seismic')
    plt.grid(color='white', ls='solid')
    plt.xlabel('Right ascension')
    plt.ylabel(' ')
    plt.title(f"Difference ({name2} minus {name1})")
    #plt.title(f"Relative difference of {name2} to {name1}")
    ya = plt.gca().coords[1]
    ya.set_ticks_visible(False)
    ya.set_ticklabel_visible(False)
    ya.set_axislabel('')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.grid(False)
    plt.gca().coords[1].set_ticks_position('r')
    plt.gca().coords[1].set_ticklabel_position('r')
    plt.gca().coords[1].set_axislabel('Intensity difference [Jy/beam]')
    #plt.gca().coords[1].set_axislabel('Relative difference [%]')
    plt.gca().coords[1].set_axislabel_position('r')
    plt.gca().coords[1].set_ticks_visible(False)
    plt.gca().coords[0].set_ticks_visible(False)
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.colorbar(cax=cax)

    # Write .png and associated .misc file
    basename = os.path.join(outdir, outname)
    file_png  = basename + '.png'
    file_misc = basename + '.misc'

    with open(file_misc, 'w') as f:
        f.write("Reference (data1)\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(info1)
        f.write("\nSolution (data2)\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(info2)
        f.write("\nDifference (data2 - data1)\n")
        f.write("-------------------------------------------------------------------\n")
        f.write(f"diff min = {np.amin(diff_data):.3f}\n")
        f.write(f"diff max = {np.amax(diff_data):.3f}\n")
        f.write(f"diff rms = {rmse(data2, data1):.3f}\n")
        f.write(f"diff rms 1s = {rmse_sig(data2, data1, 0.680):.3f}\n")
        f.write(f"diff rms 2s = {rmse_sig(data2, data1, 0.950):.3f}\n")
        f.write(f"diff rms 3s = {rmse_sig(data2, data1, 0.997):.3f}\n")
        f.write(f"diff rms check = {rmse_sig(data2, data1, 1.415):.3f}\n")
        f.write("-------------------------------------------------------------------\n")

    plt.tight_layout()

    plt.savefig(file_png)
    print("-I-", file_png)
    print("-I-", file_misc)



def plot_fits_vs_fits_old(fits_file, bipp_data_npy, ref_txt_info, bipp_txt_info, outdir, outname):

    #
    ## WARNING! Hacking wsclean fits file to injec bipp's data
    #
    import shutil
    bipp_fits = fits_file + '_bipp_hack.fits'
    shutil.copyfile(fits_file, bipp_fits)
    hdulist = fits.open(bipp_fits)
    hdulist[0].data = np.sum(np.load(bipp_data_npy).transpose(1,2,0), axis=2)
    hdulist.writeto(bipp_fits, overwrite=True)
    hdulist.close()

    from astropy.wcs import WCS

    hdu_ref = fits.open(fits_file)[0]
    wcs = WCS(hdu_ref.header)
    ref_data = hdu_ref.data.reshape(hdu_ref.data.shape[2:4])
    
    hdu_bipp = fits.open(bipp_fits)[0]
    bipp_data = hdu_bipp.data
    if args.flip_lr:
        bipp_data = np.fliplr(bipp_data)
    diff_data = bipp_data - ref_data

    #fig, ax = plt.subplots(ncols=3, figsize=(25,12), width_ratios=[1, 0.93, 1])
    #bb_title = "ABC 123" #f"{'Bluebild':8s}: {int(bb_vis):7d} vis   runtime: {bb_tot:6.2f} sec"
        
    #fp = FontProperties(family="monospace", size=22, weight="bold")
    #plt.suptitle(bb_title + "\n" + fits_title, x=0.25, y=0.92, ha='left').set_fontproperties(fp)
    #plt.suptitle("123 CBA")

    # color bar min/max
    zmin = min(np.amin(ref_data), np.amin(bipp_data))
    zmax = max(np.amax(ref_data), np.amax(bipp_data))
    print(f"-D- range = [{zmin:.3f}, {zmax:.3f}]")
    dmin = np.amin(bipp_data - ref_data)
    dmax = np.amax(bipp_data - ref_data)
    print(f"-D- diff  = [{dmin:.3f}, {dmax:.3f}]")
    
    plt.clf()
    cm = 1/2.54  # centimeters in inches
    plt.figure(figsize=(20,7))
    # ref
    ax = plt.subplot(131, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(ref_data, vmin=zmin, vmax=zmax)
    plt.grid(color='white', ls='solid')
    plt.xlabel('Right ascension')
    plt.ylabel('Declination')
    plt.title('WSClean')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.grid(False)
    plt.gca().coords[1].set_ticks_position('r')
    plt.gca().coords[1].set_ticklabel_position('r')
    plt.gca().coords[1].set_axislabel(' ')
    plt.gca().coords[1].set_axislabel_position('r')
    plt.gca().coords[1].set_ticks_visible(False)
    plt.gca().coords[0].set_ticks_visible(False)
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.colorbar(cax=cax)    

    # bipp
    ax = plt.subplot(132, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(bipp_data, vmin=zmin, vmax=zmax)
    plt.grid(color='white', ls='solid')
    plt.xlabel('Right ascension')
    plt.ylabel(' ')
    ya = plt.gca().coords[1]
    ya.set_ticks_visible(False)
    ya.set_ticklabel_visible(False)
    ya.set_axislabel('')

    plt.title('Bluebild')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.gca().coords[0].set_ticks_visible(False)
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.gca().coords[1].set_ticks_visible(False)
    plt.gca().coords[1].set_ticklabel_visible(False)
    plt.gca().axis('off')
    
    """
    cax = divider.append_axes("left", size="5%", pad=0.0)
    plt.gca().coords[0].set_ticks_visible(False)
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.gca().coords[1].set_ticks_visible(False)
    plt.gca().coords[1].set_ticklabel_visible(False)
    #plt.gca().axis('off')
    """

    # bipp minus ref
    ax = plt.subplot(133, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(diff_data)
    plt.grid(color='white', ls='solid')
    plt.xlabel('Right ascension')
    plt.ylabel(' ')
    plt.title('Difference (Bluebild minus WSClean)')
    ya = plt.gca().coords[1]
    ya.set_ticks_visible(False)
    ya.set_ticklabel_visible(False)
    ya.set_axislabel('')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.grid(False)
    plt.gca().coords[1].set_ticks_position('r')
    plt.gca().coords[1].set_ticklabel_position('r')
    plt.gca().coords[1].set_axislabel('Intensity [Jy/beam]')
    plt.gca().coords[1].set_axislabel_position('r')
    plt.gca().coords[1].set_ticks_visible(False)
    plt.gca().coords[0].set_ticks_visible(False)
    plt.gca().coords[0].set_ticklabel_visible(False)
    plt.colorbar(cax=cax)

    # Write .png and associated .misc file
    basename = os.path.join(outdir, outname)
    file_png  = basename + '.png'
    file_misc = basename + '.misc'

    with open(file_misc, 'w') as f:
        f.write("-------------------------------------------------------------------\n")
        f.write(ref_txt_info)
        f.write("-------------------------------------------------------------------\n")
        f.write(bipp_txt_info)
        f.write("-------------------------------------------------------------------\n")
        f.write(f"diff min = {np.amin(diff_data):.3f}\n")
        f.write(f"diff max = {np.amax(diff_data):.3f}\n")
        f.write(f"diff rms = {rmse(bipp_data, ref_data):.3f}\n")
        f.write(f"diff rms 1s = {rmse_sig(bipp_data, ref_data, 0.680):.3f}\n")
        f.write(f"diff rms 2s = {rmse_sig(bipp_data, ref_data, 0.950):.3f}\n")
        f.write(f"diff rms 3s = {rmse_sig(bipp_data, ref_data, 0.997):.3f}\n")
        f.write(f"diff rms check = {rmse_sig(bipp_data, ref_data, 1.415):.3f}\n")
        f.write("-------------------------------------------------------------------\n")

    plt.tight_layout()

    plt.savefig(basename + '.png')
    print("-I-", file_png)
    print("-I-", file_misc)


def rmse(y1, y2):
    return np.linalg.norm(y1 - y2) / len(y1)

def rmse_sig(y1, y2, sig):
    assert(y1.shape == y2.shape)
    assert(y1.shape[0] == y1.shape[1])
    x = np.arange(0, y1.shape[0])
    y = np.arange(0, y1.shape[0])
    cxy = y1.shape[0] / 2 
    R1 = y1.shape[0] / 2 * sig
    mask = (x[np.newaxis,:]-cxy)**2 + (y[:,np.newaxis]-cxy)**2 > R1**2
    diff = y1 - y2
    diff[mask] = 0.0
    return np.sqrt(np.sum(np.square(diff)) / y1.size)


def plot_wsclean_casa_old(wsc_fits, wsc_log, casa_fits, casa_log):
    print("==========================================================")
    print(" Plotting WSClean vs CASA")
    print("==========================================================")

    casa_header, casa_data = read_fits_file(casa_fits)
    casa_totvis, casa_t_inv, casa_info = casatb.get_casa_info_from_log(casa_log)

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
    print(f"-I- Bluebild cum energy levels [0, {nlev-1}]: min, max: {np.min(bb_eq_cum):.3f}, {np.max(bb_eq_cum):.3f}")

    """
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
    """

    # Raw intensity
    #
    fig, ax = plt.subplots(ncols=3, figsize=(25,12))
    bb_title = f"{'Bluebild':8s}: {int(bb_vis):7d} vis   runtime: {bb_tot:6.2f} sec"
    fp = FontProperties(family="monospace", size=22, weight="bold")
    plt.suptitle(bb_title + "\n" + fits_title, x=0.25, y=0.92, ha='left').set_fontproperties(fp)

    bb_eq_cum = np.zeros([bipp_data.shape[1], bipp_data.shape[2]])
    for i in range(0, bipp_data.shape[0]):
        bb_eq_cum += bipp_data[i,:,:]

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

    """
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
    """

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

    ax[0,0].set_title(f"Real (sum = {X.real.sum():.3f}, sum diag = {X.real.diagonal().sum():.3f})")
    ax[0,1].set_title(f"Imag. (sum = {X.imag.sum():.3f}, sum diag = {X.imag.diagonal().sum():.3f})")
    ax[1,0].set_title(f"Amplitude (sum = {abs(X).sum():.3f}, sum diag = {abs(X).diagonal().sum():.3f})")
    ax[1,1].set_title(f"Phase (sum = {np.angle(X).sum():.3f}, sum diag = {np.angle(X).diagonal().sum():.3f})")

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
    parser.add_argument('--outname',   help='Plots naming prefix',    required=True)
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


    if do_bb and do_wsc and do_casa:
        plot_wsc_casa_bb(args.wsc_fits, args.wsc_log,
                         args.casa_fits, args.casa_log,
                         args.bb_grid, args.bb_data, args.bb_json,
                         args.outdir, args.outname)
    print("__early_exit__")
    sys.exit(0)

    if do_wsc and do_casa:
        plot_wsclean_casa(args.wsc_fits, args.wsc_log, args.casa_fits, args.casa_log,
                          args.outname, args.outdir)

    if do_bb and do_wsc:
        plot_bluebild_wsclean(args.bb_grid, args.bb_data, args.bb_json, args.wsc_fits, args.wsc_log,
                              args.outname, args.outdir)

    if do_bb and do_casa:
        plot_bluebild_casa(args.bb_grid, args.bb_data, args.bb_json, args.casa_fits, args.casa_log,
                           args.outname, args.outdir)

        
