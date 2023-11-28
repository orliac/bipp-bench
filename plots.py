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
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

def compute_rms(x):
    return np.sqrt(np.sum(x**2) / x.size)

def compute_rmse(y1, y2):
    assert(y1.shape == y2.shape)
    assert(y1.shape[0] == y1.shape[1])
    diff = y1 - y2
    rmse = np.sqrt(np.sum(np.square(diff)) / diff.size)
    return rmse

def compute_rmse_sig(y1, y2, sig):
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


def read_fits_file(fits_file):
    hdul = fits.open(fits_file)
    print("-I- FITS hdul.info()\n", hdul.info())
    header = hdul[0].header
    data   = hdul[0].data
    #print("-I- FITS header", header)
    #print("-I- FITS data.shape:", data.shape)
    return header, data


def read_json_file(json_file):
    jf = open(json_file)
    json_data = json.load(jf)
    jf.close()
    return json_data

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


def my_subplot(plt, wcs, data, vmin, vmax, plot_grid, plot_circles, npix, cmap,
               sky_filepath, fh_sky_out, locate_max, wsc_size, wsc_scale, solution=None,
               nz_vis=None, grid_color='white'):
    
    plt.imshow(data, vmin=vmin, vmax=vmax, interpolation ="nearest", origin ="lower",
               aspect=1, cmap=cmap)

    # To add a second axis labelling with pixel scale
    """
    plt.gca().coords[0].set_ticks_position('b')
    plt.gca().coords[1].set_ticks_position('l')
    plt.gca().secondary_yaxis('right')
    plt.gca().secondary_xaxis('top')
    """
    
    if plot_grid:
        plt.grid(color=grid_color, ls='solid')

    if plot_circles:
        rad1 = 0.50
        rad2 = 0.75
        rad3 = 1.00
        half_width_pix = wsc_size / 2
        rad1_pix = rad1 * half_width_pix
        rad2_pix = rad2 * half_width_pix
        rad3_pix = rad3 * half_width_pix

        r1 = Circle((half_width_pix, half_width_pix), rad1_pix, edgecolor='darkorange', facecolor='none')
        plt.gca().add_patch(r1)
        r2 = Circle((half_width_pix, half_width_pix), rad2_pix, edgecolor='fuchsia', facecolor='none')
        plt.gca().add_patch(r2)
        r3 = Circle((half_width_pix, half_width_pix), rad3_pix, edgecolor='black', facecolor='none')
        plt.gca().add_patch(r3)

        
    # Add text vertically on left hand side of the plot
    npix = data.shape[0]
    assert(npix == wsc_size)
    opix = int(npix * 0.015)
    min = np.min(data)
    max = np.max(data)
    small_text = f"{npix} x {npix} x {wsc_scale:.3f}\""
    if nz_vis:
        small_text += f", {nz_vis} vis."
    plt.gca().text(opix, opix, small_text, fontsize=4, rotation='vertical', color='lime')
    mmr_text = f"[{min:.2e}, {max:.2e}]"
    if solution == None:
        rms = compute_rms(data)
        mmr_text += f", RMS={rms:.2e}"
    plt.gca().text(opix, npix - opix, mmr_text, fontsize=8, rotation='vertical', color='black', backgroundcolor='white',
                   verticalalignment='top')

    #print("??? wcs =", wcs)
    
    # Plot simulated sources if file is provided
    if sky_filepath:
        print("-D- sky_filepath:", sky_filepath)
        with open(sky_filepath) as sky_file:
            src_a = []
            for src_line in sky_file:
                src_line = src_line.strip()
                
                # Header lines
                if src_line.startswith('#'):
                    #print(">>>",src_line,'<<<')
                    patt = "#phase_centre_deg:\s*"
                    if re.search(patt, src_line):
                        ans = re.split("\s+", re.split(patt, src_line)[-1])
                        sim_ra_deg, sim_dec_deg = float(ans[0]), float(ans[1])
                        assert(wcs.wcs.crval[0] == sim_ra_deg)
                        assert(wcs.wcs.crval[1] == sim_dec_deg)
                    patt = "#crpix:\s*"
                    if re.search(patt, src_line):
                        ans = re.split("\s+", re.split(patt, src_line)[-1])
                        crpix0, crpix1 = float(ans[0]), float(ans[1])
                        assert(wcs.wcs.crpix[0] == crpix0)
                        assert(wcs.wcs.crpix[1] == crpix1)
                    patt = "#origin:\s*"
                    if re.search(patt, src_line):
                        ans = re.split("\s+", re.split(patt, src_line)[-1])
                        origin = float(ans[0])
                    continue
                
                id, ra, dec, px, py, intensity = src_line.strip().split(" ")
                id, ra, dec, px, py, intensity = int(id), float(ra), float(dec), float(px), float(py), float(intensity)
                sky = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                pix_ = wcs.wcs_world2pix(sky.ra, sky.dec, 100000000.0, 1.0, origin)
                
                # Make sure we recover what was simulated
                #print(px, py, pix_)
                assert(abs(px - pix_[0]) < 1E-5)
                assert(abs(py - pix_[1]) < 1E-5)
                
                S = int(npix * 0.20)
                offset = 0
                if int(py + S/2) > npix:
                    offset = npix - int(py + S/2) - int(npix * 0.01)
                if int(py - S/2) < 0:
                    offset = 0 - int(py - S/2) + int(npix * 0.01)
                axin = plt.gca().inset_axes([int(px + npix * 0.02), int(py) - int(S/2 - offset), S, S], transform = plt.gca().transData)

                test = axin.imshow(data, vmin=vmin, vmax=vmax, interpolation ="nearest", origin ="lower", aspect=1, cmap=cmap)

                # Box to highlight zoomed in region (-1 as in Python context)
                x_ref = px - 1
                y_ref = py - 1
                lhw   = 4.5  
                rhw   = lhw + 0
                x1, x2, y1, y2 = x_ref - lhw, x_ref + rhw, y_ref - lhw, y_ref + rhw
                #print(f"box: {x1}->{x2}, {y1}->{y2}")
                axin.set_xlim(x1, x2)
                axin.set_ylim(y1, y2)
                axin.set_xticks([])
                axin.set_yticks([])
                plt.gca().indicate_inset_zoom(axin, edgecolor="black")

                # Mark true position of source
                axin.plot(x_ref, y_ref, '+', linewidth=0.05, markersize=5, color='fuchsia')
            
                # Find pixel of highest intensity in zoomed in region
                if locate_max:
                    i_max = -1E10
                    x_max, y_max = -1, -1
                    for x in range(int(x1+0.5), int(x2-0.5)):
                        for y in range(int(y1+0.5), int(y2-0.5)):
                            # TODO: understand why this is so... 
                            # !!! INDICES INVERTED !!!! mem layout?
                            if data[y][x] > i_max: 
                                x_max, y_max, i_max = x, y, data[y][x]
                    #print("plotting max at", x_max, y_max)
                    axin.plot(x_max, y_max, 'x', linewidth=0.05, markersize=5, color='black')
                    max_intensity = float(data[y_max][x_max])

                    # Python -> fits
                    x_max += 1
                    y_max += 1
            
                    # Compute distance to true source in pixels
                    dist_x = x_max - px
                    dist_y = y_max - py
                    dist = np.sqrt(dist_x * dist_x + dist_y * dist_y)
                    #print(f" .. max found on pixel {x_max}, {y_max} with intensity {max_intensity:5.3f}, dist to true position = {dist:.2f} [pixel]")
                    json_src = {
                        'simulated': {
                            'ra.deg':  f"{sky.ra.deg:.8f}",
                            'ra.hms':  sky.ra.to_string(u.hour),
                            'dec.deg': f"{sky.dec.deg:.8f}",
                            'deg.hms': sky.dec.to_string(u.hour),
                            'px': px,
                            'py': py,
                            'intensity': float(intensity),
                        },
                        'recovered': {
                            'dist.x': float(f"{dist_x:.3f}"),
                            'dist.y': float(f"{dist_y:.3f}"),
                            'dist': float(f"{dist:.3f}"),
                            'px': float(f"{x_max:.3f}"),
                            'py': float(f"{y_max:.3f}"),
                            'intensity': float(f"{max_intensity:.3f}")
                        }
                    }
                    src_a.append(json_src)

                #break
            
            #if fh_sky_out:
            #    json_obj = json.dumps({solution: src_a}, indent=4)
            #    fh_sky_out.write(json_obj)
            return src_a


def plot_wsc_casa_bb(wsc_fits, wsc_log, casa_fits, casa_log, bb_grid, bb_data, bb_json,
                     outdir, outname, sky_file, wsc_size, wsc_scale):

    # Define output files
    basename = os.path.join(outdir, outname)
    file_misc = basename + '.misc'
    file_json = basename + '.json'
    fh_sky_out = None
    if sky_file:
        print("Simulated point sources file dectected:", sky_file)
        sky_out = basename + '.sky'
        print(" => sky_out:", sky_out)
        fh_sky_out = open(sky_out, "w")


    bb_json_info = read_json_file(bb_json)
    bb_vis = bb_json_info['visibilities']['ifim']
    bb_info = get_bipp_info_from_json(bb_json)
    
    if not os.path.isfile(casa_log):
        raise Warning(f"{casa_log} not found. Abort plot.")
        return

    casa_vis, _, casa_info = casatb.get_casa_info_from_log(casa_log)
    wsc_tot_vis, wsc_grid_vis, _, _, _, wsc_info = wscleantb.get_wsclean_info_from_log(wsc_log)
    #print(bb_info)
    #print(casa_info)
    #print(wsc_info)

    print(casa_vis, wsc_grid_vis, bb_vis)
    
    
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
    if args.flip_ud:
        hdulist[0].data = np.flipud(hdulist[0].data)
    hdulist.writeto(bb_fits, overwrite=True)
    hdulist.close()

    wsc_hdu  = fits.open(wsc_fits)[0]
    casa_hdu = fits.open(casa_fits)[0]
    bb_hdu   = fits.open(bb_fits)[0]
    for hdu in wsc_hdu, casa_hdu, bb_hdu:
        print(hdu.data.shape)
        if hdu.data.shape[0] == 1 and hdu.data.shape[1] == 1:
            hdu.data = hdu.data.reshape(hdu.data.shape[2:4])
        print(hdu.data.shape)

    wsc_wcs  = WCS(wsc_hdu.header)
    casa_wcs = WCS(casa_hdu.header)
    bb_wcs   = WCS(bb_hdu.header)

    wcs = wsc_wcs
    print("wcs =", wcs)

    wsc_data  = wsc_hdu.data
    casa_data = casa_hdu.data
    bb_data   = bb_hdu.data

    diff_casa_wsc = casa_data - wsc_data
    diff_bb_wsc   = bb_data   - wsc_data
    diff_bb_casa  = bb_data   - casa_data

    npix = wsc_data.shape[0]
    print(f"-D- npix = {npix}")

    # color bar min/max
    casa_min, casa_max = np.amin(casa_data), np.amax(casa_data)
    wsc_min,  wsc_max  = np.amin(wsc_data),  np.amax(wsc_data)
    bb_min,   bb_max   = np.amin(bb_data),   np.amax(bb_data)
    print(f"-I- CASA     = [{casa_min:.6e}, {casa_max:.6e}]")
    print(f"-I- WSClean  = [{wsc_min:.6e}, {wsc_max:.6e}]")
    print(f"-I- Bluebild = [{bb_min:.6e}, {bb_max:.6e}]")

    zmin = min(wsc_min, casa_min, bb_min)
    zmax = max(wsc_max, casa_max, bb_max)
    print(f"-D- overall sol range  = [{zmin:.6e}, {zmax:.6e}]")
    max_absz = max(abs(zmin), abs(zmax))
    dmin = min(np.amin(diff_casa_wsc), np.amin(diff_bb_wsc), np.amin(diff_bb_casa))
    dmax = max(np.amax(diff_casa_wsc), np.amax(diff_bb_wsc), np.amax(diff_bb_casa))
    print(f"-D- overall diff range  = [{dmin:.6e}, {dmax:.6e}]")
    abs_max = max(abs(dmin), abs(dmax))
    print(f"-D- max abs diff = {abs_max:.6e}")

    #sys.exit(0)

    plt.clf()
    fig = plt.figure(figsize=(21,14)) 

    plt.rc('axes',  titlesize=20) 
    plt.rc('axes',  labelsize=16)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    vmin, vmax = -max_absz, max_absz
    vmin, vmax = zmin, zmax
    
    plot_grid    = True
    plot_circles = False
    if sky_file:
        plot_grid = plot_circles = False

    my_viridis = plt.cm.viridis
    my_viridis.set_bad((1, 0, 0, 1))

    locate_max = True

    ### CASA
    plt.subplot(231, projection=wcs, slices=('x', 'y', 0, 0))
    casa_sky = my_subplot(plt, wcs, casa_data, vmin, vmax, plot_grid, plot_circles, npix, 'viridis',
                          sky_file, fh_sky_out, locate_max, wsc_size, wsc_scale, solution='CASA',
                          nz_vis=casa_vis)
    plt.title('(a) CASA', pad=14)
    plt.gca().coords[0].set_axislabel(" ")
    plt.gca().coords[0].set_ticklabel_visible(True)
    plt.gca().coords[1].set_axislabel("Declination")

    ### WSClean
    plt.subplot(232, projection=wcs, slices=('x', 'y', 0, 0))
    wsc_sky = my_subplot(plt, wcs, wsc_data, vmin, vmax, plot_grid, plot_circles, npix, 'viridis',
                         sky_file, fh_sky_out, locate_max, wsc_size, wsc_scale, solution='WSClean',
                         nz_vis=wsc_grid_vis)
    plt.title('(b) WSClean', pad=14)
    plt.gca().coords[0].set_axislabel(" ")
    plt.gca().coords[0].set_ticklabel_visible(True)
    plt.gca().coords[1].set_axislabel(" ")
    if sky_file:
        plt.gca().coords[1].set_ticklabel_visible(False)

    ### Bluebild
    plt.subplot(233, projection=wcs, slices=('x', 'y', 0, 0))
    bb_sky = my_subplot(plt, wcs, bb_data, vmin, vmax, plot_grid, plot_circles, npix, 'viridis',
                        sky_file, fh_sky_out, locate_max, wsc_size, wsc_scale, solution='Bluebild',
                        nz_vis=bb_vis)
    plt.title('(c) Bluebild', pad=14)
    plt.gca().coords[0].set_axislabel(" ")
    plt.gca().coords[0].set_ticklabel_visible(True)
    plt.gca().coords[1].set_axislabel(" ")
    plt.gca().coords[1].set_ticklabel_position('lr')
    if sky_file:
        plt.gca().coords[1].set_ticklabel_visible(False)
    
    cb_ax = fig.add_axes([0.94, 0.535, 0.012, 0.34])
    cbar = plt.colorbar(cax=cb_ax)
    cbar.ax.tick_params(size=0)
    cbar.set_label("Intensity [Jy/beam]", rotation=-90, va='bottom')

    ref_pix = int(wsc_size / 2)
    assert(ref_pix * 2 == wsc_size)
    
    if fh_sky_out:
        json_obj = json.dumps({'CASA':     casa_sky,
                               'WSClean':  wsc_sky,
                               'Bluebild': bb_sky,
                               'ref_px':   wcs.wcs.crpix[0],
                               'ref_py':   wcs.wcs.crpix[1]},  indent=4)
        fh_sky_out.write(json_obj)
        fh_sky_out.close()
        
        
    ### Diff plots
    
    plot_grid = True
    plot_circles = False
    locate_max = False

    plt.subplot(234, projection=wcs, slices=('x', 'y', 0, 0))
    my_subplot(plt, wcs, diff_casa_wsc, -abs_max, abs_max, plot_grid, plot_circles, npix, 'seismic',
               sky_file, None, locate_max, wsc_size, wsc_scale, grid_color='black')
    plt.title('(d) CASA minus WSClean', pad=14)
    plt.gca().coords[0].set_axislabel("Right ascension")
    plt.gca().coords[1].set_axislabel("Declination")
    plt.gca().set_autoscale_on(False)

    plt.gca().set_autoscale_on(False)
    plt.subplot(235, projection=wcs, slices=('x', 'y', 0, 0))
    my_subplot(plt, wcs, diff_bb_casa, -abs_max, abs_max, plot_grid, plot_circles, npix, 'seismic',
               sky_file, None, locate_max, wsc_size, wsc_scale, grid_color='black')
    plt.title('(e) Bluebild minus CASA', pad=14)
    plt.gca().coords[0].set_axislabel("Right ascension")
    plt.gca().coords[1].set_axislabel(" ")
    if sky_file:
        plt.gca().coords[1].set_ticklabel_visible(False)

    plt.gca().set_autoscale_on(False)
    plt.subplot(236, projection=wcs, slices=('x', 'y', 0, 0))
    my_subplot(plt, wcs, diff_bb_wsc, -abs_max, abs_max, plot_grid, plot_circles, npix, 'seismic',
               sky_file, None, locate_max, wsc_size, wsc_scale, grid_color='black')
    plt.title('(f) Bluebild minus WSClean', pad=14)
    plt.gca().coords[0].set_axislabel("Right ascension")
    plt.gca().coords[1].set_axislabel(" ")
    plt.gca().coords[1].set_ticklabel_position('lr')
    if sky_file:
        plt.gca().coords[1].set_ticklabel_visible(False)
    
    cb_ax = fig.add_axes([0.94, 0.115, 0.012, 0.34])
    cbar = plt.colorbar(cax=cb_ax)
    cbar.ax.tick_params(size=0)
    cbar.set_label("Intensity difference [Jy/beam]", rotation=-90, va='bottom')

    #for dpi in 1400, 400, 200, 100:
    for dpi in 200, 100:
        for fmt in '.png', '.pdf':
            plot_file = basename + '_dpi_'+ str(dpi) + fmt
            plt.savefig(plot_file, bbox_inches='tight', dpi=dpi)
            print("-I-", plot_file)

    casa_wsc_txt, casa_wsc_json = diff_stats_txt(casa_data, 'casa', wsc_data,  'wsc')
    bb_wsc_txt,   bb_wsc_json   = diff_stats_txt(bb_data,   'bb',   wsc_data,  'wsc')
    bb_casa_txt,  bb_casa_json  = diff_stats_txt(bb_data,   'bb',   casa_data, 'casa')

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
        f.write("\nCASA minus WSClean")
        f.write("-------------------------------------------------------------------\n")
        f.write(casa_wsc_txt)
        f.write(f"SSIM: {casa_wsc_json['casa-wsc']['ssim']:.3f}\n")
        f.write("\nBluebild minus WSClean")
        f.write("-------------------------------------------------------------------\n")
        f.write(bb_wsc_txt)
        f.write(f"SSIM: {bb_wsc_json['bb-wsc']['ssim']:.3f}\n")
        f.write("\nBluebild minus CASA")
        f.write("-------------------------------------------------------------------\n")
        f.write(bb_casa_txt)
        f.write(f"SSIM: {bb_casa_json['bb-casa']['ssim']:.3f}\n")
    print("-I-", file_misc)

    with open(file_json, 'w') as f:
        json.dump({'casa-wsc': casa_wsc_json,
                   'bb-wsc':   bb_wsc_json,
                   'bb_casa':  bb_casa_json}, f)
    print("-I-", file_json)

    #with open(file_json, 'r') as f:
    #    chk = json.load(f)
    #    print(chk)
    #    print(chk['casa-wsc']['casa']['min'])



def plot_wsc_casa_bb_OLD_but_modified(wsc_fits, wsc_log, casa_fits, casa_log, bb_grid, bb_data, bb_json,
                     outdir, outname):

    outname += f"_wsc_casa_bb"

    bb_info = get_bipp_info_from_json(bb_json)

    if not os.path.isfile(casa_log):
        raise Warning(f"{casa_log} not found. Abort plot.")
        return

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
    if args.flip_ud:
        hdulist[0].data = np.flipud(hdulist[0].data)
    hdulist.writeto(bb_fits, overwrite=True)
    hdulist.close()


    wsc_hdu  = fits.open(wsc_fits)[0]
    wcs  = WCS(wsc_hdu.header)
    print("wcs =", wcs)


    wsc_hdu  = fits.open(wsc_fits)[0]
    casa_hdu = fits.open(casa_fits)[0]
    bb_hdu   = fits.open(bb_fits)[0]
    for hdu in wsc_hdu, casa_hdu, bb_hdu:
        print(hdu.data.shape)
        if hdu.data.shape[0] == 1 and hdu.data.shape[1] == 1:
            hdu.data = hdu.data.reshape(hdu.data.shape[2:4])
        print(hdu.data.shape)

    wsc_wcs  = WCS(wsc_hdu.header)
    casa_wcs = WCS(casa_hdu.header)
    bb_wcs   = WCS(bb_hdu.header)

    wcs = wsc_wcs
    print("wcs =", wcs)

    wsc_data  = wsc_hdu.data
    casa_data = casa_hdu.data
    bb_data   = bb_hdu.data

    """
    for j in wsc_data.shape[1] / 2, :
        for i in range(0, wsc_data.shape[0] - 1, 100):
            sky0 = wcs.pixel_to_world(i,   j, 0, 0)[0]
            sky1 = wcs.pixel_to_world(i+1, j, 0, 0)[0]
            #print(sky0, sky1)
            print(f"{i} {j} {(sky0.ra.deg - sky1.ra.deg) * 3600:.6f} [asec]")
    """
    

    #extent = (1024, 0, 1024, 0)
    #extent = (0, 1024, 0, 1024)

    plt.subplot(111, projection=wcs, slices=('x', 'y', 0, 0))

    #plt.imshow(wsc_data, extent=extent, vmin=-0.1, vmax=1.1,
    plt.imshow(wsc_data, vmin=-0.1, vmax=1.1,
               interpolation ="nearest")#, origin ="upper")
    #plt.gca().invert_xaxis()

    plt.gca().coords[0].set_ticks_position('b')
    plt.gca().coords[1].set_ticks_position('l')
    plt.gca().secondary_yaxis('right')
    plt.gca().secondary_xaxis('top')

    il = 0
    with open(args.sky_file) as sky_file:
        for src_line in sky_file:
            if il%3 != 0:
                il += 1
                continue
            id, ra, dec, px, py, intensity = src_line.strip().split(" ")
            id, ra, dec, px, py, intensity = int(id), float(ra), float(dec), int(px), int(py), float(intensity)
            #print(id, ra, dec, px, py, intensity)
            sky = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
            pix = wcs.wcs_world2pix(sky.ra, sky.dec, 100000000.0, 1.0, 0)
            print(f"{id:2d} ra = {sky.ra.deg:.4f} {sky.ra.to_string(u.hour)} {dec:.4f} {intensity:.1f} ==>  {pix[0]:.2f} {pix[1]:.2f}")

            S = 100
            axin = plt.gca().inset_axes([int(pix[0]) - 1.3*S, int(pix[1]) - S/2, S, S],
                                        transform = plt.gca().transData)

            #axin.imshow(wsc_data, extent=extent,vmin=-0.1, vmax=1.1,
            axin.imshow(wsc_data, vmin=-0.1, vmax=1.1,
                        interpolation ="nearest", origin ="lower")

            # Box to highlight zoomed in region
            x1, x2, y1, y2 = int(pix[0]) -8, int(pix[0]) +9, int(pix[1]) -8, int(pix[1]) + 9
            axin.set_xlim(x1, x2)
            axin.set_ylim(y1, y2)
            axin.set_xticks([])
            axin.set_yticks([])
            plt.gca().indicate_inset_zoom(axin, edgecolor="black")

            # Mark true position of source
            axin.plot(pix[0], pix[1], '+', linewidth=0.1, markersize=.5, color='red')
            
            il += 1

    plt.savefig('abc123', bbox_inches='tight', dpi=1400)
    sys.exit(0)


    diff_casa_wsc = casa_data - wsc_data
    diff_bb_wsc   = bb_data   - wsc_data
    diff_bb_casa  = bb_data   - casa_data

    # color bar min/max
    casa_min, casa_max = np.amin(casa_data), np.amax(casa_data)
    wsc_min,  wsc_max  = np.amin(wsc_data),  np.amax(wsc_data)
    bb_min,   bb_max   = np.amin(bb_data),   np.amax(bb_data)
    print(f"-I- CASA     = [{casa_min:.6e}, {casa_max:.6e}]")
    print(f"-I- WSClean  = [{wsc_min:.6e}, {wsc_max:.6e}]")
    print(f"-I- Bluebild = [{bb_min:.6e}, {bb_max:.6e}]")

    zmin = min(wsc_min, casa_min, bb_min)
    zmax = max(wsc_max, casa_max, bb_max)
    print(f"-D- range = [{zmin:.6e}, {zmax:.6e}]")
    max_absz = max(abs(zmin), abs(zmax))
    dmin = min(np.amin(diff_casa_wsc), np.amin(diff_bb_wsc), np.amin(diff_bb_casa))
    dmax = max(np.amax(diff_casa_wsc), np.amax(diff_bb_wsc), np.amax(diff_bb_casa))
    print(f"-D- diff  = [{dmin:.6e}, {dmax:.6e}]")
    abs_max = max(abs(dmin), abs(dmax))
    print(f"-D- max abs diff = {abs_max:.6e}")

    #sys.exit(0)

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

    vmin, vmax = -max_absz, max_absz
    vmin, vmax = zmin, zmax

    plot_grid = False
    plot_circles = False

    my_viridis = plt.cm.viridis
    my_viridis.set_bad((1, 0, 0, 1))

    """
    for data in wsc_data, casa_data, bb_data:
        for x in 0, 256, 511:
            data[x, x] = np.nan
    """
    

    plt.subplot(231, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(wsc_data, vmin=vmin, vmax=vmax, cmap=my_viridis)
    if plot_grid: plt.grid(color='white', ls='solid')
    plt.title('(a) WSClean', pad=14)
    #plt.gca().coords[0].set_axislabel(" ")
    #plt.gca().coords[0].set_ticklabel_visible(True)
    #plt.gca().coords[1].set_axislabel("Declination")
    #plt.gca().set_frame_on(False)
    #plt.tick_params(axis='x', which='both', bottom=False, top=False,  labelbottom=False)
    #plt.tick_params(axis='y', which='both', right=False,  left=False, labelleft=False)
    #for pos in ['right', 'top', 'bottom', 'left']:
    #    plt.gca().spines[pos].set_visible(False)
    #plt.axis('off')

    ax = plt.gca()
    axins = ax.inset_axes([80, -40, 0.33, 0.33])#, transform = ax.transData)
    axins.imshow(wsc_data)
    axins.set_xlim(0.5, 1)
    axins.set_ylim(0.5, 1)
    ax.indicate_inset_zoom(axins, edgecolor="pink")
    plt.savefig('abc123', bbox_inches='tight')
    sys.exit(0)

    print("=======================================================")
    print(wcs.pixel_to_world(0, 0, 1, 1))
    print("=======================================================")
    """
    ax = plt.gca()
    # def add_point_sources()
    with open(args.sky_file) as sky_file:
        for src_line in sky_file:
            id, ra, dec, px, py, intensity = src_line.strip().split(" ")
            ra, dec, px, py = float(ra), float(dec), int(px), int(py)
            print(id, ra, dec, px, py, intensity)
            axins = ax.inset_axes([ra, dec, 0.5, 0.5])#, transform = ax.transData)
            axins.imshow(wsc_data)
            W = 10
            x1, x2, y1, y2 = px - W, px + W, py - W, py + W
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            ax.indicate_inset_zoom(axins, edgecolor="pink")
    """
    """
    # Create an inset axis in the bottom right corner
    axins = plt.gca().inset_axes([0.6, 0.6, 0.1, 0.1])
    axins.imshow(wsc_data)

    # subregion of the original image
    x1, x2, y1, y2 = 250, 262, 250, 262
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    plt.gca().indicate_inset_zoom(axins, edgecolor="black")
    """

    plt.subplot(232, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(casa_data, vmin=vmin, vmax=vmax, cmap=my_viridis)
    if plot_grid: plt.grid(color='white', ls='solid')
    plt.title('(b) CASA', pad=14)
    plt.gca().coords[0].set_axislabel(" ")
    plt.gca().coords[0].set_ticklabel_visible(True)
    plt.gca().coords[1].set_axislabel(" ")
    plt.gca().coords[1].set_ticklabel_visible(True)

    
    plt.subplot(233, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(bb_data, vmin=vmin, vmax=vmax, cmap=my_viridis)
    if plot_grid: plt.grid(color='white', ls='solid')
    plt.title('(c) Bluebild', pad=14)
    plt.gca().coords[0].set_axislabel(" ")
    plt.gca().coords[0].set_ticklabel_visible(True)
    plt.gca().coords[1].set_axislabel(" ")
    plt.gca().coords[1].set_ticklabel_visible(True)

    cb_ax = fig.add_axes([0.92, 0.535, 0.012, 0.34])
    cbar = plt.colorbar(cax=cb_ax)
    cbar.ax.tick_params(size=0)

    plt.subplot(234, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(diff_casa_wsc, vmin=-abs_max, vmax=abs_max, cmap='seismic')
    if plot_grid: plt.grid(color='gray', ls='solid')
    plt.title('(d) CASA minus WSClean', pad=14)
    plt.gca().coords[0].set_axislabel("Right ascension")
    plt.gca().coords[1].set_axislabel("Declination")
    plt.gca().set_autoscale_on(False)
    if plot_circles:
        r1 = Circle((half_width_pix, half_width_pix), rad1_pix, edgecolor='darkorange', facecolor='none')
        plt.gca().add_patch(r1)
        r2 = Circle((half_width_pix, half_width_pix), rad2_pix, edgecolor='fuchsia', facecolor='none')
        plt.gca().add_patch(r2)
        r3 = Circle((half_width_pix, half_width_pix), rad3_pix, edgecolor='black', facecolor='none')
        plt.gca().add_patch(r3)

    plt.gca().set_autoscale_on(False)
    plt.subplot(235, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(diff_bb_casa, vmin=-abs_max, vmax=abs_max, cmap='seismic')
    if plot_grid: plt.grid(color='gray', ls='solid')
    plt.title('(e) Bluebild minus CASA', pad=14)
    plt.gca().coords[0].set_axislabel("Right ascension")
    plt.gca().coords[1].set_axislabel(" ")
    plt.gca().coords[1].set_ticklabel_visible(True)
    if plot_circles:
        r1 = Circle((half_width_pix, half_width_pix), rad1_pix, edgecolor='darkorange', facecolor='none')
        plt.gca().add_patch(r1)
        r2 = Circle((half_width_pix, half_width_pix), rad2_pix, edgecolor='fuchsia', facecolor='none')
        plt.gca().add_patch(r2)
        r3 = Circle((half_width_pix, half_width_pix), rad3_pix, edgecolor='black', facecolor='none')
        plt.gca().add_patch(r3)

    plt.gca().set_autoscale_on(False)
    plt.subplot(236, projection=wcs, slices=('x', 'y', 0, 0))
    plt.imshow(diff_bb_wsc, vmin=-abs_max, vmax=abs_max, cmap='seismic')
    if plot_grid: plt.grid(color='gray', ls='solid')
    plt.title('(f) Bluebild minus WSClean', pad=14)
    plt.gca().coords[0].set_axislabel("Right ascension")
    plt.gca().coords[1].set_axislabel(" ")
    plt.gca().coords[1].set_ticklabel_visible(True)
    if plot_circles:
        r1 = Circle((half_width_pix, half_width_pix), rad1_pix, edgecolor='darkorange', facecolor='none')
        plt.gca().add_patch(r1)
        r2 = Circle((half_width_pix, half_width_pix), rad2_pix, edgecolor='fuchsia', facecolor='none')
        plt.gca().add_patch(r2)
        r3 = Circle((half_width_pix, half_width_pix), rad3_pix, edgecolor='black', facecolor='none')
        plt.gca().add_patch(r3)

    cb_ax = fig.add_axes([0.92, 0.115, 0.012, 0.34])
    cbar = plt.colorbar(cax=cb_ax)
    cbar.ax.tick_params(size=0)

    # Write .png and associated .misc file
    basename = os.path.join(outdir, outname)
    file_png  = basename + '.png'
    file_misc = basename + '.misc'
    file_json = basename + '.json'

    plt.savefig(file_png, bbox_inches='tight')
    print("-I-", file_png)

    casa_wsc_txt, casa_wsc_json = diff_stats_txt(casa_data, 'casa', wsc_data,  'wsc')
    bb_wsc_txt,   bb_wsc_json   = diff_stats_txt(bb_data,   'bb',   wsc_data,  'wsc')
    bb_casa_txt,  bb_casa_json  = diff_stats_txt(bb_data,   'bb',   casa_data, 'casa')

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
        f.write("\nCASA minus WSClean")
        f.write("-------------------------------------------------------------------\n")
        f.write(casa_wsc_txt)
        f.write(f"SSIM: {casa_wsc_json['casa-wsc']['ssim']:.3f}\n")
        f.write("\nBluebild minus WSClean")
        f.write("-------------------------------------------------------------------\n")
        f.write(bb_wsc_txt)
        f.write(f"SSIM: {bb_wsc_json['bb-wsc']['ssim']:.3f}\n")
        f.write("\nBluebild minus CASA")
        f.write("-------------------------------------------------------------------\n")
        f.write(bb_casa_txt)
        f.write(f"SSIM: {bb_casa_json['bb-casa']['ssim']:.3f}\n")
    print("-I-", file_misc)

    with open(file_json, 'w') as f:
        json.dump({'casa-wsc': casa_wsc_json,
                   'bb-wsc':   bb_wsc_json,
                   'bb_casa':  bb_casa_json}, f)
    print("-I-", file_json)

    #with open(file_json, 'r') as f:
    #    chk = json.load(f)
    #    print(chk)
    #    print(chk['casa-wsc']['casa']['min'])


def diff_stats_txt(data1, name1, data2, name2):

    diff_data = data1 - data2

    # Compute structural similarity index (SSIM)
    g_min = min(np.amin(data1), np.amin(data2))
    g_max = max(np.amax(data1), np.amax(data2))
    g_ssim  = ssim(data2, data1, data_range= g_max - g_min)
    print(f"-I- SSIM {name1} - {name2}  = {g_ssim:.3f}")

    """
    bw_min = min(np.amin(wsc_data), np.amin(bb_data))
    bw_max = max(np.amax(wsc_data), np.amax(bb_data))
    ssim_bb_wsc  = ssim(wsc_data, bb_data, data_range= bw_max - bw_min)
    print(f"-I- ssim_bb_wsc  = {ssim_bb_wsc:.3f}")
    bc_min = min(np.amin(casa_data), np.amin(bb_data))
    bc_max = max(np.amax(casa_data), np.amax(bb_data))
    ssim_bb_casa = ssim(casa_data, bb_data, data_range= bc_max - bc_min)
    print(f"-I- ssim_bb_casa = {ssim_bb_casa:.3f}")
    wc_min = min(np.amin(casa_data), np.amin(wsc_data))
    wc_max = max(np.amax(casa_data), np.amax(wsc_data))
    ssim_wsc_casa = ssim(casa_data, wsc_data, data_range= wc_max - wc_min)
    print(f"-I- ssim_wsc_casa = {ssim_wsc_casa:.3f}")
    """


    diff_min   = np.amin(diff_data)
    diff_max   = np.amax(diff_data)
    diff_range = diff_max - diff_min
    rmse    = compute_rmse(data2, data1)
    rmse50  = compute_rmse_sig(data2, data1, 0.50)
    rmse75  = compute_rmse_sig(data2, data1, 0.75)
    rmse100 = compute_rmse_sig(data2, data1, 1.00)
    rmsechk = compute_rmse_sig(data2, data1, 1.415)
    txt  = f"diff min       = {diff_min:.3e}\n"
    txt += f"diff max       = {diff_max:.3e}\n"
    txt += f"diff rmse      = {rmse:.3e}\n"
    txt += f"diff rmse 50%  = {rmse50:.3e}\n"
    txt += f"diff rmse 75%  = {rmse75:.3e}\n"
    txt += f"diff rmse 100% = {rmse100:.3e}\n"
    txt += f"diff rmse chk  = {rmsechk:.3e}\n"

    sol_min = np.amin(data1)
    sol_max = np.amax(data1)
    sol_range = sol_max - sol_min
    txt += f"solution min          = {sol_min:.3e}\n"
    txt += f"solution max          = {sol_max:.3e}\n"
    txt += f"solution range        = {sol_range:.3e}\n"

    ref_min = np.amin(data2)
    ref_max = np.amax(data2)
    ref_range = ref_max - ref_min
    txt += f"reference min         = {ref_min:.3e}\n"
    txt += f"reference max         = {ref_max:.3e}\n"
    txt += f"reference range       = {ref_range:.3e}\n"
    
    
    txt += f"diff scaled rmse      = {rmse    / ref_range:.3e}\n"
    txt += f"diff scaled rmse 50%  = {rmse50  / ref_range:.3e}\n"
    txt += f"diff scaled rmse 75%  = {rmse75  / ref_range:.3e}\n"
    txt += f"diff scaled rmse 100% = {rmse100 / ref_range:.3e}\n"
    txt += f"diff scaled rmse chk  = {rmsechk / ref_range:.3e}\n"

    name_diff = f"{name1}-{name2}"
    data = {
        name1 : {
            'min': sol_min.item(), 'max': sol_max.item(), 'range': sol_range.item()
        },
        name2 : {
            'min': ref_min.item(), 'max': ref_max.item(), 'range': ref_range.item()
        },
        name_diff : {
            'min': diff_min.item(), 'max': diff_max.item(), 'range': diff_range.item(),
            'rmse':    rmse.item(),
            'rmse50':  rmse50.item(),
            'rmse75':  rmse75.item(),
            'rmse100': rmse100.item(),
            'rmsechk': rmsechk.item(),
            'scaled_rmse':    rmse.item()    / ref_range.item(),
            'scaled_rmse50':  rmse50.item()  / ref_range.item(),
            'scaled_rmse75':  rmse75.item()  / ref_range.item(),
            'scaled_rmse100': rmse100.item() / ref_range.item(),
            'scaled_rmsechk': rmsechk.item() / ref_range.item(),
            'ssim': g_ssim.item()
        }
    }

    return txt, data


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
        f.write(f"diff rms = {compute_rmse(data2, data1):.3f}\n")
        f.write(f"diff rms 1s = {compute_rmse_sig(data2, data1, 0.680):.3f}\n")
        f.write(f"diff rms 2s = {compute_rmse_sig(data2, data1, 0.950):.3f}\n")
        f.write(f"diff rms 3s = {compute_rmse_sig(data2, data1, 0.997):.3f}\n")
        f.write(f"diff rms check = {compute_rmse_sig(data2, data1, 1.415):.3f}\n")
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
        f.write(f"diff rms = {compute_rmse(bipp_data, ref_data):.3f}\n")
        f.write(f"diff rms 1s = {compute_rmse_sig(bipp_data, ref_data, 0.680):.3f}\n")
        f.write(f"diff rms 2s = {compute_rmse_sig(bipp_data, ref_data, 0.950):.3f}\n")
        f.write(f"diff rms 3s = {compute_rmse_sig(bipp_data, ref_data, 0.997):.3f}\n")
        f.write(f"diff rms check = {compute_rmse_sig(bipp_data, ref_data, 1.415):.3f}\n")
        f.write("-------------------------------------------------------------------\n")

    plt.tight_layout()

    plt.savefig(basename + '.png')
    print("-I-", file_png)
    print("-I-", file_misc)



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
    print("-D-", outdir, filename)
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

def plot_2d_matrix(X, filename, outdir, title, xlabel, ylabel):

    fig, ax = plt.subplots(1, 1, figsize=(12,12))
    fig.suptitle(title, fontsize=22)

    im = ax.matshow(X, cmap = 'seismic')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

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
    #print(uv_map, uv_map.dtype, uv_map.shape)

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
    parser.add_argument('--outname',   help='Plots naming prefix',     required=True)
    parser.add_argument('--outdir',    help='Plots output directory',  required=True)
    parser.add_argument('--wsc_size',  help='WSClean size paramater',  required=True, type=int)
    parser.add_argument('--wsc_scale', help='WSClean scale parameter', required=True, type=float)
    parser.add_argument('--flip_lr',   help='Flip image left-rigth', action='store_true')
    parser.add_argument('--flip_ud',   help='Flip image up-down',    action='store_true')
    parser.add_argument('--sky_file',  help='Simulated point sources RA and DEC')
    args = parser.parse_args()
    print(args)

    # Handle non-mandatory sky file (typically OSKAR simulated point sources)
    sky_file = args.sky_file if args.sky_file else None

    do_bb   = False
    do_wsc  = False
    do_casa = False
    if args.bb_grid   and args.bb_data and args.bb_json: do_bb   = True
    if args.wsc_fits  and args.wsc_log:                  do_wsc  = True 
    if args.casa_fits and args.casa_log:                 do_casa = True
    print("-I- consider Bluebild?", do_bb)
    print("-I- consider WSClean? ", do_wsc)
    print("-I- consider CASA?    ", do_casa)
    
    if do_casa and not os.path.isfile(args.casa_log):
        print("-W- Missing CASA log, so will not consider CASA.")
        do_casa = False

    print("-I- consider Bluebild?", do_bb)
    print("-I- consider WSClean? ", do_wsc)
    print("-I- consider CASA?    ", do_casa)
        

    if do_bb and do_wsc and do_casa:
        plot_wsc_casa_bb(args.wsc_fits, args.wsc_log,
                         args.casa_fits, args.casa_log,
                         args.bb_grid, args.bb_data, args.bb_json,
                         args.outdir, args.outname, sky_file,
                         args.wsc_size, args.wsc_scale)

    print("-W- Ignore 1 to 1 plots")
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

        
