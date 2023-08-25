import os
import sys
import matplotlib
matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy
import oskar
import argparse
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta
import astropy.units as u

# Parse command line argument
parser = argparse.ArgumentParser(description='Running OSKAR simulation')
parser.add_argument('--wsc_size',   help='WSC size [pixel]', required=True, type=int)
#parser.add_argument('--wsc_scale', help="WSC scale [arcsec]", required=True, type=float)
parser.add_argument('--fov_deg', help="Field of view [degree]", required=True, type=float)
parser.add_argument('--num_time_steps', help="Number of time steps", required=True, type=int)
parser.add_argument('--input_directory', help=".tm input directory", required=True)
parser.add_argument('--telescope_lon', help="Longitude of telescope", required=True, type=float)
parser.add_argument('--telescope_lat', help="Latitude of telescope", required=True, type=float)

args = parser.parse_args()


# Basic settings. (Note that the sky model is set up later.)
params = {
    "simulator": {
        "use_gpus": False,
        "keep_log_file": True
    },
    "observation" : {
        "num_channels": 1,
        "start_frequency_hz": 100e6,
        "frequency_inc_hz": 20e6,
        "phase_centre_ra_deg": 80,
        "phase_centre_dec_deg": -40,
        "num_time_steps": args.num_time_steps,
        "start_time_utc": "2000-01-01T12:00:00.000",
        "length": "06:00:00.000"
    },
    "telescope": {
        "input_directory": args.input_directory
    },
    "interferometer": {
        "oskar_vis_filename": "oskar_bipp_paper.vis",
        "ms_filename": "oskar_bipp_paper.ms",
        "channel_bandwidth_hz": 1e6,
        "time_average_sec": 10,
    }
}

# Overwrite defaults with params above
settings = oskar.SettingsTree("oskar_sim_interferometer")
settings.from_dict(params)


# Set the numerical precision to use.
precision = "single"
if precision == "single":
    settings["simulator/double_precision"] = False

RA  = params["observation"]["phase_centre_ra_deg"]
DEC = params["observation"]["phase_centre_dec_deg"]

"""
UTC_START = Time(params["observation"]["start_time_utc"])
print("RA =", RA, ", DEC =", DEC, "UTC_START =", UTC_START)

td1h = TimeDelta(1.0 * u.hour)
td = td1h * numpy.linspace(0, 3, 25)
OBS = UTC_START + td
telesc = EarthLocation(lon    = args.telescope_lon * u.deg,
                       lat    = args.telescope_lat * u.deg,
                       height = 0)
for t_obs in OBS:
    target = SkyCoord(frame="icrs", unit="deg", ra=RA, dec=DEC, obstime=t_obs)
    target_altaz = target.transform_to(AltAz(location=telesc))
    print(f"target at {t_obs} El = {target_altaz.alt:.2f}, Az = {target_altaz.az:.2f}")

#sys.exit(1)
"""

size = args.wsc_size
assert(size % 8 == 0)


# Set up WCS projection
# ---------------------
from astropy import wcs
w = wcs.WCS(naxis=2)
w.wcs.crpix = [float(size / 2 + 1.0), float(size / 2 + 1.0)]
w.wcs.cdelt = numpy.array([-0.00111111111111111, 0.00111111111111111])
w.wcs.crval = [80.0, -40.0]
w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
print(w)

fn = params["interferometer"]["ms_filename"]
fn = os.path.splitext(fn)[0] + '.sky'
f = open(fn, "w")
i = 0
sky_data = []

eighths = (1, 4, 7)
for i in eighths:
    assert(((size / 8) * i) % 2 == 0)
    for j in eighths:
        x = i * size / 8 + 0.5
        y = j * size / 8 + 0.5
        sky = w.wcs_pix2world([[x, y]], 1)[0]
        ra, dec = sky[0], sky[1]
        intensity = 1.0 + i/10
        print(f"{i} {x} {y} {ra:.8f} {dec:.8f}")
        sky_data.append([ra, dec, intensity])
        f.write(f"{i} {ra:.8f} {dec:.8f} {x} {y} {intensity:.3f}\n")
f.close

#world = w.wcs_pix2world([[512, 512]], 0)
#print(world)

"""
# Create a sky model containing a grid of point sources separated by INC
# and centered on pix 1,1 from FoV center
SCALE_DEG = args.fov_deg / args.wsc_size
print(f"-O- scale = {SCALE_DEG:.6f} [deg], {SCALE_DEG * 3600:.6f} [arcsec]")
HALF_SCALE_DEG = SCALE_DEG / 2
print("HALF_SCALE_DEG =", HALF_SCALE_DEG,  HALF_SCALE_DEG * 3600)
inc_deg = 0.5
assert(args.wsc_size % 2 == 0)
i_range = int((args.fov_deg / 2) / inc_deg)
print(args.fov_deg, i_range)
INTENSITY = 1.0
sky_data = []
fn = params["interferometer"]["ms_filename"]
fn = os.path.splitext(fn)[0] + '.sky'
f = open(fn, "w")
i = 0
for ra in numpy.arange(-i_range * inc_deg, (i_range + 1) * inc_deg, inc_deg):
    for dec in numpy.arange(-i_range * inc_deg, (i_range + 1) * inc_deg, inc_deg):
        intensity = 1 + i/10
        print(ra, dec, intensity)
        ra_  = RA  + ra  + HALF_SCALE_DEG * 1
        dec_ = DEC + dec + HALF_SCALE_DEG * 1
        sky_data.append([ra_, dec_, intensity])
        pix_i = int((ra_  - RA)  / SCALE_DEG + numpy.sign(ra_ - RA)   * 1)
        pix_j = int((dec_ - DEC) / SCALE_DEG + numpy.sign(dec_ - DEC) * 1)
        print(f"-O- Added source at ra, dec = {ra_:.8f}, {dec_:.8f}. Pixel {pix_i}, {pix_j}")

        pix_i = int(args.wsc_size / 2 + (ra_  - RA)  / SCALE_DEG + numpy.sign(ra_ - RA)   * 0)
        pix_j = int(args.wsc_size / 2 + (dec_ - DEC) / SCALE_DEG + numpy.sign(dec_ - DEC) * 0)
        print(f"-O- Added source at ra, dec = {ra_:.8f}, {dec_:.8f}. Pixel {pix_i}, {pix_j}")
        f.write(f"{i} {ra_:.8f} {dec_:.8f} {pix_i} {pix_j} {intensity:.1f}\n")
        i += 1
f.close
"""

sky_data = numpy.array(sky_data)
sky = oskar.Sky.from_array(sky_data, precision)  # Pass precision here.

# Set the sky model and run the simulation.
sim = oskar.Interferometer(settings=settings)
sim.set_sky_model(sky)
sim.run()


"""
# Make an image
imager = oskar.Imager(precision)
imager.set(fov_deg=args.fov_deg, image_size=args.wsc_size)
imager.set(input_file="oskar_bipp_paper.vis", output_root="oskar_bipp_paper")
output = imager.run(return_images=1)
image = output["images"][0]

# Render the image using matplotlib and save it as a PNG file.
im = plt.imshow(image, cmap="jet")
plt.gca().invert_yaxis()
plt.colorbar(im)
plt.savefig("%s.png" % imager.output_root)
plt.close("all")
"""
