import os
import sys
import numpy as np
import oskar
#print(f"-I- OSKAR version = {oskar.__version__}")
import argparse
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta
import astropy.units as u
sys.path.append(os.path.dirname(__file__))
from gleam_survey import create_gleam_sky_model
from astropy.io import fits
import logging


# Parse command line argument
parser = argparse.ArgumentParser(description='Running OSKAR simulation')
parser.add_argument('--telescope_lon', help="Longitude of telescope", required=True, type=float)
parser.add_argument('--telescope_lat', help="Latitude of telescope", required=True, type=float)
parser.add_argument('--fov_deg', help="Field of view [degree]", required=False, type=float)
parser.add_argument('--num_time_steps', help="Number of time steps", required=True, type=int)
parser.add_argument('--input_directory', help=".tm input directory", required=True)
parser.add_argument('--phase_centre_ra_deg',  help="Phase centre's right ascension [deg]", required=True, type=float)
parser.add_argument('--phase_centre_dec_deg', help="Phase centre's declination [deg] ",    required=True, type=float)
parser.add_argument('--out_name', help="Output name", required=True)
parser.add_argument('--precision', help="OSKAR precision", required=True, choices=['single', 'double'])
parser.add_argument('--start_frequency_hz', help="Start frequency in MHz", required=True, type=float)
parser.add_argument('--num_channels', help="Number of channels", required=False, type=int, default=1)
parser.add_argument('--frequency_inc_hz', help="Frequency inc in Hz", required=False, type=int, default=20e6)
parser.add_argument('--use_gpus', action='store_true')
parser.add_argument('--length', help="Observation length in format hh:mm:ss.sss", type=int)

args = parser.parse_args()
print("oskar_gleam_simulation_kuma.py args =", args)

telescope = EarthLocation(lon    = args.telescope_lon * u.deg,
                          lat    = args.telescope_lat * u.deg,
                          height = 0)
print("-I- telescope =\n", telescope)
target = SkyCoord(ra = args.phase_centre_ra_deg * u.deg,
                  dec= args.phase_centre_dec_deg * u.deg)
print("-I- target =\n", target)
t_guess = Time('2025-11-10 00:00:00', scale='utc')
lst_guess = t_guess.sidereal_time('apparent', longitude=telescope.lon)
ha = lst_guess - target.ra
t_transit = t_guess - (ha / (15*u.deg/u.hour))  # convert degrees to hours
print("Transit time (UTC):", t_transit.utc.isot, "(exact)")

# Round to closest 30 minutes
sec = t_transit.unix
half_hour = 30 * 60
sec_rounded = np.round(sec / half_hour) * half_hour
t_rounded = Time(sec_rounded, format='unix', scale='utc')
print("Transit time (UTC):", t_rounded.utc.isot, "(rounded to closest half-hour)")

td_half_length = TimeDelta(args.length / 2, format='sec')
print("td_half_length =", td_half_length)

t_start = t_rounded - td_half_length
print(f"Start time (UTC): {t_start.utc.isot} for a duration of {args.length} sec")

#sys.exit(0)


handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
LOG = logging.getLogger()
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)
    
sky0 = oskar.Sky(args.precision)
hdulist = fits.open('/work/seams/gleam/GLEAM_EGC_v2.fits')
cols = hdulist[1].data[0].array
data = np.column_stack(
    (cols['RAJ2000'], cols['DEJ2000'], cols['peak_flux_wide']))
data = data[data[:, 2].argsort()[::-1]]
print("data =\n", data, data.shape)
sky_gleam = oskar.Sky.from_array(data, args.precision)
sky0.append(sky_gleam)

sky0.filter_by_radius(0.0, np.sqrt(2) * args.fov_deg / 2, args.phase_centre_ra_deg, args.phase_centre_dec_deg)
sky0.filter_by_flux(1, 5)
LOG.info("Number of sources in inner sky model: %d", sky0.num_sources)
#print("sky0 =", sky0)
#sys.exit(0)



# Basic settings. (Note that the sky model is set up later.)
#        "correlation_type": 'both'
params = {
    "simulator": {
        "use_gpus": False,
        "keep_log_file": True
    },
    "observation" : {
        "num_channels": args.num_channels,
        "start_frequency_hz": args.start_frequency_hz,
        "frequency_inc_hz": args.frequency_inc_hz,
        "phase_centre_ra_deg": args.phase_centre_ra_deg,
        "phase_centre_dec_deg": args.phase_centre_dec_deg,
        "num_time_steps": args.num_time_steps,
        "start_time_utc": t_transit.utc.isot,
        "length": args.length
    },
    "telescope": {
        "input_directory": args.input_directory,
        "station_type": 'Isotropic'
    },
    "interferometer": {
        #"oskar_vis_filename": args.out_name + ".vis",
        "ms_filename": args.out_name + ".ms",
        "channel_bandwidth_hz": 1e6,
        "time_average_sec": 10,
    }
}

if args.use_gpus:
    params["simulator"]["use_gpus"] = True
print("params ?? =\n", params, flush=True)

# Overwrite defaults with params above
settings = oskar.SettingsTree("oskar_sim_interferometer")
settings.from_dict(params)

# Set the numerical precision to use.
if args.precision == 'single':
    settings["simulator/double_precision"] = False




    
if 1 == 0:
    ## From Ruben https://github.com/SEAMS-Project/fpga-finufft/blob/master/benchmark/generation/gen_skalow_gleam.py

    ## Note: the radius of the conesearch should be half of the field of view, but twice
    ##       is taken to cover the edges of the square image.
    sky_data = create_gleam_sky_model((params["observation"]["phase_centre_ra_deg"],
                                       params["observation"]["phase_centre_dec_deg"]),
                                      args.fov_deg / 2,
                                      params["observation"]["start_frequency_hz"])
    #print("-I- sky_data =\n", sky_data, sky_data.shape)
    print("-I- sky_data =\n", sky_data[:,0:3], sky_data.shape)

    # Set the sky model and run the simulation.
    sky_data = np.array(sky_data[:,0:3])
    sky = oskar.Sky.from_array(sky_data, args.precision)  # Pass precision here.

sim = oskar.Interferometer(settings=settings)
#sim.set_sky_model(sky)
sim.set_sky_model(sky0)

print("-I- Launching simulation!", flush=True)

sim.run()

print("-I- Simulation complete", flush=True)
