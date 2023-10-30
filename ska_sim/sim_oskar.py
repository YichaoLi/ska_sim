import numpy as np
import h5py as h5
from ps_est import algebra as al
import oskar
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
from astropy.coordinates import get_sun, get_moon, FK5
from astropy.coordinates import EarthLocation
import os
import time

ra0 = 0
dec0 = -30
fov = 8
image_size = 1024

freq_idx = int(os.getenv('FREQ_IDX'))

precision = "single"

# load the telescope position
tele_model = "/home/ycli/data/ska_challenge/2023/SDC3/V1/telescope.tm"

# mock id
mock = 0

# load the sky model, HI Point Sources and Diffuse FG, target at RA 0, Dec -30
sky_model = 'imap%02d_HI_PS_FG'%mock
sky_input = '/home/DATA/ycli/ska_challenge/2023/sim_900_combined/%s.h5'%sky_model

# estimate the transit time of the target field
_Lon = 116.764 * u.deg
_Lat = -26.825 * u.deg
ska_site = EarthLocation.from_geodetic(_Lon, _Lat)
field = SkyCoord(ra = ra0 * u.deg, dec = dec0 * u.deg)
obs_time = Time('2024-1-1 00:00:00.00') + np.arange(0, 24*60, 10) * u.minute
ska_site_frame = AltAz(obstime=obs_time, location=ska_site)
field_altaz = field.transform_to(ska_site_frame)

x = obs_time.unix
#plt.plot(x - x[0], field_altaz.alt.deg)
transit_time = obs_time[np.argmax(field_altaz.alt.deg)]
print(transit_time.fits)


# load sky map and frequency information
with h5.File(sky_input, 'r') as fp:
    imap = al.load_h5(fp, 'imap')
    imap = al.make_vect(imap, axis_names=imap.info['axes'])

ra    = imap.get_axis('ra') + ra0
dec   = imap.get_axis('dec') + dec0
freq  = imap.get_axis('freq')
dfreq = imap.info['freq_delta']

# setup the output path
output_path = '/home/ycli/data/ska_challenge/oskar_sim/MS/%s/Freq%5.2fMHz/'%(
        sky_model, freq[freq_idx])
if not os.path.exists(output_path):
    os.makedirs(output_path)

# convert K to Jy
omega_B = 2. * np.pi * (imap.info['ra_delta'] * u.deg)\
                     * (imap.info['dec_delta'] * u.deg)
Jy2K = ( 1 * u.Jy/omega_B ).to(u.K, 
       equivalencies=u.brightness_temperature(freq * u.MHz)).value
imap /= Jy2K[:, None, None]

# start OSKAR simulation
DEC, RA = np.meshgrid(dec, ra)
#print(RA.shape, DEC.shape)
data = np.column_stack([RA.flat, DEC.flat, imap[freq_idx].flat])
data = data.astype('float32')
#print(data.shape)

sky = oskar.Sky.from_array(data, precision)
sky.filter_by_radius(0, 8, ra0, dec0)

num_time_steps_per_min = 2
delta_time = 30 # minute
start_time = transit_time - 2 * u.hour + np.arange(0, 240, delta_time) * u.minute
#start_time = transit_time + np.arange(0, 120, 10) * u.minute

for st in start_time:
    t0 = time.time()
    output_name = "%s_%s"%(sky_model, st.fits)
    params = {
        "simulator": {
            "use_gpus": True
        },
        "observation" : {
            "num_channels": 1,
            "start_frequency_hz": freq[freq_idx] * 1.e6,
            "frequency_inc_hz": dfreq * 1.e6,
            "phase_centre_ra_deg":  ra0,
            "phase_centre_dec_deg": dec0,
            "num_time_steps": num_time_steps_per_min * delta_time,
            "start_time_utc": st.fits,
            "length": "00:%d:00.000"%delta_time
        },
        "telescope": {
            "input_directory": tele_model,
        },
        "interferometer": {
            "oskar_vis_filename": '', #output_path + output_name + ".vis",
            "ms_filename": output_path + output_name + ".ms",
            "channel_bandwidth_hz": dfreq * 1.e6,
            "time_average_sec": 10
        }
    }

    settings = oskar.SettingsTree("oskar_sim_interferometer")
    settings.from_dict(params)
    
    if precision == "single":
        settings["simulator/double_precision"] = False
        
    sim = oskar.Interferometer(settings=settings)
    sim.set_sky_model(sky)
    sim.run()
    print('SIM %s us %6.2f min'%(st, (time.time()-t0)/60. ), flush = True)

