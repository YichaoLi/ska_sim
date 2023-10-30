import numpy as np
import h5py as h5

from ska_sim import small_scale as ss
from ska_sim import freq_corr as fc

import h5py as h5
from ps_est import algebra as al

def sim_diffuse_fg_map(temp_file, temp_key, sck_params=None, output_path=None,
                       pad=None, suffix='diffuse_fg'):
    
    params = {
        # the SCK parameters
        'A'     : 700.0, # in mk^2
        'l_fid' : 1000,
        'beta'  : 2.4,
        'alpha' : 2.8,
        'zeta'  : 4.0,
        'freq_ref' : 130.0,
    }
    
    if sck_params is not None:
        params.update(sck_params)
        
    sck_model = lambda k, p: p['A']*(p['l_fid']/k)**p['beta']
    
    #suffix = ''
    
    with h5.File(temp_file, 'r') as fp:
        try:
            imap = al.load_h5(fp, temp_key)
        except KeyError:
            msg = 'hi_map does not existe, try other keys:'
            print(msg)
            print(fp.keys())
            return
            #raise KeyError(msg)
        if pad is not None:
            imap_info = imap.info
            imap = al.info_array(np.pad(imap, ((0, 0), (pad, pad), (pad, pad))))
            imap.info = imap_info
            suffix += '_pad%d'%pad
            print(imap.shape)
        imap = al.make_vect(imap, axis_names=imap.info['axes'])

    imap[:]  = 0.
    imap_ra  = imap.get_axis_edges('ra')
    imap_dec = imap.get_axis_edges('dec')
    imap_freq= imap.get_axis('freq')

    ra_min, ra_max   = imap_ra.min(), imap_ra.max()
    dec_min, dec_max = imap_dec.min(), imap_dec.max()
    
    factor = np.cos(imap.info['dec_centre'] * np.pi / 180.)
    dx = imap.info['ra_delta'] * factor
    dy = imap.info['dec_delta']
    nx = imap.shape[1]
    ny = imap.shape[2]
    freq0 = imap_freq[0]
    dfreq = imap.info['freq_delta']
    n_freq = imap.shape[0]
    #print(dx, dy, dfreq, freq0)
    
    # usually do not need to change the values
    k_fid = 60
    dk = 40
    kbins = np.logspace(0.5, 4, 31)
    win = np.blackman(nx)[:,None]*np.blackman(ny)
    freq = np.arange(n_freq) * dfreq + freq0
    fwhm = np.deg2rad(0.2) # sufficiently small to suppress some grid effect
    
    kx    = np.fft.fftfreq(nx, d=np.deg2rad(dx))  * 2 * np.pi
    ky    = np.fft.rfftfreq(ny, d=np.deg2rad(dy)) * 2 * np.pi
    k2d   = np.sqrt(kx[:,None]**2 + ky**2)
    pk2d  = np.zeros_like(k2d, dtype=np.float64)
    valid = k2d>0
    pk2d[valid] = sck_model(k2d[valid], params)
    
    fields = ss.get_random(pk2d, (dx, dy), n_freq)
    
    sky_SCK = fc.syn_spec(fields, freq, params['freq_ref'], 
                          params['alpha'], params['zeta'])
    
    
    sky_SCK = np.rollaxis(sky_SCK, -1)
    sky_SCK = al.info_array(sky_SCK)
    sky_SCK.info = imap_info
    sky_SCK.info['unit'] = 'mK'
    sky_SCK = al.make_vect(sky_SCK, axis_names=imap_info['axes'])
    
    print(sky_SCK.shape)
    if output_path is not None:
        _output = output_path + 'sck%s.h5'%(suffix)
        with h5.File(_output, 'w') as fp:
            al.save_h5(fp, 'fg_map', sky_SCK)
    else:
        return sky_SCK 
