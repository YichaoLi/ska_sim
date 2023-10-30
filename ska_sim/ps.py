from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import numpy as np

from astropy.time import Time


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py as h5
from ps_est import algebra as al
from ps_est import plot_map as pm

from astropy import units as u

from scipy.interpolate import interp1d

from tqdm import tqdm

def sim_ps_map(ps_path, ps_catalogue, temp_file, temp_key='hi_map', output_path='./', 
        flux_cut=1000, shuffle_radec=None, polyfit=False, pad=None, plot=False):

    '''
    Generate point source map using GLMv2 LoBES catalogue

    temp_file: a temp file used to define the ra, dec, and freq ranges.
    shuffle_radec: make mock maps by shuffling the ra dec of the ps catalogue
    '''

    
    suffix='_%dmuJy'%flux_cut
    
    #temp_file = '/home/DATA/gaoliyang/data_challenge_map_21cmFAST/hi_map_000_ra600_dec600.h5'
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
    pix_size = imap.info['dec_delta'] * u.deg
    #print(imap_freq[0], imap_freq[-1])
    
    ra_min, ra_max   = imap_ra.min(), imap_ra.max()
    dec_min, dec_max = imap_dec.min(), imap_dec.max()
    #print(ra_min, ra_max)
    #print(dec_min, dec_max)
        
    with fits.open(ps_path + ps_catalogue) as hlist:
        data = hlist[1].data
        ra  = data['RA']
        dec = data['DEC']

    if shuffle_radec is not None:
        np.random.shuffle(ra)
        np.random.shuffle(dec)
        suffix += '_%02d'%shuffle_radec
    ra[ra>180] -= 360
    sel = ( ra > ra_min ) * ( ra < ra_max ) * ( dec > dec_min ) * ( dec < dec_max)
    ra = ra[sel]
    dec = dec[sel]
    
    if polyfit:
        freq = [107, 115, 122, 130, 143, 151, 158, 166, 174, 181, 189, 197, 204, 212, 220, 227]
    else:
        # use only a few of good frequenceis for interpolation
        freq = [115, 151, 181, 189]
    
    flux = []
    flux_err = []
    for f in freq:
        flux.append(data['P_FLX%d'%f][sel][:, None])
        flux_err.append(data['ERR_P_FLX%d'%f][sel][:, None])
    flux = np.concatenate(flux, axis=1)
    flux_err = np.concatenate(flux_err, axis=1)
    freq = np.array(freq)
    
    flux_cut = flux[:, 1] > flux_cut * 1.e-6
    flux_cut *= np.all(np.isfinite(flux), axis=1)
    flux_cut *= np.all(flux > 0, axis=1)
    
    flux = flux[flux_cut]
    ra   = ra[flux_cut]
    dec  = dec[flux_cut]
    
    #print(flux.max(), flux.min())
    
    # assuming psf major axis = 140 arcsec, psf minor axis = 130 arcsec, 
    # according to Table4 of https://academic.oup.com/mnras/article/464/1/1146/2280761    

    #omega_B = 2. * np.pi * (130 * u.arcsec) * (140 * u.arcsec)
    omega_B = 2. * np.pi * pix_size * pix_size
    equi = u.brightness_temperature(freq * u.MHz)
    Jy2mK = ( 1 * u.Jy/omega_B ).to(u.mK, equivalencies=equi).value
    flux *= Jy2mK[None, :]
    
    if not polyfit:
        flux_intf = interp1d(freq, flux, kind='quadratic', axis=1, fill_value="extrapolate")
        #print(flux.shape)
        flux_new = flux_intf(imap_freq)
    else:
        flux_new = np.zeros((flux.shape[0], imap_freq.shape[0]))
        suffix += '_polyfit'
        for ii in tqdm(range(flux.shape[0])):
            spec_func = np.poly1d(np.polyfit(np.log10(freq), np.log10(flux[ii]), deg=2,))
            flux_new[ii] = 10.**spec_func(np.log10(imap_freq))
            
    print('There are %d sources'%flux_new.shape[0])
    #print(flux_new.shape)

    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax  = fig.subplots()
        for i in range(flux_new.shape[0]):
            l = ax.errorbar(freq, flux[i], flux_err[i], fmt='.:')
            ax.plot(imap_freq, flux_new[i], '-', color=l[0].get_color())
        ax.set_ylim(ymin=1.e0, ymax=1.e5)
        ax.semilogy()
        plt.show()
        plt.close(fig)
        
    for ff in range(imap_freq.shape[0]):
        imap[ff] += np.histogram2d(ra, dec, bins=(imap_ra, imap_dec), 
                weights = flux_new[:, ff])[0]

    #imap *= flux2K[:, None, None]
    #print(flux_cut)
    imap.info['unit'] = 'mK'
    if output_path is not None:
        #_output = output_path + 'pointsource_%s%s.h5'%(ps_catalogue.split('.')[0], suffix)
        _output = output_path + 'pointsource%s.h5'%(suffix)
        with h5.File(_output, 'w') as fp:
            al.save_h5(fp, 'ps_map', imap)
    else:
        return imap
