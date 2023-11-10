from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import numpy as np
import copy
import h5py as h5

from astropy.time import Time
import astropy.units as u


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fpipe.ps.physical_gridding import *
from fpipe.ps import pwrspec_estimator
from fpipe.map import algebra as al
from fpipe.ps import find_modes
# 21cm transition frequency (in MHz)
__nu21__ = 1420.40575177


def show_map(data_path, data_name, diff=False, freq_slice=0, axes=None, 
             figsize=(5, 5), sigma=1, vmin=None, vmax=None):

    hlist = fits.open(data_path + data_name)[0]
    
    #print(hlist.header)
    
    w = WCS(hlist.header)
    data = hlist.data
    if data.ndim == 4:
        #print(data.shape)
        data = data[0]
    
    slices = [0, ] * w.naxis
    slices[0] = 'x'
    slices[1] = 'y'
    
    if axes is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.1, 0.1, 0.80, 0.85], projection=w, 
                          slices=tuple(slices))
        cax = fig.add_axes([0.91, 0.2, 0.2/float(figsize[0]), 0.6])
    else:
        fig, ax, cax = axes
    
    if diff:
        _d = data[freq_slice] - data[freq_slice+1]
    else:
        #print(data.shape, data.min(), data[freq_slice].max())
        #_d = np.log10( -data[freq_slice] )
        _d = data[freq_slice]
        _d = np.ma.masked_invalid(_d)
     
    if sigma == 0:
        if vmin == None: vmin = np.ma.min(_d)
        if vmax == None: vmax = np.ma.max(_d)
    else:
        mean = np.mean(_d)
        std  = np.std(_d)
        if vmax == None: vmax = mean + sigma * std
        if vmin == None: vmin = mean - sigma * std
    #print(std)
    #print(vmin, vmax)
    im = ax.pcolormesh(_d, vmin=vmin, vmax=vmax, cmap='gist_heat')
    
    if axes is None:
        ax.set_xlabel('R.A.')
        ax.set_ylabel('Dec')
    
        fig.colorbar(im, ax=ax, cax=cax)

    return vmin, vmax, im
    
def show_map_multifreq(data_path, data_name_list, figsize=(18, 16), 
                       sigma=1, vmin=None, vmax=None, n_col = 6, label=''):

    hlist = fits.open(data_path + data_name_list[0])[0]
    #print(hlist.header)
    w = WCS(hlist.header)  
    slices = [0, ] * w.naxis
    #print(w.naxis)
    slices[0] = 'x'
    slices[1] = 'y'
    
    n_plots = len(data_name_list)
    n_row   = int(np.ceil( n_plots / float(n_col)))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_row, n_col, left=0.08, bottom=0.1, top=0.9, right=0.90,
                           figure=fig, wspace=0.05, hspace=0.05)
    cax = fig.add_axes([0.91, 0.2, 0.2/float(figsize[0]), 0.6])
    
    for f in range(len(data_name_list)):
        data_name = data_name_list[f]

        #find = f * 30
    
        i = f//n_col
        j = f%n_col
    
        ax = fig.add_subplot(gs[i,j], projection=w, slices=tuple(slices), zorder=-f)
        
        vmin, vmax, im = show_map(data_path, data_name, axes=(fig, ax, cax), 
                                  sigma=sigma, vmin=vmin, vmax=vmax)
    
        lon = ax.coords[0]
        lat = ax.coords[1]
        
        lon.set_ticklabel(size=10)
        lat.set_ticklabel(size=10)
        if i != n_row-1:
            lon.set_ticklabel_visible(False)
            lon.set_axislabel('')
        else:
            ax.set_xlabel('R.A.')
            
        if j != 0:
            lat.set_ticklabel_visible(False)
            lat.set_axislabel('')
        else:
            ax.set_ylabel('Dec')

        ax.set_aspect('equal')

    fig.colorbar(im, ax=ax, cax=cax)
    cax.set_ylabel(label)
    
def combine_fits_to_cube(data_path, data_name_list, file_name, 
                         suffix='image', overwrite=True, to_mK=False):
    
    print('Combine to a cube and convert unit from Jy/bema to mK')
    
    with fits.open(data_path + data_name_list[0]) as hlist:
        w = WCS(hlist[0].header)#.celestial
        w = w.dropaxis(3) # drop stokes axis
        if to_mK:
            factor = 2 * np.sqrt(2 * np.log(2))
            bmaj = hlist[0].header['BMAJ'] * u.deg
            bmin = hlist[0].header['BMIN'] * u.deg
            freq = hlist[0].header['CRVAL3'] * u.Hz
            omega_B = 2. * np.pi * bmaj  * bmin 
            equi =u.brightness_temperature(freq)
            Jy2K = ( 1 * u.Jy/omega_B ).to(u.K, equivalencies=equi).value
            
            print(Jy2K)
        else:
            Jy2K = 1.
        
    
    data = []
    for data_name in data_name_list:
        print(data_name)
        with fits.open(data_path + data_name)as hlist:
            _d   = hlist[0].data[:, 0, ...]
            
            #data.append(_d * Jy2K * 1.e3)
            # bacause the input maps are in unit of mK, ignore 1.e3 factor
            data.append(_d * Jy2K )
            
        
    data = np.concatenate(data, axis=0)
    
    fits.writeto(data_path + '%s_cube-%s.fits'%(file_name, suffix), 
                 data, header=w.to_header(), overwrite=overwrite)
