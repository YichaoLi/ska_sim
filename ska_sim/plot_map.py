#from collections import OrderedDict as dict

from ps_est import algebra as al

import logging

import h5py as h5
import numpy as np
import scipy as sp
import copy

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.ndimage.filters import gaussian_filter as gf

from astropy.coordinates import SkyCoord
import astropy.units as u

import healpy as hp

logger = logging.getLogger(__name__)

_c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

_unit_latex = {
    'K': r'$[ {\rm K} ]$', 
    'K2': r'$[ {\rm K}^2 ]$', 
    'mK': r'$[ {\rm mK} ]$', 
    'mK2': r'$[ {\rm mK}^2 ]$', 
    'muK': r'$[ {\rm \mu K} ]$', 
    'muK2': r'$[ {\rm \mu K}^2 ]$', 
    'uK': r'$[ {\rm \mu K} ]$', 
    'uK2': r'$[ {\rm \mu K}^2 ]$', 
    }

def load_map(map_path, map_type, print_info=False):

    ext = os.path.splitext(map_path)[-1]

    if ext == '.h5':
        with h5.File(map_path, 'r') as f:
            keys = tuple(f.keys())
            try:
                imap = al.load_h5(f, map_type)
            except KeyError:
                print(f.keys())
                return
            if print_info:
                logger.info( ('%s '* len(keys))%keys )
                print(imap.info)
            try:
                mask = f['mask'][:].astype('bool')
            except KeyError:
                print('No mask')
                mask = None

            try:
                unit = imap.info['unit']
            except KeyError:
                print('No unit')
                unit = None

    elif ext == '.npy':
        imap = al.load(map_path)
        mask = None
        unit = imap.info['unit']
    else:
        raise IOError('%s not exists'%map_path)


    imap = al.make_vect(imap, axis_names = imap.info['axes'])
    freq = imap.get_axis('freq')
    ra   = imap.get_axis( 'ra')
    dec  = imap.get_axis('dec')
    ra_edges  = imap.get_axis_edges( 'ra')
    dec_edges = imap.get_axis_edges('dec')

    return imap, ra, dec, freq, ra_edges, dec_edges, mask, unit

def smoothing(imap, smoothing_fwhm=0.1):

    '''
    sigma in unit of degree

    '''

    pix_size = imap.info['dec_delta']

    _sig = smoothing_fwhm / pix_size / (8 * np.log(2))**0.5
    imap = gf(imap, _sig)

    return imap


def show_map(map_path, map_type, indx = (), figsize=(10, 4), 
        xlim=None, ylim=None, logscale=False, vmin=None, vmax=None, sigma=2., 
        unit = 'mK', title='', c_label=None, print_info=False, cmap='bwr', 
        smoothing_fwhm=None):

    #imap, ra, dec, freq, ra_edges, dec_edges, mask, map_unit\
    imap_sets = load_map(map_path, map_type, print_info=print_info)

    plot_map(imap_sets, indx = indx, figsize=figsize, 
        xlim=xlim, ylim=ylim, logscale=logscale, vmin=vmin, vmax=vmax, sigma=sigma, 
        unit = unit, title=title, c_label=c_label, print_info=print_info, cmap=cmap, 
        smoothing_fwhm=smoothing_fwhm)

def plot_map(imap_sets, indx = (), figsize=(10, 4), 
        xlim=None, ylim=None, logscale=False, vmin=None, vmax=None, sigma=2., 
        unit = 'mK', title='', c_label=None, print_info=False, cmap='bwr', 
        smoothing_fwhm=None):

    imap, ra, dec, freq, ra_edges, dec_edges, mask, map_unit = imap_sets

    imap = np.ma.masked_invalid(imap)
    imap.mask += imap == 0.

    if mask is not None:
        imap[mask] = np.ma.masked

    imap = imap[indx]
    freq = freq[indx[-1]]
    if isinstance( indx[-1], slice):
        freq = (freq[0], freq[-1])
        imap = np.ma.mean(imap, axis=0)
        imap = np.ma.masked_equal(imap, 0)
    else:
        freq = (freq,)

    if map_unit != unit and map_unit != None:
        unit_conv = getattr(u, map_unit).to(getattr(u, unit))
        imap *= unit_conv
    unit_label = _unit_latex[unit]

    if smoothing_fwhm != None:
        imap = smoothing(imap, smoothing_fwhm=smoothing_fwhm)

    if xlim is None:
        xlim = [ra_edges.min(), ra_edges.max()]
    if ylim is None:
        ylim = [dec_edges.min(), dec_edges.max()]

    if logscale:
        imap = np.ma.masked_less(imap, 0)
        if vmin is None: vmin = np.ma.min(imap)
        if vmax is None: vmax = np.ma.max(imap)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        if sigma is not None:
            if vmin is None: vmin = np.ma.mean(imap) - sigma * np.ma.std(imap)
            if vmax is None: vmax = np.ma.mean(imap) + sigma * np.ma.std(imap)
        else:
            if vmin is None: vmin = np.ma.min(imap)
            if vmax is None: vmax = np.ma.max(imap)
            #if vmax is None: vmax = np.ma.median(imap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    
    fig = plt.figure(figsize=figsize)
    l = 0.08 * 10. / figsize[0]
    b = 0.08 *  4.  / figsize[1]
    w = 1 - 0.20 * 10.  / figsize[0]
    h = 1 - 0.10 *  4.  / figsize[1]
    ax = fig.add_axes([l, b, w, h])
    l = 1 - 0.11 * 10. / figsize[0]
    b = 0.20 *  4  / figsize[1]
    w = 1 - 0.10 * 10  / figsize[0] - l
    h = 1 - 0.34 *  4  / figsize[1]
    cax = fig.add_axes([l, b, w, h])
    ax.set_aspect('equal')

    cm = ax.pcolormesh(ra_edges, dec_edges, imap.T, norm=norm, cmap=cmap)
    if len(freq) == 1:
        ax.set_title(title + r'${\rm Frequency}\, %7.3f\,{\rm MHz}$'%freq)
    else:
        ax.set_title(title + r'${\rm Frequency}\, %7.3f\,{\rm MHz}$ - $%7.3f\,{\rm MHz}$'%freq)
    ax.set_xlim(xmax = min(xlim), xmin=max(xlim))
    ax.set_ylim(ylim)
    ax.set_xlabel(r'${\rm RA}\,[^\circ]$')
    ax.set_ylabel(r'${\rm Dec}\,[^\circ]$')

    nvss_range = [ [ra_edges.min(), ra_edges.max(), 
                    dec_edges.min(), dec_edges.max()],]
    if not logscale:
        ticks = list(np.linspace(vmin, vmax, 5))
        ticks_label = []
        for x in ticks:
            ticks_label.append(r"$%5.1f$"%x)
        fig.colorbar(cm, ax=ax, cax=cax, ticks=ticks)
        cax.set_yticklabels(ticks_label, rotation=90, va='center')
    else:
        fig.colorbar(cm, ax=ax, cax=cax)
    cax.minorticks_off()
    if c_label is None:
        c_label = r'$T\,$' + unit_label
    cax.set_ylabel(c_label)

    #return xlim, ylim, (vmin, vmax), fig


