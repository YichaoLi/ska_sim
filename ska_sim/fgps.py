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


def make_kbin(kmin, kmax, knum, logk):

    if logk:
        return np.logspace(np.log10(kmin), np.log10(kmax), knum)
    else:
        return np.linspace(kmin, kmax, knum)
    
    
def svd_clean(data_all, weight=None, mode_list=[0, 5]):
    
    mode_list_ed = copy.deepcopy(mode_list)
    mode_list_st = copy.deepcopy(mode_list)
    mode_list_st[1:] = mode_list_st[:-1]
    
    # remove bad pixels
    data_all[~np.isfinite(data_all)] = 0.
    
    freq_good = np.ones(data_all.shape[0], dtype='bool')
    if weight is None:
        weight = np.ones_like(data_all)
    
    freq_cov, counts = find_modes.freq_covariance(data_all, data_all, 
                                                  weight, weight, 
                                                  freq_good, freq_good)
    svd_info = find_modes.get_freq_svd_modes(freq_cov, np.sum(freq_good))
    
    data_cleaned = []
    for (n_modes_st, n_modes_ed) in zip(mode_list_st, mode_list_ed):
        print('cleaning %d to %d modes'%(n_modes_st, n_modes_ed))
        svd_modes = svd_info[1][n_modes_st:n_modes_ed]
        data_all, amp = find_modes.subtract_frequency_modes(data_all, svd_modes, 
                                                            weight, freq_good)
        data_cleaned.append(copy.deepcopy(data_all))
        
    return data_cleaned, svd_info

def ps_est(data_all, w, weight=None, window='blackman', 
           kmin = 0.01, kmax = 1.00, knum = 30, logk = True,
           kmin_x = 0.02, kmax_x = 0.40, knum_x = 40, 
           kmin_y = 0.04, kmax_y = 0.90, knum_y = 20, 
           logk_2d = True, unitless=True):

    print('gridding map')
    cube, info = physical_grid_wcs(data_all, w, refine_frequency=1)
    Z = cube.get_axis('freq')
    X = cube.get_axis('ra')
    Y = cube.get_axis('dec')
    
    if weight is None:
        cube_w = al.ones_like(cube)
        cube_w[cube==0] = 0.
    else:
        
        cube_w, info = physical_grid_wcs(weight, w,refine_frequency=1)
    
    #print(cube.max(), cube_w.max())
    
    cube2 = cube
    cube2_w = cube_w
    
    kbin   = make_kbin(kmin,   kmax,   knum + 1,   logk   )
    kbin_x = make_kbin(kmin_x, kmax_x, knum_x + 1, logk_2d)
    kbin_y = make_kbin(kmin_y, kmax_y, knum_y + 1, logk_2d)
    
    print('ps estimation')
    ps2d, ps1d = pwrspec_estimator.calculate_xspec(cube, cube2,
                                                   cube_w, cube2_w,
                                                   window = window,
                                                   bins = kbin,
                                                   bins_x = kbin_x,
                                                   bins_y = kbin_y,
                                                   logbins = logk,
                                                   logbins_2d = logk_2d,
                                                   unitless = unitless,
                                                   nonorm=False
                                                   )
    return ps2d, ps1d

def fgclean_and_psest(data_path, data_name, mode_list, mask_edges=None, output_path=None):
    with fits.open(data_path + data_name) as hlist:
        print('load data from \n %s'%data_name)
        w = WCS(hlist[0].header)
        data_all = hlist[0].data

    weight = np.ones(data_all.shape)
    if mask_edges != None:
        print('mask edges %d'%mask_edges)
        weight[:, 0:mask_edges, :] = 0
        weight[:, -mask_edges:, :] = 0
        weight[:, :, 0:mask_edges] = 0
        weight[:, :, -mask_edges:] = 0
    data_all[weight==0] = 0
    data_cleaned, svd_info = svd_clean(data_all, weight=weight, mode_list=mode_list)
    ps2d_list = []
    ps1d_list = []

    for d in data_cleaned:
        mask = ~np.isfinite(d)
        d[mask] = 0.
        _weight = weight.copy()
        _weight[mask] = 0
        ps2d, ps1d = ps_est(d, w, weight=_weight)
        ps2d_list.append(ps2d)
        ps1d_list.append(ps1d)
    
    if output_path is not None:
        with h5.File(output_path, 'w') as results:
            results['head'] = w.to_header_string()
            results['data'] = data_cleaned
            for ii, m in enumerate(mode_list):
                for key in ps2d_list[ii].keys():
                    results['sub%02d/ps2d_%s'%(m, key)] = ps2d_list[ii][key]
                for key in ps1d_list[ii].keys():
                    results['sub%02d/ps1d_%s'%(m, key)] = ps1d_list[ii][key]
            
            results['svd_eigval']  = svd_info[0]
            results['svd_eigvec_l'] = np.array(svd_info[1])
            results['svd_eigvec_r'] = np.array(svd_info[2])
            results['svd_modes'] = np.array(mode_list)
    else:
        results = {}
        results['head'] = w.to_header_string()
        results['data'] = data_cleaned
        for ii, m in enumerate(mode_list):
            for key in ps2d_list[ii].keys():
                results['sub%02d/ps2d_%s'%(m, key)] = ps2d_list[ii][key]
            for key in ps1d_list[ii].keys():
                results['sub%02d/ps1d_%s'%(m, key)] = ps1d_list[ii][key]
            
        results['svd_eigval']  = svd_info[0]
        results['svd_eigvec_l'] = np.array(svd_info[1])
        results['svd_eigvec_r'] = np.array(svd_info[2])
        results['svd_modes'] = np.array(mode_list)
        return results


def plot_svdmaps(results, figsize=(30, 5)):
    w = results['head']
    data_cleaned = results['data']
    
    
    n_plots = len(data_cleaned)
    n_col   = n_plots
    n_row   = 1 #int(np.ceil( n_plots / float(n_col)))
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_row, n_col, left=0.08, bottom=0.1, top=0.8, right=0.90,
                           figure=fig, wspace=0.05, hspace=0.05)
    #cax = fig.add_axes([0.91, 0.2, 0.2/float(figsize[0]), 0.6])
    for i, d in enumerate(data_cleaned):

        factor = 1.
    
        slices = [0, ] * w.naxis
        slices[0] = 'x'
        slices[1] = 'y'

        ax = fig.add_subplot(gs[0,i], projection=w, slices=tuple(slices))
        pos = ax.get_position()
        pos.y0 = pos.y1 + 0.02
        pos.y1 = pos.y0 + 0.03
        pos.x0 = pos.x0 + 0.01
        pos.x1 = pos.x1 - 0.01
        #print(pos)
        cax = fig.add_axes(pos)
        #ax = fig.add_axes([0.1, 0.1, 0.80, 0.85], projection=w, slices=tuple(slices))
        #cax = fig.add_axes([0.91, 0.2, 0.2/float(figsize[0]), 0.6])
        
        mean = np.mean(d)
        std  = np.std(np.mean(d, axis=0))
        vmin = mean - 6 * std
        vmax = mean + 6 * std
    
        im = ax.pcolormesh(np.mean(d, axis=0) * factor, 
                           vmin=vmin, vmax=vmax, cmap='gist_heat')
        
        lon = ax.coords[0]
        lat = ax.coords[1]
        
        lon.set_ticklabel(size=10)
        lat.set_ticklabel(size=10)

        ax.set_xlabel('R.A.')
            
        if i != 0:
            lat.set_ticklabel_visible(False)
            lat.set_axislabel('')
        else:
            ax.set_ylabel('Dec')
        
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, cax=cax, orientation="horizontal")
        cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

def plot_ps(result_list, title_list=None, mode_list = [0, 5, 10, 15, 20, 25, 30], 
            ymin=None, ymax=None, xmin=None, xmax=None, figsize=(8, 6)):
    
    # load input ps
    result_path = '/home/DATA/ycli/ska_challenge/2023/ps_results/'
    result_name = 'lightcones_brightness_xy600_z90.h5'
    with h5.File(result_path + result_name, 'r') as fp:
        #print(fp.keys())
        eor_kbin = fp['kbin'][:]
        eor_ps1d = fp['ps1d'][:]
    
    
    n_col = len(result_list)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, n_col, left=0.08, bottom=0.1, top=0.9, right=0.95,
                           figure=fig, wspace=0.05, hspace=0.05)
    for i in range(n_col):
        ax = fig.add_subplot(gs[0,i])
        with h5.File(result_list[i], 'r') as f:
            #print(f.keys())
            mode_list = f['svd_modes'][:]
            
            for m, mode in enumerate(mode_list):
            
                xx = f['sub%02d/ps1d_bin_center'%mode][:]
                yy = f['sub%02d/ps1d_binavg'%mode][:]
                ax.plot(xx, yy, 'o-', label = 'Sub. %02d'%mode)

        ax.plot(eor_kbin, np.mean(eor_ps1d, axis=0), 'k.-')
        ax.loglog()
        
        if i != 0:
            ax.set_yticklabels([])
        else:
            ax.legend()
            ax.set_ylabel(r'$\Delta^2(k)\,[{\rm mK}^2]$')
        ax.set_xlabel(r'$k\,[{\rm Mpc}^{-1}h]$')
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)
        if title_list != None:
            ax.set_title(title_list[i])
            
def plot_svdvalue(result_list, title_list=None, mode_list = [0, 5, 10, 15, 20, 25, 30], 
            ymin=None, ymax=None, xmin=None, xmax=None, figsize=(8, 6)):

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1, left=0.08, bottom=0.1, top=0.9, right=0.95,
                           figure=fig, wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(gs[0,0])
    for i in range(len(result_list)):
        _r = result_list[i]['svd'][0]
        _x = np.arange(1, _r.shape[0]+1)
        if title_list != None:
            label=title_list[i]
        else:
            label='%02d'%i
        ax.plot(_x, _r, 'o-', label = label)

    ax.legend()
    ax.set_ylabel('SVD values')
    ax.set_xlabel('Mode NO.')
    ax.semilogy()


