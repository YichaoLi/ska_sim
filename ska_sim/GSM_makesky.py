#%%
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
import healpy as hp
from small_scale import small_scale, get_pk1d_simple
from freq_corr import GSM_interpolate
from tqdm import tqdm
#%%
GSM_file = './gsm2016_file.hdf5'
# use this line if pygdsm is installed, I did not install it
#from pygdsm.component_data import GSM2016_FILEPATH as GSM_file

save_GSM_template = './gsm_patch_template.hdf5'

ra0 = 0.0
dec0 = -30.0
dx = 16.0/3600.0
dy = 16.0/3600.0
nx = 4096
ny = 4096
freq0 = 106.0
# dfreq should be 0.1 MHz, but it is too small for demonstration
# change it when you generate the real data
#dfreq = 0.1
#n_freq = 901
dfreq = 4.0
n_freq = 20
#%%
# usually do not need to change the values
k_fid = 60
dk = 40
kbins = np.logspace(0.5, 4, 31)
win = np.blackman(nx)[:,None]*np.blackman(ny)
freq = np.arange(n_freq) * dfreq + freq0
fwhm = np.deg2rad(0.2) # sufficiently small to suppress some grid effect
#%%
# initialize the object
gsm = GSM_interpolate(GSM_file)
ss = small_scale(ra0, dec0, dx, dy, nx, ny)
#%%
#%%
# get the sky patch without small scale structures
skymaps_gsm = []
for ii in tqdm(range(gsm.n_comps)):
    ss.set_skymap(gsm.maps[ii], fwhm=fwhm);
    skymaps_gsm.append(ss.skymap)
#%%
# show the sky
plt.figure()
for ii in tqdm(range(gsm.n_comps)):
    plt.subplot(231+ii)
    plt.imshow(skymaps_gsm[ii].T, origin='lower', extent=ss.edges, aspect='auto')
    plt.title(gsm.labels[ii])
plt.tight_layout()
plt.figure()
for ii in tqdm(range(gsm.n_comps)):
    plt.loglog(*get_pk1d_simple(skymaps_gsm[ii], ss.dx, ss.dy, win=win, kbins=kbins), label=gsm.labels[ii])
plt.legend()
#%%
# if you interpolate the template to low frequency using GSM_interpolate, you will find only Synchrotron and HI is large
# and it is consistent with Figure 5 of https://arxiv.org/pdf/1605.04920.pdf
# although in fact I fail to completely reproduce Figure 5
beta = dict()
beta['Synchrotron'] = 2.4
beta['HI'] = 3.0
labels_ss = list(beta.keys())
skymaps_ss = []
for label in beta:
    imap = gsm.labels.index(label)
    ss.set_skymap(gsm.maps[imap], fwhm=fwhm);
    ss.set_k(k_fid, dk, win=win)
    ss.get_extrapolate_pk(beta[label])
    k1, pk1, pk2 = ss.show_pk_extrap(kbins);
    ss.syn_sky(gamma=1)
    ss.show_syn()
    ss.show_syn_pk(kbins)
    skymaps_ss.append(ss.skymap_ss)
    plt.show()
#%%
# save the template used to extrapolate
with h5.File(save_GSM_template, 'w') as filein:
    filein.attrs['ra0'] = ra0
    filein.attrs['dec0'] = dec0
    filein.attrs['dx'] = dx
    filein.attrs['dy'] = dy
    filein['ra'] = ss.ra
    filein['dec'] = ss.dec
    filein['x'] = ss.x
    filein['y'] = ss.y
    filein['ra'].attrs['unit'] = 'deg'
    filein['dec'].attrs['unit'] = 'deg'
    filein['x'].attrs['unit'] = 'deg'
    filein['y'].attrs['unit'] = 'deg'
    filein.create_group('GSM_sky')
    for ii in tqdm(range(gsm.n_comps)):
        filein['GSM_sky'][gsm.labels[ii]] = skymaps_gsm[ii]
    filein['GSM_sky'].attrs['unit'] = 'MJy/sr'
    filein.create_group('GSM_ss')
    for ii in tqdm(range(len(labels_ss))):
        filein['GSM_ss'][labels_ss[ii]] = skymaps_ss[ii]
    filein['GSM_ss'].attrs['unit'] = 'MJy/sr'
#%%
#%%
#%%
# After here we extrapolate to low frequency
#%%
sky_comps = np.zeros([gsm.n_comps, nx, ny], dtype=np.float64)
for ii, lb in enumerate(gsm.labels):
    with h5.File(save_GSM_template, 'r') as filein:
        if lb in filein['GSM_sky'].keys():
            print('Loading %s from GSM_sky'%lb)
            sky_comps[ii] = filein['GSM_sky'][lb][:]
        if lb in filein['GSM_ss'].keys():
            print('Loading %s from GSM_ss'%lb)
            sky_comps[ii] += filein['GSM_ss'][lb][:]
#%%
# interpolate sky in different freq for all components
# it is very large indeed, so you may split freq into sections
print('Make sky')
newsky = gsm.interpolate_skymap(sky_comps, freq)
#%%
# interpolate sky in different freq for syn and HI separately
print('Make sky for syn only')
newsky_syn = gsm.interpolate_skymap([sky_comps[0]], freq, labels=[gsm.labels[0]])
print('Make sky for HI only')
newsky_HI = gsm.interpolate_skymap([sky_comps[2]], freq, labels=[gsm.labels[2]])
#%%
plt.figure()
plt.imshow(newsky[:,:,10].T)
plt.colorbar()
plt.figure()
plt.imshow(newsky_syn[:,:,10].T+newsky_HI[:,:,10].T)
plt.colorbar()
plt.figure()
plt.imshow(newsky_syn[:,:,10].T)
plt.colorbar()
# this component is in fact negative
plt.figure()
plt.imshow(newsky_HI[:,:,10].T)
plt.colorbar()
#%%
#%%
from astropy.io import fits
with fits.open('/home/s1_tianlai/SKA/SDC3/ZW3.msn_psf.fits') as filein:
    print(repr(filein[0].header))
    psf = filein[0].data
    bmaj = filein[0].header['BMAJ']
    bmin = filein[0].header['BMIN']
#%%
beam_omega = 2*np.pi*bmaj*bmin/(8*np.log(2))
#%%
from scipy.signal import fftconvolve
#%%
newsky_conv = fftconvolve(newsky, psf[:20].T, mode='valid', axes=(0,1))
newsky_conv = newsky_conv*(dx*dy)/beam_omega
newsky0 = fftconvolve(newsky[...,0], psf[0].T, mode='valid', axes=(0,1))
newsky10 = fftconvolve(newsky[...,10], psf[10].T, mode='valid', axes=(0,1))
newsky0 = newsky0*(dx*dy)/beam_omega
newsky10 = newsky10*(dx*dy)/beam_omega
#%%
print(np.abs(newsky_conv[...,0]-newsky0).max())
print(np.abs(newsky_conv[...,10]-newsky10).max())
#%%
plt.figure()
plt.imshow(newsky[:,:,10].T)
plt.colorbar()
plt.figure()
plt.imshow(newsky_conv[:,:,10].T)
plt.colorbar()
#%%
#%%
#%%
#%%
inds = np.random.choice(np.arange(np.prod(newsky.shape[:2])), 20, replace=False)
i, j = np.unravel_index(inds, newsky.shape[:2])
point_src = np.zeros(newsky.shape[:-1]+(3,), dtype=np.float64)
point_src[i,j,:] = 1.0
psrc_conv = fftconvolve(point_src, psf[0:900:300].T, mode='valid', axes=(0,1))/beam_omega
#%%
from matplotlib.colors import SymLogNorm
plt.figure()
plt.imshow(psrc_conv[...,0], norm=SymLogNorm(linthresh=1))
plt.colorbar()
#%%
