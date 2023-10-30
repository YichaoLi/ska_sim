#%%
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
import healpy as hp
from small_scale import get_random, get_pk1d_simple
from freq_corr import syn_spec, SCK_freq_corr
#%%
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
# the SCK parameters
A = 700.0 # in mk^2
l_fid = 1000
beta = 2.4
alpha = 2.8
zeta = 4.0
freq_ref = 130.0
#%%
# usually do not need to change the values
k_fid = 60
dk = 40
kbins = np.logspace(0.5, 4, 31)
win = np.blackman(nx)[:,None]*np.blackman(ny)
freq = np.arange(n_freq) * dfreq + freq0
fwhm = np.deg2rad(0.2) # sufficiently small to suppress some grid effect
#%%
kx = np.fft.fftfreq(nx, d=np.deg2rad(dx)) * 2*np.pi
ky = np.fft.rfftfreq(ny, d=np.deg2rad(dy)) * 2*np.pi
k2d = np.sqrt(kx[:,None]**2 + ky**2)
pk2d = np.zeros_like(k2d, dtype=np.float64)
valid = k2d>0
pk2d[valid] = A*(l_fid/k2d[valid])**beta
#%%
fields = get_random(pk2d, (dx, dy), n_freq)
#%%
sky_SCK = syn_spec(fields, freq, freq_ref, alpha, zeta)
#%%
plt.figure()
plt.imshow(sky_SCK[:,:,0])
plt.colorbar()
plt.figure()
plt.imshow(sky_SCK[:,:,10])
plt.colorbar()
#%%
plt.plot(freq, sky_SCK[1000,1000,:])
# %%
x = sky_SCK.reshape(-1, n_freq)
x = x - x.mean(axis=0, keepdims=True)
cov = (x.T@x)/x.shape[0]
# %%
cov_SCK = SCK_freq_corr(freq, freq_ref, alpha, zeta)
# %%
plt.figure()
plt.imshow(cov_SCK)
plt.colorbar()
plt.figure()
plt.imshow(cov)
plt.colorbar()
# %%
norm = np.sqrt(np.diag(cov))
norm = norm[:,None] * norm
cov = cov/norm

norm = np.sqrt(np.diag(cov_SCK))
norm = norm[:,None] * norm
cov_SCK = cov_SCK/norm
# %%
plt.figure()
plt.imshow(cov_SCK)
plt.colorbar()
plt.figure()
plt.imshow(cov)
plt.colorbar()
plt.figure()
plt.imshow(cov-cov_SCK)
plt.colorbar()
# %%
kbins = np.logspace(0.5, 4, 31)
win = np.blackman(nx)[:,None]*np.blackman(ny)
pk_SCK_freq = np.diag(SCK_freq_corr(freq, freq_ref, alpha, zeta))
# %%
ifreq = 10
k1d, pk1d = get_pk1d_simple(sky_SCK[...,ifreq], dx, dy, win=win, kbins=kbins)
pk_SCK = A*(l_fid/k1d[k1d>0])**beta * pk_SCK_freq[ifreq]
# %%
plt.loglog(k1d, pk1d)
plt.loglog(k1d[k1d>0], pk_SCK)
# %%
