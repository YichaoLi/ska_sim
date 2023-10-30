#%%
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt
import healpy as hp
from scipy.spatial.transform import Rotation
from glob import glob
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
#%%
def get_grid(ra0, dec0, dx, nx, dy, ny):
    '''
    Make grid in flat sky approximation at arbitrary center (ra0, dec0)\n
    all quantity in deg\n
    return\n
        ra: ra for each point in deg\n
        dec: dec for each point in deg\n
        (x, y): relative position for each point with respect to (ra0, dec0) in deg\n
    '''
    x = np.arange(nx)*dx
    x = x-x.mean()
    y = np.arange(ny)*dy
    y = y-y.mean()
    xm, ym = np.meshgrid(x, y, indexing='ij')
    shp = xm.shape
    xm = np.deg2rad(xm.reshape(-1))
    ym = np.deg2rad(ym.reshape(-1))
    zm = np.sqrt(1-xm**2-ym**2)
    pos = np.asfortranarray([zm, xm, ym]).T # nx*ny, 3
    rot = Rotation.from_euler('yz', [-dec0, ra0], degrees=True)
    pos = rot.apply(pos)
    ra, dec = hp.vec2ang(pos, lonlat=True)
    ra = ra.reshape(shp)
    dec = dec.reshape(shp)
    return ra, dec, (x, y)
#%%
def get_pk2d_simple(skymap, dx, dy, win=None, remove_mean=True):
    '''
    Calculate 2d power spectrum from a sky slice\n
    skymap: skymap in grid\n
    dx, dy: x and y interval in deg\n
    win: the window function when calculating the power spectrum\n
    remove_mean: if True, remove the mean of skymap, it usually should be True\n
    return\n
        k: |k| in rad^-1\n
        Pk: the power spectrum\n
        (kx, ky): the x and y components of k\n
    '''
    skymap = np.asarray(skymap)
    if remove_mean:
        skymap = skymap - skymap.mean()
    dV = np.deg2rad(dx) * np.deg2rad(dy)
    if win is not None:
        win = np.asarray(win)
        skymap = skymap * win
        norm = dV/np.sum(win**2)
    else:
        norm = dV/skymap.size
    fk2d = np.fft.rfft2(skymap)
    kx = np.fft.fftfreq(skymap.shape[0], d=np.deg2rad(dx)) * 2*np.pi
    ky = np.fft.rfftfreq(skymap.shape[1], d=np.deg2rad(dy)) * 2*np.pi
    k2d = np.sqrt(kx[:,None]**2 + ky**2)
    pk2d = np.abs(fk2d)**2 * norm
    return k2d, pk2d, (kx, ky)

def pk2d_to_pk1d(k2d, pk2d, kbins):
    '''
    Calculate 1d power spectrum from 2d power spectrum
    '''
    pk1d, kbins = np.histogram(k2d[k2d!=0], bins=kbins, weights=pk2d[k2d!=0])
    k1d, _ = np.histogram(k2d[k2d!=0], bins=kbins, weights=k2d[k2d!=0])
    n, _ = np.histogram(k2d[k2d!=0], bins=kbins)
    valid = n>0
    pk1d = pk1d[valid]/n[valid]
    k1d = k1d[valid]/n[valid]
    return k1d, pk1d

def get_pk1d_simple(skymap, dx, dy, win=None, remove_mean=True, kbins=50):
    '''
    Calculate 1d power spectrum from a sky slice\n
    In flat sky approximation, P(k) = C(l=k)\n
    skymap: skymap in grid\n
    dx, dy: x and y interval in deg\n
    win: the window function when calculating the power spectrum\n
    remove_mean: if True, remove the mean of skymap, it usually should be True\n
    kbins: kbins in rad^-1 when calculating P(k), NOTE kbins is equivalent to l bins\n
    return\n
        k: k in rad^-1\n
        Pk: the power spectrum\n
    '''
    k2d, pk2d, _ = get_pk2d_simple(skymap, dx, dy, win=win, remove_mean=remove_mean)
    return pk2d_to_pk1d(k2d, pk2d, kbins)

def get_random(pk2d, d, size=None):
    '''
    Synthesize random field for given pk\n
    pk2d: 2d array, the power spectrum of the output field\n
    d: array_like with length 2, the x and y interval of the output field\n
    return\n
        the synthesized field
    '''
    if size is None:
        n_rel = 1
    else:
        n_rel = size
    d = np.deg2rad(d)
    shp_k = pk2d.shape
    shp = list(shp_k)
    shp[-1] = (shp_k[-1]-1)*2
    nk = np.random.randn(*shp_k, n_rel) + np.random.randn(*shp_k, n_rel)*1.J
    amp = np.prod(shp)/np.sqrt(np.prod(shp)*np.prod(d))/np.sqrt(2)
    nk = nk * np.expand_dims(np.sqrt(pk2d), axis=-1) * amp
    nk[0, 0] = 0.0
    nk[nk.shape[0]//2,0] = 0.0
    nk[0,-1] = 0.0
    nk[nk.shape[0]//2,-1] = 0.0
    newsky = np.fft.irfft2(nk, axes=(0, 1))
    if size is None:
        newsky = newsky[...,0]
    return newsky

def get_pk_ss(skymap, dx, dy, k_fid, dk, beta, win=None):
    '''
    Extrapolate the power spectrum of skymap with a powerlaw at some k with smooth transition.\n
    skymap: the input low resolution skymap\n
    dx, dy: the x and y interval for skymap, in deg\n
    k_fid: in rad^-1, from where to extrapolate\n
    dk: in rad^-1, the k interval to estimate the amplitude of power spectrum for normalization\n
    beta: the powerlaw index, P(k)\propto k^-beta\n
    win: the window function when calculating the power spectrum\n
    return\n
        pk2d_extrap: 2d array, the extrapolated power spectrum\n
        (k2d, pk2d_input): 2d array, the k and input power spectrum\n
    '''
    k_bin_fid = np.array([k_fid-dk/2.0, k_fid+dk/2.0])
    _, norm = get_pk1d_simple(skymap, dx, dy, kbins=k_bin_fid, win=win, remove_mean=True)
    norm = norm[0]
    k1, pk1, _ = get_pk2d_simple(skymap, dx, dy, win=win, remove_mean=True)
    pk2 = np.zeros_like(pk1, dtype=np.float64)
    valid = np.abs(k1)>k_fid
    pk2[valid] = np.abs(k1[valid]/k_fid)**(-beta) * norm
    pk2[valid] = pk2[valid] - pk1[valid]
    pk2 = np.maximum(pk2, 0.0)
    return pk2, (k1, pk1)

def rescale_map(skymap_ss, skymap_temp, dx, dy, k_fid, dk, gamma=1.0, win=None):
    '''
    Rescale the synthesized map with some template\n
    new_skymap_ss\propto skymap_ss * skymap_temp^gamma\n
    skymap_ss: the synthesized random field\n
    skymap_temp: the low resolution template to rescale the skymap_ss\n
    dx, dy: the x and y interval in deg\n
    k_fid: in rad^-1, from where to extrapolate\n
    dk: in rad^-1, the k interval to estimate the amplitude of power spectrum for normalization\n
    gamma: the power law index, new_skymap_ss\propto skymap_ss * skymap_temp^gamma\n
    win: the window function when calculating the power spectrum\n
    return\n
        new_skymap_ss\n
    '''
    newsky_ss = skymap_ss * skymap_temp**gamma
    k_bin_fid = np.array([k_fid-dk/2.0, k_fid+dk/2.0])
    _, norm0 = get_pk1d_simple(skymap_ss, dx, dy, kbins=k_bin_fid, win=win, remove_mean=True)
    _, norm1 = get_pk1d_simple(newsky_ss, dx, dy, kbins=k_bin_fid, win=win, remove_mean=True)
    norm0 = norm0[0]
    norm1 = norm1[0]
    #print(norm0, norm1)
    newsky_ss = newsky_ss * np.sqrt(norm0/norm1)
    #print(norm0, get_pk1d_simple(newsky_ss, dx, dy, kbins=k_bin_fid))
    return newsky_ss
#%%
class small_scale(object):
    '''
    This class summarize the functions above to synthesize a random map\n
    '''
    def __init__(self, ra0, dec0, dx, dy, nx, ny):
        self.ra0 = ra0
        self.dec0 = dec0
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny
        self.ra, self.dec, (self.x, self.y) = get_grid(ra0, dec0, dx, nx, dy, ny)
        self.edges = [self.x[0]-dx/2.0, self.x[-1]+dx/2.0, self.y[0]-dy/2.0, self.y[-1]+dy/2.0]
        self.skymap = None
        self.k_fid = None
        self.dk = None

    def set_skymap(self, skymap, fwhm=None):
        if fwhm is not None:
            skymap = hp.smoothing(skymap, fwhm=fwhm)
        self.skymap = hp.get_interp_val(skymap, self.ra, self.dec, lonlat=True)
        return self.skymap, skymap

    def set_k(self, k_fid, dk, win=None, k_offset=None):
        self.k_fid = k_fid
        self.dk = dk
        if k_offset is None:
            k_offset = dk
        self.k_offset = k_offset
        self.win = win

    def get_extrapolate_pk(self, beta):
        self.beta = beta
        self.pk2d_extrap, (self.k2d, self.pk2d_input) = get_pk_ss(self.skymap, self.dx, self.dy, self.k_fid, self.dk, beta, win=self.win)


    def show_pk_extrap(self, kbins, auto_scale=True):
        k1, pk1 = pk2d_to_pk1d(self.k2d, self.pk2d_input, kbins=kbins)
        _, pk2 = pk2d_to_pk1d(self.k2d, self.pk2d_extrap, kbins=kbins)
        plt.figure()
        plt.loglog(k1, pk1+pk2, label='total')
        plt.loglog(k1, pk1, '--', label='input')
        plt.loglog(k1, pk2, '--', label='extrapolate')
        plt.xlabel('k/rad^-1')
        plt.ylabel('P(k)')
        plt.legend()
        if auto_scale:
            ymax = pk1[0]
            ymin = pk2[-1]
            plt.ylim(ymin/10.0, ymax*10.0)
        return k1, pk1, pk2

    def syn_sky(self, gamma=1.0):
        skymap_ss = get_random(self.pk2d_extrap, (self.dx, self.dy))
        if gamma != 0:
            skymap_ss_rescale = rescale_map(skymap_ss, self.skymap, self.dx, self.dy, self.k_fid+self.k_offset, self.dk, gamma=gamma, win=self.win)
            self.skymap_ss = skymap_ss_rescale
        else:
            self.skymap_ss = skymap_ss
        self.skymap_ss_uniform = skymap_ss

    def show_syn(self):
        plt.figure()
        plt.imshow(self.skymap.T, origin='lower', extent=self.edges, aspect='auto')
        plt.colorbar()
        plt.figure()
        plt.imshow(self.skymap_ss.T, origin='lower', extent=self.edges, aspect='auto')
        plt.colorbar()
        plt.figure()
        plt.imshow((self.skymap+self.skymap_ss).T, origin='lower', extent=self.edges, aspect='auto')
        plt.colorbar()

    def show_syn_pk(self, kbins, auto_scale=True):
        k1, pk1 = pk2d_to_pk1d(self.k2d, self.pk2d_input, kbins=kbins)
        _, pk2 = pk2d_to_pk1d(self.k2d, self.pk2d_extrap, kbins=kbins)
        ksyn, pksyn = get_pk1d_simple(self.skymap_ss, self.dx, self.dy, kbins=kbins, win=self.win)
        kall, pkall = get_pk1d_simple(self.skymap+self.skymap_ss, self.dx, self.dy, kbins=kbins, win=self.win)
        plt.figure()
        plt.loglog(k1, pk1, label='input')
        plt.loglog(ksyn, pksyn, label='syn')
        plt.loglog(kall, pkall, label='total')
        plt.xlabel('k/rad^-1')
        plt.ylabel('P(k)')
        plt.legend()
        if auto_scale:
            ymax = pk1[0]
            ymin = pk2[-1]
            plt.ylim(ymin/10.0, ymax*10.0)

#%%
# test the pipeline to make small scale structure
if __name__ == '__main__':
    ra0 = 120.0
    dec0 = 40.0
    dx = 0.1
    dy = 0.1
    nx = 512
    ny = 512
    k_fid = 60
    dk = 20
    kbins = np.logspace(0.5, 4, 31)
    win = np.blackman(nx)[:,None]*np.blackman(ny)
    #%%
    with h5.File('./gsm2016_file.hdf5', 'r') as filein:
        skymap = filein['highres_Synchrotron_map'][:]
    skymap = hp.reorder(skymap, n2r=True)
    #%%
    ss = small_scale(ra0, dec0, dx, dy, nx, ny)
    #%%
    ss.set_skymap(skymap, fwhm=np.deg2rad(0.5));
    #ss.set_skymap(skymap);
    #%%
    skymap0 = hp.get_interp_val(skymap, ss.ra, ss.dec, lonlat=True)
    #%%
    plt.figure()
    plt.loglog(*get_pk1d_simple(ss.skymap, ss.dx, ss.dy, win=win, kbins=kbins))
    plt.loglog(*get_pk1d_simple(skymap0, ss.dx, ss.dy, win=win, kbins=kbins))
    #%%
    plt.figure()
    plt.imshow(ss.skymap)
    plt.colorbar()
    plt.figure()
    plt.imshow(skymap0)
    plt.colorbar()
    plt.figure()
    plt.imshow(win)
    #%%
    ss.set_k(k_fid, dk, win=win)
    ss.get_extrapolate_pk(2.0)
    k1, pk1, pk2 = ss.show_pk_extrap(kbins);
    #%%
    ss.syn_sky(gamma=1)
    ss.show_syn()
    ss.show_syn_pk(kbins)
    #%%
    # whether the region is correct
    plt.figure()
    plt.imshow(ss.ra.T, origin='lower', extent=ss.edges)
    plt.colorbar()
    plt.figure()
    plt.imshow(ss.dec.T, origin='lower', extent=ss.edges)
    plt.colorbar()
    #%%
    plt.figure()
    plt.imshow(ss.skymap.T, origin='lower', extent=ss.edges)
    plt.colorbar()
    # NOTE our projection is not exactly the same, and there is some distortion
    plt.figure()
    hp.cartview(skymap, lonra=[ra0-dx*nx//2, ra0+dx*nx//2], latra=[dec0-dy*ny//2, dec0+dy*ny//2], flip='geo')
#%%
#%%
#%%
#%%
#%%
# test the individual function
if __name__ == '__main__':
    with h5.File('./gsm2016_file.hdf5', 'r') as filein:
        skymap = filein['highres_Synchrotron_map'][:]
    skymap = hp.reorder(skymap, n2r=True)
    skymap_s = hp.smoothing(skymap, fwhm=np.deg2rad(0.5))
    #%%
    ra0 = 60
    dec0 = 40.0
    dx = 0.2
    dy = 0.2
    nx = 256
    ny = 256
    l_fid = 60
    n_l = 20

    ra, dec, _ = get_grid(ra0, dec0, dx, nx, dy, ny)
    skymap_grid = hp.get_interp_val(skymap, ra, dec, lonlat=True)
    skymap_grid_s = hp.get_interp_val(skymap_s, ra, dec, lonlat=True)
    #%%
    plt.figure()
    plt.imshow(skymap_grid)
    plt.figure()
    plt.imshow(skymap_grid_s)
    #%%
    win = np.blackman(nx)[:,None]*np.blackman(ny)
    pk2d, (k2d, pk2d_input) = get_pk_ss(skymap_grid, dx, dy, l_fid, n_l, 2.0)
    pk2d_s, (k2d_s, pk2d_input_s) = get_pk_ss(skymap_grid_s, dx, dy, l_fid, n_l, 2.0)
    pk2d_w, (k2d_w, pk2d_input_w) = get_pk_ss(skymap_grid, dx, dy, l_fid, n_l, 2.0, win=win)
    pk2d_s_w, (k2d_s_w, pk2d_input_s_w) = get_pk_ss(skymap_grid_s, dx, dy, l_fid, n_l, 2.0, win=win)
    #%%
    k1, pk1 = pk2d_to_pk1d(k2d, pk2d_input, kbins=50)
    _, pk2 = pk2d_to_pk1d(k2d, pk2d, kbins=50)
    k1_s, pk1_s = pk2d_to_pk1d(k2d_s, pk2d_input_s, kbins=50)
    _, pk2_s = pk2d_to_pk1d(k2d_s, pk2d_s, kbins=50)
    k1_w, pk1_w = pk2d_to_pk1d(k2d_w, pk2d_input_w, kbins=50)
    _, pk2_w = pk2d_to_pk1d(k2d_w, pk2d_w, kbins=50)
    k1_s_w, pk1_s_w = pk2d_to_pk1d(k2d_s_w, pk2d_input_s_w, kbins=50)
    _, pk2_s_w = pk2d_to_pk1d(k2d_s_w, pk2d_s_w, kbins=50)
    #%%
    plt.figure()
    plt.loglog(k1, pk1, 'b')
    plt.loglog(k1, pk2, 'r')
    plt.loglog(k1, pk1+pk2, 'k')
    plt.loglog(k1_s, pk1_s, 'b--')
    plt.loglog(k1_s, pk2_s, 'r--')
    plt.loglog(k1_s, pk1_s+pk2_s, 'k--')
    plt.figure()
    plt.loglog(k1, pk1, 'g.')
    plt.loglog(k1_w, pk1_w, 'b')
    plt.loglog(k1_w, pk2_w, 'r')
    plt.loglog(k1_w, pk1_w+pk2_w, 'k')
    plt.loglog(k1_s_w, pk1_s_w, 'b--')
    plt.loglog(k1_s_w, pk2_s_w, 'r--')
    plt.loglog(k1_s_w, pk1_s_w+pk2_s_w, 'k--')
    #%%
    skymap_ss = get_random(pk2d_w, (dx, dy))
    #skymap_ss = get_random(np.ones_like(pk2d, dtype=np.float64), (dx, dy))
    #%%
    k_new, pk_new = get_pk1d_simple(skymap_ss, dx, dy, win=win)
    plt.plot(k_new, pk_new)
    #%%
    plt.imshow(skymap_ss.T)
    plt.colorbar()
    #%%
    k_new, pk_new = get_pk1d_simple(skymap_ss+skymap_grid, dx, dy, win=win)
    #k_new, pk_new = get_pk1d_simple(skymap_ss, dx, dy)
    #%%
    plt.loglog(k_new, pk_new)
    plt.loglog(k1_w, pk2_w)
    plt.loglog(k1_w, pk1_w)
    plt.loglog(k1_w, pk1_w+pk2_w)
    #%%
    skymap_ss_rescale = rescale_map(skymap_ss, skymap_grid, dx, dy, l_fid, n_l, gamma=1.0, win=win)
    #%%
    plt.figure()
    plt.imshow(skymap_ss_rescale)
    plt.colorbar()
    plt.figure()
    plt.imshow(skymap_ss)
    plt.colorbar()
    plt.figure()
    plt.imshow(skymap_grid)
    plt.colorbar()
    #%%
    plt.loglog(*get_pk1d_simple(skymap_ss_rescale+skymap_grid, dx, dy, win=win))
    plt.loglog(*get_pk1d_simple(skymap_ss+skymap_grid, dx, dy, win=win))
    plt.loglog(*get_pk1d_simple(skymap_grid, dx, dy, win=win))
#%%
#%%
#%%
# test the relation between C_l and P(k)
if __name__ == '__main__':
    lmax = 800
    beta = 2.0
    nside = 512
    l = np.arange(lmax+1)
    cl = np.zeros(l.shape, dtype=np.float64)
    cl[1:] = l[1:]**-beta
    skymap = hp.synfast(cl, nside)
    skymap_s = hp.smoothing(skymap, fwhm=np.deg2rad(0.5))
    #%%
    ra0 = 60
    dec0 = 40.0
    dx = 0.1
    dy = 0.1
    nx = 512
    ny = 512
    ra, dec, _ = get_grid(ra0, dec0, dx, nx, dy, ny)
    newmap = hp.get_interp_val(skymap, ra, dec, lonlat=True)
    newmap_s = hp.get_interp_val(skymap_s, ra, dec, lonlat=True)
    plt.figure()
    plt.imshow(newmap)
    plt.figure()
    plt.imshow(newmap_s)
    #%%
    win = np.blackman(newmap.shape[0])[:,None]*np.blackman(newmap.shape[-1])
    k1d, pk1d = get_pk1d_simple(newmap, dx, dy)
    k1d_w, pk1d_w = get_pk1d_simple(newmap, dx, dy, win=win)
    k1d_s, pk1d_s = get_pk1d_simple(newmap_s, dx, dy)
    k1d_w_s, pk1d_w_s = get_pk1d_simple(newmap_s, dx, dy, win=win)
    #%%
    plt.figure()
    plt.loglog(k1d, pk1d)
    plt.loglog(k1d_w, pk1d_w)
    plt.loglog(k1d_s, pk1d_s)
    plt.loglog(k1d_w_s, pk1d_w_s)
    plt.loglog(k1d, k1d**-beta)
#%%
#%%
#%%
#%%



