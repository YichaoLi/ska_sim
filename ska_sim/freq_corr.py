#%%
import numpy as np
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from matplotlib import pyplot as plt
import h5py as h5
import healpy as hp
from tqdm import tqdm
from scipy.linalg import eigh
#%%
_kB = 1.38065e-23
_C = 2.99792e8
_h = 6.62607e-34
_TCMB = 2.725
_hoverk = _h / _kB

def K_CMB2MJysr(K_CMB, nu):#in Kelvin and Hz
    B_nu = 2 * (_h * nu)* (nu / _C)**2 / (np.exp(_hoverk * nu / _TCMB) - 1)
    conversion_factor = (B_nu * _C / nu / _TCMB)**2 / 2 * np.exp(_hoverk * nu / _TCMB) / _kB
    return  K_CMB * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

def K_RJ2MJysr(K_RJ, nu):#in Kelvin and Hz
    conversion_factor = 2 * (nu / _C)**2 * _kB
    return  K_RJ * conversion_factor * 1e20#1e-26 for Jy and 1e6 for MJy

#%%
class GSM_interpolate(object):
    def __init__(self, gsm_data_file, interpolator=lambda x, y: UnivariateSpline(x, y, s=0, k=2), load_maps=True):
        with h5.File(gsm_data_file, 'r') as filein:
            spec_nf = filein['spectra'][:]
            self.labels = ['Synchrotron', 'CMB', 'HI', 'Dust1', 'Dust2', 'Free-Free']
            if load_maps:
                self.maps = np.array([filein['highres_%s_map'%lb][:] for lb in self.labels])
                self.maps = hp.reorder(self.maps, n2r=True)
            else:
                self.maps = None
        self.pca_freqs_ghz = spec_nf[0]
        self.pca_scaling   = spec_nf[1]
        self.pca_comps     = spec_nf[2:]
        self.n_comps = len(self.pca_comps)
        self.ln_pca_freqs = np.log(self.pca_freqs_ghz)
        self.spl_lnscaling = interpolator(self.ln_pca_freqs, np.log(self.pca_scaling))
        self.spl_comps = dict()
        for ii in range(self.n_comps):
            self.spl_comps[self.labels[ii]] = interpolator(self.ln_pca_freqs, self.pca_comps[ii])

    def interpolate_freq(self, freq, labels):
        '''
        freq in MHz\n
        return\n
            comps: (n_comps, n_freq), interpolated amplitude of comps
        '''
        n_comps = len(labels)
        lnfreq = np.log(np.asarray(freq)/1.e3)
        scaling = np.exp(self.spl_lnscaling(lnfreq))
        comps = np.zeros((n_comps,)+lnfreq.shape, dtype=np.float64)
        for ii in range(n_comps):
            comps[ii] = self.spl_comps[labels[ii]](lnfreq)*scaling
        return comps

    def interpolate_skymap(self, skymap_comps, freq, labels=None, data_unit='TCMB', include_cmb=True):
        '''
        skymap_comps: (n_comps, N1, N2, ...) the sky map for different components, in MJy/sr, which is the default unit for GSM\n
        freq: (n_freq, ) in MHz
        include_cmb: whether to include the CMB. Defaults to False. A value of T_CMB = 2.725 K is used.
        return\n
            skymap: (N1, N2, ..., n_freq) in data_unit\n
        '''
        if labels is None:
            labels = self.labels

        if data_unit not in ['MJysr', 'TCMB', 'TRJ']:
            raise RuntimeError("UNIT ERROR: %s not supported. Only MJysr, TCMB, TRJ are allowed." % data_unit)

        freq = np.asarray(freq)
        try:
            assert freq.min() >= 10
            assert freq.max() <= 5000e3
        except AssertionError:
            raise RuntimeError("Frequency values lie outside 10 MHz < f < 5 THz")

        skymap_comps = np.asarray(skymap_comps)
        shp = skymap_comps.shape
        n_comps = len(labels)
        assert shp[0] == n_comps
        skymap_comps = skymap_comps.reshape(shp[0], -1) # n_comps, n_pix
        amp = self.interpolate_freq(freq, labels)
        skymap = skymap_comps.T@amp # n_pix, n_freq
        if not include_cmb:
            skymap =  skymap - K_CMB2MJysr(_TCMB, 1e6 * freq)
        if data_unit == 'TCMB':
            skymap = skymap / K_CMB2MJysr(1., 1e6 * freq)
        elif data_unit == 'TRJ':
            skymap = skymap / K_RJ2MJysr(1., 1e6 * freq)
        skymap = skymap.reshape(*shp[1:], -1)
        return skymap
#%%
def SCK_freq_corr(freq, freq_ref, alpha, zeta):
    x = np.asarray(freq)/freq_ref
    y1 = x**-alpha
    return y1[:,None] * y1 * np.exp(-np.log(x[:,None]/x)**2/2./zeta**2)

def get_proj(cov):
    cov = np.asarray(cov)
    w, v = eigh(cov)
    w[w<0] = 0.0
    return v*np.sqrt(w), (w, v)

def syn_spec(fields, freq, freq_ref, alpha, zeta):
    fields = np.asarray(fields)
    freq = np.asarray(freq)
    assert fields.shape[-1] == freq.shape[0]
    shp = fields.shape
    fields = fields.reshape(-1, shp[-1])
    cov = SCK_freq_corr(freq, freq_ref, alpha, zeta)
    K, _ = get_proj(cov)
    fields = fields @ K.T
    fields = fields.reshape(*shp[:-1], -1)
    return fields
#
#cov = []
#cov.append([1.0, 0.1, 0.0])
#cov.append([0.1, 2.0, 0.2])
#cov.append([0.0, 0.2, 0.5])
#cov = np.array(cov)
#print(cov)
#K, _ = get_proj(cov)
#x = np.random.randn(10000, K.shape[0])
#y = x@K.T
#dy = y - y.mean(axis=0, keepdims=True)
#cov_y = (y.T@y)/y.shape[0]
#print(cov_y)
#
#%%
#%%
##%%
#from small_scale import small_scale, get_random, get_pk1d_simple
#ra0 = 0.0
#dec0 = -30.0
#dx = 16.0/3600.0
#dy = 16.0/3600.0
#nx = 2048
#ny = 2048
#n_freq = 901
#dfreq = 0.1
#freq0 = 106.0
##%%
#k_fid = 60
#dk = 40
#kbins = np.logspace(0.5, 4, 31)
#win = np.blackman(nx)[:,None]*np.blackman(ny)
#freq = np.arange(n_freq) * dfreq + freq0
##%%
#gsm = GSM_interpolate('./gsm2016_file.hdf5')
##%%
#ss = small_scale(ra0, dec0, dx, dy, nx, ny)
##%%
#skytemps = []
#for ii in tqdm(range(gsm.n_comps)):
#    ss.set_skymap(gsm.maps[ii], fwhm=None);
#    skytemps.append(ss.skymap)
##%%
#skymaps0 = []
#for ii in tqdm(range(gsm.n_comps)):
#    skymaps0.append(gsm.interpolate_skymap(skytemps[ii:ii+1], freq, labels=[gsm.labels[ii]]))
##%%
#plt.figure()
#for ii in tqdm(range(gsm.n_comps)):
#    plt.plot(freq, skymaps0[ii].mean(axis=(0,1)), label=gsm.labels[ii])
#plt.legend()
##%%
##%%
#beta = dict()
#beta['Synchrotron'] = 2.4
#beta['HI'] = 3.0
#fwhm = np.deg2rad(0.25)
##%%
#skymaptemp_ss = []
#for label in beta:
#    imap = gsm.labels.index(label)
#    ss.set_skymap(gsm.maps[imap], fwhm=fwhm);
#    ss.set_k(k_fid, dk, win=win)
#    ss.get_extrapolate_pk(beta[label])
#    k1, pk1, pk2 = ss.show_pk_extrap(kbins);
#    ss.syn_sky(gamma=1)
#    ss.show_syn()
#    ss.show_syn_pk(kbins)
#    skymaptemp_ss.append(ss.skymap_ss)
#    plt.show()
##%%
#skymaps_ss = []
#for ii in tqdm(range(len(skymaptemp_ss))):
#    skymaps_ss.append(gsm.interpolate_skymap(skymaptemp_ss[ii:ii+1], freq, labels=[list(beta.keys())[ii]]))
##%%
#with h5.File('foreground_patch_sky.hdf5', 'w') as filein:
#    filein.attrs['ra0'] = ra0
#    filein.attrs['dec0'] = dec0
#    filein.attrs['dx'] = dx
#    filein.attrs['dy'] = dy
#    filein['ra'] = ss.ra
#    filein['dec'] = ss.dec
#    filein['x'] = ss.x
#    filein['y'] = ss.y
#    filein['freq'] = freq
#    filein.create_group('GSM_smooth')
#    for ii in tqdm(range(gsm.n_comps)):
#        filein['GSM_smooth'][gsm.labels[ii]] = skymaps0[ii]
#    filein.create_group('GSM_ss')
#    for ii in tqdm(range(len(skymaps_ss))):
#        filein['GSM_ss'][list(beta.keys())[ii]] = skymaps_ss[ii]
##%%
##%%
##%%
##%%
##%%
#gsm = GSM_interpolate('./gsm2016_file.hdf5')
#freq = np.linspace(106, 196, 21)
##%%
#skymaps = []
#for ii in range(len(gsm.labels)):
#    skymaps.append(gsm.interpolate_skymap([gsm.maps[ii]], freq, labels=[gsm.labels[ii]]))
##%%
#plt.figure()
#for ii in range(len(skymaps)):
#    plt.plot(skymaps[ii].mean(axis=0), label=gsm.labels[ii])
#plt.legend()
##%%
##%%
#