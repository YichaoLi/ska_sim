#%%
import numpy as np
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from matplotlib import pyplot as plt
import h5py as h5
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
    def __init__(self, gsm_data_file, interpolator=lambda x, y: UnivariateSpline(x, y, s=0, k=2)):
        self.labels = ['Synchrotron', 'CMB', 'HI', 'Dust1', 'Dust2', 'Free-Free']
        with h5.File(gsm_data_file, 'r') as filein:
            spec_nf = filein['spectra'][:]
        self.pca_freqs_ghz = spec_nf[0]
        self.pca_scaling   = spec_nf[1]
        self.pca_comps     = spec_nf[2:]
        self.n_comps = len(self.pca_comps)
        self.ln_pca_freqs = np.log(self.pca_freqs_ghz)
        self.spl_lnscaling = interpolator(self.ln_pca_freqs, np.log(self.pca_scaling))
        self.spl_comps = []
        for ii in range(self.n_comps):
            self.spl_comps.append(interpolator(self.ln_pca_freqs, self.pca_comps[ii]))

    def interpolate_freq(self, freq):
        '''
        freq in MHz\n
        return\n
            comps: (n_comps, n_freq), interpolated amplitude of comps
        '''
        lnfreq = np.log(np.asarray(freq)/1.e3)
        scling = np.exp(self.spl_lnscaling(lnfreq))
        comps = np.zeros((self.n_comps,)+lnfreq.shape, dtype=np.float64)
        for ii in range(self.n_comps):
            comps[ii] = self.spl_comps[ii](lnfreq)*scling
        return comps

    def interpolate_skymap(self, skymap_comps, freq, T_unit='TCMB', include_cmb=False):
        '''
        skymap_comps: (n_comps, N1, N2, ...) the sky map for different components, in MJy/sr, which is the default unit for GSM\n
        freq: (n_freq, ) in MHz
        include_cmb: whether to include the CMB. Defaults to False. A value of T_CMB = 2.725 K is used.
        return\n
            skymap: (N1, N2, ..., n_freq) in T_unit\n
        '''
        if T_unit not in ['MJysr', 'TCMB', 'TRJ']:
            raise RuntimeError("UNIT ERROR: %s not supported. Only MJysr, TCMB, TRJ are allowed." % T_unit)

        freq = np.asarray(freq)
        skymap_comps = np.asarray(skymap_comps)
        shp = skymap_comps.shape
        assert shp[0] == self.n_comps
        skymap_comps = skymap_comps.reshape(shp[0], -1) # n_comps, n_pix
        amp = self.interpolate_freq(freq)
        skymap = skymap_comps.T@amp # n_pix, n_freq
        if not include_cmb:
            skymap =  skymap - K_CMB2MJysr(_TCMB, 1e6 * freq)
        if T_unit == 'TCMB':
            skymap = skymap / K_CMB2MJysr(1., 1e6 * freq)
        elif T_unit == 'TRJ':
            skymap = skymap / K_RJ2MJysr(1., 1e6 * freq)
        skymap = skymap.reshape(*shp[1:], -1)
        return skymap
#%%