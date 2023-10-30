from ska_sim import plot_map as pm
import numpy as np
from ps_est import algebra as al
import astropy.units as u
import h5py as h5


def stack_maps(hi_file, hi_key='delta_T', hi_unit='mK',
                 ps_file=None, ps_key='ps_map', 
                 fg_file=None, fg_key='ps_map', 
                 output_path=None, pad=None, suffix='imap'):
    
    suffix += '_HI'
    
    hi_map_sets = pm.load_map(hi_file, hi_key)
    
    
    #pad = 300
    imap_combined  = hi_map_sets[0]
    ra_n, dec_n = imap_combined.shape[1:]
    _info = hi_map_sets[0].info
    _info['unit'] = hi_unit
    
    pad_shp = ((0, 0), (pad, pad), (pad, pad))
    imap_combined = np.pad(imap_combined, pad_shp, mode='wrap')
    
    if ps_file != None:
        suffix += '_PS'
        _ps_map = pm.load_map(ps_file, ps_key)[0]
        _ps_map_unit = _ps_map.info['unit']
        unit_conv = 1.
        if _ps_map_unit != hi_unit:
            print('convert from unit of %s to mK'%_ps_map_unit)
            unit_conv = getattr(u, _ps_map_unit).to(getattr(u, hi_unit))
        #print(unit_conv)
        #print(_ps_map.info)
        imap_combined += _ps_map * unit_conv
        
    if fg_file != None:
        suffix += '_FG'
        _fg_map = pm.load_map(fg_file, fg_key)[0]
        _fg_map_unit = _fg_map.info['unit']
        unit_conv = 1.
        if _fg_map_unit != hi_unit:
            print('convert from unit of %s to mK'%_fg_map_unit)
            unit_conv = getattr(u, _fg_map_unit).to(getattr(u, hi_unit))
        imap_combined += _fg_map * unit_conv
    
    imap_combined = al.info_array(imap_combined)
    imap_combined.info = _info
    imap_combined = al.make_vect(imap_combined, axis_names=_info['axes'])
    
    if output_path is not None:
        _output = output_path + '%s.h5'%suffix
        with h5.File(_output, 'w') as fp:
            al.save_h5(fp, 'imap', imap_combined)
    else:
        return imap_combined
