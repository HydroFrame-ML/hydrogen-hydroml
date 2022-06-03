from .scalers import MinMaxScaler

TAYLOR_DEFAULTS = {
    'precipitation': (
        MinMaxScaler,
        {'x_min': 0.0, 'x_max': 2.3250430690836765, 'feature_range': (0,1)}
    ),
    'temperature': (
        MinMaxScaler,
        {'x_min': 243.04002434890052, 'x_max': 294.7861684163411, 'feature_range': (0,1)}
    ),
    'wind velocity': (
        MinMaxScaler,
        {'x_min': -9.555846213189705, 'x_max': 15.570362051328024, 'feature_range': (-1,1)}
    ),
    'UGRD': (
        MinMaxScaler,
        {'x_min': -9.555846213189705, 'x_max': 15.570362051328024, 'feature_range': (-1,1)}
    ),
    'VGRD': (
        MinMaxScaler,
        {'x_min': -9.459658953082725, 'x_max': 10.973674557960384, 'feature_range': (-1,1)}
    ),
    'VGRD': (
        MinMaxScaler,
        {'x_min': -9.459658953082725, 'x_max': 10.973674557960384, 'feature_range': (-1,1)}
    ),
    'atmospheric pressure': (
        MinMaxScaler,
        {'x_min': 62042.070149739586, 'x_max': 75857.17057291667, 'feature_range': (0,1)}
    ),
    'specific humidity': (
        MinMaxScaler,
        {'x_min': 0.00034727433576045206, 'x_max': 0.010973041411489248, 'feature_range': (0,1)}
    ),
    'downward shortwave radiation': (
        MinMaxScaler,
        {'x_min': 44.300002892812095, 'x_max': 341.0555042037224, 'feature_range': (0,1)}
    ),
    'downward longwave radiation': (
        MinMaxScaler,
        {'x_min': 93.16834259771652, 'x_max': 356.44537989298504, 'feature_range': (0,1)}
    ),
    'saturation': (
        MinMaxScaler,
        {'x_min': 0.006598270240534006, 'x_max': 1.0, 'feature_range': (0,1)}
    ),
    'pressure': (
        MinMaxScaler,
        {'x_min': -12.673498421769821, 'x_max': 53.29893832417906, 'feature_range': (0,1)}
    ),
    'soil_moisture': (
        MinMaxScaler,
        {'x_min': 0.0025535305830866597, 'x_max': 0.48199999999999993, 'feature_range': (0,1)}
    ),
    'wtd': (
        MinMaxScaler,
        {'x_min': 0.0, 'x_max': 54.86268596956115, 'feature_range': (0,1)}
    ),
    'eflx_lh_tot': (
        MinMaxScaler,
        {'x_min': -11.987887556301255, 'x_max': 227.2745242502459, 'feature_range': (-1,1)}
    ),
    'eflx_lwrad_out': (
        MinMaxScaler,
        {'x_min': 188.73408048992417, 'x_max': 428.3168634458776, 'feature_range': (0,1)}
    ),
    'eflx_sh_tot': (
        MinMaxScaler,
        {'x_min': -212.63476598064483, 'x_max': 231.30395973560096, 'feature_range': (0,1)}
    ),
    'eflx_soil_grnd': (
        MinMaxScaler,
        {'x_min': -225.05620842421095, 'x_max': 190.92417048181622, 'feature_range': (0,1)}
    ),
    'qflx_evap_tot': (
        MinMaxScaler,
        {'x_min': -0.017156259474353067, 'x_max': 0.3255699677768812, 'feature_range': (-1,1)}
    ),
    'qflx_evap_grnd': (
        MinMaxScaler,
        {'x_min': 0.0, 'x_max': 0.14114173688490758, 'feature_range': (0,1)}
    ),
    'qflx_evap_soi': (
        MinMaxScaler,
        {'x_min': -0.03406682732982543, 'x_max': 0.14114173688490758, 'feature_range': (-1,1)}
    ),
    'qflx_evap_veg': (
        MinMaxScaler,
        {'x_min': -0.017161818440162336, 'x_max': 0.3219445254210888, 'feature_range': (-1,1)}
    ),
    'qflx_tran_veg': (
        MinMaxScaler,
        {'x_min': 0.0, 'x_max': 0.1636195226512655, 'feature_range': (0,1)}
    ),
    'qflx_infl': (
        MinMaxScaler,
        {'x_min': -0.14098597960181578, 'x_max': 1.7733137195552644, 'feature_range': (-1,0)}
    ),
    'swe_out': (
        MinMaxScaler,
        {'x_min': 0.0, 'x_max': 754.746199964657, 'feature_range': (0,1)}
    ),
    't_grnd': (
        MinMaxScaler,
        {'x_min': 239.6187892890265, 'x_max': 298.3340490161149, 'feature_range': (0,1)}
    ),
    't_soil': (
        MinMaxScaler,
        {'x_min': 273.1600036621094, 'x_max': 298.3340490161149, 'feature_range': (0,1)}
    ),
    'computed_porosity': (
        MinMaxScaler,
        {'x_min': 0.33, 'x_max': 0.482, 'feature_range': (0,1)}
    ),
    'porosity': (
        MinMaxScaler,
        {'x_min': 0.33, 'x_max': 0.482, 'feature_range': (0,1)}
    ),
    'slope_x': (
        MinMaxScaler,
        {'x_min': -0.40505003929138184, 'x_max': 0.4567600190639496, 'feature_range': (-1,1)}
    ),
    'slope_y': (
        MinMaxScaler,
        {'x_min': -0.3405400514602661, 'x_max': 0.46806982159614563, 'feature_range': (-1,1)}
    ),
    'computed_permeability': (
        MinMaxScaler,
        {'x_min': 0.004675077, 'x_max': 0.06, 'feature_range': (0,1)}
    ),
    'permeability': (
        MinMaxScaler,
        {'x_min': 0.004675077, 'x_max': 0.06, 'feature_range': (0,1)}
    )
}
