from collections.abc import Mapping

import numpy as np
import pandas as pd
import re

from sheap.SheaProducts.Utils.TableUtils import TablExtractor,combine_table_result


def sheaposterior_as_table(sheapspectral, low=0.16, high=0.84, method=None,selected_index = [],calculate_host=True,calculate_fe=True ,
                      continuum_wavelengths = [2500.0, 3000.0, 4200.0, 5100.0,6200.0,],fe_windows = {"FE_UV": (2250.0, 2650.0),"FE_OPT": (4434.0, 4684.0),}
                      , merge_columns=("obj_name", "n_obj"),calculate_lum=True,calculate_extraproducts=True,verbose=True):
    if len(selected_index)==0:
        coords = sheapspectral.coords
    else:
        coords = sheapspectral.coords[selected_index]
    coords = np.atleast_2d(coords)
    (df_extra,df_cont,df_basic,df_chi) = TablExtractor(sheapspectral, low=low, high=high, method=method,
                                                       selected_index = selected_index,calculate_host=calculate_host,calculate_fe=calculate_fe
                                                        ,continuum_wavelengths = continuum_wavelengths,fe_windows = fe_windows, calculate_lum=calculate_lum,
                                                        calculate_extraproducts=calculate_extraproducts,verbose=verbose)
    
    return  combine_table_result(df_chi, df_cont, df_basic, df_extra, coords=coords,merge_columns=merge_columns)
