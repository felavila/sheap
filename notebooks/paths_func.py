import glob
import pandas as pd 
import os 

import jax 
from jax import jit 
import jax.numpy as jnp
import glob 
import os 
import pandas as pd 
import matplotlib.pyplot as plt 

###
#Halpha 5600,7300
#Hbeta 4400, 5600
#MII 2500, 3000
#cIII 1600, 2000
#CIV 1100, 2000
#La 1000,1500
####
def cross_pandas_spectra(path_dr16,path_data,name_csv):
    """
        Small code to cross math objects in a panda with the actual spectra in certain directory 
    """
    #Objects_low_stellar_mass_with_stellar_contribution.csv
    #bernal_sdss_fits
    file_paths = glob.glob(f"{path_dr16}/{path_data}/*.fits")
    objs_panda = pd.read_csv(f"{path_dr16}/{name_csv}")
    objs_panda["dr_name"] = [f"{PLATE:04d}-{MJD:05d}-{FIBERID:04d}" for PLATE,MJD,FIBERID in objs_panda[["PLATE","MJD","FIBERID"]].values]
    #return file_paths,objs_panda
    objs_panda_paths_list = [os.path.basename(path).replace(".fits", "") for path in file_paths]
    objs_panda_paths_filtered = objs_panda[objs_panda["dr_name"].isin(objs_panda_paths_list)].reset_index(drop=True)
    if len(objs_panda_paths_filtered)==0:
        raise "Check the paths your panda is empty"
    else:
        print(f"You cross match found {len(objs_panda_paths_filtered)}")
    objs_panda_paths_filtered["fit_path"] = objs_panda_paths_filtered["dr_name"].apply(lambda x: os.path.join(path_dr16,f"{path_data}/{x}.fits")).values
    return file_paths,objs_panda_paths_filtered



# Objects_low_stellar_mass_with_stellar_contribution_all = pd.read_csv("Objects_low_stellar_mass_with_stellar_contribution_slope_all_130225.csv")
# paths = glob.glob(f"{path_dr16}/bernal_sdss_fits/*.fits")
# dr = pd.read_csv(f"{path_dr16}/Objects_low_stellar_mass_with_stellar_contribution.csv")#[0:10]
# dr["dr_name"] = [f"{PLATE:04d}-{MJD:05d}-{FIBERID:04d}" for PLATE,MJD,FIBERID in dr[["PLATE","MJD","FIBERID"]].values]
# dr_paths = [os.path.basename(path).replace(".fits", "") for path in paths]
# dr_filtered = dr[dr["dr_name"].isin(dr_paths)].reset_index(drop=True)
# dr_filtered["fit_path"] = dr_filtered["dr_name"].apply(lambda x: os.path.join(path_dr16,f"bernal_sdss_fits/{x}.fits")).values

#from paths_func import cross_pandas_spectra