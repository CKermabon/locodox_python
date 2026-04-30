import numpy as np
import xarray as xr
import os
from datetime import date, datetime
from m_users_fonctions import cherche_info_ctd_ref
import matplotlib.colors as mcolors

fic_txt = '/Users/chemon/ARGO_NEW/NEW_LOCODOX/locodox_python/BDD_TR/bdd_REF_ARGO.txt'
fic_mat = '/Users/chemon/ARGO_NEW/NEW_LOCODOX/locodox_python/BDD_TR/bddo2ref_all_2025.mat'
rep_data_argo = '/Users/chemon/tmp/' #'/Volumes/argo/gdac/dac/coriolis/'
num_float = '6903035'

# CTD comparison to estimate a supplement gain.
##################################################
# cmp_ctd = 1 : we used CTD, 0 otherwise
cmp_ctd = 1
fic_txt = '/Users/chemon/ARGO_NEW/NEW_LOCODOX/locodox_python/BDD_TR/bdd_REF_ARGO.txt'
fic_mat = '/Users/chemon/ARGO_NEW/NEW_LOCODOX/locodox_python/BDD_TR/bddo2ref_all_2025.mat'
if cmp_ctd==1:
    num_ctd,num_cycle,cruise_name, ds_cruise = cherche_info_ctd_ref(fic_txt,fic_mat,num_float)
else:
    num_ctd = []
    num_cycle = []


# Cycle to use to estimate the correction
first_cycle_to_use = 1
last_cycle_to_use = 113

# Piecewise or not
test_piece = 1 
# Compute automatically NCEP breakpoint or force it
compute_NCEP_breakpoint=1
if test_piece==1:
    nb_segment_WOA = 2
# Compute automatically NCEP breakpoint or force it
    if compute_NCEP_breakpoint==1:
        breakpoint_NCEP_user = 18 #delta_T_NCEP[7] # To be 
        nb_segment_NCEP=2
    else:
        nb_segment_NCEP=2

# Results Directory.
# A subdirectory named numfloat_date will be created for each run in the result directory.
# The ASCII file will be created in it.
# The B corrected files and plots will be created in a subdirectory (nc and plot subdirectory)
###################################################
rep_fic_res = '/Users/chemon/ARGO_NEW/NEW_LOCODOX/locodox_python/fic_res_janvier2026'
now = datetime.now()
date_str = now.strftime("%Y%m%d%H%M")
rep_fic_res_final = os.path.join(rep_fic_res,num_float + '_' + date_str)
rep_fic_fig = os.path.join(rep_fic_res_final,'plot') # Plot
rep_fic_nc = os.path.join(rep_fic_res_final,'nc') # NetCDF
os.makedirs(rep_fic_res_final, exist_ok=True)
os.makedirs(rep_fic_fig, exist_ok=True)
os.makedirs(rep_fic_nc, exist_ok=True)
# ASCII file name
racine_res = 'locodox_res_'
fic_res_ASCII = os.path.join(rep_fic_res_final,racine_res + num_float)

# Default pressure coefficient used in ARGO
# sensor_aanderaa = 1 if aanderaa sensor, 0 if rinko sensor
sensor_aanderaa = 1
if sensor_aanderaa == 1:
    racine_res = racine_res + 'aanderaa_' 
    pcoef2 = 0.00022
    pcoef3 = 0.0419
else:
    racine_res = racine_res + 'rinko_' 
    pcoef2 = 0
    pcoef3 = 0.04

# Relative error to be written in the BD files
percent_relative_error = 2.0

# Which plot ?
###############
# info_plot = 1 : all plots
# info_plot = 0 : not all plots are created.
info_plot = 1 

# Which ARGO variables (PRES/PSAL/TEMP) to be used to estimate correction.
#################################################
# which_var = 1 : RAW Data
# which_var = 2 : Adjusted Data
# which_var = 3 : Adjusted Data if available, otherwise Raw Data
which_var = 3
if which_var==1:
    str_chaine = ''
else:
    str_chaine='_ADJUSTED'

# Which QC used for pressure, temperature, salinity and oxygen.
# In Sprof, we got interpolated data (flag=8)
pres_qc = [1,2,8]
temp_qc = [1,2,8]
sal_qc = [1,2,8]
doxy_qc = [1,2,3,8]

# ARGO InAir code
code_inair = [699,711,799]
# Argo InWater code
code_inwater = [690,710]
# Min and max pressure to extract the salinity for inwater data (because for inwater data, the pump is off.
# So, we decided to take the salinity from the profile (with the pump on).
min_pres = 0
max_pres = 10





# Pressure effect 
###################
# Pressure for pressure effect estimation. We use pressure > pressure_threshold
pressure_threshold = 1000  

# Bathymetry plot
# Bathymetry file
fic_bathy = '/Users/chemon/ARGO_NEW/LOCODOX/DATA/LOCODOX_EXTERNAL_DATA/TOPOGRAPHY/ETOPO2v2c_f4.nc'
#  Dataset associated
ds_bathy = xr.open_dataset(fic_bathy)
# extension for Position plot
extend_lon_lat = 1
# Depth for bathymetry contouring
depths = np.arange(-7000,500,500)

# WOA correction
##################
# fic_woa : WOA file
# Tis file contains the WOA variables (doxywoa,Psatwoa/density/preswoa/PSAL_WOA/TEMP_WOA).
# Ex : doxywoa(time,Depth,lat,lon) with time=12,lat=180,lon=360,Depth=102 : contains the monthly average of doxy
# This file is created by an internal LOPS routine. !!! To change !!!
#
# WOA file
fic_WOA = '/Users/chemon/ARGO_NEW/LOCODOX/DATA/LOCODOX_EXTERNAL_DATA/WOA/WOA2018_DECAV_monthly_5500_1deg.nc'
# Min and max pressure used to estimate WOA correction. 
min_pres_interp = 0
max_pres_interp = 25

# NCEP Correction
######################
# NCEP directory : where the NCEP slp/air.sig995/rhum.sig995 can be found or downloaded if needed
rep_NCEP_data= '/Users/chemon/ARGO_NEW/NEW_LOCODOX/NCEP_DATA/'
# NCEP ftp server
ftp_server = 'ftp.cdc.noaa.gov'
# Ncep ftp directory
rep_ftp = 'Datasets/ncep.reanalysis/surface'
# NCEP variables needed.
ncep_variables = ['slp','air.sig995','rhum.sig995']

couleurs = [
        (1, 0, 0),       # Rouge
        (0, 1, 0),       # Vert
        (0, 0, 1),       # Bleu
        (1, 1, 0),       # Jaune
        (1, 0.65, 0),    # Orange
        (0, 1, 1),       # Cyan
        (1, 0.75, 0.8),  # Rose
        (0.65, 0.16, 0.16), # Marron
        (0, 0.5, 0.5),    # Bleu-vert
        (0.5, 0, 0.5)   # Violet

    ]

my_cmap = mcolors.ListedColormap(couleurs, name='ma_palette')
