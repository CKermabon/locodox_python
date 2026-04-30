#!/usr/bin/env python
# coding: utf-8

# # Loading modules/functions

# In[1]:


#
# Jupyter python notebook to estimate ARGO Oxygen Correction.
# DOXY_ADJUSTED is corrected via the correction of the partial pressure PPOX as in 'Bittig and al (2018)'
# https://dx.doi.org/10.3389/fmars.2017.00429

#instruction to be able to zoom on matplotlib figure

# Path to LOCODOX python
import sys
sys.path.insert(0,'/Users/chemon/ARGO_NEW/NEW_LOCODOX/locodox_python/source')

# Import python module
import os
import argopy
import xarray as xr
import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.rcParams["figure.dpi"] = 100 
#matplotlib.use('module://ipympl.backend_nbagg')
from scipy.optimize import curve_fit
import gsw
from datetime import date, datetime
import copy
import pwlf
import shutil
import time
#from wrangling import interp_climatology


# Import module developped for LOCODOX
from m_argo_data import open_argo_multi_profile_file, get_argo_launch_date, get_argo_optode_height, get_argo_data_for_WOA
from m_argo_data import get_argo_data_for_NCEP
from m_WOA_data import open_WOA_file, interp_WOA_on_ARGO
from m_NCEP_data import open_NCEP_file, interp_NCEP_on_ARGO, calcul_NCEP_PPOX
from m_NCEP_data import download_NCEP_if_needed
from m_users_fonctions import interp_pres_grid,O2stoO2p, O2ctoO2p, O2ptoO2c, O2ctoO2s,umolkg_to_umolL, diff_time_in_days, copy_attr, write_param_results, interp_climatology_float
from m_users_fonctions import cherche_info_ctd_ref, calcul_R2_ARGO_CTD
from m_model_curve_fit import model_Gain, model_Gain_Derive, model_Gain_CarryOver, model_Gain_Derive_CarryOver, model_Gain_pres, model_AXplusB
from m_users_plot import plot_WMO_position, plot_DOXY_QC, plot_QC_cycle, plot_DOXY_cycle, plot_ppox_Inair_Inwater_Ncep, plot_cmp_corr_NCEP, plot_cmp_corr_WOA
from m_users_plot import plot_cmp_ARGO_CTD, plot_cmp_corr_oxy_woa, plot_Theta_S, plot_CTD_Argo_Pos, plot_cmp_corr_NCEP_with_error, plot_cmp_corr_WOA_with_error, plot_ref_div_argo
from m_users_plot import plot_cmp_correction_with_WOA
from m_read_write_netcdf import corr_file, corr_file_with_ppox


# # User initialization

plt.ion()
# In[2]:


#####################
# Initialization.
###################
from init_for_processing import *

# In[3]:


if num_float=='6903035':
    test_piece =1
    compute_NCEP_breakpoint=1
    breakpoint_NCEP_user = 18
    nb_segment_NCEP = 2
    nb_segment_WOA = 2
    print(compute_NCEP_breakpoint,test_piece)


# In[4]:


ds_cruise


# # ARGO reading and selection
# ## We use the NetCDF Argo file 
# - meta file
# - Sprof file
# - Rtraj file

# In[5]:


# Read ARGO files 
ds_argo_meta = open_argo_multi_profile_file(num_float,rep_data_argo,'meta')
ds_argo_Sprof = open_argo_multi_profile_file(num_float,rep_data_argo,'Sprof')
ds_argo_Rtraj = open_argo_multi_profile_file(num_float,rep_data_argo,'Rtraj')


# In[6]:


# Launch data
launch_date = get_argo_launch_date(ds_argo_meta)
print(launch_date)
optode_height = get_argo_optode_height(ds_argo_meta)
print(optode_height)
# Delta time from launch date
delta_T_Sprof = diff_time_in_days(ds_argo_Sprof['JULD'],launch_date)


# In[7]:


# Select the cycles to be used
ds_argo_Sprof = ds_argo_Sprof.where((ds_argo_Sprof['CYCLE_NUMBER']>=first_cycle_to_use) & (ds_argo_Sprof['CYCLE_NUMBER']<=last_cycle_to_use),drop=True)
ds_argo_Sprof = ds_argo_Sprof.where(ds_argo_Sprof['DIRECTION']=='A',drop=True)
ds_argo_Rtraj = ds_argo_Rtraj.where( (ds_argo_Rtraj['CYCLE_NUMBER']>=first_cycle_to_use) & (ds_argo_Rtraj['CYCLE_NUMBER']<=last_cycle_to_use),drop=True)
ds_argo_Sprof['PLATFORM_NUMBER'] = ds_argo_Sprof['PLATFORM_NUMBER'].astype(int) # The where transform the nan from int to float ...


# In[8]:


# Check ds_argo_Sprof
ds_argo_Sprof


# # Plots

# In[9]:


# Maps plotting
if info_plot==1:
    ds_bathy = xr.open_dataset(fic_bathy)
    depths = np.arange(-7000,500,500)
    fig = plot_WMO_position(ds_argo_Sprof, ds_bathy, depths,extend_lon_lat)
    #fig.canvas.draw_idle()  
    fig.savefig(os.path.join(rep_fic_fig,num_float +'_pos.png'))


# In[10]:


# DOXY RAW DATA
fig=plot_DOXY_cycle(ds_argo_Sprof,qc_keep=[1,2,3,8])
plt.show()
#fig.canvas.draw_idle()  


# In[11]:


fig.savefig(os.path.join(rep_fic_fig,num_float +'_doxy_cycle.png'))


# In[12]:


fig=plot_DOXY_cycle(ds_argo_Sprof) # qc_keep = [1,2,3,4,8] by default
plt.show()


# In[13]:


fig.savefig(os.path.join(rep_fic_fig,num_float +'_doxy_cycle_2.png'))


# In[14]:


# DOXY_QC plot with PRES/PSAL/TEMP
if info_plot==1:
    fig=plot_DOXY_QC(ds_argo_Sprof,doxy_qc)
    plt.show()
    plt.savefig(os.path.join(rep_fic_fig,num_float +'_doxy_qc.png'))


# In[15]:


# Same with DATA ADJUSTED (PRES,PSAL,TEMP)
if info_plot==1:
    fig = plot_DOXY_QC(ds_argo_Sprof,doxy_qc,'_ADJUSTED')
    plt.show()
    fig.savefig(os.path.join(rep_fic_fig,num_float +'_doxy_qc_PTS_adjusted.png'))


# In[16]:


# PRES/PSAL/TEMP QC
if info_plot==1:
    fig=plot_QC_cycle(ds_argo_Sprof)
    fig.savefig(os.path.join(rep_fic_fig,num_float +'_PTS_QC.png'))


# In[17]:


# Same with adjusted DATA
if info_plot==1:
    fig=plot_QC_cycle(ds_argo_Sprof,'_ADJUSTED')
    fig.savefig(os.path.join(rep_fic_fig,num_float +'_PTS_Adjusted_QC.png'))


# In[18]:


# Theta/S
#_=plot_Theta_S(ds_argo_Sprof)
#_=plot_Theta_S(ds_argo_Sprof,'_ADJUSTED')
fig=plot_Theta_S(ds_argo_Sprof,qc_keep=[1,2,8])
fig.savefig(os.path.join(rep_fic_fig,num_float +'_theta_S.png'))
fig=plot_Theta_S(ds_argo_Sprof,'_ADJUSTED',qc_keep=[1,2,8])
fig.savefig(os.path.join(rep_fic_fig,num_float +'_theta_S_adjusted.png'))


# # Get WOA and NCEP DATA
# - Attention : The WOA file is a LOPS home-made file

# In[19]:


# WOA file reading
ds_woa = open_WOA_file(fic_WOA)
download_NCEP_if_needed(ds_argo_Sprof['JULD'],ftp_server,rep_ftp,rep_NCEP_data,ncep_variables)


# # Prepare data for WOA Correction

# In[20]:


################################## 
# Correction estimated with WOA
##################################
# ARGO data needed for WOA correction.
# ds_argo contains variables needed to estimate correction with WOA.
ds_argo = get_argo_data_for_WOA(ds_argo_Sprof,pres_qc,temp_qc,sal_qc,doxy_qc,which_var)
ds_argo


# In[21]:


# Delta Times from launch_date
delta_T_WOA = diff_time_in_days(ds_argo['JULD'].values,launch_date)


# In[22]:


# Interp WOA data on ARGO time
ds_woa_interp_on_ARGO = interp_WOA_on_ARGO(ds_woa, ds_argo)
ds_woa_interp_on_ARGO


# In[23]:


# Interpolation WOA et ARGO on regular grid (defined at the beginning)
print(f"WOA/ARGO Interpolation between {min_pres_interp} et {max_pres_interp}")
var_to_interpol = [var for var in ds_woa_interp_on_ARGO.data_vars if "N_LEVELS" in ds_woa_interp_on_ARGO[var].dims]
ds_woa_interp = interp_pres_grid(min_pres_interp,max_pres_interp,var_to_interpol,ds_woa_interp_on_ARGO,'preswoa','Depth')

var_to_interpol = [var for var in ds_argo.data_vars if "N_LEVELS" in ds_argo[var].dims]
ds_argo_interp = interp_pres_grid(min_pres_interp,max_pres_interp,var_to_interpol,ds_argo,'PRES_ARGO','N_LEVELS')


# In[24]:


ds_woa_interp


# In[25]:


ds_argo_interp['LONGITUDE'] = ds_argo['LONGITUDE']
ds_argo_interp['LATITUDE'] = ds_argo['LATITUDE']
ds_argo_interp


# In[26]:


# Add attributes in data interp 
ds_argo_interp = copy_attr(ds_argo,ds_argo_interp)
ds_woa_interp = copy_attr(ds_woa_interp_on_ARGO,ds_woa_interp)


# In[27]:


# PPOX WOA.
ppox_WOA = O2stoO2p(ds_woa_interp['Psatwoa'],ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PSAL_ARGO']) 
ppox_WOA_mean = np.nanmean(ppox_WOA,axis=1)


# In[28]:


# PPOX ARGO

psal = ds_argo_interp['PSAL_ARGO']  
pres = ds_argo_interp['PRES_ARGO'] 
temp = ds_argo_interp['TEMP_ARGO'] 
lon = ds_argo_interp['LONGITUDE'].values
lat = ds_argo_interp['LATITUDE'].values

lon_2d = lon[:, np.newaxis]  # Résultat de forme (154, 1)
lon_2d = lon_2d * np.ones((1, psal.shape[1])) 
lat_2d = lat[:, np.newaxis]  # Résultat de forme (154, 1)
lat_2d = lat_2d * np.ones((1, psal.shape[1])) 

psal_absolue  = gsw.SA_from_SP(psal,pres,lon_2d,lat_2d)
ct = gsw.CT_from_t(psal_absolue, temp, pres)
ana_dens = gsw.rho(psal_absolue,ct,0)

#ana_dens = sw.pden(ds_argo_interp['PSAL_ARGO'],ds_argo_interp['TEMP_ARGO'],np.arange(min_pres_interp,max_pres_interp+1,1),0)
O2_ARGO_umolL = umolkg_to_umolL(ds_argo_interp['DOXY_ARGO'],ds_argo['DOXY_ARGO'].units,ana_dens)

ppox_ARGO = O2ctoO2p(O2_ARGO_umolL,ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PSAL_ARGO']) # On calcule PPOX pour P=0
ppox_ARGO_mean = np.nanmean(ppox_ARGO,axis=1)


# # Prepare data for NCEP Correction
# ## We recover the InAir data and InWater Argo Data from the Rtraj file.
# - In the Rtraj file, the pump is off. So, we decide to take the salinity from the Sprof file, near the surface (between min_pres and max_pres defined by the user)
# - In the RTraj file, all TEMP_QC is 3. But we decide to take it into account because the temperature is important to calculate the watervapor. <br>
#   The temperature in the Rtraj is closer to the InAir and InWater data. So, we use it. We compare it to the Temperature from the Sprof and a message is written if the difference is >0.5.
# - So, this salinity and temperature are used to calculate NCEP PPOX
# - NCEP PPOX is calculated from the article 'Tackling Oxygen Optode Drift : Near Surface and In-Air Oxygen optode Measurements on a float provide an accurate in situ Reference' <br>
#  (Bittg and al. 2015) : <br>
#      - PPOX_air = 0.20946 * (P_air - P_vap) where P_air = Atmospheric pressure, P_vap : water vapor scaled to the optode height
#      - P_vap = P_vap_S + (Rhum_10m * P_vap_10m - P_vap_S) * ln(optode_height / 1e-4)/ln(10/1e-4) avec <br>
#          - P_vap_S : seasurface water vapor pressure
#          - P_vap_1Om : ncep water vapor at 10m
#          - Rhum_10m : relative humidity at 10m

# In[29]:


# Get ARGO PPOX inair/inwater for NCEP correction
dsair,dsinwater = get_argo_data_for_NCEP(ds_argo_Rtraj,ds_argo_Sprof,which_var,code_inair,code_inwater,min_pres,max_pres)
PPOX1 = dsair['PPOX_DOXY'].values
PPOX2 = dsinwater['PPOX_DOXY'].values
plt.savefig(os.path.join(rep_fic_fig,num_float +'_InAir_InWater.png'))


# In[30]:


# Compute NCEP data at ARGO time
ds_NCEP_air,ds_NCEP_rhum, ds_NCEP_slp = open_NCEP_file(rep_NCEP_data)
ds_NCEP_air,ds_NCEP_rhum,ds_NCEP_slp = interp_NCEP_on_ARGO(ds_NCEP_air,ds_NCEP_rhum,ds_NCEP_slp,dsair['LONGITUDE_ARGO'],dsair['LATITUDE_ARGO'],dsair['JULD'])
z0q = 1e-4
NCEP_PPOX = calcul_NCEP_PPOX(dsinwater,ds_NCEP_air,ds_NCEP_rhum,ds_NCEP_slp,optode_height,z0q)


# In[31]:


#dsinwater


# In[32]:


delta_T_NCEP = diff_time_in_days(dsair['JULD'].values,launch_date)


# # Plot REF/ARGO
# The correction used is :
# - Ref = (G * (1 + D/100 * deltaT / 365)) * (data + Offset) <br>
#  Attention : Offset is supposed to be 0 (and so is not estimated) <br>
# G is the gain <br>
# D is the time drift <br>
# deltaT is the time, in days, from the launch date
# - So, Ref/data is : (G * (1 + D/100 * deltaT / 365)) = A * deltaT + B <br>
#   The correction is a straight line as a function of deltaT.
# 
#   The plot shows this correction. It can help the user to choose if he has to choose a piece wise correction or not.
# ## Ref_WOA/ppox_Argo
# 

# In[33]:


fig = plot_ref_div_argo(delta_T_WOA,ppox_WOA_mean,ppox_ARGO_mean,num_float)
#fig.canvas.draw()
plt.show()
fig.savefig(os.path.join(rep_fic_fig,num_float +'_ref_div_WOA_deltaDays.png'))


# ## Ref_NCEP/ppox_ARGO

# In[34]:


fig=plot_ref_div_argo(delta_T_NCEP,NCEP_PPOX,PPOX1,num_float)
#fig.canvas.draw()
plt.show()
fig.savefig(os.path.join(rep_fic_fig,num_float +'_ref_div_NCEP_deltaDays.png'))


# # Correction estimation (without piece, just 1 segment)
# We calculate the correction on the the oxygen partial pressure (PPOX)
# ## WOA Correction
# - 2 possible corrections :
#     - a gain : Ref/data = G 
#     - a gain and a time drift : Ref/data = (G + (1 * D * deltaT/365)) <br>
# We use curve_fit for G (gain) and D (time drift) estimation

# In[35]:


# Correction WOA : Gain estimation
initial_guess = 1  # Valeurs initiales pour G 
params_Gain_WOA, covariance = curve_fit(model_Gain, ppox_ARGO_mean/ppox_ARGO_mean, ppox_WOA_mean/ppox_ARGO_mean, p0=initial_guess,nan_policy='omit')
perr_Gain_WOA = np.sqrt(np.diag(covariance))
print(f"WOA Gain estimated : {params_Gain_WOA} with an error {perr_Gain_WOA}")


# In[36]:


# Correction WOA : Gain and Drift Time estimation
initial_guess = [1, 0]  # Valeurs initiales pour G et D et C
params_Gain_Derive_WOA, covariance,info,mesg,ier = curve_fit(model_Gain_Derive, [ppox_ARGO_mean/ppox_ARGO_mean,delta_T_WOA], ppox_WOA_mean/ppox_ARGO_mean, p0=initial_guess,nan_policy='omit',full_output=True)
perr_Gain_Derive_WOA = np.sqrt(np.diag(covariance))
print(f"WOA Gain/Drift estimated : {params_Gain_Derive_WOA} with an error {perr_Gain_Derive_WOA}")


# ## NCEP Correction
# - 4 possible corrections:
#     - a gain (G)
#     - a gain (G) estimated with CarryOver (C)
#     - a gain (G) and a time drift (D)
#     - a gain (G) and a time drift (D) estimated with CarryOver (C) <br>
# We use curve_fit for G, D and C estimation. <br>
# The Carryover represents the fact that the InAir Argo PPOX may be polluted by water (waves) <br>
# The CarryOver is determinated by the article 'Oxygen Optode Sensors : Principle, characterization, calibration and application in the Ocean' <br>
# (Bittig and al. 2018) : <br>
#         - G * PPOX_obs_sufr - PPOX_air = C * (G * PPOX_obs_water - PPOX_air)

# In[37]:


# Plot NCEP and ARGO PPOX
fig = plot_ppox_Inair_Inwater_Ncep(dsair,dsinwater,NCEP_PPOX)
#fig.canvas.draw()
plt.show()
fig.savefig(os.path.join(rep_fic_fig,num_float +'_NCEP_InAir_InWater.png'))


# In[38]:


# Estimate Gain correction with NCEP 
# without CarryOver
initial_guess = 1
# Gain
params_Gain_NCEP, covariance,info,mesg,ier = curve_fit(model_Gain, PPOX1/PPOX1, NCEP_PPOX/PPOX1, p0=initial_guess,nan_policy='omit',full_output=True)
perr_Gain_NCEP = np.sqrt(np.diag(covariance))
print(f"NCEP Gain estimated : {params_Gain_NCEP} with an error {perr_Gain_NCEP}")

# with CarryOver
initial_guess = [1, 0]  # Valeurs initiales pour G et C
params_Gain_NCEP_CarryOver, covariance = curve_fit(model_Gain_CarryOver, [PPOX1,PPOX2], NCEP_PPOX, p0=initial_guess,nan_policy='omit')
perr_Gain_NCEP_CarryOver = np.sqrt(np.diag(covariance))
print(f"NCEP Gain/CarryOver estimated : {params_Gain_NCEP_CarryOver} with an error {perr_Gain_NCEP_CarryOver}")

# Test pour estimation de l'offset.
# Ca ne fonctionne pas. On trouve un offset très grand.
# !!!! A retravailler !!!!
def model_Gain_Derive_offset(X,G,D,Offset):
    """ Function to estimate, with curve_fit, a correction with a Time Drift and a Gain 
    
    Parameters
    ----------
    X : Oxygen Values (X[0]) and delta_T from launch_date (X[1])

    Returns
    --------
    G * (1 + (D * X[1])/(365*100)) * X[0]
    """
    return (G * (1 + (D * X[1])/(365*100)) * (X[0]+Offset) )

initial_guess = [1,0,0]
params_Gain_Derive_NCEP_Offset, covariance,info,mesg,ier = curve_fit(model_Gain_Derive_offset, [PPOX1/PPOX1,delta_T_NCEP], NCEP_PPOX/PPOX1, p0=initial_guess,nan_policy='omit',full_output=True)
perr_Gain_Derive_NCEP_Offset = np.sqrt(np.diag(covariance))
print(f"Gain/Drift estimated : {params_Gain_Derive_NCEP_Offset} with an error {perr_Gain_Derive_NCEP_Offset}")

params_Gain_Derive_NCEP_Offset, covariance,info,mesg,ier = curve_fit(model_Gain_Derive_offset, [PPOX1,delta_T_NCEP], NCEP_PPOX, p0=initial_guess,nan_policy='omit',full_output=True)
perr_Gain_Derive_NCEP_Offset = np.sqrt(np.diag(covariance))
print(f"Gain/Drift estimated : {params_Gain_Derive_NCEP_Offset} with an error {perr_Gain_Derive_NCEP_Offset}")


initial_guess = [1, 0]  # G/C
params_Gain_Derive_NCEP, covariance,info,mesg,ier = curve_fit(model_Gain_Derive, [PPOX1/PPOX1,delta_T_NCEP], NCEP_PPOX/PPOX1, p0=initial_guess,nan_policy='omit',full_output=True)
perr_Gain_Derive_NCEP = np.sqrt(np.diag(covariance))
print(f"Gain/Drift estimated : {params_Gain_Derive_NCEP} with an error {perr_Gain_Derive_NCEP}")

def model_offset(X,Offset):
    """ Function to estimate, with curve_fit, a correction with a Time Drift and a Gain 
    
    Parameters
    ----------
    X : Oxygen Values (X[0]) and delta_T from launch_date (X[1])

    Returns
    --------
    G * (1 + (D * X[1])/(365*100)) * X[0]
    """
    return  (X[0]+Offset) 

initial_guess = 0
ppox_corr = PPOX1*(params_Gain_Derive_NCEP[0] * (1 + params_Gain_Derive_NCEP[1]*delta_T_NCEP/365/100))
params_offset, covariance,info,mesg,ier = curve_fit(model_offset, ppox_corr,NCEP_PPOX, p0=initial_guess,nan_policy='omit',full_output=True)
perr_offset = np.sqrt(np.diag(covariance))
print(f"Offset : {params_offset} with an error {perr_offset}")


# In[39]:


# Estimate Gain/Drift correction with NCEP 
# without CarryOver
initial_guess = [1, 0]  # G/D
params_Gain_Derive_NCEP, covariance,info,mesg,ier = curve_fit(model_Gain_Derive, [PPOX1/PPOX1,delta_T_NCEP], NCEP_PPOX/PPOX1, p0=initial_guess,nan_policy='omit',full_output=True)
perr_Gain_Derive_NCEP = np.sqrt(np.diag(covariance))
print(f"Gain/Drift estimated : {params_Gain_Derive_NCEP} with an error {perr_Gain_Derive_NCEP}")

# with CarryOver
initial_guess = [1, 0, 0]  # G/C/D
params_Gain_Derive_NCEP_CarryOver, covariance = curve_fit(model_Gain_Derive_CarryOver, [PPOX1,PPOX2,delta_T_NCEP], NCEP_PPOX, p0=initial_guess,nan_policy='omit')
perr_Gain_Derive_NCEP_CarryOver = np.sqrt(np.diag(covariance))
print(f"Gain/CarryOver/Drift estimated : {params_Gain_Derive_NCEP_CarryOver} with an error {perr_Gain_Derive_NCEP_CarryOver}")


# In[40]:


# We don't need the CarryOver value.
params_Gain_NCEP_CarryOver = np.array([params_Gain_NCEP_CarryOver[0]])
params_Gain_Derive_NCEP_CarryOver = np.array(params_Gain_Derive_NCEP_CarryOver[[0,2]])
perr_Gain_NCEP_CarryOver = np.array([perr_Gain_NCEP_CarryOver[0]])
perr_Gain_Derive_NCEP_CarryOver = np.array(perr_Gain_Derive_NCEP_CarryOver[[0,2]])


# In[41]:


dict_corr = {'Gain WOA' : np.stack((params_Gain_WOA,perr_Gain_WOA),axis=0),'Gain/Derive WOA':np.stack((params_Gain_Derive_WOA,perr_Gain_Derive_WOA),axis=0),
             'Gain NCEP' : np.stack((params_Gain_NCEP,perr_Gain_NCEP),axis=0),'Gain Ncep CarryOver' : np.stack((params_Gain_NCEP_CarryOver,perr_Gain_NCEP_CarryOver),axis=0),
             'Gain/Derive Ncep':np.stack((params_Gain_Derive_NCEP,perr_Gain_Derive_NCEP),axis=0),'Gain/Derive Ncep CarryOver':np.stack((params_Gain_Derive_NCEP_CarryOver,perr_Gain_Derive_NCEP_CarryOver),axis=0)}
write_param_results(dict_corr, num_float)


# In[42]:


# Compare corrections
# User can change dict_corr.
# Example : dict_corr = {'GAIN WOA' : params_Gain_WOA,'Gain NCEP' : params_Gain_NCEP} compares Gain WOA Correction
# with Gain NCEP Correction.
dict_corr = {'GAIN WOA' : params_Gain_WOA,'Gain NCEP' : params_Gain_NCEP,'Gain/Derive WOA' : params_Gain_Derive_WOA,
             'Gain/Derive Ncep' : params_Gain_Derive_NCEP,'Gain Ncep CarryOver' : params_Gain_NCEP_CarryOver,'Gain/Derive Ncep CarryOver' : params_Gain_Derive_NCEP_CarryOver}
breakpoint_list=[[]] * len(dict_corr)
_=plot_cmp_corr_NCEP(dict_corr,breakpoint_list,dsair,NCEP_PPOX,delta_T_NCEP,my_cmap)


# In[43]:


plt.savefig(os.path.join(rep_fic_fig,num_float +'_cmp_corr_NCEP.png'))


# In[44]:


_=plot_cmp_corr_WOA(dict_corr, breakpoint_list,ds_argo_interp, ds_woa_interp, delta_T_WOA,my_cmap)


# In[45]:


plt.savefig(os.path.join(rep_fic_fig,num_float +'_cmp_corr_PSATWOA.png'))


# # Piece wise Correction
# - If the user chooses a piece correction, that means there is a drift.
# - If the user wants a piece correction only with a simple gain, then he can process the soft with different first_cycle_to_use/last_cycle_to_use
# - When pieces :
#     - We use polyfit to estimate the straight line by pieces.
#     - We estimate the gain and drift so that the corrected data coresponds to this straight line.
# 

# ## Number of pieces defined by the user

# ### WOA

# In[46]:


# Number of piece to be defined by the user
if test_piece==1:

    params_morceaux_Gain_Derive_WOA = []
    perr_morceaux_Gain_Derive_WOA = []
    bid = ppox_WOA_mean/ppox_ARGO_mean
    mask = ~np.isnan(bid)
    y_noisy = bid[mask]
    x = delta_T_WOA[mask]


    model = pwlf.PiecewiseLinFit(x, y_noisy)
    model.fit(nb_segment_WOA)
    breakpoints = model.fit_breaks  # Coordonnées X des breakpoints
    print(f"Breakpoints (X) : {breakpoints}")
    x_pred = delta_T_WOA
    y_pred = model.predict(x_pred)

    plt.figure(figsize=(10,6))
    plt.scatter(x, y_noisy, label="Données bruitées", alpha=0.5)
    plt.plot(x,y_noisy,'-b')
    breaks_WOA = model.fit_breaks
    slopes = model.slopes
    intercepts = model.intercepts

    print("Breaks :", model.fit_breaks)
    print("Slopes :", model.slopes)
    print("Intercepts :", model.intercepts)
    # formule  = Y = Slopes * X + Intercept

    for i in range(nb_segment_WOA):
        print(f"\n\nCorrection WOA : Piece {i+1}\n")
        x0, x1 = breaks_WOA[i], breaks_WOA[i+1]
        mask = (delta_T_WOA >=x0) & (delta_T_WOA<=x1)
        xs = delta_T_WOA[mask]            
        ys = slopes[i] * xs + intercepts[i]
        # To have a continue line between the segment, for plotting, we create segment from xo to x1
        xs_plot = np.array([x0, x1]) #np.linspace(x0, x1, 100)
        ys_plot = slopes[i] * xs_plot + intercepts[i]
        plt.plot(xs_plot, ys_plot, color="red", lw=2)  
        slope_woa = intercepts[i]
        drift_woa = slopes[i] * 36500/slope_woa
        print(f"Gain/Drift WOA calculated directly from PWLF module : {slope_woa},{drift_woa}")

        # La version de pwlf ne permet pas d'avoir acces aux erreurs sur l'estimation de slope et intercept.
        # cov = model.calc_covariance() plante.
        # Pour contourner cela et obtenir une erreur, on passe par curve_fit.
        bid = ppox_WOA_mean/ppox_ARGO_mean
        popt, pcov = curve_fit(model_AXplusB, xs, bid[mask], p0=[1,0],nan_policy='omit')
        perr_bid_WOA = np.sqrt(np.diag(pcov))
        #print(f"popt : {popt}, error : {perr_bid_WOA}")

        perr_bid_WOA_final = perr_bid_WOA.copy()
        perr_bid_WOA_final[0] = perr_bid_WOA[1] # The error on slope_woa = error on intercepts (because slope_woa = intercepts[i])
        perr_bid_WOA_final[1] = 36500 * np.sqrt( (perr_bid_WOA[0] / intercepts[i])**2 + ((slopes[i] * perr_bid_WOA[1]) / (intercepts[i]**2))**2 ) # because drift_woa = slopes[i] * 36500/slope_woa

        initial_guess = [1, 0]  # Valeurs initiales pour G et D 
        mask = (delta_T_WOA >=x0) & (delta_T_WOA<=x1)
        xs = delta_T_WOA[mask]            # dense sur le segment
        ys = slopes[i] * xs + intercepts[i]
        var1_seg = ppox_ARGO_mean[mask]
        delta_T_seg = delta_T_WOA[mask]
        #val_bid1,covariance,info,mesg,ier = curve_fit(model_Gain_Derive, [var1_seg/var1_seg,delta_T_seg], ppox_WOA_mean[mask]/var1_seg, p0=initial_guess,nan_policy='omit',full_output=True)
        #print(f"val_bid WOA1: {val_bid1}")
        val_bid,covariance,info,mesg,ier = curve_fit(model_Gain_Derive, [var1_seg/var1_seg,delta_T_seg], ys, p0=initial_guess,nan_policy='omit',full_output=True)
        #perr_bid = np.sqrt(np.diag(covariance))
        perr_morceaux_Gain_Derive_WOA.append(perr_bid_WOA_final)
        print(f"Gain/Drift WOA calculated with curve_fit : {val_bid} error : {perr_bid_WOA_final}")
        params_morceaux_Gain_Derive_WOA.append(val_bid)   
        
    
    for b in breaks_WOA:
        plt.axvline(b, color="green", ls="--", alpha=0.7)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Fit par morceaux", "Données"])
    plt.grid(True)
    plt.show()


# ### NCEP

# In[47]:


if (test_piece==1) & (compute_NCEP_breakpoint==0):

    # The breakpoint are determinated automatically.
    params_morceaux_Gain_Derive_NCEP = []
    params_morceaux_Gain_Derive_CarryOver = []
    perr_morceaux_Gain_Derive_NCEP = []
    perr_morceaux_Gain_Derive_CarryOver = []

    bid = NCEP_PPOX/PPOX1
    mask = np.isfinite(bid)
    y_noisy = bid[mask]
    x = delta_T_NCEP[mask]

    model = pwlf.PiecewiseLinFit(x, y_noisy)
    model.fit(nb_segment_NCEP)

    breakpoints = model.fit_breaks  # Coordonnées X des breakpoints
    print(f"Breakpoints (X) : {breakpoints}")
    x_pred = delta_T_NCEP
    y_pred = model.predict(x_pred)


    plt.figure(figsize=(10,6))
    plt.scatter(x, y_noisy, label="Données bruitées", alpha=0.5)
    plt.plot(x,y_noisy,'-b')
    breaks_NCEP = model.fit_breaks
    slopes = model.slopes
    intercepts = model.intercepts

    print("Breaks :", model.fit_breaks)
    print("Slopes :", model.slopes)
    print("Intercepts :", model.intercepts)
    # formule  = Y = Slopes * X + Intercept


    nb_day_carryover = 90
    for i in range(nb_segment_NCEP):
        print(f"\n\nCorrection NCEP : Piece {i+1}\n")
        x0, x1 = breaks_NCEP[i], breaks_NCEP[i+1]
        mask = (delta_T_NCEP >=x0) & (delta_T_NCEP<=x1)
        xs = delta_T_NCEP[mask]            # dense sur le segment
        ys = slopes[i] * xs + intercepts[i]
        # To have a continue line between the segment, for plotting, we create segment from xo to x1
        xs_plot = np.array([x0, x1]) #np.linspace(x0, x1, 100)
        ys_plot = slopes[i] * xs_plot + intercepts[i]
        plt.plot(xs_plot, ys_plot, color="red", lw=2)        
        slope_ncep = intercepts[i]
        drift_ncep = slopes[i] * 36500/slope_ncep
        print(f"Gain/Drift NCEP calculated directly from PWLF module : {slope_ncep},{drift_ncep}")


        # La version de pwlf ne permet pas d'avoir acces aux erreurs sur l'estimation de slope et intercept.
        # cov = model.calc_covariance() plante.
        # Pour contourner cela et obtenir une erreur, on passe par curve_fit.
        bid = NCEP_PPOX/dsair['PPOX_DOXY']
        popt, pcov = curve_fit(model_AXplusB, xs, bid[mask], p0=[1,0],nan_policy='omit')
        perr_bid_NCEP = np.sqrt(np.diag(pcov))
    
        var1_seg = PPOX1[mask]
        var2_seg = PPOX2[mask]
        delta_T_seg = delta_T_NCEP[mask]
        initial_guess = [1, 0]  # Valeurs initiales pour G et D 
        #val_bid1,covariance,info,mesg,ier= curve_fit(model_Gain_Derive, [var1_seg/var1_seg,delta_T_seg], NCEP_PPOX[mask]/var1_seg, p0=initial_guess,nan_policy='omit',full_output=True)
        #print(f"val_bid NCEP 1: {val_bid1}")
        #print(f"Erreur Gain/Derive NCEP : {np.sqrt(np.diag(covariance))}")

        val_bid_NCEP,covariance,info,mesg,ier= curve_fit(model_Gain_Derive, [var1_seg/var1_seg,delta_T_seg], ys, p0=initial_guess,nan_policy='omit',full_output=True)
        # perr_bid_NCEP = np.sqrt(np.diag(covariance)) # Les erreurs sont mal estimees (infimes voire nulles).
    
        #
        # Calcul de l'erreur sur slope et drift final.
        #
        perr_bid_NCEP_final = perr_bid_NCEP.copy()
        perr_bid_NCEP_final[0] = perr_bid_NCEP[1]
        perr_bid_NCEP_final[1] = 36500 * np.sqrt( (perr_bid_NCEP[0] / intercepts[i])**2 + ((slopes[i] * perr_bid_NCEP[1]) / (intercepts[i]**2))**2 )
        perr_morceaux_Gain_Derive_NCEP.append(perr_bid_NCEP_final)
        print(f"Gain/Drift NCEP calculated with curve_fit : {val_bid_NCEP} error : {perr_bid_NCEP_final}")



        params_morceaux_Gain_Derive_NCEP.append(val_bid_NCEP)

        if (breaks_NCEP[i+1]-breaks_NCEP[i]>nb_day_carryover):
            print(f"Piece {i+1} lasts more than {nb_day_carryover} days : CarryOver Estimation")
            initial_guess = [1, 0, 0]  # Valeurs initiales pour G et D et C
            val_bid_NCEP_CarryOver, covariance,info,mesg,ier = curve_fit(model_Gain_Derive_CarryOver, [var1_seg,var2_seg,delta_T_seg],NCEP_PPOX[mask],p0=initial_guess,nan_policy='omit',full_output=True)
            #val_bid2, covariance,info,mesg,ier = curve_fit(model_Gain_Derive_CarryOver, [var1_seg,var2_seg,delta_T_seg],ys*dsair['PPOX_DOXY'][mask],p0=initial_guess,nan_policy='omit',full_output=True)
            #print(f"val_bid NCEP CarryOver2: {val_bid2}")
            params_morceaux_Gain_Derive_CarryOver.append(val_bid_NCEP_CarryOver)  
            perr_bid_CarryOver = np.sqrt(np.diag(covariance))
            perr_morceaux_Gain_Derive_CarryOver.append(perr_bid_CarryOver)
            print(f"Gain/Drift NCEP CarryOver calculated with curve_fit : {val_bid_NCEP_CarryOver} error : {perr_bid_CarryOver}")


        else:
            print(f"Piece {i+1} lasts less than {nb_day_carryover} days : No CarryOver Estimation")
            print(f"Gain/Drift NCEP calculated with curve_fit : {val_bid_NCEP} error : {perr_bid_NCEP_final}")
            val_bid = np.array([val_bid_NCEP[0],0,val_bid_NCEP[1]])
            perr_bid = np.array([perr_bid_NCEP_final[0],0,perr_bid_NCEP_final[1]])
            params_morceaux_Gain_Derive_CarryOver.append(val_bid)   
            perr_morceaux_Gain_Derive_CarryOver.append(perr_bid)

        
    
    for b in breaks_NCEP:
        plt.axvline(b, color="green", ls="--", alpha=0.7)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Fit par morceaux", "Données"])
    plt.grid(True)
    plt.show()


# In[48]:


# The user force the breakpoint for the piecewise linear fit.
# To be adapt by the user
if (test_piece==1) & (compute_NCEP_breakpoint==1):
    params_morceaux_Gain_Derive_NCEP = []
    params_morceaux_Gain_Derive_CarryOver = []
    perr_morceaux_Gain_Derive_NCEP = []
    perr_morceaux_Gain_Derive_CarryOver = []

    bid = NCEP_PPOX/PPOX1
    mask = np.isfinite(bid)
    y_noisy = bid[mask]
    x = delta_T_NCEP[mask]

    model = pwlf.PiecewiseLinFit(x, y_noisy)
    model.fit_with_breaks(np.array([delta_T_NCEP[0],breakpoint_NCEP_user,delta_T_NCEP[-1]]))
    #breakpoints = np.array([delta_T_NCEP[0],delta_T_NCEP[7],delta_T_NCEP[-1]])  # Coordonnées X des breakpoints
    #print(f"Breakpoints (X) : {breakpoints}")
    x_pred = delta_T_NCEP
    y_pred = model.predict(x_pred)


    plt.figure(figsize=(10,6))
    plt.scatter(x, y_noisy, label="Données bruitées", alpha=0.5)
    plt.plot(x,y_noisy,'-b')
    breaks_NCEP = model.fit_breaks
    slopes = model.slopes
    intercepts = model.intercepts

    print("Breaks :", model.fit_breaks)
    print("Slopes :", model.slopes)
    print("Intercepts :", model.intercepts)
    # formule  = Y = Slopes * X + Intercept


    nb_day_carryover = 90
    for i in range(nb_segment_NCEP):
        print(f"\n\nCorrection NCEP : Piece {i+1}\n")
        x0, x1 = breaks_NCEP[i], breaks_NCEP[i+1]
        mask = (delta_T_NCEP >=x0) & (delta_T_NCEP<=x1)
        xs = delta_T_NCEP[mask]            
        ys = slopes[i] * xs + intercepts[i]
        # To have a continue line between the segment, for plotting, we create segment from xo to x1
        xs_plot = np.array([x0, x1]) #np.linspace(x0, x1, 100)
        ys_plot = slopes[i] * xs_plot + intercepts[i]
        plt.plot(xs_plot, ys_plot, color="red", lw=2)
        slope_ncep = intercepts[i]
        drift_ncep = slopes[i] * 36500/slope_ncep
        print(f"Gain/Drift NCEP calculated directly from PWLF module : {slope_ncep},{drift_ncep}")


        # La version de pwlf ne permet pas d'avoir acces aux erreurs sur l'estimation de slope et intercept.
        # cov = model.calc_covariance() plante.
        # Pour contourner cela et obtenir une erreur, on passe par curve_fit.
        bid = NCEP_PPOX/dsair['PPOX_DOXY']
        popt, pcov = curve_fit(model_AXplusB, xs, bid[mask], p0=[1,0],nan_policy='omit')
        perr_bid_NCEP = np.sqrt(np.diag(pcov))
    
        var1_seg = PPOX1[mask]
        var2_seg = PPOX2[mask]
        delta_T_seg = delta_T_NCEP[mask]
        initial_guess = [1, 0]  # Valeurs initiales pour G et D 
        #val_bid1,covariance,info,mesg,ier= curve_fit(model_Gain_Derive, [var1_seg/var1_seg,delta_T_seg], NCEP_PPOX[mask]/var1_seg, p0=initial_guess,nan_policy='omit',full_output=True)
        #print(f"val_bid NCEP 1: {val_bid1}")
        #print(f"Erreur Gain/Derive NCEP : {np.sqrt(np.diag(covariance))}")

        val_bid_NCEP,covariance,info,mesg,ier= curve_fit(model_Gain_Derive, [var1_seg/var1_seg,delta_T_seg], ys, p0=initial_guess,nan_policy='omit',full_output=True)
        # perr_bid_NCEP = np.sqrt(np.diag(covariance)) # Les erreurs sont mal estimees (infimes voire nulles).
    
        #
        # Calcul de l'erreur sur slope et drift final.
        #
        perr_bid_NCEP_final = perr_bid_NCEP.copy()
        perr_bid_NCEP_final[0] = perr_bid_NCEP[1]
        perr_bid_NCEP_final[1] = 36500 * np.sqrt( (perr_bid_NCEP[0] / intercepts[i])**2 + ((slopes[i] * perr_bid_NCEP[1]) / (intercepts[i]**2))**2 )
        perr_morceaux_Gain_Derive_NCEP.append(perr_bid_NCEP_final)
        print(f"Gain/Drift NCEP calculated with curve_fit : {val_bid_NCEP} error : {perr_bid_NCEP_final}")



        params_morceaux_Gain_Derive_NCEP.append(val_bid_NCEP)

        if (breaks_NCEP[i+1]-breaks_NCEP[i]>nb_day_carryover):
            print(f"Piece {i+1} lasts more than {nb_day_carryover} days : CarryOver Estimation")
            initial_guess = [1, 0, 0]  # Valeurs initiales pour G et D et C
            val_bid_NCEP_CarryOver, covariance,info,mesg,ier = curve_fit(model_Gain_Derive_CarryOver, [var1_seg,var2_seg,delta_T_seg],NCEP_PPOX[mask],p0=initial_guess,nan_policy='omit',full_output=True)
            #val_bid2, covariance,info,mesg,ier = curve_fit(model_Gain_Derive_CarryOver, [var1_seg,var2_seg,delta_T_seg],ys*dsair['PPOX_DOXY'][mask],p0=initial_guess,nan_policy='omit',full_output=True)
            #print(f"val_bid NCEP CarryOver2: {val_bid2}")
            params_morceaux_Gain_Derive_CarryOver.append(val_bid_NCEP_CarryOver)  
            perr_bid_CarryOver = np.sqrt(np.diag(covariance))
            perr_morceaux_Gain_Derive_CarryOver.append(perr_bid_CarryOver)
            print(f"Gain/Drift NCEP CarryOver calculated with curve_fit : {val_bid_NCEP_CarryOver} error : {perr_bid_CarryOver}")


        else:
            print(f"Piece {i+1} lasts less than {nb_day_carryover} days : No CarryOver Estimation")
            print(f"Gain/Drift NCEP calculated with curve_fit : {val_bid_NCEP} error : {perr_bid_NCEP_final}")
            val_bid = np.array([val_bid_NCEP[0],0,val_bid_NCEP[1]])
            perr_bid = np.array([perr_bid_NCEP_final[0],0,perr_bid_NCEP_final[1]])
            params_morceaux_Gain_Derive_CarryOver.append(val_bid)   
            perr_morceaux_Gain_Derive_CarryOver.append(perr_bid)

        
    
    for b in breaks_NCEP:
        plt.axvline(b, color="green", ls="--", alpha=0.7)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Fit par morceaux", "Données"])
    plt.grid(True)
    plt.show()


# ### Results creation

# In[49]:


if test_piece==1:
    params_morceaux_Gain_Derive_NCEP = np.array(params_morceaux_Gain_Derive_NCEP)
    params_morceaux_Gain_Derive_CarryOver = np.array(params_morceaux_Gain_Derive_CarryOver)
    params_morceaux_Gain_Derive_CarryOver = params_morceaux_Gain_Derive_CarryOver[:, [0, 2]]
    params_morceaux_Gain_Derive_WOA = np.array(params_morceaux_Gain_Derive_WOA)

    perr_morceaux_Gain_Derive_NCEP = np.array(perr_morceaux_Gain_Derive_NCEP)
    perr_morceaux_Gain_Derive_CarryOver = np.array(perr_morceaux_Gain_Derive_CarryOver)
    perr_morceaux_Gain_Derive_CarryOver = perr_morceaux_Gain_Derive_CarryOver[:, [0, 2]]
    perr_morceaux_Gain_Derive_WOA = np.array(perr_morceaux_Gain_Derive_WOA)

    dict_corr = {'Gain/Derive Piece WOA' : np.stack((params_morceaux_Gain_Derive_WOA, perr_morceaux_Gain_Derive_WOA), axis=0),
                 'Gain/Derive Piece Ncep' : np.stack((params_morceaux_Gain_Derive_NCEP,perr_morceaux_Gain_Derive_NCEP),axis=0),
                 'Gain/Derive Ncep Piece CarryOver' : np.stack((params_morceaux_Gain_Derive_CarryOver,perr_morceaux_Gain_Derive_CarryOver),axis=0),
                 'Gain/Derive Ncep':np.stack((params_Gain_Derive_NCEP,perr_Gain_Derive_NCEP),axis=0),'Gain NCEP':np.stack((params_Gain_NCEP,perr_Gain_NCEP),axis=0),
                 'Gain/Derive NCEP CarryOver':np.stack((params_Gain_Derive_NCEP_CarryOver,perr_Gain_Derive_NCEP_CarryOver),axis=0)}

    write_param_results(dict_corr,num_float)


# ### Plots

# In[50]:


dict_corr = {'GAIN WOA' : params_Gain_WOA,'Gain NCEP' : params_Gain_NCEP,'Gain/Derive WOA' : params_Gain_Derive_WOA,
             'Gain/Derive Ncep' : params_Gain_Derive_NCEP,'Gain Ncep CarryOver' : params_Gain_NCEP_CarryOver,'Gain/Derive Ncep CarryOver' : params_Gain_Derive_NCEP_CarryOver}
breakpoint_list=[[]] * len(dict_corr)
_=plot_cmp_corr_NCEP(dict_corr,breakpoint_list,dsair,NCEP_PPOX,delta_T_NCEP,my_cmap)
_=plot_cmp_corr_WOA(dict_corr, breakpoint_list,ds_argo_interp, ds_woa_interp, delta_T_WOA,my_cmap)


# In[51]:


if test_piece==1:
    dict_corr = {'Gain/Derive Piece WOA' : params_morceaux_Gain_Derive_WOA,
                 'Gain/Derive Piece Ncep' : params_morceaux_Gain_Derive_NCEP,'Gain/Derive Ncep Piece CarryOver' : params_morceaux_Gain_Derive_CarryOver}
    breaks_WOA_plot = breaks_WOA.copy()
    breaks_NCEP_plot = breaks_NCEP.copy()
    breaks_NCEP_plot[-1]=np.max([delta_T_WOA[-1],delta_T_NCEP[-1]])
    breaks_WOA_plot[-1]=np.max([delta_T_WOA[-1],delta_T_NCEP[-1]])
    breaks_NCEP_plot[0]=np.min([delta_T_WOA[0],delta_T_NCEP[0]])
    breaks_WOA_plot[0]=np.min([delta_T_WOA[0],delta_T_NCEP[0]])
    breakpoint_list=[breaks_WOA_plot,breaks_NCEP_plot,breaks_NCEP_plot]
    _=plot_cmp_corr_NCEP(dict_corr,breakpoint_list,dsair,NCEP_PPOX,delta_T_NCEP,my_cmap)
    plt.show()
    plt.savefig(os.path.join(rep_fic_fig,num_float +'_cmp_corr_NCEP_piece.png'))
    _=plot_cmp_corr_WOA(dict_corr, breakpoint_list,ds_argo_interp, ds_woa_interp, delta_T_WOA,my_cmap)
    plt.show()
    plt.savefig(os.path.join(rep_fic_fig,num_float +'_cmp_corr_PSATWOA_piece.png'))


# ### The user must decide which correction to apply

# In[53]:


# Which correction to keep/apply
# Here, the user must decide which correction to keep to estimate a supplement gain with CTD.
# 
# corr_to_keep = 1 ==> WOA GAIN
# corr_to_keep = 2 ==> WOA Gain/Drift
# corr_to_keep = 3 ==> NCEP Gain without CarryOver
# corr_to_keep = 4 ==> NCEP Gain with CarryOver
# corr_to_keep = 5 ==> NCEP Gain/Drift without CarryOver
# corr_to_keep = 6 ==> NCEP Gain/Drift with CarryOver
# corr_to_keep = 7 ==> Piece NCEP Gain/Drift
# corr_to_keep = 8 ==> Piece NCEP Gain/Drift with Carrryover
# corr_to_keep = 9 ==> Piece WOA Gain/Drift

options = {
    1: 'WOA Gain',
    2: 'WOA Gain/Drift',
    3: 'NCEP Gain without CarryOver',
    4: 'NCEP Gain with CarryOver',
    5: 'NCEP Gain/Drift without Carryover',
    6: 'NCEP Gain/Drift with CarryOver',
    7: 'Piece NCEP Gain/Drift',
    8: 'Piece NCEP Gain/Drift with CarryOver',
    9: 'Piece WOA Gain/Drift'
}

for key, val in options.items():
    print(f"{key} = '{val}'")

while True:
    try:
        corr_to_keep = int(input("Which Correction ? "))
        if corr_to_keep in options:
            break
        else:
            print("Choice not defined")
    except ValueError:
        print("Enter a integer")

correction_choosen = options[corr_to_keep]
print(f"Selection : {correction_choosen}")

breaks_to_keep = np.array([min(delta_T_Sprof),max(delta_T_Sprof)])
match corr_to_keep:
    case 1:
        params_to_keep = copy.deepcopy(params_Gain_WOA)
        comment_corr = 'Correction with a WOA Gain'
        perr_to_keep = copy.deepcopy(perr_Gain_WOA)
        nb_param = 1
    case 2:
        params_to_keep = copy.deepcopy(params_Gain_Derive_WOA)
        comment_corr = 'Correction with a WOA Gain/Drift'
        perr_to_keep = copy.deepcopy(perr_Gain_Derive_WOA)
        nb_param = 2
    case 3:
        params_to_keep = copy.deepcopy(params_Gain_NCEP)
        comment_corr = 'Correction with a NCEP Gain'
        perr_to_keep = copy.deepcopy(perr_Gain_NCEP)
        nb_param = 1
    case 4:
        params_to_keep = copy.deepcopy(params_Gain_NCEP_CarryOver)
        comment_corr = 'Correction with a NCEP CarryOver Gain'
        perr_to_keep = copy.deepcopy(perr_Gain_NCEP_CarryOver)
        nb_param = 1
    case 5:
        params_to_keep = copy.deepcopy(params_Gain_Derive_NCEP)
        comment_corr = 'Correction with a NCEP Gain/Drift'
        perr_to_keep = copy.deepcopy(perr_Gain_Derive_NCEP)
        nb_param = 2
    case 6:
        params_to_keep = copy.deepcopy(params_Gain_Derive_NCEP_CarryOver)
        comment_corr = 'Correction with a NCEP CarryOver Gain/Drift'
        perr_to_keep = copy.deepcopy(perr_Gain_Derive_NCEP_CarryOver)
        nb_param = 2
    
    case 7 :
        params_to_keep = copy.deepcopy(params_morceaux_Gain_Derive_NCEP)
        comment_corr = 'Correction by piece with a NCEP  Gain/Drift'
        perr_to_keep = perr_morceaux_Gain_Derive_NCEP
        nb_param = 2
        breaks_to_keep = breaks_NCEP

    case 8 :
        params_to_keep = copy.deepcopy(params_morceaux_Gain_Derive_CarryOver)
        comment_corr = 'Correction by piece with a NCEP CarryOver Gain/Drift'
        perr_to_keep = perr_morceaux_Gain_Derive_CarryOver
        nb_param = 2
        breaks_to_keep = breaks_NCEP


    case 9 :
        params_to_keep = copy.deepcopy(params_morceaux_Gain_Derive_WOA)
        comment_corr = 'Correction by piece with a WOA Gain/Drift'
        perr_to_keep = perr_morceaux_Gain_Derive_WOA
        nb_param = 2
        breaks_to_keep = breaks_WOA

nb_segment = len(breaks_to_keep)-1

print(f'breakpoint : {breaks_to_keep}')
breaks_to_keep[0] = min(delta_T_Sprof)
breaks_to_keep[-1] = max(delta_T_Sprof)

print(comment_corr)
print(f'Correction used :  {params_to_keep}')
print(f'Error on correction : {perr_to_keep}')
print(f'Final breakpoint : {breaks_to_keep}')
print(f'nb_piece : {nb_segment}')


# ### Supplement gain estimation from CTD
# - We can estimate the gain from PPOX or DOXY. It's the same only if offset = 0
# - The user can indicate several CTD/cycles comparisons.
# - The Gain is estimated by combining all the datas
# - The Gain is estimated on CTD interpolated on theta ARGO.

# In[54]:


# We estimate a supplement Gain from the CTD without pressure effect.
params_no_corr_pressure = np.array([0])
params_Gain_CTD = np.array([1])
perr_Gain_CTD = np.array([0])
ppox_cruise_theta_tot = np.array([])
doxy_cruise_theta_tot = np.array([])
ppox_cruise_pres_tot = np.array([])
doxy_cruise_pres_tot = np.array([])
ppox_cycle_corr_tot = np.array([])
doxy_cycle_corr_tot = np.array([])

# Compare ARGO Profil  with  CTD. We compare OXYGEN directly
if cmp_ctd==1:
    for i_ctd in np.arange(len(num_ctd)):

        ds_cruise2 = copy.deepcopy(ds_cruise)
        ds_cruise2 = ds_cruise2.where((ds_cruise2['STATION_NUMBER']==num_ctd[i_ctd]) & (ds_cruise2['STATION_CRUISE']==cruise_name[i_ctd]),drop=True)
        ds_cycle = ds_argo_Sprof.where((ds_argo_Sprof['CYCLE_NUMBER']==num_cycle[i_ctd]) & (ds_argo_Sprof['DIRECTION']=='A'),drop=True)

        fig=plot_CTD_Argo_Pos(ds_cycle, ds_cruise, ds_bathy,depths,extend_lon_lat)
        fig.canvas.draw()
        fig.savefig(os.path.join(rep_fic_fig,num_float +'_cmp_CTD_argo.png'))
        
        delta_T_Sprof_en_cours = diff_time_in_days(ds_cycle['JULD'].values,launch_date)
        tab_delta_T= np.tile(delta_T_Sprof_en_cours,(1,len(ds_cycle['N_LEVELS'])))

        if nb_segment>1:
            #index = np.argmax(np.array(breakpoints_cycle) >= num_cycle[i_ctd]) if np.any(np.array(breakpoints_cycle) >= num_cycle[i_ctd]) else -1
            #index = index - 1
            index = next(x for x, val in enumerate(np.array(breaks_to_keep)) if val>= delta_T_Sprof_en_cours)
            if index > 0:
                index = index -1
            params_ok = params_to_keep[index,:]
        else:
            params_ok = params_to_keep
            

        print(f'param_ok : {params_ok}')

        var_bid = ['PSAL','TEMP','PRES']
        info_nok = 0
        str_chaine_to_use = str_chaine
        if which_var==3:
            for var_en_cours in var_bid:
                var_name = var_en_cours + str_chaine
                if var_name in ds_cycle:
                    if np.all(np.isnan(ds_cycle[var_name].values)):
                        print(f" {var_name}: exists, but all NaN.")
                        info_nok = 1     
                else:
                    print(f" {var_name} doesn't exist")
                    info_nok = 1
            if info_nok==1:
                str_chaine_to_use = ''

        if str_chaine_to_use=='':
            print(f"We use RAW Data for CTD/cycle comparaison")
        else:
            print(f"We use Adjusted Data for CTD/cycle comparaison")

        psal = ds_cycle['PSAL'+str_chaine_to_use]  
        pres = ds_cycle['PRES'+str_chaine_to_use]
        temp = ds_cycle['TEMP'+str_chaine_to_use] 
        lon = ds_cycle['LONGITUDE'].values
        lat = ds_cycle['LATITUDE'].values

        lon_2d = lon[:, np.newaxis]  # Résultat de forme (154, 1)
        lon_2d = lon_2d * np.ones((1, psal.shape[1])) 
        lat_2d = lat[:, np.newaxis]  # Résultat de forme (154, 1)
        lat_2d = lat_2d * np.ones((1, psal.shape[1])) 

        psal_absolue_cycle  = gsw.SA_from_SP(psal,pres,lon_2d,lat_2d)
        ct_cycle = gsw.CT_from_t(psal_absolue_cycle, temp, pres)
        ana_dens_cycle = gsw.rho(psal_absolue_cycle,ct_cycle,0)
        O2_cycle_umolL = umolkg_to_umolL(ds_cycle['DOXY'],ds_cycle['DOXY'].units,ana_dens_cycle[0])
        ppox_cycle = O2ctoO2p(O2_cycle_umolL,ds_cycle['TEMP'+str_chaine_to_use].isel(N_PROF=0),ds_cycle['PSAL'+str_chaine_to_use].isel(N_PROF=0),ds_cycle['PRES'+str_chaine_to_use].isel(N_PROF=0)) 
        # Apply Initial Correction (from NCEP or WOA)
        if nb_param == 1:
            ppox_cycle_corr = model_Gain(ppox_cycle,*params_ok)
            doxy_cycle_corr = model_Gain(ds_cycle['DOXY'],*params_ok)
        else:
            ppox_cycle_corr = model_Gain_Derive([ppox_cycle,tab_delta_T],*params_ok)
            doxy_cycle_corr = model_Gain_Derive([ds_cycle['DOXY'],tab_delta_T],*params_ok)

        # We concatenate all cycles
        doxy_cycle_corr_tot = np.concatenate((doxy_cycle_corr_tot,doxy_cycle_corr),axis=None)
        ppox_cycle_corr_tot = np.concatenate((ppox_cycle_corr_tot,ppox_cycle_corr),axis=None)


        # PPOX CTD
        psal = ds_cruise2['PSAL']  
        pres = ds_cruise2['PRES']
        temp = ds_cruise2['TEMP'] 
        lon = ds_cruise2['LONGITUDE'].values
        lat = ds_cruise2['LATITUDE'].values

        lon_2d = lon[:, np.newaxis]  # Résultat de forme (154, 1)
        lon_2d = lon_2d * np.ones((1, psal.shape[1])) 
        lat_2d = lat[:, np.newaxis]  # Résultat de forme (154, 1)
        lat_2d = lat_2d * np.ones((1, psal.shape[1])) 

        psal_absolue_cruise  = gsw.SA_from_SP(psal,pres,lon_2d,lat_2d)
        ct_cruise = gsw.CT_from_t(psal_absolue_cruise, temp, pres)
        ana_dens_cruise = gsw.rho(psal_absolue_cruise,ct_cruise,0)
        O2_cruise_umolL = umolkg_to_umolL(ds_cruise2['DOXY'],ds_cycle['DOXY'].units,ana_dens_cruise[0])
        ppox_cruise = O2ctoO2p(O2_cruise_umolL,ds_cruise2['TEMP'].isel(PROF=0),ds_cruise2['PSAL'].isel(PROF=0),ds_cruise2['PRES'].isel(PROF=0)) 
        
        # Interp CTD on ARGO pressure
        ppox_cruise_interp = np.interp(ds_cycle['PRES'],ds_cruise2['PRES'].isel(PROF=0),ppox_cruise[0])
        doxy_cruise_interp = np.interp(ds_cycle['PRES'],ds_cruise2['PRES'].isel(PROF=0),ds_cruise2['DOXY'].isel(PROF=0)) 
        ppox_cruise_pres_tot = np.concatenate((ppox_cruise_pres_tot,ppox_cruise_interp),axis=None)
        doxy_cruise_pres_tot = np.concatenate((doxy_cruise_pres_tot,doxy_cruise_interp),axis=None)

         # Compute ARGO theta.
        pres_cycle = ds_cycle['PRES'+str_chaine_to_use].values 
        theta_cycle = gsw.pt_from_CT(psal_absolue_cycle, ct_cycle)
        
        # Interp CTD on theta Float (interp_climatology_float from first version from pyowc)
        theta_cruise = gsw.pt_from_CT(psal_absolue_cruise, ct_cruise)
        oxy_cruise_interp_on_theta, pres_cruise_interp_on_theta = interp_climatology_float(
            ds_cruise2['DOXY'].transpose(), theta_cruise.transpose(),ds_cruise2['PRES'].transpose(), doxy_cycle_corr[0,:], theta_cycle[0,:], pres_cycle[0,:])
        oxy_cruise_interp_on_theta = np.transpose(oxy_cruise_interp_on_theta)
        pres_cruise_interp_on_theta = np.transpose(pres_cruise_interp_on_theta)
        ppox_cruise_interp_on_theta, pres_cruise_interp_on_theta_bid = interp_climatology_float(
            ppox_cruise.transpose(), theta_cruise.transpose(),ds_cruise['PRES'].transpose(), ppox_cycle_corr[0,:], theta_cycle[0,:], pres_cycle[0,:])
        ppox_cruise_interp_on_theta = np.transpose(ppox_cruise_interp_on_theta)


        diff_pres = np.abs(pres_cruise_interp_on_theta-pres_cycle)
        # If pres_cruise_interp_on_theta is too far from intial pressure ==> NaN
        isbad = diff_pres[0] > 200 # 50
        oxy_cruise_interp_on_theta[0][isbad]=np.nan
        ppox_cruise_interp_on_theta[0][isbad]=np.nan

        ppox_cruise_theta_tot = np.concatenate((ppox_cruise_theta_tot,ppox_cruise_interp_on_theta),axis=None)
        doxy_cruise_theta_tot = np.concatenate((doxy_cruise_theta_tot,oxy_cruise_interp_on_theta),axis=None)
        

        ds_cruise2.close()
        ds_cycle.close()
 
    
    # Solution without pressure correction
    # CTD Gain estimation on all cycles/CTDs On pressure
    initial_guess = 0
    params_Gain_CTD_V1, covariance = curve_fit(model_Gain, ppox_cycle_corr_tot/ppox_cruise_pres_tot, ppox_cruise_pres_tot/ppox_cruise_pres_tot, p0=initial_guess,nan_policy='omit')
    perr_Gain_CTD_V1 = np.sqrt(np.diag(covariance))

    params_Gain_CTD_V2, covariance = curve_fit(model_Gain, doxy_cycle_corr_tot/doxy_cruise_pres_tot, doxy_cruise_pres_tot/doxy_cruise_pres_tot, p0=initial_guess,nan_policy='omit')
    perr_Gain_CTD_V2 = np.sqrt(np.diag(covariance))

    # Solution without pressure correction
    # CTD Gain estimation on all cycles/CTDs On theta
    initial_guess = 0
    params_Gain_CTD_V3, covariance = curve_fit(model_Gain, ppox_cycle_corr_tot/ppox_cruise_theta_tot, ppox_cruise_theta_tot/ppox_cruise_theta_tot, p0=initial_guess,nan_policy='omit')
    perr_Gain_CTD_V3 = np.sqrt(np.diag(covariance))

    params_Gain_CTD_V4, covariance = curve_fit(model_Gain, doxy_cycle_corr_tot/doxy_cruise_theta_tot, doxy_cruise_theta_tot/doxy_cruise_theta_tot, p0=initial_guess,nan_policy='omit')
    perr_Gain_CTD_V4 = np.sqrt(np.diag(covariance))

    print(f"Gain/Error calculated with PPOX on Pressure : {params_Gain_CTD_V1}/{perr_Gain_CTD_V1}")
    print(f"Gain/Error calculated with DOXY on Pressure : {params_Gain_CTD_V2}/{perr_Gain_CTD_V2}")
    print(f"Gain/Error calculated with PPOX on Theta : {params_Gain_CTD_V3}/{perr_Gain_CTD_V3}")
    print(f"Gain/Error calculated with DOXY on Theta : {params_Gain_CTD_V4}/{perr_Gain_CTD_V4}")

    # 08/01/2026 : 
    # We keep the Gain estimated using DOXY on pressure
    print("We keep the gain estimated with DOXY on Pressure")
    params_Gain_CTD = params_Gain_CTD_V2
    perr_Gain_CTD = perr_Gain_CTD_V2

    print("Supplement Gain from CTD (without effect pressure) : ",",".join(f"{val:.46}" for val in params_Gain_CTD))

    for i_ctd in np.arange(len(num_ctd)):
        ds_cruise2 = copy.deepcopy(ds_cruise)
        ds_cruise2 = ds_cruise2.where((ds_cruise2['STATION_NUMBER']==num_ctd[i_ctd]) & (ds_cruise2['STATION_CRUISE']==cruise_name[i_ctd]),drop=True)
        ds_cycle = ds_argo_Sprof.where((ds_argo_Sprof['CYCLE_NUMBER']==num_cycle[i_ctd]) & (ds_argo_Sprof['DIRECTION']=='A'),drop=True)
        plt.figure()
        h2 = plt.plot(doxy_cycle_corr,ds_cycle['PRES'],'.-r')
        h1 = plt.plot(ds_cruise2['DOXY'],ds_cruise['PRES'],'^-b')
        h3 = plt.plot(doxy_cycle_corr*params_Gain_CTD_V2,ds_cycle['PRES'],'xg-')
        h4 = plt.plot(doxy_cycle_corr*params_Gain_CTD_V1,ds_cycle['PRES'],'+-c')

        plt.grid()
        plt.gca().invert_yaxis()
        plt.xlabel('DOXY')
        plt.ylabel('PRES')
        plt.legend([h1[0],h2[0],h3[0],h4[0]],['CTD','DOXY Corr','DOXY Corr with CTD Gain','DOXY Corr with CTD Gain PPOX'])
        plt.show()
        
        ds_cycle.close()
        ds_cruise2.close()
        


# # Supplement Gain and effect pressure estimation from CTD
# ## We estimate the effect pressure and the gain simultaneously
# - The pressure effect must be calculated on DOXY only. <br>
#   The pressure effect is applied on the DOXY (not on PPOX)
# - The pressure effect is calculated on theta levels.
# 
# ## Remarque
# - When “undoing” the pressure effect, PRES and TEMP are used (not PRES_ADJUSTED and TEMP_ADJUSTED). <br>
#   This is because PRES and TEMP are used during ARGO decoding.  <br>
# - When correcting BD files, TEMP and PRES (Raw Data) are also used to undo the initial pressure effect <br>
#   and apply the new pressure effect..
# - The user can use several CTD/cycle comparison. There is one (Gain/pressure effect) for each. <br>
#   Attention : the result given is for the last (CTD/cycle) comparison. The user must adapt to choose the best estimation.

# In[55]:


#from scipy.interpolate import interp1d

# We estimate a pressure effect by using potential temperature.
# Init Pressure correction (0 by default) and a supplement gain from CTD (1 by default)
params_Gain_CTD_with_pressure = np.array([1])
params_corr_pressure = np.array([0])
perr_pressure = np.array([0,0])

print(params_to_keep)
# Pressure effect estimation.
# We compare ARGO O2 with CTD O2.
# According to V. Thierry, we must estimate a Gain and a pressure effect.
# Estimate only a pressure effect seems to be incorrect.
# We do thatfrom the 'first' correction.


#pressure_threshold = 1000. Pour le flotteur 6903080, si on force pressure_threshold à 1500 voire 2000, les resultats en theta sont
# plus proches de la CTD que les resultats en pression. Le code en theta parait OK (17/03/2026)

if cmp_ctd==1:
    for i_ctd in np.arange(len(num_ctd)):
        ds_cruise2 = copy.deepcopy(ds_cruise)
        ds_cruise2 = ds_cruise2.where((ds_cruise2['STATION_NUMBER']==num_ctd[i_ctd]) & (ds_cruise2['STATION_CRUISE']==cruise_name[i_ctd]),drop=True)
        ds_cycle = ds_argo_Sprof.where((ds_argo_Sprof['CYCLE_NUMBER']==num_cycle[i_ctd]) & (ds_argo_Sprof['DIRECTION']=='A'),drop=True)
        delta_T_Sprof_en_cours = diff_time_in_days(ds_cycle['JULD'].values,launch_date)
        tab_delta_T= np.tile(delta_T_Sprof_en_cours,(1,len(ds_cycle['N_LEVELS'])))

        if nb_segment>1:
            #index = np.argmax(np.array(breakpoints_cycle) >= num_cycle[i_ctd]) if np.any(np.array(breakpoints_cycle) >= num_cycle[i_ctd]) else -1
            #index = index - 1
            index = next(x for x, val in enumerate(np.array(breaks_to_keep)) if val>= delta_T_Sprof_en_cours)
            if index > 0:
                index = index -1
            params_ok = params_to_keep[index,:]
        else:
            params_ok = params_to_keep
            

        print(f'param_ok : {params_ok}')


        var_bid = ['PSAL','TEMP','PRES']
        info_nok = 0
        str_chaine_to_use = str_chaine
        if which_var==3:
            for var_en_cours in var_bid:
                var_name = var_en_cours + str_chaine
                if var_name in ds_cycle:
                    if np.all(np.isnan(ds_cycle[var_name].values)):
                        print(f" {var_name}: exists, but all NaN.")
                        info_nok = 1     
                else:
                    print(f" {var_name} doesn't exist")
                    info_nok = 1
            if info_nok==1:
                str_chaine_to_use = ''

        if str_chaine_to_use=='':
            print(f"We use RAW Data for CTD/cycle comparaison")
        else:
            print(f"We use Adjusted Data for CTD/cycle comparaison")

        psal = ds_cruise2['PSAL']  
        pres = ds_cruise2['PRES']
        temp = ds_cruise2['TEMP'] 
        lon = ds_cruise2['LONGITUDE'].values
        lat = ds_cruise2['LATITUDE'].values

        lon_2d = lon[:, np.newaxis]  # Résultat de forme (154, 1)
        lon_2d = lon_2d * np.ones((1, psal.shape[1])) 
        lat_2d = lat[:, np.newaxis]  # Résultat de forme (154, 1)
        lat_2d = lat_2d * np.ones((1, psal.shape[1])) 

        psal_absolue_cruise  = gsw.SA_from_SP(psal,pres,lon_2d,lat_2d)
        ct_cruise = gsw.CT_from_t(psal_absolue_cruise, temp, pres)
        ana_dens_cruise = gsw.rho(psal_absolue_cruise,ct_cruise,0)
        O2_cruise_umolL = umolkg_to_umolL(ds_cruise['DOXY'],ds_cycle['DOXY'].units,ana_dens_cruise[0])
        ppox_cruise = O2ctoO2p(O2_cruise_umolL,ds_cruise2['TEMP'].isel(PROF=0),ds_cruise2['PSAL'].isel(PROF=0),ds_cruise2['PRES'].isel(PROF=0)) 


        psal = ds_cycle['PSAL'+str_chaine_to_use]  
        pres = ds_cycle['PRES'+str_chaine_to_use]
        temp = ds_cycle['TEMP'+str_chaine_to_use] 
        lon = ds_cycle['LONGITUDE'].values
        lat = ds_cycle['LATITUDE'].values

        lon_2d = lon[:, np.newaxis]  # Résultat de forme (154, 1)
        lon_2d = lon_2d * np.ones((1, psal.shape[1])) 
        lat_2d = lat[:, np.newaxis]  # Résultat de forme (154, 1)
        lat_2d = lat_2d * np.ones((1, psal.shape[1])) 

        psal_absolue_cycle  = gsw.SA_from_SP(psal,pres,lon_2d,lat_2d)
        ct_cycle = gsw.CT_from_t(psal_absolue_cycle, temp, pres)
        ana_dens_cycle = gsw.rho(psal_absolue_cycle,ct_cycle,0)
        O2_cycle_umolL = umolkg_to_umolL(ds_cycle['DOXY'],ds_cycle['DOXY'].units,ana_dens_cycle[0])
        ppox_cycle = O2ctoO2p(O2_cycle_umolL,ds_cycle['TEMP'+str_chaine_to_use].isel(N_PROF=0),ds_cycle['PSAL'+str_chaine_to_use].isel(N_PROF=0),ds_cycle['PRES'+str_chaine_to_use].isel(N_PROF=0)) 
        if nb_param == 1:
            ppox_cycle_corr = model_Gain(ppox_cycle,*params_ok)
            doxy_cycle_corr = model_Gain(ds_cycle['DOXY'],*params_ok)
        else:
            ppox_cycle_corr = model_Gain_Derive([ppox_cycle,tab_delta_T],*params_ok)
            doxy_cycle_corr = model_Gain_Derive([ds_cycle['DOXY'],tab_delta_T],*params_ok)


        

        # Compute ARGO and CTD theta.
        # pres_corr_pres_cycle/temp_corr_pres_cycle : useful for pressure effect
        psal_cycle = ds_cycle['PSAL'+str_chaine_to_use].values
        pres_cycle = ds_cycle['PRES'+str_chaine_to_use].values 
        pres_corr_pres_cycle = ds_cycle['PRES'].values 
        temp_cycle = ds_cycle['TEMP'+str_chaine_to_use].values
        temp_corr_pres_cycle = ds_cycle['TEMP'].values
        theta_cycle = gsw.pt_from_CT(psal_absolue_cycle, ct_cycle)

        # Interpolation CTD on ARGO Pressure
        #psal_cruise_interp = np.interp(ds_cycle['PRES'],ds_cruise['PRES'].isel(N_PROF=0),ds_cruise['PSAL'].isel(N_PROF=0))
        #temp_cruise_interp = np.interp(ds_cycle['PRES'],ds_cruise['PRES'].isel(N_PROF=0),ds_cruise['TEMP'].isel(N_PROF=0))
        #pres_cruise_interp = np.interp(ds_cycle['PRES'],ds_cruise['PRES'].isel(N_PROF=0),ds_cruise['PRES'].isel(N_PROF=0))
        #ppox_cruise_interp = np.interp(ds_cycle['PRES'],ds_cruise['PRES'].isel(N_PROF=0),ppox_cruise[0])
        #theta_cruise = sw.ptmp(psal_cruise_interp,temp_cruise_interp,pres_cruise_interp,0)
        doxy_cruise_interp = np.interp(ds_cycle['PRES'],ds_cruise2['PRES'].isel(PROF=0),ds_cruise2['DOXY'].isel(PROF=0))
        mask = pres_cycle[0] >= pressure_threshold
        print('Pressure effect Estimation on Pressure')
        initial_guess = [1,0]   
        params_corr_pressure_V1, covariance = curve_fit(lambda X, G, Gp: model_Gain_pres(X, G, Gp, pcoef2, pcoef3),
                       [doxy_cycle_corr[0][mask],pres_corr_pres_cycle[0][mask],temp_corr_pres_cycle[0][mask]], doxy_cruise_interp[0][mask], p0=initial_guess,nan_policy='omit')
        perr_pressure_V1 = np.sqrt(np.diag(covariance))
        print(f'CTD pressure effect estimated with pressure effect : {params_corr_pressure_V1[1]:.6f} with error {perr_pressure_V1[1]:.6f}')
        print(f'CTD Gain estimated with pressure effect : {params_corr_pressure_V1[0]:.6f} with error {perr_pressure_V1[0]:.6f}')
        
        
        theta_cruise = gsw.pt_from_CT(psal_absolue_cruise, ct_cruise)
        oxy_cruise_interp_on_theta, pres_cruise_interp_on_theta = interp_climatology_float(
            ds_cruise2['DOXY'].transpose(), theta_cruise.transpose(),ds_cruise['PRES'].transpose(), doxy_cycle_corr[0,:], theta_cycle[0,:], pres_cycle[0,:])
        oxy_cruise_interp_on_theta = np.transpose(oxy_cruise_interp_on_theta)
        pres_cruise_interp_on_theta = np.transpose(pres_cruise_interp_on_theta)

        #plt.figure()
        #h1=plt.plot(ds_cruise2['DOXY'],ds_cruise2['PRES'],'.b')
        #h2=plt.plot(oxy_cruise_interp_on_theta[0],pres_cycle[0,:],'.g')
        #h3=plt.plot(doxy_cycle_corr[0],pres_cycle[0,:],'.r')
        #plt.gca().invert_yaxis()
        #plt.grid()
        #_=plt.legend([h1[0],h2[0],h3[0]],['CTD','CTD InterpOnTheta','Argo'])
        #plt.xlabel('DOXY')
        #plt.ylabel('PRES')
        #plt.show()
        
        
        diff_pres = np.abs(pres_cruise_interp_on_theta-pres_cycle)
        # If pres_cruise_interp_on_theta is too far from intial pressure ==> NaN
        isbad = diff_pres[0] > 200
        oxy_cruise_interp_on_theta[0][isbad]=np.nan
        ppox_cruise_interp_on_theta[0][isbad]=np.nan
        
        
        # Keep data under a pressure threshold
        mask = pres_cycle[0] >= pressure_threshold
        print('Pressure effect Estimation on Theta')
        initial_guess = [1,0]   
        params_corr_pressure_V2, covariance = curve_fit(lambda X, G, Gp: model_Gain_pres(X, G, Gp, pcoef2, pcoef3),
                       [doxy_cycle_corr[0][mask],pres_corr_pres_cycle[0][mask],temp_corr_pres_cycle[0][mask]], oxy_cruise_interp_on_theta[0][mask], p0=initial_guess,nan_policy='omit')
        perr_pressure_V2 = np.sqrt(np.diag(covariance))
        print(f'CTD pressure effect estimated with pressure effect : {params_corr_pressure_V2[1]:.6f} with error {perr_pressure_V2[1]:.6f}')
        print(f'CTD Gain estimated with pressure effect : {params_corr_pressure_V2[0]:.6f} with error {perr_pressure_V2[0]:.6f}')

        #
        # 08/01/26 : On choisit l'effet de pression estime sur les niveaux theta (_V2) ou niveaux de pression (_V1). 
        # On decide de garder l'effet de pression estime sur les niveaux de pression. Sur les flotteurs bi-tetes, ca donne un meilleur resultat.
        params_corr_pressure = params_corr_pressure_V1
        perr_pressure = perr_pressure_V1
        

        plt.figure()
        h1 = plt.plot(ds_cruise2['DOXY'][0],ds_cruise2['PRES'][0],'+-b')
        #h1 = plt.plot(oxy_cruise_interp_on_theta[0],ds_cycle['PRES'][0],'+-b')
        h2 = plt.plot(doxy_cycle_corr[0],ds_cycle['PRES'][0],'x-r')
        h3=plt.plot(doxy_cycle_corr[0]*1/(1+(pcoef2*ds_cycle['TEMP']+pcoef3)*ds_cycle['PRES'][0]/1000)*(1+(pcoef2*ds_cycle['TEMP']+ params_corr_pressure[1])*ds_cycle['PRES'][0]/1000),ds_cycle['PRES'][0],'o-g')  
        h4=plt.plot(doxy_cycle_corr[0]*1/(1+(pcoef2*ds_cycle['TEMP']+pcoef3)*ds_cycle['PRES'][0]/1000)*(1+(pcoef2*ds_cycle['TEMP']+params_corr_pressure[1])*ds_cycle['PRES'][0]/1000)*params_corr_pressure[0],ds_cycle['PRES'][0],'o-y')        
        h5=plt.plot(doxy_cycle_corr[0]*1/(1+(pcoef2*ds_cycle['TEMP']+pcoef3)*ds_cycle['PRES'][0]/1000)*(1+(pcoef2*ds_cycle['TEMP']+params_corr_pressure_V2[1])*ds_cycle['PRES'][0]/1000)*params_corr_pressure_V2[0],ds_cycle['PRES'][0],'o-m')        
        plt.gca().invert_yaxis()
        plt.grid()
        _=plt.legend([h1[0],h2[0],h3[0], h4[0],h5[0]],['CTD','ARGO Corr','ARGO Corr+Pressure Effect','ARGO Corr + CTD Gain + CTD Pressure effect','ARGO Corr + CTD Gain + CTD Pressure effect (theta)'])
        plt.xlabel('DOXY')
        plt.ylabel('PRES')

        params_Gain_CTD_with_pressure = np.array([params_corr_pressure[0]])
        params_corr_pressure = np.array([params_corr_pressure[1]])
        
        print("We keep the gain estimated with DOXY on Pressure")
        print(params_Gain_CTD_with_pressure,params_corr_pressure)

        #initial_guess =[1]
        #params_Gain_CTD_CK, covariance = curve_fit(model_Gain, doxy_cycle_pour_pres, doxy_cruise_pour_pres, p0=initial_guess,nan_policy='omit')
        #print(params_Gain_CTD_CK)
        
        ds_cruise.close()
        ds_cycle.close()


# ## Error and correction creation
# ### Add 1 or 2 column of 0 in the error array (column for the error on drift if necessary and for the pressure effect)
# ###  Create 4 solutions and the associated error
# - The initial one
# - The initial one with a CTD gain
# - The initial one with a pressure effect
# - The initial one with a CTD gain and pressure effect

# In[56]:


# Create error array with or without pressure effect.
print(perr_to_keep)
perr_final = perr_to_keep.copy()
derive_final = 0
if nb_param == 1:
    if nb_segment==1:
        perr_final = np.append(perr_final, 0) # error on drift
        perr_final = np.append(perr_final, 0) # error on pressure effect
    else: # On ne passe jamais la. Si plusieurs segments, alors on estime gain et derive, donc nb_param>1
        zeros_column = np.zeros((perr_final.shape[0], 2))  # Créer 2 colonne de zéros
        perr_final = np.hstack((perr_final, zeros_column)) 
else :
    if nb_segment==1:
        derive_final = params_to_keep[1]
        perr_final = np.append(perr_final, 0) # error on pressure effect
    else:
        zeros_column = np.zeros((perr_final.shape[0], 1))  # Créer une colonne de zéros
        perr_final = np.hstack((perr_final, zeros_column)) 
        derive_final = params_to_keep[:,1]

print(perr_to_keep)
print(perr_final)


# In[57]:


if nb_segment==1:
    corr_final_without_pressure_correction = np.array([params_to_keep[0],derive_final,params_no_corr_pressure[0]])
    corr_final_with_pressure_correction = np.array([params_to_keep[0],derive_final,params_corr_pressure[0]])
    corr_final_CTD_without_pressure_correction = np.array([params_Gain_CTD[0] * params_to_keep[0],derive_final,params_no_corr_pressure[0]])
    corr_final_CTD_with_pressure_correction = np.array([params_Gain_CTD_with_pressure[0] * params_to_keep[0],derive_final,params_corr_pressure[0]])
    perr_final_pressure = perr_final.copy()
    perr_final_pressure[2] = perr_pressure[1] # correction to keep with pressure effect
    perr_final_CTD = [np.sqrt(perr_final[0]*perr_final[0] + perr_Gain_CTD[0]*perr_Gain_CTD[0]),perr_final[1],perr_final[2]] # correction to keep with CTD supplement Gain
    perr_final_CTD_with_pressure = [np.sqrt(perr_final[0]*perr_final[0] +  perr_Gain_CTD[0]*perr_Gain_CTD[0] + perr_pressure[0]*perr_pressure[0]),perr_final[1],perr_pressure[1]] # correction to keep with CTD effect pressure and CTD Gain
else :
    corr_final_without_pressure_correction = np.append(params_to_keep, np.zeros((params_to_keep.shape[0],1)), axis=1) # Ajout colonne de 0 pour effet de pression
    corr_final_with_pressure_correction = np.append(params_to_keep, np.full((params_to_keep.shape[0],1),params_corr_pressure[0]), axis=1)
    corr_final_CTD_without_pressure_correction = corr_final_without_pressure_correction.copy()
    corr_final_CTD_without_pressure_correction[:,0] = corr_final_CTD_without_pressure_correction[:,0] * params_Gain_CTD[0]
    corr_final_CTD_with_pressure_correction = corr_final_with_pressure_correction.copy()
    corr_final_CTD_with_pressure_correction[:,0] = corr_final_CTD_with_pressure_correction[:,0] * params_Gain_CTD_with_pressure[0]
    perr_final_pressure = perr_final.copy()
    perr_final_pressure[:,2] = perr_pressure[1]
    perr_final_CTD = perr_final.copy()
    perr_final_CTD[:,0] = np.sqrt(perr_final_CTD[:,0]*perr_final_CTD[:,0] + perr_Gain_CTD[0]*perr_Gain_CTD[0]) # correction to keep with CTD supplement Gain
    perr_final_CTD_with_pressure = perr_final_pressure.copy()
    perr_final_CTD_with_pressure[:,0] = np.sqrt(perr_final[:,0]*perr_final[:,0] + perr_final_CTD[:,0]*perr_final_CTD[:,0] + perr_pressure[0]*perr_pressure[0]) # correction to keep with CTD effect pressure and CTD Gain
    
    


# In[58]:


#corr_test_CK = np.array([params_to_keep[0]*params_Gain_CTD_CK[0],derive_final,params_no_corr_pressure[0]])
if cmp_ctd==1:
    for i_ctd in np.arange(len(num_ctd)):
        ds_cruise2 = copy.deepcopy(ds_cruise)
        ds_cruise2 = ds_cruise2.where((ds_cruise2['STATION_NUMBER']==num_ctd[i_ctd]) & (ds_cruise2['STATION_CRUISE']==cruise_name[i_ctd]),drop=True)
        ds_cycle = ds_argo_Sprof.where((ds_argo_Sprof['CYCLE_NUMBER']==num_cycle[i_ctd]) & (ds_argo_Sprof['DIRECTION']=='A'),drop=True)
        delta_T_Sprof_en_cours = diff_time_in_days(ds_cycle['JULD'].values,launch_date)
        if nb_segment>1:
            #index = np.argmax(np.array(breakpoints_cycle) >= num_cycle[i_ctd]) if np.any(np.array(breakpoints_cycle) >= num_cycle[i_ctd]) else -1
            #index = index - 1
            index = next(x for x, val in enumerate(np.array(breaks_to_keep)) if val>= delta_T_Sprof_en_cours)
            if index > 0:
                index = index -1
            dict_corr = {'Initial Correction' : corr_final_without_pressure_correction[index,:],'initial Correction with PresEffect' : corr_final_with_pressure_correction[index,:],'Correction CTD' : corr_final_CTD_without_pressure_correction[index,:],
                     'Correction CTD with PresEffect' : corr_final_CTD_with_pressure_correction[index,:]}
        else:
            dict_corr = {'Initial Correction' : corr_final_without_pressure_correction,'initial Correction with PresEffect' : corr_final_with_pressure_correction,'Correction CTD' : corr_final_CTD_without_pressure_correction,
                     'Correction CTD with PresEffect' : corr_final_CTD_with_pressure_correction}
            
        R2=calcul_R2_ARGO_CTD(ds_cruise2,ds_cycle,dict_corr,launch_date,pcoef2,pcoef3) 
        _=plot_cmp_ARGO_CTD(ds_cruise2,ds_cycle,dict_corr,launch_date,pcoef2,pcoef3) 


# In[59]:


plt.savefig(os.path.join(rep_fic_fig,num_float +'_cmp_differents_corr_with_CTD.png'))




# # Choose between 4 corrections 
# - Initial correction
# - Intial correction with CTD gain
# - Initial correction with pressure effect
# - Initial correction with CTD gain and pressure effect

# In[60]:


#
# Which correction to apply (without or with pressure correction).
#
options = {
    1: 'Initial Correction',
    2: 'initial Correction with pressure effect',
    3: 'Correction with CTD Gain',
    4: 'Correction with CTD Gain and pressure effect'
}

# afficher les choix
for key, val in options.items():
    print(f"{key} = '{val}'")

# lire le choix de l'utilisateur
while True:
    try:
        corr_to_apply = int(input("Choose final correction: "))
        if corr_to_apply in options:
            break
        else:
            print("Choice not defined.")
    except ValueError:
        print("Enter an integer.")

correction_choosen = options[corr_to_apply]
print(f"Selection : {correction_choosen}")

if corr_to_apply == 1:
    corr_final_to_use = corr_final_without_pressure_correction
    perr_to_use = perr_final
elif corr_to_apply == 2 :
    corr_final_to_use = corr_final_with_pressure_correction
    comment_corr = comment_corr + ' and a pressure effect'
    perr_to_use = perr_final_pressure
elif corr_to_apply == 3:
    corr_final_to_use = corr_final_CTD_without_pressure_correction
    comment_corr = comment_corr + ' and CTD Gain'
    perr_to_use = perr_final_CTD
else:
    corr_final_to_use = corr_final_CTD_with_pressure_correction
    comment_corr = comment_corr + ' and CTD Gain and pressure effect'
    perr_to_use = perr_final_CTD_with_pressure

comment_corr = 'Final correction : '+ comment_corr
print(comment_corr)
print(corr_final_to_use)
print(perr_to_use)


# In[61]:


if cmp_ctd == 1:
    if nb_segment>1:
            #index = np.argmax(np.array(breakpoints_cycle) >= num_cycle[i_ctd]) if np.any(np.array(breakpoints_cycle) >= num_cycle[i_ctd]) else -1
            #index = index - 1
            index = next(x for x, val in enumerate(np.array(breaks_to_keep)) if val>= delta_T_Sprof_en_cours)
            if index > 0:
                index = index -1
                dict_corr = {'Final Result' : corr_final_to_use[index,:]}

            else:
                dict_corr = {'Final Result' : corr_final_to_use}
            
    r2=calcul_R2_ARGO_CTD(ds_cruise2,ds_cycle,dict_corr,launch_date,pcoef2,pcoef3) 
    print(f"R2 Value for {comment_corr} : {r2}")


# In[62]:


dict_corr1 = {'Gain WOA' : np.stack((params_Gain_WOA,perr_Gain_WOA),axis=0),'Gain/Derive WOA':np.stack((params_Gain_Derive_WOA,perr_Gain_Derive_WOA),axis=0),
                  'Gain NCEP' : np.stack((params_Gain_NCEP,perr_Gain_NCEP),axis=0),'Gain Ncep CarryOver' : np.stack((params_Gain_NCEP_CarryOver,perr_Gain_NCEP_CarryOver),axis=0),
                  'Gain/Derive Ncep':np.stack((params_Gain_Derive_NCEP,perr_Gain_Derive_NCEP),axis=0),'Gain/Derive Ncep CarryOver':np.stack((params_Gain_Derive_NCEP_CarryOver,perr_Gain_Derive_NCEP_CarryOver),axis=0)}


dict_corr3 = {'Initial' : np.stack((corr_final_without_pressure_correction, perr_final), axis=0),
                 'Initial with pressure effect' : np.stack((corr_final_with_pressure_correction,perr_final_pressure),axis=0),'Initial with CTD Gain' : np.stack((corr_final_CTD_without_pressure_correction,perr_final_CTD),axis=0),
                 'Initial with CTD Gain and pressure effect' : np.stack((corr_final_CTD_with_pressure_correction,perr_final_CTD_with_pressure),axis=0)}

dict_corr4 = {comment_corr : np.stack((corr_final_to_use,perr_to_use),axis=0)}
if test_piece==1:
    dict_corr2 = {'Gain/Derive Piece WOA' : np.stack((params_morceaux_Gain_Derive_WOA, perr_morceaux_Gain_Derive_WOA), axis=0),
                 'Gain/Derive Piece Ncep' : np.stack((params_morceaux_Gain_Derive_NCEP,perr_morceaux_Gain_Derive_NCEP),axis=0),'Gain/Derive Ncep Piece CarryOver' : np.stack((params_morceaux_Gain_Derive_CarryOver,perr_morceaux_Gain_Derive_CarryOver),axis=0)}
    dict_corr = dict_corr1 | dict_corr2 | dict_corr3 | dict_corr4
else:
    dict_corr = dict_corr1 | dict_corr3 | dict_corr4

write_param_results(dict_corr,num_float,fic_res_ASCII,cmp_ctd,num_ctd,num_cycle,cruise_name,pressure_threshold)


# In[63]:


# Comparaison correction finale avec NCEP ET PSAT WOA
#dict_corr = {'Final Correction' : np.stack((corr_final_without_pressure_correction, perr_final_pressure), axis=0),'Final Correction2' : np.stack((corr_final_without_pressure_correction, perr_final_pressure), axis=0)}
dict_corr = {'Final Correction' : np.stack((corr_final_to_use, perr_to_use), axis=0)}
#if nb_segment==1:
#    breakpoint_list=[[]]*len(dict_corr)
#else:
#    breakpoint_list = [breaks_to_keep]*len(dict_corr)

breakpoint_list = [breaks_to_keep]*len(dict_corr)

_=plot_cmp_corr_NCEP_with_error(dict_corr,breakpoint_list,dsair,NCEP_PPOX,delta_T_NCEP,my_cmap)


# In[64]:


plt.savefig(os.path.join(rep_fic_fig,num_float +'_cmp_NCEP_final_corr.png'))


# In[65]:


_=plot_cmp_corr_WOA_with_error(dict_corr,breakpoint_list,ds_argo_interp, ds_woa_interp, delta_T_WOA,my_cmap)


# In[66]:


plt.savefig(os.path.join(rep_fic_fig,num_float +'_cmp_PSATWOA_final_corr.png'))


# # Final plots

# In[67]:


plot_cmp_correction_with_WOA(ds_argo_Sprof,delta_T_Sprof,breaks_to_keep,corr_final_without_pressure_correction,corr_final_to_use,ds_woa_interp_on_ARGO,str_chaine,
                            pcoef2,pcoef3)


# # B File Correction

# In[57]:


# BR/BD files correction
val_bid = os.path.join(rep_data_argo,num_float,'profiles','B?' +num_float + '_???.nc')
fic_argo = glob.glob(val_bid)
fic_argo.sort()


#print(breaks_to_keep)

for i_fic in range(0,len(fic_argo)):
    fic_en_cours = fic_argo[i_fic]
    fic_res = os.path.join(rep_fic_nc,os.path.basename(fic_en_cours))
    fic_res = fic_res.replace('BR','BD',1)
    #fic_res2 = os.path.join(rep_fic_nc,os.path.basename(fic_en_cours))
    #fic_res2 = fic_res.replace('BR','BK',1)
    #fic_res2 = fic_res.replace('BD','BK',1)
    ds = xr.open_dataset(fic_en_cours,engine='argo')
    cycle_en_cours = ds['CYCLE_NUMBER'].values[0]
    juld_en_cours = ds['JULD'].values[0]
    delta_T_en_cours = diff_time_in_days(juld_en_cours,launch_date)
    if (cycle_en_cours >= first_cycle_to_use) & (cycle_en_cours<=last_cycle_to_use):
        if nb_segment>1:
            index = next((x for x, val in enumerate(np.array(breaks_to_keep)) if val>= delta_T_en_cours),None)
            if index is None :
                if delta_T_en_cours < breaks_to_keep[0] :
                    index = 0
                elif delta_T_en_cours > breaks_to_keep[-1] :
                    index = len(breaks_to_keep) - 1
            if index > 0:
                index = index -1
            corr_final = corr_final_to_use[index,:]
        else :
            corr_final = corr_final_to_use
        
        coef_pres = corr_final[2]
        gain_final = corr_final[0]
        derive_final = corr_final[1]  
        coef_corr = f'INCLINE_T=0, SLOPE={gain_final}, DRIFT={derive_final}, COEF_PRES ={coef_pres},OFFSET=0.000000'
        eq_corr = 'DOXY2 = DOXY / (1 + (Pcoef2*TEMP + coef3)*PRES/1000) * (1 + (Pcoef2*TEMP + coef_pres)*PRES/1000),DOXY_ADJUSTED=OFFSET+(SLOPE*(1+DRIFT/100.*(profile_date_juld-launch_date_juld)/365)+INCLINE_T*TEMP)*DOXY2'
        print(f'Correction de {fic_en_cours} avec {corr_final}')
        if os.path.exists(fic_res):
            os.remove(fic_res)
        corr_file(fic_en_cours,fic_res,launch_date,comment_corr,coef_corr,eq_corr,pcoef2,pcoef3,gain_final,derive_final,coef_pres,percent_relative_error)
        #corr_file_with_ppox(fic_en_cours,fic_res2,launch_date,comment_corr,coef_corr,eq_corr,pcoef2,pcoef3,gain_final,derive_final,coef_pres,percent_relative_error)

# Rename final directory with the user's choice for firt and final correction.
new_rep = os.path.join(os.path.dirname(rep_fic_res_final),num_float+'_Corr'+str(corr_to_keep)+'_Corr'+str(corr_to_apply))
if os.path.exists(new_rep):
    print(f"Directoty '{new_rep}' already exists.")
    answer = input("Delete it ? (y/n) ").strip().lower()
    if answer == 'y':
        shutil.rmtree(new_rep)
        os.rename(rep_fic_res_final, new_rep)
        print(f"Directory '{rep_fic_res_final}' renamed in '{new_rep}'.")
    else:
        print("No directory rename.")
else:
    os.rename(rep_fic_res_final, new_rep)
    print(f"Directory '{rep_fic_res_final}' renamed in '{new_rep}'.")


# In[ ]:




