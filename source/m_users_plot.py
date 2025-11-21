# Import modules
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
import seawater as sw
from m_users_fonctions import O2ctoO2s, diff_time_in_days, umolkg_to_umolL
import copy

def plot_WMO_position(ds_WMO: xr.Dataset,ds_bathy: xr.Dataset,depths: np.ndarray,extend_val: float) -> None :
    """ Function to plot longitude/latitude with bathymetry

    Parameters
    -----------
    ds_WMO : xr.Dataset
        contains :
        LONGITUDES
        LATITUDES
        CYCLE_NUMBER
        PLATFORM_NUMBER (WMO)
        (from Sprof Netcdf ARGO file)
    ds_bathy : xr.Dataset
        contains : 
        x : Longitudes
        y : Latitude
        z : bathymetry
    depths :np.ndarray
        Bathymetry depths to contour
    extend_val : float
        The value to define the limits of the plot
        (min(LONGITUDES)-extend val max(LONGITUDES+extend_val)
        (min(LATITUDES)-extend val max(LATITUDES+extend_val)

    Returns
    -------
    None
    Only a plot is shown.
    The plot plots the lon/lat and the bathymetry.
    We use a Mercator projection.

    """
    ds_bathy_filtered = ds_bathy.where(
        (ds_bathy['x'] >= ds_WMO['LONGITUDE'].min() - extend_val) &
        (ds_bathy['x'] <= ds_WMO['LONGITUDE'].max() + extend_val) &
        (ds_bathy['y'] >= ds_WMO['LATITUDE'].min() - extend_val) &
        (ds_bathy['y'] <= ds_WMO['LATITUDE'].max() + extend_val),
        drop=True
    )

    # Position on projection Mercator
    N = len(depths)
    nudge = 0.01  # shift bin edge slightly to include data
    boundaries = [min(depths)] + sorted(depths+nudge)  # low to high
    norm = matplotlib.colors.BoundaryNorm(boundaries, N)
    blues_cm = matplotlib.colormaps['Blues_r'].resampled(N)
    colors_depths = blues_cm(norm(depths))

    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([ds_WMO['LONGITUDE'].min()-extend_val,ds_WMO['LONGITUDE'].max()+extend_val,ds_WMO['LATITUDE'].min()-extend_val,ds_WMO['LATITUDE'].max()+extend_val],
                  crs=ccrs.PlateCarree())

    # Land in grey
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='grey')
    # Ocean
    ax.add_feature(cfeature.OCEAN) #, edgecolor='none', facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)

    cs = ax.contourf(ds_bathy_filtered['x'],ds_bathy_filtered['y'], ds_bathy_filtered['z'], levels=depths.tolist(),transform=ccrs.PlateCarree(),cmap=plt.cm.bone)
    sm = plt.cm.ScalarMappable(cmap=blues_cm, norm=norm)
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', label='Depth (m)', shrink=0.7,fraction=0.05, pad=0.2)

    # Positions
    longitudes = ds_WMO['LONGITUDE']
    latitudes = ds_WMO['LATITUDE']
    num_positions = len(longitudes)
    colors = plt.cm.tab10(np.arange(num_positions) // 10 % 10)  # 1 color by 10
    cycle_number_plot = ds_WMO['CYCLE_NUMBER'].values.astype(int)
    # plot position with color and add the cycle number
    ax.plot(longitudes, latitudes, color='black', linewidth=0.8, transform=ccrs.PlateCarree(), zorder=1)
    for i, (lon, lat) in enumerate(zip(longitudes, latitudes)):
        ax.scatter(lon, lat, color=colors[i], s=20, transform=ccrs.PlateCarree(), zorder=2,label=f'Group {i // 10}' if i % 10 == 0 else "")
        if i % 5 == 0:
            ax.text(lon, lat, str(cycle_number_plot[i]),color='black', fontsize=8, fontweight='bold',transform=ccrs.PlateCarree(),
                    ha='left', va='center',zorder=3)
    ax.text(lon, lat, str(cycle_number_plot[-1]),color='black', fontsize=8, fontweight='bold',transform=ccrs.PlateCarree(),
             ha='left', va='center',zorder=3)
    # dd a legend by group
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels, loc='upper right', title='Groups')
    ax.set_xlabel('LONGITUDE')
    ax.set_ylabel('LATITUDE')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='-')

    plt.title(f"{ds_WMO['PLATFORM_NUMBER'].isel(N_PROF=0).values} : Region where the floats derived") 
    plt.show()

    return None

def plot_CTD_Argo_Pos(ds_WMO : xr.Dataset, ds_bathy: xr.Dataset,depths: np.ndarray,extend_val: float,rep_ctd : str,fic_ctd : str,num_ctd : np.ndarray,num_cycle:np.ndarray):
    
    """ Function to plot CTD and ARGO longitude/latitude used to compare with bathymetry

    Parameters
    -----------
    ds_WMO : xr.Dataset
        contains :
        LONGITUDES
        LATITUDES
        CYCLE_NUMBER
        PLATFORM_NUMBER (WMO)
        (from Sprof Netcdf ARGO file)
    ds_bathy : xr.Dataset
        contains : 
        x : Longitudes
        y : Latitude
        z : bathymetry
    depths :np.ndarray
        Bathymetry depths to contour
    extend_val : float
        The value to define the limits of the plot
        (min(LONGITUDES)-extend val max(LONGITUDES+extend_val)
        (min(LATITUDES)-extend val max(LATITUDES+extend_val)
    rep_ctd : str
        CTD Directoy
    fic_ctd : str
        CTD NetCDF Files
    num_ctd : np.ndarray
        CTD Number
    num_cycle : np.ndarray
        CYCLE Number
    Returns
    -------
    None
    Only a plot is shown.
    The plot plots the CTD and ARGO lon/lat used for comparison.
    We also plot the bathymetry.
    We use a Mercator projection.

    """
    ds_bathy_filtered = ds_bathy.where(
        (ds_bathy['x'] >= ds_WMO['LONGITUDE'].min() - extend_val) &
        (ds_bathy['x'] <= ds_WMO['LONGITUDE'].max() + extend_val) &
        (ds_bathy['y'] >= ds_WMO['LATITUDE'].min() - extend_val) &
        (ds_bathy['y'] <= ds_WMO['LATITUDE'].max() + extend_val),
        drop=True
    )

    ds_WMO2 = copy.deepcopy(ds_WMO)
    ds_WMO2 = ds_WMO2.where(ds_WMO2['CYCLE_NUMBER'].isin(num_cycle),drop=True)
    # Position on projection Mercator
    N = len(depths)
    nudge = 0.01  # shift bin edge slightly to include data
    boundaries = [min(depths)] + sorted(depths+nudge)  # low to high
    norm = matplotlib.colors.BoundaryNorm(boundaries, N)
    blues_cm = matplotlib.colormaps['Blues_r'].resampled(N)
    colors_depths = blues_cm(norm(depths))

    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([ds_WMO2['LONGITUDE'].min()-extend_val,ds_WMO2['LONGITUDE'].max()+extend_val,ds_WMO2['LATITUDE'].min()-extend_val,ds_WMO2['LATITUDE'].max()+extend_val],
                  crs=ccrs.PlateCarree())

    # Land in grey
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='grey')
    # Ocean
    ax.add_feature(cfeature.OCEAN) #, edgecolor='none', facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)

    cs = ax.contourf(ds_bathy_filtered['x'],ds_bathy_filtered['y'], ds_bathy_filtered['z'], levels=depths.tolist(),transform=ccrs.PlateCarree(),cmap=plt.cm.bone)
    sm = plt.cm.ScalarMappable(cmap=blues_cm, norm=norm)
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical', label='Depth (m)', shrink=0.7,fraction=0.05, pad=0.2)

    colors = plt.cm.jet(np.linspace(0, 1, len(num_ctd)))
    # Positions
    for i_ctd in np.arange(len(num_ctd)):
        ds_cruise = xr.open_dataset(rep_ctd[i_ctd] + fic_ctd[i_ctd])
        ds_cruise = ds_cruise.where(ds_cruise['STATION_NUMBER']==num_ctd[i_ctd],drop=True)
        ds_cycle = ds_WMO2.where((ds_WMO2['CYCLE_NUMBER']==num_cycle[i_ctd]) & (ds_WMO2['DIRECTION']=='A'),drop=True)
        lon = ds_cruise['LONGITUDE']
        lat = ds_cruise['LATITUDE']
        h_ctd = ax.scatter(lon, lat, color=colors[i_ctd], s=20, marker='D',transform=ccrs.PlateCarree())#, zorder=1)
        lon = ds_cycle['LONGITUDE']
        lat = ds_cycle['LATITUDE']
        h_argo = ax.scatter(lon, lat, color=colors[i_ctd], s=20, marker = 'o',transform=ccrs.PlateCarree())#, zorder=1)

    
    ax.set_xlabel('LONGITUDE')
    ax.set_ylabel('LATITUDE')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='-')

    plt.title(f"{ds_cycle['PLATFORM_NUMBER'].isel(N_PROF=0).values:.0f} : CTD/ARGO") 
    ax.legend([h_ctd,h_argo],['CTD','Argo'])
    return None


def plot_Theta_S(ds_WMO1 : xr.Dataset,strvar : str='',pr : int=0,qc_keep : list=[1,2,3,4,8])->None:
    """ Function to plot potential temperature at pr  vs Salinity with a color by  cycle

    Parameters
    ----------
    ds_WMO1 : xr.Dataset
     contains ARGO data (from Sprof Netcdf ARGO file)
    strvar : str (default : '')
     Indicates if we plot Raw Data (strvar='') or Adjusted Data (strvar = '_ADJUSTED')
    pr : int (default : 0)
        Indicates the pressure for which to calculate the potential temperature.

    Returns
    -------
    None
    Only plot is shown. The colormap is jet. 
    So, the blue plots are associated to the first cycles and the red for the last cycles.
    """
    ds_WMO = copy.deepcopy(ds_WMO1)
    ds_WMO = ds_WMO.where(ds_WMO['DIRECTION']=='A',drop=True)
    platform_number = ds_WMO['PLATFORM_NUMBER'].values.astype(int)
    ds_WMO['PSAL' + strvar] = ds_WMO['PSAL' + strvar].where(ds_WMO['PSAL' + strvar + '_QC'].isin(qc_keep),np.nan)
    ds_WMO['TEMP' + strvar] = ds_WMO['TEMP' + strvar].where(ds_WMO['TEMP' + strvar + '_QC'].isin(qc_keep),np.nan)
    ds_WMO['PRES' + strvar] = ds_WMO['PRES' + strvar].where(ds_WMO['PRES' + strvar + '_QC'].isin(qc_keep),np.nan)
    
    psal = ds_WMO['PSAL'+strvar].values  
    pres = ds_WMO['PRES'+strvar].values 
    temp = ds_WMO['TEMP'+strvar].values  
    cycles = ds_WMO['CYCLE_NUMBER'].values  
    theta_argo = sw.ptmp(psal,temp,pres,pr)
    
    norm = plt.Normalize(vmin=np.min(cycles), vmax=np.max(cycles))
    cmap = matplotlib.colormaps.get_cmap('jet')  # Dégradé bleu -> rouge
    colors = cmap(norm(cycles))  # Couleurs pour chaque profil

    plt.figure()
    for i, cycle in enumerate(cycles):
        h = plt.plot(ds_WMO['PSAL'+strvar].isel(N_PROF=i), theta_argo[i], '.-',color=colors[i],markersize=1,label='ARGO')[0]


    plt.gca().invert_yaxis()

    plt.xlabel('Salinity (PSAL' + strvar +')')
    plt.ylabel('Potential Temperature (+ ' + strvar + ' at ' + str(pr) + 'db)')
    plt.title(platform_number[0])
    plt.grid()
    plt.gca().invert_yaxis()
    plt.show()

    return h


def plot_DOXY_cycle(ds_WMO1 : xr.Dataset,strvar : str='',qc_keep : list=[1,2,3,4,8])->None:
    """ Function to plot DOXY Data vs Pressure with a color by  cycle

    Parameters
    ----------
    ds_WMO1 : xr.Dataset
     contains ARGO data (from Sprof Netcdf ARGO file)
    strvar : str (default : '')
     Indicates if we plot Raw Data (strvar='') or Adjusted Data (strvar = '_ADJUSTED')

    Returns
    -------
    None
    Only plot is shown. The colormap is jet. 
    So, the blue plots are associated to the first cycles and the red for the last cycles.
    """
    ds_WMO = copy.deepcopy(ds_WMO1)
    ds_WMO['DOXY' + strvar] = ds_WMO['DOXY' + strvar].where(ds_WMO['DOXY' + strvar + '_QC'].isin(qc_keep),np.nan)
    ds_WMO['PRES' + strvar] = ds_WMO['PRES' + strvar].where(ds_WMO['PRES' + strvar + '_QC'].isin(qc_keep),np.nan)
    oxy = ds_WMO['DOXY'+strvar].values  
    pres = ds_WMO['PRES'+strvar].values  
    cycles = ds_WMO['CYCLE_NUMBER'].values  

    norm = plt.Normalize(vmin=np.min(cycles), vmax=np.max(cycles))
    cmap = matplotlib.colormaps.get_cmap('jet')  # Dégradé bleu -> rouge
    colors = cmap(norm(cycles))  # Couleurs pour chaque profil

    plt.figure()
    for i, cycle in enumerate(cycles):
        h = plt.plot(ds_WMO['DOXY'+strvar].isel(N_PROF=i), ds_WMO['PRES'+strvar].isel(N_PROF=i), '.-',color=colors[i],markersize=1,label='ARGO')[0]


    plt.gca().invert_yaxis()

    plt.xlabel(' Oxygen (OXY' + strvar +')')
    plt.ylabel(' Pressure (PRES' + strvar+')')
    plt.title(ds_WMO['PLATFORM_NUMBER'].values[0])
    plt.grid()

    plt.show()

    return h
    

def plot_DOXY_QC(ds_WMO : xr.Dataset, doxy_qc : list, strvar:str='')->None :
    """ Function to plot DOXY vs pressure considering QC (all QC and (1/2/3 QC)

    Parameters
    ----------
    ds_WMO : wr.Dataset
     contains ARGO data 
     (from Sprof NetCDF ARGO file)
    doxy_qc : list
        DOXY QC to keep (defined by user)
    strvar : str (default : '')
        indicates if we plot RAW Data  (strvar='') or ADJUSTED Data (strvar ='_ADJUSTED')

    Returns
    None
    Only a plot showing density and DOXY vs Pressure considering QC
    
    """
    fig, axes = plt.subplots(3,1)

    # Density plot
    _=axes[0].plot(sw.pden(ds_WMO['PSAL'+strvar],ds_WMO['TEMP'+strvar],ds_WMO['PRES'+strvar],0)-1000,ds_WMO['PRES'+strvar],'.k',markersize=1)
    axes[0].grid()
    axes[0].set_xlabel('Argo ' +strvar + ' Density')
    axes[0].set_ylabel('Presure ' + strvar + ' (dbar)')
    _=axes[0].set_ylim(0,None)
    axes[0].invert_yaxis()

    # DOXY plot for all QC
    GRBColor = [[0,1,0],[0,1,1],[1,0.5,0],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    qc_tot = [1,2,3,4,5,8,9]
    for i in range(len(GRBColor)):
        _=axes[1].plot(ds_WMO['DOXY'].where(ds_WMO['DOXY_QC']==qc_tot[i],drop=False),
                       ds_WMO['PRES'+ strvar].where(ds_WMO['DOXY_QC']==qc_tot[i],drop=False),'.',color=GRBColor[i],markersize=1)
    axes[1].grid()
    _=axes[1].set_ylim(0,None)
    axes[1].set_ylabel('Presure ' + strvar + ' (dbar)')
    axes[1].invert_yaxis()
    _=axes[1].set_xlim(200,400)
    axes[1].set_title('[O2] data and their QC (1=G,2=C,3=O,4=R,5=M,8=Y,9=B)')

    # Doxy plot for QC 1/2/3
    _=axes[2].plot(ds_WMO['DOXY'].where(ds_WMO['DOXY_QC'].isin(doxy_qc),drop=True),
                   ds_WMO['PRES'+strvar].where(ds_WMO['DOXY_QC'].isin(doxy_qc),drop=True),'.c',markersize=1)
    axes[2].grid()
    _=axes[2].set_ylim(0,None)
    axes[2].set_ylabel('Presure ' + strvar + ' (dbar)')
    axes[2].invert_yaxis()
    _=axes[2].set_xlim(200,400)
    axes[2].set_title("[O2] data (QC " + str(doxy_qc) + " )")

    plt.tight_layout()
#plt.subplots_adjust(top=0.85, wspace=5)

    plt.show()

    return None
    

def plot_QC_cycle(ds_WMO : xr.Dataset,strvar : str='') -> None :
    """ Function to plot QC for each cycle/pressure for PRES/TEMP/PSAL

    Parameters
    ----------
    ds_WMO : xr.Dataset
     contains ARGO Data (from Sprof Netcdf ARGO file)
    strvar : str (default : '')
     Indicates if we plot QC for Raw (strvar='')) or adjusted Data (strvar = '_ADJUSTED')

     Returns
     --------
     None
     Only plots with QC variable
    """

    list_var = ['PRES' + strvar,'TEMP'+strvar,'PSAL'+strvar]

    plt.figure()
    bid = ds_WMO['CYCLE_NUMBER'].expand_dims(N_LEVELS=np.arange(len(ds_WMO['N_LEVELS']))).transpose()
    Wcolor=[[0.8,0.8,0.8], #  Gray      (QC = 0: no QC was performed)
            [0.3,1, 0.3], # Green     (QC = 1: Good Data)
            [1,1,0.3],  # Yellow    (QC = 2: Probably Good Data)
            [1,0.7,0.3], # Orange    (QC = 3: Bad Data, Potentially Correctable)
            [1,0.3,0.3], # Red       (QC = 4: Bad Data)
            [0.8,0,0.8], # Magenta   (QC = 5: Value Changed)
            [0,0,0],  # Black     (QC = 6: not used)
            [0,0,0],# Black     (QC = 7: not used)
            [0.3,0.8,0.8], # Cyan      (QC = 8: Interpolated Value)
            [0.3,0.3,1]]  # Blue (QC = 9: Missing Value)
    
    cmap = matplotlib.colors.ListedColormap(Wcolor)
    bounds = np.arange(len(Wcolor)+1)  # Définir les limites pour chaque couleur
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    tab_qc = [9,8,7,6,0,1,2,3,4,5]
    i_plot = 0
    for i_var in list_var:
        print(i_var)
        i_plot = i_plot + 1
        plt.subplot(3,1,i_plot)
        for i in tab_qc: #range(len(Wcolor),-1,-1):
            try:
                bid2 = bid.where(ds_WMO[i_var + '_QC']==i,np.nan)
                bid3 = ds_WMO['PRES'+strvar].where(ds_WMO[i_var + '_QC']==i,np.nan)
                plt.scatter(bid2.values,bid3.values,color=Wcolor[i],label=f"QC={i}")
            except:
                pass
        
        plt.gca().invert_yaxis()
        plt.title(i_var + '_QC')
        plt.ylabel('PRES'+strvar)
        if i_plot<=2:
            plt.xticks([])
        if i_plot==3:
            plt.xlabel('CYCLE')

    cbar_ax = plt.axes([0.92, 0.1, 0.02, 0.8])  # Position [left, bottom, width, height]
    cbar = matplotlib.colorbar.ColorbarBase(cbar_ax,cmap=cmap, norm=norm, boundaries=bounds, ticks=(np.arange(0, 10)+np.arange(1, 11))/2, 
                                            spacing='proportional', orientation='vertical')
    tab_qc.sort()
    cbar.set_ticklabels(tab_qc)  
    cbar.set_label('QC Flags')

    #_=plt.tight_layout()

    plt.show()

    return None

def plot_ppox_Inair_Inwater_Ncep(dsair : xr.Dataset, dsinwater : xr.Dataset, ncep_data : np.ndarray) -> None:
    """ Function to plot InAir/InWater PPOX compared to NCEP PPOX

    Parameters
    ----------
    dsair : xr.Dataset
        InAIR data
    dsinwater : xr.Dataset
        InWater data
    ncep_data : np.ndarray
        NCEP PPOX

    Returns
    -------
    None
    A plot is created
    """
    plt.figure()
    plt.plot(dsair['CYCLE_NUMBER'],ncep_data,'.-k')
    plt.plot(dsair['CYCLE_NUMBER'],dsair['PPOX_DOXY'],'.b-')
    plt.plot(dsinwater['CYCLE_NUMBER'],dsinwater['PPOX_DOXY'],'.r-')
    plt.grid()
    plt.xlabel('CYCLE_NUMBER')
    plt.ylabel('PPOX')
    _=plt.legend(['NCEP','INAIR','INWATER'])

    return None

def plot_cmp_corr_NCEP(dict_corr : dict, list_pieceT : list, dsair : xr.Dataset,ncep_data : np.ndarray,delta_T : np.ndarray) -> None:
    """ Function to compare different PPOX dsair correction

    Parameters
    ----------
    dict_corr : dict
        dict of Correction (Name/Value).
    list_pieceT : list
        list of time to cut the correction in piece.
        same length as dict_corr.
    dsair : xr.Dataset
        Contains InAir data
    ncep_data : np.ndarray
        NCEP PPOX
    delta_T : np.ndarray
        For each data : (JULD - launch_date)

    Returns
    -------
    None
    A plot is created
    """
    norm = plt.Normalize(vmin=0, vmax=len(dict_corr))
    cmap = matplotlib.colormaps.get_cmap('jet')  # Dégradé bleu -> rouge
    colors = cmap(norm(np.arange(0,len(dict_corr))))  # Couleurs pour chaque profil

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(dsair['CYCLE_NUMBER'],ncep_data,'.-k',markersize=1,label='NCEP')
    plt.plot(dsair['CYCLE_NUMBER'],dsair['PPOX_DOXY'],'.--k',markersize=1,label='RAW')

    plt.subplot(2,1,2)
    plt.plot(delta_T,ncep_data/dsair['PPOX_DOXY'],'.-k',label='NCEP')

    i_coul = -1
    for index, (key, value) in enumerate(dict_corr.items()):
        print(key)
        i_coul = i_coul + 1
        val_corr = value
        pieceT = list_pieceT[index]
        if len(pieceT) == 0:
            nb_morceaux = 1
        else:
            nb_morceaux = len(pieceT)-1
        print(f"Number of pieces : {nb_morceaux}")        
        
        bid = dsair['PPOX_DOXY'].copy()
        for i_morceaux in range(0,nb_morceaux):
            mask = np.ones(delta_T.shape,dtype=bool)
            if nb_morceaux==1:
                val_corr_en_cours = val_corr
            else:
                val_corr_en_cours = val_corr[i_morceaux]
                mask = (delta_T >= pieceT[i_morceaux]) & (delta_T < pieceT[i_morceaux+1])
                
            print(val_corr_en_cours)
                                                          
            if len(val_corr_en_cours)==1:
                bid[mask] = val_corr_en_cours[0]*dsair['PPOX_DOXY'][mask]
            else:
                bid[mask] = (val_corr_en_cours[0]*(1+val_corr_en_cours[1]/100*delta_T[mask]/365))*dsair['PPOX_DOXY'][mask]
                
        label_corr = f'{key}'  # Nom personnalisé de la courbe dans la légende
        plt.subplot(2,1,1)
        plt.plot(dsair['CYCLE_NUMBER'],bid,'.-',color=colors[i_coul],markersize=1,label=label_corr)
        plt.subplot(2,1,2)
        plt.plot(delta_T,bid/dsair['PPOX_DOXY'],'.-',color=colors[i_coul],markersize=1,label=label_corr)
    
    plt.subplot(2,1,1)    
    plt.grid()
    plt.xlabel('CYCLE_NUMBER')
    plt.ylabel('PPOX')
    leg=plt.legend(draggable=True) 
#_=plt.legend() #loc='lower left', bbox_to_anchor=(1, 0))

    plt.subplot(2,1,2)    
    plt.grid()
    plt.xlabel('JULD')
    plt.ylabel('Time Drift Gain')
    #leg=plt.legend(draggable=True) 

    return None


def plot_cmp_corr_NCEP_old(dict_corr : dict, dsair : xr.Dataset,ncep_data : np.ndarray,delta_T : np.ndarray) -> None:
    """ Function to compare different PPOX dsair correction

    Parameters
    ----------
    dict_corr : dict
        dict of Correction (Name/Value).
    dsair : xr.Dataset
        Contains InAir data
    ncep_data : np.ndarray
        NCEP PPOX
    delta_T : np.ndarray
        For each data : (JULD - launch_date)

    Returns
    -------
    None
    A plot is created
    """
    norm = plt.Normalize(vmin=0, vmax=len(dict_corr))
    cmap = matplotlib.colormaps.get_cmap('jet')  # Dégradé bleu -> rouge
    colors = cmap(norm(np.arange(0,len(dict_corr))))  # Couleurs pour chaque profil

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(dsair['CYCLE_NUMBER'],ncep_data,'.-k',markersize=1,label='NCEP')
    plt.plot(dsair['CYCLE_NUMBER'],dsair['PPOX_DOXY'],'.--k',markersize=1,label='RAW')

    plt.subplot(2,1,2)
    plt.plot(delta_T,ncep_data/dsair['PPOX_DOXY'],'.-k',label='NCEP')

    i_coul = -1
    for corr in dict_corr.items():
        i_coul = i_coul + 1
        val_corr = corr[1]
        if len(val_corr)==1:
            bid = val_corr[0]*dsair['PPOX_DOXY']
        else:
            bid = (val_corr[0]*(1+val_corr[1]/100*delta_T/365))*dsair['PPOX_DOXY']
        label_corr = f'{corr}'  # Nom personnalisé de la courbe dans la légende
        plt.subplot(2,1,1)
        plt.plot(dsair['CYCLE_NUMBER'],bid,'.-',color=colors[i_coul],markersize=1,label=corr[0])
        plt.subplot(2,1,2)
        plt.plot(delta_T,bid/dsair['PPOX_DOXY'],'.-',color=colors[i_coul],markersize=1,label=corr[0])
    
    plt.subplot(2,1,1)    
    plt.grid()
    plt.xlabel('CYCLE_NUMBER')
    plt.ylabel('PPOX')
    leg=plt.legend(draggable=True) 
#_=plt.legend() #loc='lower left', bbox_to_anchor=(1, 0))

    plt.subplot(2,1,2)    
    plt.grid()
    plt.xlabel('JULD')
    plt.ylabel('Time Drift Gain')
    #leg=plt.legend(draggable=True) 

    return None


def plot_cmp_corr_NCEP_with_error(dict_corr : dict, perr_to_use : np.ndarray, list_pieceT : list, dsair : xr.Dataset,ncep_data : np.ndarray,delta_T : np.ndarray) -> None:
    """ Function to compare different PPOX dsair correction

    Parameters
    ----------
    dict_corr : dict
        dict of Correction (Name/Value).
    perr_to_use : np.ndarray
        error for correction
    list_pieceT : list
        list of time to cut the correction in piece.
        same length as dict_corr.
    dsair : xr.Dataset
        Contains InAir data
    ncep_data : np.ndarray
        NCEP PPOX
    delta_T : np.ndarray
        For each data : (JULD - launch_date)

    Returns
    -------
    None
    A plot is created
    """
    norm = plt.Normalize(vmin=0, vmax=len(dict_corr))
    cmap = matplotlib.colormaps.get_cmap('jet')  # Dégradé bleu -> rouge
    colors = cmap(norm(np.arange(0,len(dict_corr))))  # Couleurs pour chaque profil

    plt.figure()
    plt.plot(dsair['CYCLE_NUMBER'],ncep_data,'.-k',markersize=1,label='NCEP')
    plt.plot(dsair['CYCLE_NUMBER'],dsair['PPOX_DOXY'],'.--k',markersize=1,label='RAW')

    d_val_dslope = np.full_like(delta_T, np.nan)
    d_val_ddrift = np.full_like(delta_T, np.nan)
    delta_val_corr = np.full_like(delta_T, np.nan)
    
    i_coul = -1
    for index, (key, value) in enumerate(dict_corr.items()):
        print(key)
        i_coul = i_coul + 1
        val_corr = value
        pieceT = list_pieceT[index]
        perr_corr = perr_to_use[index]
        if len(pieceT)==0 :
            nb_morceaux = 1
        else:
            nb_morceaux = len(pieceT) - 1
            
        print(f"Number of pieces : {nb_morceaux}")
        
        bid = dsair['PPOX_DOXY'].copy()
        bid_min = dsair['PPOX_DOXY'].copy()
        bid_max = dsair['PPOX_DOXY'].copy()

        for i_morceaux in range(0,nb_morceaux):
            mask = np.ones(delta_T.shape,dtype=bool)
            if nb_morceaux==1:
                val_corr_en_cours = val_corr
                perr_en_cours = perr_corr
            else:
                val_corr_en_cours = val_corr[i_morceaux]
                perr_en_cours = perr_corr[i_morceaux]
                mask = (delta_T >= pieceT[i_morceaux]) & (delta_T < pieceT[i_morceaux+1])
                
            print(val_corr_en_cours)
            
            
            d_val_dslope[mask] = (1 + val_corr_en_cours[1] / 100 * delta_T[mask] / 365)  * dsair['PPOX_DOXY'][mask]
            d_val_ddrift[mask] = val_corr_en_cours[0] * (delta_T[mask] / 36500) * dsair['PPOX_DOXY'][mask]
            delta_val_corr[mask] = np.sqrt((d_val_dslope[mask] * perr_en_cours[0]) ** 2 +(d_val_ddrift[mask] * perr_en_cours[1]) ** 2)                                   


            if len(val_corr_en_cours)==1:
                bid[mask] = val_corr_en_cours[0]*dsair['PPOX_DOXY'][mask]
                bid_min[mask] = (val_corr_en_cours[0]-perr_en_cours[0])*dsair['PPOX_DOXY'][mask]
                bid_mask[mask] = (val_corr_en_cours[0]+perr_en_cours[0])*dsair['PPOX_DOXY'][mask]

            else:
                bid[mask] = (val_corr_en_cours[0]*(1+val_corr_en_cours[1]/100*delta_T[mask]/365))*dsair['PPOX_DOXY'][mask]
                bid_min[mask] = ((val_corr_en_cours[0]-perr_en_cours[0])*(1+(val_corr_en_cours[1]-perr_en_cours[1])/100*delta_T[mask]/365))*dsair['PPOX_DOXY'][mask]
                bid_max[mask] = ((val_corr_en_cours[0]+perr_en_cours[0])*(1+(val_corr_en_cours[1]+perr_en_cours[1])/100*delta_T[mask]/365))*dsair['PPOX_DOXY'][mask]

        label_corr = f'{key}'  # Nom personnalisé de la courbe dans la légende
        plt.plot(dsair['CYCLE_NUMBER'],bid,'.-',color=colors[i_coul],markersize=1,label=label_corr)
        plt.fill_between(dsair['CYCLE_NUMBER'], bid_min, bid_max, color=colors[i_coul], alpha=0.3, label="Incertitude (min/max)")
        plt.errorbar(dsair['CYCLE_NUMBER'], bid, yerr=delta_val_corr, fmt='o', color=colors[i_coul], capsize=4, label='Incertitude (propagation)')

    plt.grid()
    plt.xlabel('CYCLE_NUMBER')
    plt.ylabel('PPOX')
    leg=plt.legend(draggable=True) 
#_=plt.legend() #loc='lower left', bbox_to_anchor=(1, 0))

    return None
    


def plot_cmp_corr_WOA(dict_corr : dict, list_pieceT : list, ds_argo_interp : xr.Dataset, ds_woa_interp : xr.Dataset, delta_T : np.ndarray)-> None:
    """ Function to compare different correction with PSATWOA

    Parameters
    -----------
    dict_corr : dict
        dict of correction (Name/Value)
    list_pieceT : list
        list of time to cut the correction in piece.
        same length as dict_corr.
    ds_argo_interp : xr.Dataset
        Contains ARGO data interpolated on a regular grid (to calculate the mean of ARGO PSAT on ths grid)
    ds_woa_interp : xr.Dataset
        Contains WOA DATA interpolated on the same regular grid (to calculate the mean of WOA PASAT on ths grid)
    delta_T : np.ndarray
        Difference (JULD - launch_date)

    Returns
    -------
    None
    A plot is created
    """
    ana_dens = sw.pden(ds_argo_interp['PSAL_ARGO'],ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PRES_ARGO'],0)
    O2_umolL = umolkg_to_umolL(ds_argo_interp['DOXY_ARGO'],ds_argo_interp['DOXY_ARGO'].units,ana_dens)
    psatargo = O2ctoO2s(O2_umolL,ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PSAL_ARGO'])
    psatargo_mean = psatargo.mean(dim='N_LEVELS')
    psatWOA_mean = ds_woa_interp['Psatwoa'].mean(dim='N_LEVELS')

    norm = plt.Normalize(vmin=0, vmax=len(dict_corr))
    cmap = matplotlib.colormaps.get_cmap('jet')  # Dégradé bleu -> rouge
    colors = cmap(norm(np.arange(0,len(dict_corr))))  # Couleurs pour chaque profil

    plt.figure()
    plt.plot(delta_T,psatWOA_mean,'.-k',label='WOA')
    plt.plot(delta_T,psatargo_mean,'.--k',label='RAW')

    tab_delta_T = np.vstack([delta_T]*len(ds_argo_interp['N_LEVELS'])).transpose()

    i_coul = -1
    for index, (key, value) in enumerate(dict_corr.items()):
        print(key)
        i_coul = i_coul + 1
        val_corr = value
        pieceT = list_pieceT[index]
        if len(pieceT)==0:
            nb_morceaux = 1
        else:
            nb_morceaux = len(pieceT) - 1

        print(f"Number of pieces : {nb_morceaux}")
        
        bid = ds_argo_interp['DOXY_ARGO'].copy()
        for i_morceaux in range(0,nb_morceaux):
            mask = np.ones(tab_delta_T.shape,dtype=bool)
            if nb_morceaux==1:
                val_corr_en_cours = val_corr
            else:
                val_corr_en_cours = val_corr[i_morceaux]
                mask = (tab_delta_T >= pieceT[i_morceaux]) & (tab_delta_T < pieceT[i_morceaux+1])
                
            print(val_corr_en_cours)
                                                          
            if len(val_corr_en_cours)==1:
                bid.values[mask] = val_corr_en_cours[0]*ds_argo_interp['DOXY_ARGO'].values[mask]
            else:
                bid.values[mask] = (val_corr_en_cours[0]*(1+val_corr_en_cours[1]/100*tab_delta_T[mask]/365))*ds_argo_interp['DOXY_ARGO'].values[mask]
                
        O2_umolL = umolkg_to_umolL(bid,ds_argo_interp['DOXY_ARGO'].units,ana_dens)
        psatargo_corr = O2ctoO2s(O2_umolL,ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PSAL_ARGO'])
        psatargo_corr_mean = psatargo_corr.mean(dim='N_LEVELS')
        
        label_corr = f'{key}'  # Nom personnalisé de la courbe dans la légende
        plt.plot(delta_T,psatargo_corr_mean,'.-',color=colors[i_coul],markersize=1,label=label_corr)        
    plt.grid()
    plt.xlabel('DELTA JULD')
    plt.ylabel('PSAT')
    _=plt.legend(draggable=True)

    return None


def plot_cmp_corr_WOA_with_error(dict_corr : dict, perr_to_use : np.ndarray, list_pieceT : list, ds_argo_interp : xr.Dataset, ds_woa_interp : xr.Dataset, delta_T : np.ndarray)-> None:
    """ Function to compare different correction with PSATWOA

    Parameters
    -----------
    dict_corr : dict
        dict of correction (Name/Value)
    perr_to_use : np.ndarray
        error for correction
    list_pieceT : list
        list of time to cut the correction in piece.
        same length as dict_corr.
    ds_argo_interp : xr.Dataset
        Contains ARGO data interpolated on a regular grid (to calculate the mean of ARGO PSAT on ths grid)
    ds_woa_interp : xr.Dataset
        Contains WOA DATA interpolated on the same regular grid (to calculate the mean of WOA PASAT on ths grid)
    delta_T : np.ndarray
        Difference (JULD - launch_date)

    Returns
    -------
    None
    A plot is created
    """
    ana_dens = sw.pden(ds_argo_interp['PSAL_ARGO'],ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PRES_ARGO'],0)
    O2_umolL = umolkg_to_umolL(ds_argo_interp['DOXY_ARGO'],ds_argo_interp['DOXY_ARGO'].units,ana_dens)
    psatargo = O2ctoO2s(O2_umolL,ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PSAL_ARGO'])
    psatargo_mean = psatargo.mean(dim='N_LEVELS')
    psatWOA_mean = ds_woa_interp['Psatwoa'].mean(dim='N_LEVELS')

    norm = plt.Normalize(vmin=0, vmax=len(dict_corr))
    cmap = matplotlib.colormaps.get_cmap('jet')  # Dégradé bleu -> rouge
    colors = cmap(norm(np.arange(0,len(dict_corr))))  # Couleurs pour chaque profil

    plt.figure()
    plt.plot(delta_T,psatWOA_mean,'.-k',label='WOA')
    plt.plot(delta_T,psatargo_mean,'.--k',label='RAW')

    tab_delta_T = np.vstack([delta_T]*len(ds_argo_interp['N_LEVELS'])).transpose()

    d_val_dslope = np.full_like(tab_delta_T, np.nan)
    d_val_ddrift = np.full_like(tab_delta_T, np.nan)
    delta_val_corr = np.full_like(tab_delta_T, np.nan)

    i_coul = -1
    for index, (key, value) in enumerate(dict_corr.items()):
        print(key)
        i_coul = i_coul + 1
        val_corr = value
        perr_corr = perr_to_use[index]
        pieceT = list_pieceT[index]
        if len(pieceT)==0:
            nb_morceaux = 1
        else:
            nb_morceaux = len(pieceT) - 1

        print(f"Number of pieces : {nb_morceaux}")
        
        bid = ds_argo_interp['DOXY_ARGO'].copy()
        bid_min = ds_argo_interp['DOXY_ARGO'].copy()
        bid_max = ds_argo_interp['DOXY_ARGO'].copy()

        for i_morceaux in range(0,nb_morceaux):
            mask = np.ones(tab_delta_T.shape,dtype=bool)
            if nb_morceaux==1:
                val_corr_en_cours = val_corr
                perr_en_cours = perr_corr
            else:
                val_corr_en_cours = val_corr[i_morceaux]
                perr_en_cours = perr_corr[i_morceaux]
                mask = (tab_delta_T >= pieceT[i_morceaux]) & (tab_delta_T < pieceT[i_morceaux+1])

            print(val_corr_en_cours)
            
            #val_bid1 = np.where(mask,(1 + val_corr_en_cours[1] / 100 * tab_delta_T / 365) * ds_argo_interp['DOXY_ARGO'].values,np.nan)
            #d_val_dslope[mask] = val_bid1[mask]
            #val_bid2 = np.where(mask,(val_corr_en_cours[0] * (tab_delta_T / 36500)) * ds_argo_interp['DOXY_ARGO'].values ,np.nan)
            #d_val_ddrift[mask] = val_bid2[mask]
            #delta_val_corr[mask] = np.sqrt((d_val_dslope[mask] * perr_en_cours[0]) ** 2 + (d_val_ddrift[mask] * perr_en_cours[1]) ** 2)                                   

            
            d_val_dslope[mask] = (1 + val_corr_en_cours[1] / 100 * tab_delta_T[mask] / 365) * ds_argo_interp['DOXY_ARGO'].values[mask]
            d_val_ddrift[mask] = (val_corr_en_cours[0] * (tab_delta_T[mask] / 36500)) * ds_argo_interp['DOXY_ARGO'].values[mask]
            delta_val_corr[mask] = np.sqrt((d_val_dslope[mask] * perr_en_cours[0]) ** 2 + (d_val_ddrift[mask] * perr_en_cours[1]) ** 2) 
            #print(delta_val_corr.shape)

            if len(val_corr_en_cours)==1:
                bid.values[mask] = val_corr_en_cours[0]*ds_argo_interp['DOXY_ARGO'].values[mask]
                bid_min.values[mask] = (val_corr_en_cours[0]-perr_en_cours[0])*ds_argo_interp['DOXY_ARGO'].values[mask]
                bid_mask.values[mask] = (val_corr_en_cours[0]+perr_en_cours[0])*ds_argo_interp['DOXY_ARGO'].values[mask]
            else:
                bid.values[mask] = (val_corr_en_cours[0]*(1+val_corr_en_cours[1]/100*tab_delta_T[mask]/365))*ds_argo_interp['DOXY_ARGO'].values[mask]
                bid_min.values[mask] = ((val_corr_en_cours[0]-perr_en_cours[0])*(1+(val_corr_en_cours[1]-perr_en_cours[1])/100*tab_delta_T[mask]/365))*ds_argo_interp['DOXY_ARGO'].values[mask]
                bid_max.values[mask] = ((val_corr_en_cours[0]+perr_en_cours[0])*(1+(val_corr_en_cours[1]+perr_en_cours[1])/100*tab_delta_T[mask]/365))*ds_argo_interp['DOXY_ARGO'].values[mask]   
                
        O2_umolL = umolkg_to_umolL(bid,ds_argo_interp['DOXY_ARGO'].units,ana_dens)
        psatargo_corr = O2ctoO2s(O2_umolL,ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PSAL_ARGO'])
        psatargo_corr_mean = psatargo_corr.mean(dim='N_LEVELS')
        O2_umolL_min = umolkg_to_umolL(bid_min,ds_argo_interp['DOXY_ARGO'].units,ana_dens)
        psatargo_corr_min = O2ctoO2s(O2_umolL_min,ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PSAL_ARGO'])
        psatargo_corr_min_mean = psatargo_corr_min.mean(dim='N_LEVELS')
        O2_umolL_max = umolkg_to_umolL(bid_max,ds_argo_interp['DOXY_ARGO'].units,ana_dens)
        psatargo_corr_max = O2ctoO2s(O2_umolL_max,ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PSAL_ARGO'])
        psatargo_corr_max_mean = psatargo_corr_max.mean(dim='N_LEVELS')

        
        label_corr = f'{key}'  # Nom personnalisé de la courbe dans la légende
        plt.plot(delta_T,psatargo_corr_mean,'.-',color=colors[i_coul],markersize=1,label=label_corr) 
        plt.fill_between(delta_T, psatargo_corr_min_mean, psatargo_corr_max_mean, color=colors[i_coul], alpha=0.3, label="Incertitude (min/max)")
        plt.errorbar(delta_T, psatargo_corr_mean, yerr=np.sqrt(np.nanmean(delta_val_corr,axis=1)), fmt='o', color=colors[i_coul], capsize=4, label='Incertitude (propagation)')


    plt.grid()
    plt.xlabel('DELTA JULD')
    plt.ylabel('PSAT')
    _=plt.legend(draggable=True)

    return None

    

def plot_cmp_corr_WOA_old(dict_corr : dict, ds_argo_interp : xr.Dataset, ds_woa_interp : xr.Dataset, delta_T : np.ndarray)-> None:
    """ Function to compare different correction with PSATWOA

    Parameters
    -----------
    dict_corr : dict
        dict of correction (Name/Value)
    ds_argo_interp : xr.Dataset
        Contains ARGO data interpolated on a regular grid (to calculate the mean of ARGO PSAT on ths grid)
    ds_woa_interp : xr.Dataset
        Contains WOA DATA interpolated on the same regular grid (to calculate the mean of WOA PASAT on ths grid)
    delta_T : np.ndarray
        Difference (JULD - launch_date)

    Returns
    -------
    None
    A plot is created
    """
    ana_dens = sw.pden(ds_argo_interp['PSAL_ARGO'],ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PRES_ARGO'],0)
    O2_umolL = umolkg_to_umolL(ds_argo_interp['DOXY_ARGO'],ds_argo_interp['DOXY_ARGO'].units,ana_dens)
    psatargo = O2ctoO2s(O2_umolL,ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PSAL_ARGO'])
    psatargo_mean = psatargo.mean(dim='N_LEVELS')
    psatWOA_mean = ds_woa_interp['Psatwoa'].mean(dim='N_LEVELS')

    norm = plt.Normalize(vmin=0, vmax=len(dict_corr))
    cmap = matplotlib.colormaps.get_cmap('jet')  # Dégradé bleu -> rouge
    colors = cmap(norm(np.arange(0,len(dict_corr))))  # Couleurs pour chaque profil

    plt.figure()
    plt.plot(delta_T,psatWOA_mean,'.-k',label='WOA')
    plt.plot(delta_T,psatargo_mean,'.--k',label='RAW')

    i_coul = -1
    for corr in dict_corr.items():
        i_coul = i_coul + 1
        val_corr = corr[1]
        if len(val_corr)==1:
            bid = val_corr[0]*ds_argo_interp['DOXY_ARGO']
        else:
            tab_delta_T = np.vstack([delta_T]*len(ds_argo_interp['N_LEVELS'])).transpose()
            bid = (val_corr[0]*(1+val_corr[1]/100*tab_delta_T/365))*ds_argo_interp['DOXY_ARGO']
            
        O2_umolL = umolkg_to_umolL(bid,ds_argo_interp['DOXY_ARGO'].units,ana_dens)
        psatargo_corr = O2ctoO2s(O2_umolL,ds_argo_interp['TEMP_ARGO'],ds_argo_interp['PSAL_ARGO'])
        psatargo_corr_mean = psatargo_corr.mean(dim='N_LEVELS')
        label_corr = f'{corr}'  # Nom personnalisé de la courbe dans la légende
        plt.plot(delta_T,psatargo_corr_mean,'.-',color=colors[i_coul],markersize=1,label=corr[0])
    plt.grid()
    plt.xlabel('DELTA JULD')
    plt.ylabel('PSAT')
    _=plt.legend(draggable=True)

    return None

def plot_cmp_ARGO_CTD(dsctd : xr.Dataset,ds_cycle : xr.Dataset, dict_corr : dict, launch_date : np.datetime64, coef2 : float, coef3 : float) -> None:
    """ Function to compare an ARGO DOXY profile with a CTD Doxy profile

    Parameters
    ----------
    dsctd : xr.Dataset
     Contains OXYK (O2 in mmol/Kg) and PRES
    ds_cycle : xr.Dataset
     Cycle to compare
    dict_corr : dict
     Correction to compare
    launch_date : np.datetime64
     Launch Date
    coef2, coef3 : float
         coefficient used in constructor pressure effect : (1 + (coef2 * Temp + coef3)*Pres/1000)     
     Returns
     -------
     None
     A plot is created
     """
    
    norm = plt.Normalize(vmin=0, vmax=len(dict_corr))
    cmap = matplotlib.colormaps.get_cmap('jet')  # colormap
    colors = cmap(norm(np.arange(0,len(dict_corr))))  # Color
    
    delta_T_cycle = diff_time_in_days(ds_cycle['JULD'].values,launch_date)
    tab_delta_T = np.tile(delta_T_cycle,(1,len(ds_cycle['N_LEVELS'])))
    #tab_delta_T = np.vstack([delta_T_cycle1]*len(ds_cycle1['N_LEVELS'])).transpose()

    plt.figure()
    plt.plot(dsctd['OXYK'].isel(N_PROF=0), dsctd['PRES'].isel(N_PROF=0), 'x--b', label='CTD')[0]
    plt.plot(ds_cycle['DOXY'].isel(N_PROF=0),ds_cycle['PRES'].isel(N_PROF=0),'x--k',label='RAW')[0]

    i_coul = -1
    for corr in dict_corr.items():
        i_coul = i_coul + 1
        val_corr = corr[1]    
        if len(val_corr)==1:
            bid = val_corr[0]*ds_cycle['DOXY']
        elif len(val_corr)==2:
            bid = (val_corr[0]*(1+val_corr[1]/100*tab_delta_T/365))*ds_cycle['DOXY']
        else:
            if val_corr[2] == 0.0:
                bid = (val_corr[0]*(1+val_corr[1]/100*tab_delta_T/365))*ds_cycle['DOXY']
            else:
                bid = (val_corr[0]*(1+val_corr[1]/100*tab_delta_T/365))*ds_cycle['DOXY']
                bid = bid/(1 + (coef2*ds_cycle['TEMP'] + coef3) * ds_cycle['PRES']/1000)
                bid = (1 + (coef2*ds_cycle['TEMP'] + val_corr[2]) * ds_cycle['PRES']/1000) * bid 
            
        plt.plot(bid.isel(N_PROF=0),ds_cycle['PRES'].isel(N_PROF=0),'.-',color=colors[i_coul],label=corr[0])[0]
    
    plt.grid()
    plt.gca().invert_yaxis()
    plt.xlabel('DOXY')
    plt.ylabel('PRES')
    _=plt.legend(draggable=True)

    return None

def plot_cmp_corr_oxy_woa(ds_argo_Sprof : xr.Dataset, ds_woa : xr.Dataset) -> None :
    """ Function to plot RAW and Corrected DOXY compared to DOXY WOA

    Parameters
    ----------
    ds_argo_Sprof : xr.Dataset
     Data from Sprof ARGO Netcdf file
    ds_woa : xr.Dataset
     Data from WOA interpolated on ARGO position/time

     Returns
     -------
     None
     A plot is created
     """
    
    plt.figure()
    h1=plt.plot(ds_argo_Sprof['DOXY_ADJUSTED'],ds_argo_Sprof['PRES'],'+-r',label='ADJUSTED')[0]
    h2=plt.plot(ds_argo_Sprof['DOXY'],ds_argo_Sprof['PRES'],'x-b',label='RAW')[0]
    h3=plt.plot(ds_woa['doxywoa'],ds_woa['preswoa'],'o-k',label='WOA')[0]
    plt.grid()
    plt.gca().invert_yaxis()
    plt.xlabel('DOXY')
    plt.ylabel('PRES')
    _=plt.legend([h1,h2,h3],['ADJUSTED','RAW','WOA'],draggable=True)

    return None
    

def plot_ref_div_argo(x : np.ndarray,ref : np.ndarray,data : np.ndarray):
    """ Function to plot the linear line correction

    Parameters
    -----------
    x :  days from launch date
    ref : Reference data (NCEP or WOA)
    data : Argo data

    Returns
    --------
    None 
    A plot is created
    """
    plt.figure()
    plt.plot(x,ref/data)
    mask = np.isfinite(ref) & np.isfinite(data)
    poly_data = np.polyfit(x[mask], ref[mask]/data[mask], 1)
    plt.plot(x,np.polyval(poly_data,x),'*-r')
    plt.grid()
    plt.xlabel('DeltaT')
    plt.ylabel('REF/ARGO_DATA')
    plt.show()
    return None