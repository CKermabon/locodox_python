# Import modules
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
import seawater as sw

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
    # plot position with color and add the cycle number
    for i, (lon, lat) in enumerate(zip(longitudes, latitudes)):
        ax.scatter(lon, lat, color=colors[i], s=10, transform=ccrs.PlateCarree(), label=f'Group {i // 10}' if i % 10 == 0 else "")
        ax.text(lon + 0.1, lat + 0.1, ds_WMO['CYCLE_NUMBER'].values[i], color='black', fontsize=4, transform=ccrs.PlateCarree())

    # dd a legend by group
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels, loc='upper right', title='Groups')
    ax.set_xlabel('LONGITUDE')
    ax.set_ylabel('LATITUDE')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='-')

    plt.title(f"{ds_WMO['PLATFORM_NUMBER']} : Region where the floats derived") 
    plt.show()

    return None


def plot_DOXY_cycle(ds_WMO : xr.Dataset,strvar : str='')->None:
    """ Function to plot DOXY Data vs Pressure with a color by  cycle

    Parameters
    ----------
    ds_WMO : xr.Dataset
     contains ARGO data (from Sprof Netcdf ARGO file)
    strvar : str (default : '')
     Indicates if we plot Raw Data (strvar='') or Adjusted Data (strvar = '_ADJUSTED')

    Returns
    -------
    None
    Only plot is shown. The colormap is jet. 
    So, the blue plots are associated to the first cycles and the red for the last cycles.
    """
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