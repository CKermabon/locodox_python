# Definition de fonctions utiles
import numpy as np

#########################################################
# Fonctions permettant d'estimer le gain et/ou la derive.
#########################################################
def model_Gain(X,G):
    """
    Modele utilise via curvefit pour estimer un gain.
    """
    return G * X 

def model_Gain_Derive(X,G,D):
    """
    Modele utilise via curvefit pour estimer un gain et une derive
    """
    return (G * (1 + (D * X[1])/(365*100)) * X[0] )

def model_Gain_CarryOver(X,G,C):
    """
    Modele utilise via curvefit pour estimer un gain avec prise en compte CarryOver (utile pour calcul via donnees dans l'air NCEP)
    """
    return G * (X[0] - C * X[1]) / (1 - C) # C : Carry-over

def model_Gain_Derive_CarryOver(X,G,C,D):
    """
    Modele utilise via curvefit pour estimer un gain et une derive avec prise en compte CarryOver (utile pour calcul via donnees dans l'air NCEP)
    """
    return (G / (1-C) * (1 + D / 100 * X[2]/365) * (X[0] - C * X[1]) )

#
# Fonction de calcul de la vapeur d'eau.
#
def watervapor(T,S):
	"""
	Cette fonction calcule la vapeur d'eau a partir de la temperature et salinite
    Fonction issue de Locodox Matlab
	"""
	pw=(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S))
	return pw	
    
##############################
# Fonctions de conversion O2.
##############################
def umolkg_to_umolL(O2mmolKg,units,ana_dens=1000):
    """
    Cette fonction convertit l'oxygene ARGO (en micromole/kg) en micromole/L.
    En entree :
        O2mmolKg : O2 en micromole/Kg
        an_dens : densite potentielle a 0.

    En sortie :
        O2mmolL : O2 en micromole par Litre.
    """
    if units == 'micromole/kg':
        O2mmolL = O2mmolKg / (1000 / ana_dens) 
        return O2mmolL
    else:
        print(f'Les donnees O2 doivent etre en micromole/kg')
        return None
        

def O2ctoO2p(O2conc,T,S,P=0):
    """
    Cette fonction convertit les donnees de concentration O2 (en umol/L) en pression partielle O2 a partir de T (temperature) et S (salinite)
    Si P (pression) n'est pas fournie, on la force a 0
    Fonction issue de Locodox Matlab
    """
    xO2     = 0.20946 # mole fraction of O2 in dry air (Glueckauf 1951)
    pH2Osat = 1013.25*(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S)) # saturated water vapor in mbar 
    sca_T   = np.log((298.15-T)/(273.15+T)) # scaled temperature for use in TCorr and SCorr
    TCorr   = 44.6596*np.exp(2.00907+3.22014*sca_T+4.05010*sca_T**2+4.94457*sca_T**3-2.56847e-1*sca_T**4+3.88767*sca_T**5) # temperature correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit mL(STP) L-1; and conversion from mL(STP) L-1 to umol L-1
    Scorr   = np.exp(S*(-6.24523e-3-7.37614e-3*sca_T-1.03410e-2*sca_T**2-8.17083e-3*sca_T**3)-4.88682e-7*S**2) # salinity correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit ml(STP) L-1
    Vm      = 0.317 # molar volume of O2 in m3 mol-1 Pa dbar-1 (Enns et al. 1965)
    R       = 8.314 # universal gas constant in J mol-1 K-1

    pO2=O2conc*(xO2*(1013.25-pH2Osat))/(TCorr*Scorr)*np.exp(Vm*P/(R*(T+273.15)))
    return pO2


def O2stoO2p(O2sat,T,S,P=0,P_atm=1013.25):
    """
    Cette fonction convertit les donnees de pourcentage de saturation en pression partielle O2 a partir de T (temperature) et S (salinite)
    ainsi que P (pression hydrostatique. 0 par defaut) et P_atm (pression atmospherique. 1013.25 mbar par defaut.
    Fonction issue de Locodox Matlab
    """
    xO2     = 0.20946 # mole fraction of O2 in dry air (Glueckauf 1951)
    pH2Osat = 1013.25*(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S)); # saturated water vapor in mbar
    Vm      = 0.317 # molar volume of O2 in m3 mol-1 Pa dbar-1 (Enns et al. 1965)
    R       = 8.314 # universal gas constant in J mol-1 K-1

    pO2=O2sat/100*(xO2*(P_atm-pH2Osat))

    return pO2

def O2ctoO2s(O2conc,T,S,P=0,P_atm=1013.25):
    """
    Cette fonction convertit les donnees de concentration O2 (en umol/L) en pourcentage de saturation a partir de T (temperature), S (salinite)
    ainsi que P (pression hydrostatique. O par defaut) et P_atm (pression atmospherique. 1013.25 mbar par defaut.
    Fonction issue de Locodox Matlab.
    """
    pH2Osat = 1013.25*(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S)) # saturated water vapor in mbar
    sca_T   = np.log((298.15-T)/(273.15+T)) # scaled temperature for use in TCorr and SCorr
    TCorr   = 44.6596*np.exp(2.00907+3.22014*sca_T+4.05010*sca_T**2+4.94457*sca_T**3-2.56847e-1*sca_T**4+3.88767*sca_T**5)# temperature correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit mL(STP) L-1; and conversion from mL(STP) L-1 to umol L-1
    Scorr   = np.exp(S*(-6.24523e-3-7.37614e-3*sca_T-1.03410e-2*sca_T**2-8.17083e-3*sca_T**3)-4.88682e-7*S**2)# salinity correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit ml(STP) L-1
    Vm      = 0.317 # molar volume of O2 in m3 mol-1 Pa dbar-1 (Enns et al. 1965)
    R       = 8.314 # universal gas constant in J mol-1 K-1

    O2sat=O2conc*100/(TCorr*Scorr)/(P_atm-pH2Osat)*(1013.25-pH2Osat)*np.exp(Vm*P/(R*(T+273.15)))  

    return O2sat

def O2ptoO2c(pO2,T,S,P=0):
    """
    Fonction permettant de convertir la pression partielle O2 en concentration O2 (en mmol/L)
    a partir de la temperature, la salinite et la pression hydrostatique (a 0 par defaut).
    Fonction issue de Locodox Matlab
    """
    xO2     = 0.20946 # mole fraction of O2 in dry air (Glueckauf 1951)
    pH2Osat = 1013.25*(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S)) #saturated water vapor in mbar
    sca_T   = np.log((298.15-T)/(273.15+T)) #scaled temperature for use in TCorr and SCorr
    TCorr   = 44.6596*np.exp(2.00907+3.22014*sca_T+4.05010*sca_T**2+4.94457*sca_T**3-2.56847e-1*sca_T**4+3.88767*sca_T**5)# temperature correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit mL(STP) L-1; and conversion from mL(STP) L-1 to umol L-1
    Scorr   = np.exp(S*(-6.24523e-3-7.37614e-3*sca_T-1.03410e-2*sca_T**2-8.17083e-3*sca_T**3)-4.88682e-7*S**2) #salinity correction part from Garcia and Gordon (1992), Benson and Krause (1984) refit ml(STP) L-1
    Vm      = 0.317 # molar volume of O2 in m3 mol-1 Pa dbar-1 (Enns et al. 1965)
    R       = 8.314 # universal gas constant in J mol-1 K-1

    O2conc=pO2/(xO2*(1013.25-pH2Osat))*(TCorr*Scorr)/np.exp(Vm*P/(R*(T+273.15)))
    return O2conc
     
###################################
# Fonction d'interpolation sur une grille reguliere en pression.
###################################
def interp_pres_grid(min_pres,max_pres,nb_profil,var_to_interpol,ds,var_name_pres):
    interpol_var = {}
    new_pres = np.arange(min_pres,max_pres+1,1)
    for var in var_to_interpol:
        print(f'Interpolation de la variable {var} sur grille reguliere en pression')
        interpol_data = np.zeros(shape=(nb_profil,max_pres-min_pres+1))
        for i_cycle in (range(0,nb_profil)):
            data_en_cours = ds[var][i_cycle,:]
            isok = np.isfinite(data_en_cours)
            nb_indices = np.count_nonzero(isok)
            if nb_indices>0:
                #print(data_en_cours[isok])
                #print(ds[var_name_pres][i_cycle,isok].values)
                interpol_data[i_cycle,:] = np.interp(new_pres,ds[var_name_pres][i_cycle,isok].values,data_en_cours[isok])
            else:
                interpol_data[i_cycle,:] = np.nan
        interpol_var[var] = interpol_data  
    return interpol_var
    