# Definition de fonctions utiles
import numpy as np

def watervapor(T,S):
	"""
	Cette fonction calcule la vapeur d'eau a partir de la temperature et salinite
	"""
#	print(f"Valeur de T :{T}")
	pw=(np.exp(24.4543-(67.4509*(100/(T+273.15)))-(4.8489*np.log(((273.15+T)/100)))-0.000544*S))
	return pw	
