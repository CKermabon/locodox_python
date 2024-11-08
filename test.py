import matplotlib.pyplot as plt
from m_argo_data import read_argo_data

if __name__ == "__main__":
	rep_argo_data = '/Users/chemon/ARGO_NEW/NEW_LOCODOX/data_test/'
	num_float = '6902802'
	ds_argo_inair, ds_argo_inwater,ds_argo_Sprof, optode_height = read_argo_data(num_float,rep_argo_data)
	print(optode_height) 
#	print(ds_argo_inair)
#	print(ds_argo_inwater)
#	plt.figure()
#	plt.subplot(2,2,1)
#	plt.plot(ds_argo_inair['LONGITUDE'])
#	plt.subplot(2,2,2)
#	plt.plot(ds_argo_inair['PPOX_DOXY'])
#	plt.subplot(2,2,3)
#	plt.plot(ds_argo_inwater['LONGITUDE'])
#	plt.subplot(2,2,4)
#	plt.plot(ds_argo_inwater['PPOX_DOXY'])
#	plt.show()
	plt.figure()
	plt.plot(ds_argo_inair['CYCLE_NUMBER'],ds_argo_inair['LONGITUDE'],'+b')
	plt.plot(ds_argo_Sprof['CYCLE_NUMBER'],ds_argo_Sprof['LONGITUDE'],'xr') 
	plt.show()
