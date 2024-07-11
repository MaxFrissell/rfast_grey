# import statements
import emcee
import os
import time
import sys
import shutil
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
import scipy.stats
import random
import h5py
import csv
from multiprocessing     import Pool
from astropy.io          import ascii
from astropy.table       import Table, Column, MaskedColumn
from rfast_routines      import spectral_grid
from rfast_routines      import gen_spec
from rfast_routines      import kernel_convol
from rfast_routines      import gen_spec_grid
from rfast_routines      import inputs
from rfast_routines      import init
from rfast_routines      import init_3d
from rfast_atm_routines  import set_gas_info
from rfast_atm_routines  import setup_atm
from rfast_atm_routines  import mmr2vmr
from rfast_atm_routines  import vmr2mmr
from rfast_opac_routines import opacities_info
from rfast_user_models   import surfalb
from rfast_user_models   import cloud_optprops
from rfast_user_models   import cloud_struct

nsample = 1000

# read inputs
filename_scr   = 'rfast_inputs.scr'
fnr,fnn,fns,dirout,Nlev,pmin,pmax,bg,\
species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,mmri,\
tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,\
species_l,species_c,\
lams,laml,res,regrid,smpl,opdir,\
Rp,Mp,gp,a,Apars,em,\
cld,phfc,opars,cpars,lamc0,fc,\
ray,ref,sct,fixp,pf,fixt,tf,p10,fp10,\
src,\
alpha,ntg,\
Ts,Rs,\
ntype,snr0,lam0,rnd,\
clr,fmin,mmrr,nwalkers,nstep,nburn,thin,restart,progress = inputs(filename_scr)

# read input data
data        = ascii.read(fnn + '.dat',data_start=1,delimiter='|')
lam         = data['col2'][:]
dlam        = data['col3'][:]
albedo      = data['col4'][:]
dat         = data['col6'][:]
err         = data['col7'][:]

reader = emcee.backends.HDFBackend(fnr + '.h5')
# get samples, discarding burn-in and apply thinning
samples  = reader.get_chain(discard = nburn)
nstep    = samples.shape[0]
nwalkers = samples.shape[1]
ndim     = samples.shape[2]

#flatten the chain
flatchain = samples.reshape(samples.shape[0] * samples.shape[1], samples.shape[2])

# Code setting every chain value to the truth to check the way spectra are generated
#real_values = [np.log10(0.78 * pmax), np.log10(0.21 * pmax), np.log10(3.e-3 * pmax), np.log10(7.e-7 * pmax), np.log10(4.e-4 * pmax), #np.log10(1.e-8 * pmax), np.log10(2.e-6 * pmax), -1, np.log10(0.2), 0, 0, np.log10(1.e4), np.log10(4.e6), 1, np.log10(0.5)]
#for i in range(len(flatchain)):
#	flatchain[i] = real_values

F1_array = []
F2_array = []

data = pd.read_csv('grey_surfcerr_A0_0.1.csv', delim_whitespace = True)

expected_F2 = np.array(data.albedo)
expected_err = np.mean(data.uncertainty)

r = random.sample(range(0, nstep*nwalkers), nsample)
n2 = []

print_vals = False
use_dat_file = False
if use_dat_file == False:
	for i in range(0,nsample):
		print('Walker ',i)
		print('Sample ',r[i], '\n')
		print('Albedo length is ',len(F1_array), '\n')

		#retreived atmospheric abundances
		#pp values
		lpn2     = flatchain[r[i],0]
		lpo2     = flatchain[r[i],1]
		lph2o    = flatchain[r[i],2]
		lpo3     = flatchain[r[i],3]
		lpco2    = flatchain[r[i],4]
		lpco     = flatchain[r[i],5]
		lpch4    = flatchain[r[i],6]
		pmax    = 10**lpn2+10**lpo2+10**lph2o+10**lpco2+10**lpo3+10**lpco+10**lpch4
        
		lfn2  = np.log10(10**lpn2/pmax)
		lfo2  = np.log10(10**lpo2/pmax)
		lfh2o = np.log10(10**lph2o/pmax)
		lfo3  = np.log10(10**lpo3/pmax)
		lfco2 = np.log10(10**lpco2/pmax)
		lfco  = np.log10(10**lpco/pmax)
		lfch4 = np.log10(10**lpch4/pmax)
  
		n2,o2,h2o,o3,co2,co,ch4= 10**(lfn2),10**(lfo2),10**(lfh2o),10**(lfo3),10**(lfco2),10**(lfco),10**(lfch4)

		f0[species_r=='n2'],f0[species_r=='o2'],f0[species_r=='h2o'],f0[species_r=='o3'],f0[species_r=='co2'],f0[species_r=='co'],f0[species_r=='ch4'] = n2,o2,h2o,o3,co2,co,ch4

		# retrieved planetary parameters
		A0     = 10**flatchain[r[i],7] # If you change this back to multicomponent,
		Rp     = 10**flatchain[r[i],8] # need to add A1 and shift every index up 1
		Mp     = 10**flatchain[r[i],9]
		dpc    = 10**flatchain[r[i],10]
		pt     = 10**flatchain[r[i],11]
		tauc0  = 10**flatchain[r[i],12]
		fc     = 10**flatchain[r[i],13]
        

		Ngas,gasid,mmw0,ray0,nu0,mb,rayb = set_gas_info(bg)
		
		print('A0 = ', A0)
		#print('A1 = ', A1)
        
        	# generate wavelength grids
		Nres           = 3 # no. of res elements to extend beyond grid edges to avoid edge sensitivity (2--4)
		lam,dlam       = gen_spec_grid(lams,laml,np.float_(res),Nres=0)
		lam_hr,dlam_hr = gen_spec_grid(lams,laml,np.float_(res)*smpl,Nres=np.rint(Nres*smpl))
			
        	# initialize disk integration quantities
		threeD = init_3d(src,ntg)
	
		# inform user of key opacities information
		opacities_info(opdir)
        
        	# initialize opacities and convolution kernels
		sigma_interp,cia_interp,ncia,ciaid,kern = init(lam,dlam,lam_hr,species_l,species_c,opdir,pf,tf)

		# initialize atmospheric model
		p,t,t0,z,grav,f,fb,m,nu = setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,
                                    	species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
                                    	mmri,mb,Mp,Rp,p10,fp10,src,ref,nu0)
		
		Apars = A0 #,A1 # read albedo parameters from chains
		As = surfalb(Apars,lam_hr) #for forward model

        	# cloud optical properties: asymmetry parameter, single scattering albedo, extinction efficiency
		gc,wc,Qc = cloud_optprops(opars,cld,opdir,lam_hr)

		# cloud vertical structure model
		cpars = pt,dpc,tauc0 # actually use chain values
		dtauc0,atm = cloud_struct(cpars,cld,p,t,z,grav,f,fb,m)
		p,t,z,grav,f,fb,m = atm

		# timing
		tstart = time.time()

        	# call forward model
        	#tstartg = time.time()
		F1_hr,F2_hr = gen_spec(Nlev,Rp,a,As,em,p,t,t0,m,z,grav,Ts,Rs,ray,ray0,rayb,f,fb,
                    mb,mmw0,mmrr,ref,nu,alpha,threeD,
                    gasid,ncia,ciaid,species_l,species_c,
                    cld,sct,phfc,fc,gc,wc,Qc,dtauc0,lamc0,
                    src,sigma_interp,cia_interp,lam_hr,pf=pf,tf=tf)
                               
        	# degrade resolution
		F1   = kernel_convol(kern,F1_hr)
		F2   = kernel_convol(kern,F2_hr)
		F1_array.extend(F1)
		F2_array.extend(F2)
		
        	# timing
		tend = time.time()
		print('Timing (s): ',tend-tstart, '\n')

        	# write data file
		names = ['wavelength (um)','d wavelength (um)','albedo','flux ratio']
		data_out = Table([lam,dlam,F1,F2], names=names)
		ascii.write(data_out,'spectra_gen.raw',format='fixed_width',overwrite=True)

        	# plot raw spectrum
		#plt.plot(lam, F1, alpha = 0.05)
        	#plt.plot(lam, F2, alpha = 0.05)
ci_bot   = []
ci_lower = []
ci_mid   = []
ci_upper = []
ci_max   = []
#generate error relative to each wavelength value


#when loading F2_array from a file
if use_dat_file == True:
    F1_array = pd.read_csv('F1_array_reshape.csv', header=0, index_col=0)

F1_array = np.asarray(F1_array)

#reshape the spectra planet-to-star flux ratios into rows for each case, and remove NaN values from array
spectra_samples = F2_array.reshape((nsample,len(expected_F2)))

dat_rows, dat_cols = np.shape(spectra_samples)

#saving the array to a csv file
if use_dat_file == False:
    F2_df = pd.DataFrame(spectra_samples)
    F2_df.to_csv('F2_array_reshape.csv')

for j in range(0,len(expected_F2)):
    globals()['ci_bot_%s' % j]=scipy.stats.scoreatpercentile(spectra_samples[:,j], 0., interpolation_method='fraction',axis=0)
    globals()['ci_lower_%s' % j]=scipy.stats.scoreatpercentile(spectra_samples[:,j], 2.5, interpolation_method='fraction',axis=0)
    globals()['ci_mid_%s' % j]=scipy.stats.scoreatpercentile(spectra_samples[:,j], 50.0, interpolation_method='fraction',axis=0)
    globals()['ci_upper_%s' % j]=scipy.stats.scoreatpercentile(spectra_samples[:,j], 97.5, interpolation_method='fraction',axis=0)
    globals()['ci_max_%s' % j]=scipy.stats.scoreatpercentile(spectra_samples[:,j], 100., interpolation_method='fraction',axis=0)
    
for q in range(0,len(expected_F2)):
    ci_bot.append(globals()['ci_bot_%s' % q])
    ci_lower.append(globals()['ci_lower_%s' % q])
    ci_mid.append(globals()['ci_mid_%s' % q])
    ci_upper.append(globals()['ci_upper_%s' % q])
    ci_max.append(globals()['ci_max_%s' % q])

ci_avg = np.add(ci_lower,ci_upper) / 2

lam         = np.asarray(lam)
expected_F2 = np.asarray(expected_F2)

expected_err_lower = [v - 2*expected_err for v in expected_F2]
expected_err_upper = [v + 2*expected_err for v in expected_F2]

if use_dat_file == True:
	for num in range(0,dat_rows):
		plt.plot(lam, expected_F2, color = 'red')
		plt.plot(lam,spectra_samples[num,:])
		plt.ylabel('Albedo')
		plt.xlabel('Wavelength ($\mu$m)')
		plt.grid(alpha = 0.5)
	plt.savefig('model_fits.png', dpi = 300)
	plt.close()
        
        
ax = plt.subplot(1,1,1)

ax.plot(lam, expected_F2, color='r')
#ax.plot(lam, ci_avg, color = 'g')
#ax.errorbar(lam, expected_F2, yerr=expected_err, color = 'r', ecolor='r', linestyle ='-')
ax.fill_between(lam, expected_err_upper, expected_err_lower, color = 'r', alpha = 0.5)
#ax.fill_between(lam, expected_err_max, expected_err_upper, color = 'y', alpha = 0.5, label='_nolegend_')
#ax.fill_between(lam, expected_err_lower, expected_err_bot, color = 'y', alpha = 0.5, label='_nolegend_')
ax.plot(lam, ci_mid, color='b')
ax.fill_between(lam, ci_upper, ci_lower, color='b',  alpha=0.33, label='95% CI')
#ax.fill_between(lam, ci_max, ci_bot, color = 'xkcd:sky blue', alpha = 0.33, label='100% CI')

plt.legend(['Expected Albedo', 'Median Albedo From 95% CI','95% CI'])
plt.ylabel('Albedo')
plt.xlabel('Wavelength ($\mu$m)')
plt.grid(alpha = 0.5)
plt.savefig('spectra_gen.png', dpi=300)
plt.close()
