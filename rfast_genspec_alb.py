# import statements
import time
import shutil
import os
import sys
import numpy             as np
import matplotlib.pyplot as plt
from astropy.table       import Table, Column, MaskedColumn
from astropy.io          import ascii
from scipy.interpolate   import interp1d
from rfast_routines      import gen_spec
from rfast_routines      import gen_spec_grid
from rfast_routines      import kernel_convol
from rfast_routines      import init
from rfast_routines      import init_3d
from rfast_routines      import inputs
from rfast_atm_routines  import set_gas_info
from rfast_atm_routines  import setup_atm
from rfast_atm_routines  import set_press_grid
from rfast_opac_routines import opacities_info
from rfast_user_models   import surfalb
from rfast_user_models   import cloud_optprops
from rfast_user_models   import cloud_struct

# limits calculations to single thread/processor
os.environ["OMP_NUM_THREADS"] = "1"

# get input script filename
if len(sys.argv) >= 2:
  filename_scr = sys.argv[1] # if script name provided at command line
else:
  filename_scr = input("rfast inputs script filename: ") # otherwise ask for filename

# obtain input parameters from script
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

# set info for all radiatively active gases, including background gas
Ngas,gasid,mmw0,ray0,nu0,mb,rayb = set_gas_info(bg)

# generate wavelength grids
Nres           = 3 # no. of res elements to extend beyond grid edges to avoid edge sensitivity (2--4)
lam,dlam       = gen_spec_grid(lams,laml,np.float_(res),Nres=0)
lam_hr,dlam_hr = gen_spec_grid(lams,laml,np.float_(res)*smpl,Nres=np.rint(Nres*smpl))


# initialize disk integration quantities
threeD   = init_3d(src,ntg)

# inform user of key opacities information
opacities_info(opdir)

# initialize opacities and convolution kernels
sigma_interp,cia_interp,ncia,ciaid,kern = init(lam,dlam,lam_hr,species_l,species_c,opdir,pf,tf)

# timing
tstart = time.time()

# initialize atmospheric model
#tstartp = time.time()
p,t,t0,z,grav,f,fb,m,nu = setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,
                                    tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,
                                    species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
                                    mmri,mb,Mp,Rp,p10,fp10,src,ref,nu0)
#print('Atmospheric setup timing (ms): ',(time.time()-tstartp)*1e3)

# surface albedo model
#tstartp = time.time()

albedo_model = surfalb(Apars,lam_hr)
As = albedo_model#[0] #returns the parametrisation

if any(As) > 1:
  print('Warning: albedo values larger than 1')
  quit()

#surface_1 = albedo_model[1] #interpolated reflectance data for first material
#surface_2 = albedo_model[2] #interpolated reflectance data for second material 
#surface_3 = albedo_model[3] #interpolated reflectance data for third material

#name1 = albedo_model[4] #names of the surfaces for plotting
#name2 = albedo_model[5]
#name3 = albedo_model[6]

#print('Run parameters: ', name1, name2, name3)
#print('Land fractions = ', Apars)
#print('Ocean fractions = ', (1 - Apars[0] - Apars[1]))
#print('Cloud coverage = ', fc)

#plot the interpolated reflectance spectra
#plt.plot(lam_hr, surface_1, label = name1, color = 'palevioletred')
#plt.plot(lam_hr, surface_2, label = name2, color = 'mediumslateblue')
#plt.plot(lam_hr, surface_3, label = name3, color = 'mediumseagreen')
#plt.title('Interpolations')
#plt.grid(alpha = 0.5)
#plt.ylabel('Reflectance')
#plt.xlabel('Wavelength ($\mu$m)')
#plt.legend()
#plt.savefig(dirout+'{}_{}'.format(name1,name2)+'_Interp'+'.png',format='png',bbox_inches='tight')
#plt.close()

# cloud optical properties: asymmetry parameter, single scattering albedo, extinction efficiency
gc,wc,Qc = cloud_optprops(opars,cld,opdir,lam_hr)

# cloud vertical structure model
dtauc0,atm = cloud_struct(cpars,cld,p,t,z,grav,f,fb,m)
p,t,z,grav,f,fb,m = atm
#print('Surface/cloud setup timing (ms): ',(time.time()-tstartp)*1e3)
 
# call forward model
#tstartp = time.time()
F1_hr,F2_hr = gen_spec(Nlev,Rp,a,As,em,p,t,t0,m,z,grav,Ts,Rs,ray,ray0,rayb,f,fb,
                       mb,mmw0,mmri,ref,nu,alpha,threeD,
                       gasid,ncia,ciaid,species_l,species_c,
                       cld,sct,phfc,fc,gc,wc,Qc,dtauc0,lamc0,
                       src,sigma_interp,cia_interp,lam_hr,pf=pf,tf=tf)
#print('Radiative transfer timing (ms): ',(time.time()-tstartp)*1e3)

# degrade resolution
#tstartp = time.time()
F1   = kernel_convol(kern,F1_hr)
F2   = kernel_convol(kern,F2_hr)
#print('Convolution timing (ms): ',(time.time()-tstartp)*1e3)

# timing
print('Total setup and forward model timing (ms), spectral points: ',(time.time()-tstart)*1e3,lam_hr.shape[0])

# write data file
if (src == 'diff' or src == 'cmbn'):
  names = ['wavelength','d_wavelength','albedo','flux_ratio']
if (src == 'thrm'):
  names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux (W/m**2/um)']
if (src == 'scnd'):
  names = ['wavelength (um)','d wavelength (um)','Tb (K)','flux ratio']
if (src == 'trns'):
  names = ['wavelength (um)','d wavelength (um)','zeff (m)','transit depth']
if (src == 'phas'):
  names = ['wavelength (um)','d wavelength (um)','reflect','flux ratio']
data_out = Table([lam,dlam,F1,F2], names=names)
ascii.write(data_out,dirout+fns+'.raw',format = 'fixed_width', overwrite = True)

#modified data write line only for albedo_modeling.py purposes
#ascii.write(data_out,dirout+'{}_{}'.format(name1,name2)+'_A0_'+str(Apars)+'.csv', overwrite = True)

# document parameters to file
shutil.copy(filename_scr,dirout+fns+'.log')
if not os.path.isfile(dirout+filename_scr):
  shutil.copy(filename_scr,dirout+filename_scr)

# useful to indicate radius at p10
if (src == 'trns'):
  Re = 6.378e6 # Earth radius (m)
  r  = Rp + z/Re
  r_interp = interp1d(np.log(p),r,fill_value="extrapolate")
  print("Radius at p10 (Re): ",r_interp(np.log(p10)))

# plot raw spectrum
if (src == 'diff' or src == 'scnd' or src == 'cmbn' or src == 'phas'):
  ylab = 'Albedo'
if (src == 'thrm'):
  ylab = r'Specific flux (W/m$^2$/${\rm \mu}$m)'
if (src == 'trns'):
  ylab = r'Transit depth'

#fig, ax = plt.subplots()
#ax.plot(lam, F1, c = 'cornflowerblue')
#ax.set_ylabel(ylab)
#ax.grid(alpha = 0.5)
#ax.set_xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
#ax.axvspan(0.7, 1.5, alpha = 0.3, color = 'palevioletred', label = 'NIR')
#ax.set_title('Surface Albedo Spectrum: {} & {}'.format(name1,name2))
#ax.legend()
#fig.savefig(dirout+'{}_{}_{}'.format(name1,name2,name3)+'.png',format='png',bbox_inches='tight', dpi = 150)
#plt.close()
