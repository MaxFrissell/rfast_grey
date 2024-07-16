# import statements
import emcee
import os
import time
import sys
import shutil
import h5py
import numpy             as np
import matplotlib.pyplot as plt
import pandas            as pd
from multiprocessing     import Pool
from astropy.io          import ascii
from rfast_routines      import spectral_grid
from rfast_routines      import gen_spec
from rfast_routines      import kernel_convol
from rfast_routines      import gen_spec_grid
from rfast_routines      import inputs
from rfast_routines      import init
from rfast_routines      import init_3d
from rfast_routines      import readdat
from rfast_atm_routines  import set_gas_info
from rfast_atm_routines  import setup_atm
from rfast_atm_routines  import mmr2vmr
from rfast_atm_routines  import vmr2mmr
from rfast_opac_routines import opacities_info
from rfast_user_models   import surfalb
from rfast_user_models   import cloud_optprops
from rfast_user_models   import cloud_struct
#from rfast_user_models   import surfalb_fast # Don't need
from rfast_user_models   import cloud_optprops_fast

## FROM RFAST_ANALYZE_PP
## Need this to make mcmc restart from the .h5 file and run steps until the
## Number of steps is reached

# simple routine for importing emcee chain from h5 file
def reademceeh5(fn,nburn,thin,flatten=False):

  # open file, important data
  hf       = h5py.File(fn,'r')
  grps     = [item for item in hf['mcmc'].values()]

  # extract samples chain and log-likelihood, remove burn-in
  if (nburn >= 0):
    samples  = grps[1][nburn:,:,:]
    lnprob   = grps[2][nburn:,:]
  else:
    samples  = grps[1][nburn:,:,:]
    lnprob   = grps[2][nburn:,:]

  # thin
  samples  = samples[0::thin,:,:]
  lnprob   = lnprob[0::thin,:]

  # flatten
  if flatten:
    samples  = samples.reshape(samples.shape[0]*samples.shape[1],samples.shape[2])
    lnprob   = lnprob.reshape(lnprob.shape[0]*lnprob.shape[1])

  # close h5 file
  hf.close()

  return samples,lnprob
  
## END RFAST_ANALYZE_PP

# recommended to prevent interference with emcee parallelization
#os.environ["OMP_NUM_THREADS"] = "1"

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

## AREA WHERE MAX IS TRYING TO LOAD FILES ONCE AT THE START
  
# Load all surface and cloud files
#granite = pd.read_csv('granite_solid.csv')
#open_ocean = pd.read_csv('open_ocean_usgs.csv')
#weathered_basalt = pd.read_csv('basalt_weathered_usgs.csv')
liq_cloud_data = readdat(opdir+'strato_cum.mie',19)
ice_cloud_data = readdat(opdir+'baum_cirrus_de100.mie',1)

# If we're restarting from a .h5 file, check out current step
if restart:
# Find how many steps have been completed already in .h5
  samples_for_restart, lnprob_for_restart = reademceeh5(dirout+fnr+'.h5',0,1) # Last variables are burnin and thin
  current_step = samples_for_restart.shape[0]

#CHECKED VALUES:
#pmax = 10100.0
#species_r = n2, o2, h2o, o3, co2, co, ch4

# unpackage parameters from user-defined routines
tiso = tpars[0]
# A0 = Apars[0]
# A1 = Apars[1]
# no parameters for cloud optical properties model
pt,dpc,tauc0 = cpars

# input data filename
fn_dat = fnn + '.dat' #comes from rfast_noise.py

# set info for all radiatively active gases
Ngas,gasid,mmw0,ray0,nu0,mb,rayb = set_gas_info(bg)

#Ngas = 14
#gasid = ar, ch4, co2, h2, h2o, he, n2, o2, o3, n2o, co, so2, nh3, c2h2

# get initial gas mixing ratios
p,t,t0,z,grav,f,fb,m,nu = setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,
                                    tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,
                                    species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
                                    mmri,mb,Mp,Rp,p10,fp10,src,ref,nu0)


# convert between mmr and vmr, if needed
#if (mmri != mmrr):
 # if mmri: # convert input mmr to vmr
  #  f,fb,f0 = mmr2vmr(mmw0,gasid,species_r,m,mb,f0,fb,f)
 # else: # otherwise convert input vmr to mmr
  #  f,fb,f0 = vmr2mmr(mmw0,gasid,species_r,m,mb,f0,fb,f)


# read input data
data        = ascii.read(dirout+fn_dat,data_start=1,delimiter='|')
lam         = data['col2'][:]
dlam        = data['col3'][:]
albedo      = data['col4'][:] ##added january 11th
dat         = data['col6'][:] #flux ratio
err         = data['col7'][:] #flux ratio error

# save input radius for thermal emission case
Rpi = Rp

# generate wavelength grids
Nres             = 3 # no. of res elements to extend beyond grid edges to avoid edge sensitivity (2--4)
if regrid: #regrid == True diverges here
  lam_hr,dlam_hr = gen_spec_grid(lams,laml,np.float_(res)*smpl,Nres=np.rint(Nres*smpl))
else:
  x_low = min(0.01,min(lam)-dlam[0]*Nres) # note: prevent min wavelength of 0 um
  x_hgh = max(lam)+dlam[-1]*Nres
  lam_hr,dlam_hr = spectral_grid(x_low,x_hgh,res=lam/dlam*smpl,lamr=lam)

# assign photometric vs. spectroscopic points
mode           = np.copy(lam_hr)
mode[:]        = 1 #sets all values in mode to 1...why?

# initialize disk integration quantities
threeD   = init_3d(src,ntg)

# initialize opacities and convolution kernels
sigma_interp,cia_interp,ncia,ciaid,kern = init(lam,dlam,lam_hr,species_l,species_c,opdir,pf,tf,mode=mode)

# initialize atmospheric model, this seems redundant jan 31
#p,t,t0,z,grav,f,fb,m,nu = setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,
                                   # tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,
                                   # species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
                                   # mmri,mb,Mp,Rp,p10,fp10,src,ref,nu0)


# surface albedo model
As = surfalb(Apars,lam_hr) #only one argument returned by surfalb for grey surface

# cloud optical properties: asymmetry parameter, single scattering albedo, extinction efficiency
gc,wc,Qc = cloud_optprops(opars,cld,opdir,lam_hr)

# cloud vertical structure model
dtauc0,atm = cloud_struct(cpars,cld,p,t,z,grav,f,fb,m)
p,t,z,grav,f,fb,m = atm

# min and max center-log ratio, if doing clr retrieval
if clr:
  n     = len(f0) + 1
  ximin = (n-1.)/n*(np.log(fmin) - np.log((1.-fmin)/(n-1.)))
  ximax = (n-1)/n*(np.log(1-n*fmin) - np.log(fmin))

#define priors for retreived parameters (from previous code added january 11th)
gaslower,   gasupper   = 10**(-2.0), 10**(7.0)
tracelower             = 10**(-12.0) # Adding so trace gases are in prior range
pmaxlower,  pmaxupper  = 1.,         1.e7
A0lower,    A0upper    = 0.01,       1.
#A1lower,    A1upper    = 0.01,       1. #for land fraction surface albedo model
Rplower,    Rpupper    = 10**(-0.5), 10**(0.5)
Mplower,    Mpupper    = 0.1,        10.
dpclower,   dpcupper   = 1.,         1.e7
ptlower,    ptupper    = 1.,         1.e7
tauc0lower, tauc0upper = 0.001,      1000.
fclower,    fcupper    = 0.001,      1.

lgaslower,   lgasupper   = np.log10(gaslower),   np.log10(gasupper)
ltracelower              = np.log10(tracelower)
lpmaxlower,  lpmaxupper  = np.log10(pmaxlower),  np.log10(pmaxupper)
lA0lower,    lA0upper    = np.log10(A0lower),    np.log10(A0upper)
#lA1lower,    lA1upper    = np.log10(A1lower),    np.log10(A1upper) #for more than one albedo parameter
lRplower,    lRpupper    = np.log10(Rplower),    np.log10(Rpupper)
lMplower,    lMpupper    = np.log10(Mplower),    np.log10(Mpupper)
ldpclower,   ldpcupper   = np.log10(dpclower),   np.log10(dpcupper)
lptlower,    lptupper    = np.log10(ptlower),    np.log10(ptupper)
ltauc0lower, ltauc0upper = np.log10(tauc0lower), np.log10(tauc0upper)
lfclower,    lfcupper    = np.log10(fclower),    np.log10(fcupper)

#pmax = 10100.0

# log-prior function
def lnprior(x):
  lpN2, lpO2,lpH2O,lpO3,lpCO2,lpCO,lpCH4,lA0,lRp,lMp,ldpc,lpt,ltauc0,lfc = x #changed order to match initialised priors

##added from previous code january 11th

  pmax  = 10**lpN2+10**lpO2+10**lpH2O+10**lpO3+10**lpCO2+10**lpCO+10**lpCH4
  lpmax = np.log10(pmax)
  lfN2  = np.log10(10**lpN2/pmax)
  lfO2  = np.log10(10**lpO2/pmax)
  lfH2O = np.log10(10**lpH2O/pmax)
  lfO3  = np.log10(10**lpO3/pmax)
  lfCO2 = np.log10(10**lpCO2/pmax)
  lfCO  = np.log10(10**lpCO/pmax)
  lfCH4 = np.log10(10**lpCH4/pmax)

##^^^^
  
  fN2,fO2,fH2O,fO3,fCO2,fCO,fCH4,pmax,Rp,Mp,fc,A0,dpc,pt,tauc0 = 10**(lfN2),10**(lfO2),10**(lfH2O),10**(lfO3),10**(lfCO2),10**(lfCO),10**(lfCH4),10**(lpmax),10**(lRp),10**(lMp),10**(lfc),10**(lA0),10**(ldpc),10**(lpt),10**(ltauc0)
  f0[species_r=='n2'],f0[species_r=='o2'],f0[species_r=='h2o'],f0[species_r=='o3'],f0[species_r=='co2'],f0[species_r=='co'],f0[species_r=='ch4'] = fN2,fO2,fH2O,fO3,fCO2,fCO,fCH4

  # sum gaussian priors
  lng = 0.0 

  # prior limits modified from previous code january 11th
  if lgaslower   <= lpN2  <= lgasupper   and \
     ltracelower   <= lpO2  <= lgasupper   and \
     lgaslower   <= lpH2O <= lgasupper   and \
     ltracelower   <= lpO3  <= lgasupper   and \
     lgaslower   <= lpCO2 <= lgasupper   and \
     ltracelower   <= lpCO  <= lgasupper   and \
     lgaslower   <= lpCH4 <= lgasupper   and \
     A0lower    <= A0    <= A0upper    and \
     Rplower    <= Rp    <= Rpupper    and \
     Mplower    <= Mp    <= Mpupper    and \
     dpclower   <= dpc   <= dpcupper   and \
     ptlower    <= pt    <= ptupper    and \
     tauc0lower <= tauc0 <= tauc0upper and \
     fclower    <= fc    <= fcupper    and \
     np.sum(f0) <= 1                   and \
     pt + dpc   <  pmax:
    return 0.0 + lng
  return -np.inf
##^^^

# log-likelihood function
def lnlike(x):

  # reverts to using Mp if gp is not retrieved
  gp = -1

  lpN2,lpO2,lpH2O,lpO3,lpCO2,lpCO,lpCH4,lA0,lRp,lMp,ldpc,lpt,ltauc0,lfc = x #matches order of x above 


##added from previous code january 11th
  pmax  = 10**lpN2+10**lpO2+10**lpH2O+10**lpO3+10**lpCO2+10**lpCO+10**lpCH4
  lpmax = np.log10(pmax)
  lfN2  = np.log10(10**lpN2/pmax)
  lfO2  = np.log10(10**lpO2/pmax)
  lfH2O = np.log10(10**lpH2O/pmax)
  lfO3  = np.log10(10**lpO3/pmax)
  lfCO2 = np.log10(10**lpCO2/pmax)
  lfCO  = np.log10(10**lpCO/pmax)
  lfCH4 = np.log10(10**lpCH4/pmax)
##^^^^
  
  # not performing clr retrieval
  fN2,fO2,fH2O,fO3,fCO2,fCO,fCH4,pmax,Rp,Mp,fc,A0,dpc,pt,tauc0 = 10**(lfN2),10**(lfO2),10**(lfH2O),10**(lfO3),10**(lfCO2),10**(lfCO),10**(lfCH4),10**(lpmax),10**(lRp),10**(lMp),10**(lfc),10**(lA0),10**(ldpc),10**(lpt),10**(ltauc0)

  f0[species_r=='n2'],f0[species_r=='o2'],f0[species_r=='h2o'],f0[species_r=='o3'],f0[species_r=='co2'],f0[species_r=='co'],f0[species_r=='ch4'] = fN2,fO2,fH2O,fO3,fCO2,fCO,fCH4

  # package parameters for user-defined routines
  tpars = tiso
  Apars = A0#,A1
  # no parameters for cloud optical properties model
  cpars = pt,dpc,tauc0

  # package parameters for call to forward model - ie this x & y are the ones being called in the forward model repackaged from genspec
  x0 = f0,pmax,Rp,Mp,gp,Apars,tpars,opars,cpars,fc,a,alpha,mb,rayb
  y  = lam_hr,pmin,Ngas,mmw0,nu0,em,Ts,Rs,ray0,gasid,ncia,ciaid,species_l,species_c,\
       Rpi,sigma_interp,cia_interp,kern,ray,ref,cld,sct,phfc,fixp,pf,fixt,tf,mmrr,\
       p10,fp10,src,threeD,fntmp,skptmp,colt,colpt,psclt,species_r,fnatm,skpatm,\
       colr,colpr,psclr,Nlev
  
  return -0.5*(np.sum((dat-Fx(x0,y))**2/err**2)) #should this be changed to albedo?


# log-probability from Bayes theorem
'''
def lnprob(x):

  lp = lnprior(x)
  
  if not np.isfinite(lp):
    return -np.inf
  return lp + lnlike(x)

'''

def lnprob(x):
  try:
      lp = lnprior(x)
  
      if not np.isfinite(lp):
        return -np.inf
      return lp + lnlike(x)
  except:
      #print(x) #returns segmentation fault error?
      return -np.inf
   
# forward model for emcee and analysis purposes; re-packages gen_spec routine
def Fx(x,y):

  f0,pmax,Rp,Mp,gp,Apars,tpars,opars,cpars,fc,a,alpha,mb,rayb = x #this uses .scr input values
  lam,pmin,Ngas,mmw0,nu0,em,Ts,Rs,ray0,gasid,ncia,ciaid,species_l,species_c,\
  Rpi,sigma_interp,cia_interp,kern,ray,ref,cld,sct,phfc,fixp,pf,fixt,tf,mmrr,\
  p10,fp10,src,threeD,fntmp,skptmp,colt,colpt,psclt,species_r,fnatm,skpatm,\
  colr,colpr,psclr,Nlev = y

  # do not read in thermal structure
  rdtmp = False

  # do not read in atmospheric structure
  rdgas = False

  # initialize atmospheric model
  p,t,t0,z,grav,f,fb,m,nu = setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,
                                      tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,
                                      species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
                                      mmrr,mb,Mp,Rp,p10,fp10,src,ref,nu0,gp=gp)
  #goes through setup_atm from rfast_atm_routines here

  # grey surface albedo model
  As = surfalb(Apars,lam)

  # multicomponent surface albedo model
  # As = surfalb_fast(Apars,lam,granite,open_ocean,weathered_basalt)[0]

  # cloud optical properties: asymmetry parameter, single scattering albedo, extinction efficiency
  gc,wc,Qc = cloud_optprops_fast(opars,cld,opdir,lam,liq_cloud_data,ice_cloud_data)

  # cloud vertical structure model
  dtauc0,atm = cloud_struct(cpars,cld,p,t,z,grav,f,fb,m)
  p,t,z,grav,f,fb,m = atm

  # call forward model
  F1,F2 = gen_spec(Nlev,Rp,a,As,em,p,t,t0,m,z,grav,Ts,Rs,ray,ray0,rayb,f,fb,
                    mb,mmw0,mmrr,ref,nu,alpha,threeD,
                    gasid,ncia,ciaid,species_l,species_c,
                    cld,sct,phfc,fc,gc,wc,Qc,dtauc0,lamc0,
                    src,sigma_interp,cia_interp,lam,pf=pf,tf=tf)

  # degrade resolution
  F_out = kernel_convol(kern,F1) ##changed to F1 on january 11th, degrades from 1752 to 158

  # "distance" scaling for thermal emission case
  if (src == 'thrm'):
    F_out = F_out*(Rp/Rpi)**2

  return F_out #length 158

# syntax to identify main core of program
##added from previously modified code on january 11th
if __name__ == '__main__': #why do these have to be redefined if they were initialised outside a function previously
  gaslower,    gasupper    = 10**(-2.0),           10**(7.0)
  tracelower               = 10**(-12)
  lgaslower,   lgasupper   = np.log10(gaslower),   np.log10(gasupper)
  ltracelower              = np.log10(tracelower)
  lA0lower,    lA0upper    = np.log10(0.01),       np.log10(1.0)
  #lA1lower,    lA1upper    = np.log10(0.01),       np.log10(1.0)
  lRplower,    lRpupper    = np.log10(10**(-0.5)), np.log10(10.0**(0.5))
  lMplower,    lMpupper    = np.log10(0.1),        np.log10(10.0)
  ldpclower,   ldpcupper   = np.log10(1.),         np.log10(1.e7)
  lptlower,    lptupper    = np.log10(1.),         np.log10(1.e7)
  ltauc0lower, ltauc0upper = np.log10(0.001),      np.log10(1000.0)
  lfclower,    lfcupper    = np.log10(0.001),      np.log10(1.0)
  
# initialize walkers in a uniform distribution of values within their prior range
  lpN2_pos   = [np.random.uniform(low=lgaslower, high=lgasupper) for i in range(nwalkers)]
  lpO2_pos   = [np.random.uniform(low=ltracelower, high=lgasupper) for i in range(nwalkers)]
  lpH2O_pos  = [np.random.uniform(low=lgaslower, high=lgasupper) for i in range(nwalkers)]
  lpO3_pos   = [np.random.uniform(low=ltracelower, high=lgasupper) for i in range(nwalkers)]
  lpCO2_pos  = [np.random.uniform(low=lgaslower, high=lgasupper) for i in range(nwalkers)]
  lpCO_pos   = [np.random.uniform(low=tracelower, high=lgasupper) for i in range(nwalkers)]
  lpCH4_pos  = [np.random.uniform(low=lgaslower, high=lgasupper) for i in range(nwalkers)]
  lA0_pos    = [np.random.uniform(low=lA0lower, high=lA0upper) for i in range(nwalkers)]
  #lA1_pos    = [np.random.uniform(low=lA1lower, high=lA1upper) for i in range(nwalkers)]
  lRp_pos    = [np.random.uniform(low=lRplower, high=lRpupper) for i in range(nwalkers)]
  lMp_pos    = [np.random.uniform(low=lMplower, high=lMpupper) for i in   range(nwalkers)]
  ldpc_pos   = [np.random.uniform(low=ldpclower, high=ldpcupper) for i in range(nwalkers)]
  dpc_pos    = [10**x for x in ldpc_pos]
  lpt_pos    = [np.random.uniform(low=lptlower, high=lptupper) for i in range(nwalkers)]
  pt_pos     = [10**x for x in lpt_pos]
  ltauc0_pos = [np.random.uniform(low=ltauc0lower,high=ltauc0upper) for i in range(nwalkers)]
  lfc_pos    = [np.random.uniform(low=lfclower,high=lfcupper) for i in range(nwalkers)]
  
  pmax_pos  = [10**i for i in lpN2_pos]+[10**i for i in lpO2_pos]+[10**i for i in lpH2O_pos]+[10**i for i in lpO3_pos]+[10**i for i in lpCO2_pos]+[10**i for i in lpCO_pos]+[10**i for i in lpCH4_pos]

# ensure the cloud top pressure and change in cloud pressure sum to less than pmax
##added from previous code, ensures that the clouds aren't on the surface
  #while (sum(pt_pos)+sum(dpc_pos) >= sum(pmax_pos))or(10**lA0_pos+10**lA1_pos>=1.0): #this condition is not called, code jumps straight to pos under else:
  for i in range(nwalkers):
      while (pt_pos[i]+dpc_pos[i] >= pmax_pos[i]):
          lpN2_pos[i] = np.random.uniform(low=lgaslower, high=lgasupper)
          lpO2_pos[i] = np.random.uniform(low=ltracelower, high=lgasupper)
          lpH2O_pos[i]  = np.random.uniform(low=lgaslower, high=lgasupper)
          lpO3_pos[i]   = np.random.uniform(low=ltracelower, high=lgasupper)
          lpCO2_pos[i]  = np.random.uniform(low=lgaslower, high=lgasupper)
          lpCO_pos[i]   = np.random.uniform(low=ltracelower, high=lgasupper)
          lpCH4_pos[i]  = np.random.uniform(low=lgaslower, high=lgasupper)
          lA0_pos[i]    = np.random.uniform(low=lA0lower, high=lA0upper)
          #lA1_pos[i]    = np.random.uniform(low=lA1lower, high=lA1upper)
          pmax_pos[i]  = 10**lpN2_pos[i]+10**lpO2_pos[i]+10**lpH2O_pos[i]+10**lpO3_pos[i]+10**lpCO2_pos[i]+10**lpCO_pos[i]+10**lpCH4_pos[i]
          ldpc_pos[i]   = np.random.uniform(low=ldpclower, high=ldpcupper)
          dpc_pos[i]    = 10**ldpc_pos[i] 
          lpt_pos[i]    = np.random.uniform(low=lptlower, high=lptupper)
          pt_pos[i]     = 10**lpt_pos[i]
  else:
    pos = np.vstack([lpN2_pos, lpO2_pos, lpH2O_pos, lpO3_pos, lpCO2_pos, lpCO_pos, lpCH4_pos, lA0_pos, lRp_pos, lMp_pos, ldpc_pos, lpt_pos, ltauc0_pos, lfc_pos]).T
    #why the transpose?
  
  # inform user of key opacities information
  opacities_info(opdir)

  # test forward model
  x  = f0,pmax,Rp,Mp,gp,Apars,tpars,opars,cpars,fc,a,alpha,mb,rayb #same x as above?
  y  = lam_hr,pmin,Ngas,mmw0,nu0,em,Ts,Rs,ray0,gasid,ncia,ciaid,species_l,species_c,\
       Rpi,sigma_interp,cia_interp,kern,ray,ref,cld,sct,phfc,fixp,pf,fixt,tf,mmrr,\
       p10,fp10,src,threeD,fntmp,skptmp,colt,colpt,psclt,species_r,fnatm,skpatm,\
       colr,colpr,psclr,Nlev

  if (src == 'diff' or src == 'scnd' or src == 'cmbn' or src == 'phas'):
    ylab = 'Albedo'
  if (src == 'thrm'):
    ylab = r'Specific flux (W/m$^2$/${\rm \mu}$m)'
  if (src == 'trns'):
    ylab = r'Transit depth'
  
  #plt.scatter(lam, albedo, s = 12, c = 'palevioletred') #not including error bars on the flux ratio
  plt.errorbar(lam, albedo, yerr = err, fmt = '.k') #error should be in albedo
  plt.plot(lam,Fx(x,y)) #jumps back to Fx function above at line 283
  plt.ylabel(ylab)
  plt.xlabel(r'Wavelength (' + u'\u03bc' + 'm)')
  plt.savefig(dirout+fnr+'_test.png',format='png',bbox_inches='tight')
  plt.close()

  # document parameters to file
  shutil.copy(filename_scr,dirout+fnr+'.log')

  # g(x) after benneke & seager (2012); only needed if doing clr retrieval
  if clr:
    gx = np.exp((np.sum(np.log(f0)) + np.log(max(fmin,1-np.sum(f0))))/(len(f0) + 1))

  # unpackage parameters from user-defined routines
  tiso = tpars[0]
  A0 = Apars#[0]
  #A1 = Apars[1]
  # no parameters for cloud optical properties model
  pt,dpc,tauc0 = cpars

  # retrieved parameters initial guess
  lfN2,lfO2,lfH2O,lfO3,lfCO2,lfCO,lfCH4 = np.log10(f0[species_r=='n2'])[0],np.log10(f0[species_r=='o2'])[0],np.log10(f0[species_r=='h2o'])[0],np.log10(f0[species_r=='o3'])[0],np.log10(f0[species_r=='co2'])[0],np.log10(f0[species_r=='co'])[0],np.log10(f0[species_r=='ch4'])[0]
  #log10 of the values from f0

##added january 11th
  lpN2  = np.log10((10**lfN2)*pmax) #convert to partial pressures 
  lpO2  = np.log10((10**lfO2)*pmax)
  lpH2O = np.log10((10**lfH2O)*pmax)
  lpO3  = np.log10((10**lfO3)*pmax)
  lpCO2 = np.log10((10**lfCO2)*pmax)
  lpCO  = np.log10((10**lfCO)*pmax)
  lpCH4 = np.log10((10**lfCH4)*pmax)
  lA0,lRp,lMp,ldpc,lpt,ltauc0,lfc = np.log10(A0),np.log10(Rp),np.log10(Mp),np.log10(dpc),np.log10(pt),np.log10(tauc0),np.log10(fc)

  guess = [lpN2,lpO2,lpH2O,lpO3,lpCO2,lpCO,lpCH4,lA0,lRp,lMp,ldpc,lpt,ltauc0,lfc] #consistent with the paper
  #same order as the x variables in prob and prior functions
  #want to initialise the walkers around these values?

  ndim  = len(guess)

  # create backup / save file; prevent h5 overwrite or check if restart h5 exists
  if not restart:
    if os.path.isfile(dirout+fnr+'.h5'):
      print("rfast warning | major | h5 file already exists")
      quit()
    else: #directs to this conditions
      backend  = emcee.backends.HDFBackend(dirout+fnr+'.h5')
      backend.reset(nwalkers, ndim)
      # initialize walkers as a cloud around guess

      #pos = guess + 1e-4*np.random.randn(nwalkers,ndim)
  else:
    if not os.path.isfile(dirout+fnr+'.h5'):
      print("rfast warning | major | h5 does not exist for restart")
      quit()
    else:
      # otherwise initialize walkers from existing backend
      new_backend = emcee.backends.HDFBackend(dirout + fnr+ '.h5')
      pos = new_backend.get_last_sample()
      #backend  = emcee.backends.HDFBackend(dirout+fnr+'.h5')
      #pos = backend.get_last_sample()
  
  # timing
  tstart = time.time()

  # multiprocessing implementation
  with Pool() as pool:

    if not restart:
      # initialize the sampler
      sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool, moves=[(emcee.moves.DEMove(sigma=5e-04, gamma0=2.38/np.sqrt(2*ndim)), 0.8), (emcee.moves.DESnookerMove(gammas=2.0), 0.2),])

      # run the mcmc
      sampler.run_mcmc(pos, nstep, progress=progress) #failing here

    else:
      print('redirected to added condition')
      
      sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=new_backend, pool=pool, moves=[(emcee.moves.DEMove(sigma=5e-04, gamma0=2.38/np.sqrt(2*ndim)), 0.8), (emcee.moves.DESnookerMove(gammas=2.0), 0.2),])
      
      # run mcmc for as many steps as needed to reach nstep
      extra_steps = nstep - current_step
      sampler.run_mcmc(None, extra_steps, progress = progress)

    tend = time.time()
    print('Retrieval timing (s): ',tend-tstart)

'''
    #initialise the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool,  moves=[(emcee.moves.DEMove(sigma=5e-04, gamma0=2.38/np.sqrt(2*ndim)), 0.8), (emcee.moves.DESnookerMove(gammas=2.0), 0.2),])

    #run the mcmc
    sampler.run_mcmc(pos, nstep, progress = progress)


  # timing
  tend = time.time()
  print('Retrieval timing (s): ',tend-tstart)
'''
