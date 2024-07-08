import numpy             as     np
import math
import time
from   scipy.interpolate import interp1d
from   rfast_routines    import readdat  
from   rfast_user_models import atm_temp
#
# set up hard-coded gas parameters
#
# inputs:
#
#       bg   - background gas identifier (e.g., 'ar')
#
def set_gas_info(bg):

  # radiatively active gas info:
  #   name
  #   molar weight (g/mole)
  #   cross section (m**2/molecule) at 0.4579 um for
  #   STP refractivity
  #   note: zero is used as placeholder in ray0 and nu0 for gases that will "always" be trace
  #   note: see sneep & ubacks 2005, jqsrt 92:293--310
  gasid =             [    'ar',    'ch4',    'co2',    'h2',   'h2o',    'he',     'n2',    'o2',   'o3',    'n2o',    'co',   'so2',   'nh3', 'c2h2']
  mmw0  =  np.float32([  39.948,    16.04,    44.01,  2.0159,  18.015,  4.0026,   28.013,  31.999, 48.000,   44.013,   28.01,  64.066,  17.031, 26.038])
  ray0  =  np.float32([8.29e-31,2.035e-30,2.454e-30,2.16e-31,8.93e-31,1.28e-32,10.38e-31,8.39e-31, 0.0000,2.903e-30,11.3e-31,  0.0000,  0.0000, 0.0000])
  nu0   =  np.float32([0.000281, 0.000444, 0.000449,0.000132,0.000261,0.000035, 0.000298,0.000271, 0.0000, 0.000483,0.000325,0.000686,0.000376, 0.0000])

  # number of available radiatively active gases
  Ngas  = len(gasid)

  # set background gas parameters, noting that rayb is relative to argon
  ib    = gasid.index(bg.lower())
  mb    = mmw0[ib]
  rayb  = ray0[ib]/ray0[gasid.index('ar')]

  return Ngas,gasid,mmw0,ray0,nu0,mb,rayb
#
#
# set up atmospheric model
#
# inputs:
#
#      Nlev   - number of atmospheric levels
#      pmin   - top of atmosphere pressure (Pa)
#      pmax   - bottom of atmosphere pressure (Pa)
#      tpars  - parameters for user-defined thermal structure model
#     rdtmp   - if true, read gas mixing ratios from file
#     fntmp   - filename for thermal structure
#    skptmp   - lines to skip for header in fntmp
#      colt   - column of temperature in fntmp
#     colpt   - pressure column in fntmp
#     psclt   - factor to convert pressure to Pa
#      Ngas   - number of potential radiatively active gases, from set_gas_info
# species_r   - identifiers of user-requested radiatively active gases
#        f0   - gas mixing ratios for vertically-constant case
#     rdgas   - if true, read gas mixing ratios from file
#     fnatm   - filename for gas mixing ratios
#    skpatm   - lines to skip for header in fnatm
#      colr   - columns of gas mixing ratios corresponding to species_r
#     colpr   - pressure column in fnatm
#     psclr   - factor to convert pressure to Pa
#      mmw0   - gas mean molar weights (see set_gas_info)
#       mmr   - if true, mixing ratios are interpreted as mass mixing ratios
#       cld   - flag that indicates if clouds are included
#        pt   - cloud top pressure (Pa)
#       dpc   - cloud thickness (dpc)
#     tauc0   - cloud optical depth at user-specified wavelength
#       ref   - flag to indicate if refraction is included
#        mb   - background gas mean molar weight (g/mole)
#        Mp   - planetary mass (Mearth)
#        Rp   - planetary radius (Rearth)
#       src   - model type (diff,thrm,cmbn,scnd,trns)
#       p10   - pressure for fixed-pressure planetary radius (Pa)
#      fp10   - if true, Rp is interpreted as being given at p10
#       ref   - flag to indicate if refraction is included for transit case
#       nu0   - gas STP refractivities (see set_gas_info)
#        gp   - (optional) planetary surface gravity (m s**-2)
#
# outputs:
#
#         p   - pressure grid (Pa) (Nlev)
#         t   - temperature profile (K) (Nlev)
#         z   - altitude profile (m) (Nlev)
#      grav   - gravity profile (m/s/s) (Nlev)
#         f   - gas mixing ratio profiles (Ngas x Nlev)
#         m   - atmospheric mean molecular weight (Nlev)
#       nu0   - atmospheric refractivity at STP (float)
#
def setup_atm(Nlev,Ngas,gasid,mmw0,pmin,pmax,
              tpars,rdtmp,fntmp,skptmp,colt,colpt,psclt,
              species_r,f0,rdgas,fnatm,skpatm,colr,colpr,psclr,
              mmr,mb,Mp,Rp,p10,fp10,src,ref,nu0,gp=-1):

  # small mixing ratio for non-active gases
  rsmall = 1e-10

  # set pressure grid (ie, vertical grid)
  p = set_press_grid(Nlev,pmin,pmax)

  # ensure p10 is in pressure grid, for transit spectra and if used
  if (src == "trns" and fp10):
    ip    = np.where(abs(p-p10) == min(abs(p-p10)))[0]
    p[ip] = p10     

  # set gas profiles -- either vertically constant or read in
  f    = np.zeros([Ngas,Nlev])
  f[:] = rsmall
  if not rdgas:
    i = 0
    for id in species_r:
      ig = gasid.index(id.lower())
      f[ig,:] = f0[i]
      i  = i + 1
  else:
    dat  = readdat(fnatm,skpatm) # [ levels x input columns]
    pd   = dat[:,colpr-1]*psclr
    i    = 0
    for id in species_r:
      fd        = dat[:,colr[i]-1]
      fd_interp = interp1d(pd,np.log10(fd),fill_value="extrapolate")
      fi = np.power(10,fd_interp(p))
      ig = gasid.index(id.lower())
      f[ig,:] = fi
      i  = i + 1    

  # mixing ratios summed across all species
  ft = np.sum(f,axis=0)

  # fill amount, cannot be less than zero
  fb         = 1 - ft
  fb[fb < 0] = 0

  # set mean molecular weight (kg/molec)
  m = mmw(f,fb,mb,mmw0,mmr)

  # set refractivity at STP
  nu = 0.
  if (ref and src == 'trns'):
    nu = refractivity(f,nu0,mmw0,mmr,m)

  # set temperature profile, either from user-defined model or read-in
  if not rdtmp:
    t,t0  = atm_temp(tpars,p,Ngas,gasid,f,fb,mmr,m,mb)
  else:
    dat  = readdat(fntmp,skptmp) # [ levels x input columns]
    pd   = dat[:,colpt-1]*psclt
    td   = dat[:,colt-1]
    td_interp = interp1d(pd,td,fill_value="extrapolate")
    t    = td_interp(p)
    t[-1]= t0 # set surface / lower boundary temperature

  # do hydrostatic calculation
  #tstart = time.time()
  if (src == "trns" and fp10):
    z,grav = hydrostat(Nlev,p,t,m,Mp,Rp,gp,ip=ip[0])
  else:
    z,grav = hydrostat(Nlev,p,t,m,Mp,Rp,gp)
  #print('Hydrostatic calculation timing (s): ',time.time()-tstart)

  return p,t,t0,z,grav,f,fb,m,nu
#
#
# hydrostatic calculation
#
# inputs:
#
#      Nlev   - number of atmospheric levels
#         p   - pressure grid (Pa)
#         t   - temperature profile (K)
#         m   - mean molecular weight profile (kg/molec)
#        Mp   - planetary mass (Mearth)
#        Rp   - planetary radius (Rearth)
#        gp   - planetary surface gravity (m s**-2; optional; supersedes Mp if used)
#        ip   - if set, pressure index where assumed Rp, gp apply
#
# outputs:
#
#         z   - altitude profile (m)
#      grav   - gravity profile (m/s/s)
#
def hydrostat(Nlev,p,t,m,Mp,Rp,gp,ip=-1):

  # earth radius (m)
  Re   = 6.378e6

  # surface gravity (m/s/s)
  if (gp == -1):
    grav0 = 9.798*Mp/Rp**2
  else:
    grav0 = gp

  # universal gas constant
  kB = 1.38064852e-23 # m**2 kg s**-2 K**-1

  # altitude and gravity grids
  z    = np.zeros(Nlev)
  grav = np.zeros(Nlev)

  # mean layer mean molecular weight
  mm = 0.5*(m[1:] + m[:-1])

  # case where Rp, gp apply at bottom of pressure profile
  if (ip == -1):

    # surface values
    z[Nlev-1]    = 0.
    grav[Nlev-1] = grav0

    # iterate upward from surface
    for i in range(1,Nlev):
      k    = Nlev-i-1
      a    = kB/grav0/mm[k]
      fac  = a*((t[k+1]-(t[k]-t[k+1])/(p[k]-p[k+1])*p[k+1])*np.log(p[k+1]/p[k])-(t[k]-t[k+1]))
      fac  = (Rp*Re)/(1+(z[k+1]/Rp/Re)) - fac
      z[k] = (Rp*Re/fac - 1)*Rp*Re
      grav[k] = grav0*(Rp*Re)**2/(Rp*Re + z[k])**2

  # case where Rp, gp apply somewhere else in profile
  else:

    # values at ip
    z[ip]    = 0.
    grav[ip] = grav0

    # iterate upward from ip
    for i in range(1,ip+1):
      k    = ip-i
      a    = kB/grav0/mm[k]
      fac  = a*(t[k+1]-(t[k]-t[k+1])/(p[k]-p[k+1]))*np.log(p[k+1]/p[k])
      fac  = fac - a*((t[k]-t[k+1])/(p[k]-p[k+1])*(p[k]-p[k+1]))
      fac  = (Rp*Re)/(1+(z[k+1]/Rp/Re)) - fac
      z[k] = (Rp*Re/fac - 1)*Rp*Re
      grav[k] = grav0*(Rp*Re)**2/(Rp*Re + z[k])**2

    # iterate downward from ip
    for i in range(ip+1,Nlev):
      k    = i
      a    = kB/grav0/mm[k-1]
      fac  = a*(t[k]-(t[k-1]-t[k])/(p[k-1]-p[k]))*np.log(p[k]/p[k-1])
      fac  = fac - a*((t[k-1]-t[k])/(p[k-1]-p[k])*(p[k-1]-p[k]))
      fac  = (Rp*Re)/(1-(z[k-1]/Rp/Re)) - fac
      z[k] = -(Rp*Re/fac - 1)*Rp*Re
      grav[k] = grav0*(Rp*Re)**2/(Rp*Re + z[k])**2

  return z,grav
#
#
# mean molecular weight calculation
#
# inputs:
#
#        f    - gas mixing ratios
#       fb    - background gas volume mixing ratio
#       mb    - background gas mean molar weight (g/mole)
#     mmw0    - gas molar weights (see set_gas_info)
#      mmr    - if true mixing ratios are mass mixing ratios; volume mixing ratios otherwise
#
# outputs:
#
#        m    - mean molecular weight (kg/molecule)
#
def mmw(f,fb,mb,mmw0,mmr):

  # constants
  Na    = 6.0221408e23   # avogradro's number

  # mean molar weight if mixing ratios are mass versus volume
  if mmr:
    de    = np.sum(f/mmw0[:,np.newaxis],axis=0) + fb/mb
    id    = np.copy(de)
    id[:] = 1
    m     = np.divide(id,de)/1.e3/Na
  else:
    id    = np.copy(fb)
    id[:] = 1
    m     = (np.sum(f*mmw0[:,np.newaxis],axis=0) + fb*mb)/1.e3/Na

  return m
#
#
# refractivity at STP
#
#   inputs:
#
#     f       -  gas mixing ratios
#   nu0       -  species refractivity at STP
#  mmw0       -  species mean molar weight (grams per mole)
#   mmr       -  indicates if mixing ratios are mass vs. volume
#     m       -  atmospheric mean molecular weight (kg per molec)
#
#   outputs:
#
#     nu0     -  refractivity at STP
#
def refractivity(f,nu0,mmw0,mmr,m):

  # constants
  Na    = 6.0221408e23   # avogradro's number

  # volume mixing ratio-weighted refractivity
  if mmr:
    fm  = np.divide(np.mean(f,axis=1),mmw0)*np.mean(m*Na*1.e3)
  else:
    fm  = np.mean(f,axis=1)
  nu  = np.sum(fm*nu0)

  return nu
#
#
# set vertical pressure grid
#
# inputs:
#
#       Nlev  - number of vertical levels
#       pmax  - top-of-model pressure
#       pmax  - bottom-of-model pressure
#        cld  - boolean to indicate if clouds are requested
#         pt  - cloud top pressure
#      tauc0  - cloud optical depth at user-provided wavelength
#        src  - model type (e.g., thermal, transit, diffuse reflectance)
#
# outputs:
#
#          p  - pressure grid
#
# notes:
#        all input pressures must have same units
#
def set_press_grid(Nlev,pmin,pmax):

  # simple logarithmic gridpoint spacing
  p = np.logspace(np.log10(pmin),np.log10(pmax),Nlev)

  return p
#
#
# function for converting rfast mmr quantities to vmr
#
# inputs:
#
#       mmw0  - gas molar weights, from set_gas_info
#  species_r  - radiatively active species names
#          m  - level-dependent atmospheric molecular weight (kg/molec)
#         mb  - background gas molar weight
#         f0  - user-input gas mmrs
#          f  - gas mmrs [Ngas x Nlev]
#         fb  - background gas mmr
#
# outputs:
#
#       f,fb,f0 converted from mmr to vmr
#
def mmr2vmr(mmw0,gasid,species_r,m,mb,f0,fb,f):
  Na = 6.0221408e23   # avogradro's number
  f  = np.multiply(np.outer(1/mmw0,m*1e3*Na),f)
  fb = m*1e3*Na/mb*fb
  i = 0
  for id in species_r:
    ig    = gasid.index(id.lower())
    f0[i] = m[-1]*1e3*Na/mmw0[ig]*f0[i]
    i     = i + 1
  ret = f,fb,f0
  return ret
#
#
# function for converting rfast vmr quantities to mmr
#
# inputs:
#
#       mmw0  - gas molar weights, from set_gas_info
#  species_r  - radiatively active species names
#          m  - level-dependent atmospheric molecular weight (kg/molec)
#         mb  - background gas molar weight
#         f0  - user-input gas mmrs
#          f  - gas mmrs [Ngas x Nlev]
#         fb  - background gas mmr
#
# outputs:
#
#       f,fb,f0 converted from vmr to mmr
#
def vmr2mmr(mmw0,gasid,species_r,m,mb,f0,fb,f):
  Na = 6.0221408e23   # avogradro's number
  f  = np.multiply(np.outer(mmw0,1/(m*1e3*Na)),f)
  fb = mb/(m*1e3*Na)*fb
  i = 0
  for id in species_r:
    ig    = gasid.index(id.lower())
    f0[i] = mmw0[ig]/(m[-1]*1e3*Na)*f0[i]
    i     = i + 1
  ret = f,fb,f0
  return ret