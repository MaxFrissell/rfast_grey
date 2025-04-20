import h5py
import numpy    as     np
from   scipy    import interpolate
from astropy.io import ascii
#
#
# routine to read-in cia data
#
#   inputs:
#
#   species   -  list of strings, choosing from 'co2','h2','n2','o2',
#              which will determine species included and ordering of species
#   lam       -  wavelength grid to place data on (um)
#    opdir    - directory where hi-res opacities are located (string)
#
#   outputs:
#
#      temp   -  temperature grid for cia data (K)
#       cia   -  cia coefficients [Ncases,Ntemp,Nwavelength] (m**-6 m**-1)
#
#   notes:
#
#   to add a new CIA instance, either increment the number of CIA 
#   cases (if gas already included) or add a new entry to ncia (if adding
#   a new gas). for the already-included gas scenario, go to the appropriate 
#   "if" check and repeat the logic for the new CIA file. for a wholly new 
#   species, and a new "elif" check and update the logic.
#
def cia_read(species,lam,datdir):

  # uniform temperature grid to interpolate onto
  temp = [50.,75.,100.,200.,300.,400.,500.,750.,1000.]
  temp = np.float_(temp)

  # number of cia cases associated with co2, h2, n2, o2
  ngas  = 4
  ncia0 = [1,2,1,3]
  gasn  = ['co2','h2','n2','o2']

  # gas and partner info (must agree with read-in order below)
  # first entry is the absorber and subsequent entries are partners
  ciaid0  = np.transpose(np.array([ ['     ']*(max(ncia0)+1) for i in range(ngas)]))
  ciaid0[0:ncia0[0]+1,0] = ['co2','co2']
  ciaid0[0:ncia0[1]+1,1] = ['h2','h2','he']
  ciaid0[0:ncia0[2]+1,2] = ['n2','n2']
  ciaid0[0:ncia0[3]+1,3] = ['o2','o2','n2','x']

  # determine number of cia cases
  icia    = 0
  ncia    = [0]*len(species)
  ciaid   = np.transpose(np.array([ ['     ']*(max(ncia0)+1) for i in range(len(species))]))
  for isp in range(len(species)):
    idg  = gasn.index(np.char.lower(species[isp]))
    ncia[isp] = ncia0[idg]
    ciaid[:,isp] = ciaid0[:,idg]
    icia = icia + ncia0[idg]

  # variable to store cia data
  kcia  = np.zeros([max(icia,1),len(temp),len(lam)])

  # loop over included species, interpolate onto uniform grid
  icia = 0
  for isp in range(len(species)):

    if (species[isp].lower() == 'co2'):

      # read data
      fn              = 'CO2-CO2_abs.cia'
      temp0,lam0,cia0 = read_cia(datdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)
      cia0[np.where(cia0<=0)] = 0. # interpolated values cannot be less than zero

      # interpolate onto lam
      cia_interp    = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia,:,:] = cia_interp(lam)
      kcia[np.where(kcia<=0)] = 0. # interpolated values cannot be less than zero

      # update cia counter
      icia = icia + ncia0[0]

    elif (species[isp].lower() == 'h2'):

      # read data
      fn              = 'H2-H2_abs.cia'
      temp0,lam0,cia0 = read_cia(datdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)
      cia0[np.where(cia0<=0)] = 0. # interpolated values cannot be less than zero

      # interpolate onto lam
      cia_interp     = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia,:,:] = cia_interp(lam)
      kcia[np.where(kcia<=0)] = 0. # interpolated values cannot be less than zero

      # read data
      fn              = 'H2-He_abs.cia'
      temp0,lam0,cia0 = read_cia(datdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)
      cia0[np.where(cia0<=0)] = 0. # interpolated values cannot be less than zero

      # interpolate onto lam
      cia_interp       = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia+1,:,:] = cia_interp(lam)
      kcia[np.where(kcia<=0)] = 0. # interpolated values cannot be less than zero

      # update cia counter
      icia = icia + ncia0[1]

    elif (species[isp].lower() == 'n2'):

      # read data
      fn              = 'N2-N2_abs.cia'
      temp0,lam0,cia0 = read_cia(datdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)
      cia0[np.where(cia0<=0)] = 0. # interpolated values cannot be less than zero

      # interpolate onto lam
      cia_interp     = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia,:,:] = cia_interp(lam)
      kcia[np.where(kcia<=0)] = 0. # interpolated values cannot be less than zero

      # update cia counter
      icia = icia + ncia0[2]

    elif (species[isp].lower() == 'o2'):

      # read data
      fn              = 'O2-O2_abs.cia'
      temp0,lam0,cia0 = read_cia(datdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)
      cia0[np.where(cia0<=0)] = 0. # interpolated values cannot be less than zero

      # interpolate onto lam
      cia_interp     = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia,:,:] = cia_interp(lam)
      kcia[np.where(kcia<=0)] = 0. # interpolated values cannot be less than zero

      # read data
      fn              = 'O2-O2_abs_Herzberg.cia'
      temp0,lam0,cia0 = read_cia(datdir+fn)

      # copy onto temp (only one temperature point)
      cia0 = np.squeeze(cia0)
      cia0 = np.repeat(cia0[np.newaxis,:], len(temp), axis=0)
      cia0[np.where(cia0<=0)] = 0. # interpolated values cannot be less than zero

      # interpolate onto lam
      cia_interp       = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia,:,:]   = kcia[icia,:,:] + cia_interp(lam)
      kcia[np.where(kcia<=0)] = 0. # interpolated values cannot be less than zero

      # read data
      fn              = 'O2-N2_abs.cia'
      temp0,lam0,cia0 = read_cia(datdir+fn)

      # copy onto temp (only one temperature point)
      cia0 = np.squeeze(cia0)
      cia0 = np.repeat(cia0[np.newaxis,:], len(temp), axis=0)
      cia0[np.where(cia0<=0)] = 0. # interpolated values cannot be less than zero

      # interpolate onto lam
      cia_interp       = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia+1,:,:] = cia_interp(lam)
      kcia[np.where(kcia<=0)] = 0. # interpolated values cannot be less than zero

      # read data
      fn              = 'O2-X_abs.cia'
      temp0,lam0,cia0 = read_cia(datdir+fn)

      # interpolate onto temp
      cia_interp = interpolate.interp1d(temp0,cia0,axis=0,assume_sorted=True,fill_value="extrapolate")
      cia0       = cia_interp(temp)
      cia0[np.where(cia0<=0)] = 0. # interpolated values cannot be less than zero

      # interpolate onto lam
      cia_interp       = interpolate.interp1d(lam0,cia0,axis=1,fill_value="extrapolate")
      kcia[icia+2,:,:] = cia_interp(lam)
      kcia[np.where(kcia<=0)] = 0. # interpolated values cannot be less than zero

      # update cia counter
      icia = icia + ncia0[3]

  # convert to m**-6 m**-1
  amg = 2.6867774e25
  kcia = kcia/(amg**2)*1.e2

  return temp,kcia,ncia,ciaid
#
#
# routine to read-in opacity database
#
#   inputs:
#
#   species   - list of strings, choosing from 'ch4','co2','h2o','o2','o3',
#                which will determine species included and ordering of species
#      lam    - desired output opacity wavelength grid (um)
#    opdir    - directory where hi-res opacities are located (string)
#
#   outputs:
#
#     press   -  opacity pressure grid (Pa)
#     sigma   -  opacities (m**2/molec) with size [Nspecies,Npressure,Nwavelength]
#
#   notes:
#
#   to add a new absorber, first ensure the relevant .abs file exists in the 
#   hires_opacities folder. add the species id/name to spc, indicate if a uv xsec
#   file exists in the hires folder and the lines to skip at the top of this file.
#
def opacities_read(species,lam,opdir):

  form  = '_hitran2020_10_100000cm-1_1cm-1.h5'
  forx  = 'xsec.dat'
  fn    = [opdir + s.lower() + form for s in species]
  fnx   = [opdir + s.lower() + forx for s in species]
  Nspec = len(species) # number of species included

  # species names and mean molar weight, if they have xsec file & lines to skip in header
  spc   = ['ch4','co2','h2o', 'o2', 'o3','n2o', 'co', 'h2', 'n2','so2','nh3','c2h2']
  lbl   = [  'y',  'y',  'y',  'y',  'y',  'y',  'y',  'y',  'y',  'y',  'y',   'y']
  xsc   = [  'y',  'y',  'y',  'y',  'y',  'y',  'y',  'n',  'n',  'y',  'y',   'n']
  lskp  = [    8,    8,    8,    8,    8,    8,    8,    0,    0,    8,    8,     0]

  # small cross section (cm**2/molec) to replace all zeroes
  smallsig = 1e-40

  # if there are >0 species included
  if (Nspec > 0):

    # query the h2o file to get number of p, T points
    fn0 = opdir + 'h2o' + form

    # open h5 file for reading
    hf = h5py.File(fn0,'r')

    # get pressure and temperature
    press = hf.get('p')[:]    # Pa
    temp  = hf.get('tatm')[:] # K
    Np    = len(press)
    Nt    = len(temp)

    # close h5 file
    hf.close()

    # initialize output opacities array
    sigma = np.zeros([Nspec,Np,Nt,len(lam)])

    # loop over other species and store opacities
    for j in range(0,Nspec):

      # species index
      isp = spc.index(species[j].lower())

      # if species has an lbl file
      if (lbl[isp] == 'y'):

        # open h5 file for reading
        hf = h5py.File(fn[j],'r')

        # get wavenumber, set wavelength
        wn0   = hf.get('wn')[:]   # cm**-1
        lam0  = 1.e4/wn0          # um

        # get opacities
        sig   = hf.get('abc')[:]  # cm**2 molec**-1

        # close h5 file
        hf.close()

        # remove <=0 values
        sig   = np.nan_to_num(sig)
        sig[np.where(sig<=smallsig)] = smallsig

        # interpolate onto output wavelength grid
        sig_interp      = interpolate.interp1d(wn0[:-2],np.log10(sig[:,:,:-2]),axis=2,assume_sorted=True,fill_value="extrapolate")
        sigma[j,:,:,:]  = np.power(10,sig_interp(1.e4/lam))

      # read xsec file, if applicable
      if (xsc[isp] == 'y'):
        data  = ascii.read(fnx[j],data_start=lskp[isp])
        lamx  = data['col1']
        xsecx = data['col2'] # cm**2/molecule
        xsecx[np.where(xsecx<=0)] = smallsig
        xsec_interp = interpolate.interp1d(lamx,np.log10(xsecx),assume_sorted=True,fill_value="extrapolate")
        xsec        = np.power(10,xsec_interp(lam))
        xsec[np.where(xsec<smallsig)] = smallsig
        xsec        = np.repeat(xsec[np.newaxis,:],   sigma.shape[1], axis=0)
        xsec        = np.repeat(xsec[:,np.newaxis,:], sigma.shape[2], axis=1)
        sigma[j,:,:,:] = sigma[j,:,:,:] + xsec

  else: # case with no species -> zero opacities across all press, temp
    press = np.float32([1.,1.e7])
    temp  = np.float32([50.,650.])
    Np    = len(press)
    Nt    = len(temp)
    sigma = np.zeros([1,Np,Nt,len(lam)])
    sigma[:,:,:,:] = smallsig

  # convert to m**2/molecule
  sigma = sigma*1.e-4

  return press,temp,sigma
#
#
# inform the user of key parameters for hi-res opacities
#
# inputs:
#
#    opdir    - high-res opacities directory
#
# outputs:
#
def opacities_info(opdir):

  fn    = 'info.txt'
  file  = open(opdir+fn,'r')
  lines = file.readlines()
  print("rfast note | vib/rot opacity " + lines[0], end = '')
  print("rfast note | vib/rot opacity " + lines[1], end = '')
  print("rfast note | vib/rot opacity " + lines[2])

  return
#
#
# rayleigh scattering cross section (m**2/molecule)
#
#   inputs:
#
#     lam     -  wavelength (um)
#     ray0    -  xsec (m**2/molec) at 0.4579 (see set_gas_info)
#     f       -  gas vmr profiles (ordered as set_gas_info)
#     fb      -  background gas vmr
#     rayb    -  background gas rayleigh cross section relative to Ar
#
#   outputs:
#
#     sigma   -  cross section (m**2/molecule) at each lam point
#
def rayleigh(lam,ray0,f,fb,rayb):

  # cross section at all wavelengths
  sigma = np.multiply.outer(np.sum(f*ray0[:,np.newaxis],axis=0),(0.4579/lam)**4.)
  sigma = sigma + np.multiply.outer(fb*rayb*ray0[0,np.newaxis],(0.4579/lam)**4.)

  return sigma
#
#
# read a *_abs.cia file
#
#   inputs:
#
#   fn        -  filename
#
#   outputs:
#
#        t    -  temperature grid for cia data (K)
#      cia    -  cia coefficients (cm**-1 amagat**-2)
#
def read_cia(fn):

  # open and read data file
  data  = ascii.read(fn,data_start=0,delimiter=' ')

  # number of temperature, wavelength points
  Ntemp = int(data['col1'][0])
  Nlam  = len(data['col1'][1:-1])

  # variables for storing temperature, cia
  temp  = np.zeros(Ntemp)
  lam   = np.zeros(Nlam)
  cia   = np.zeros([Ntemp,Nlam])

  # store wavelength points
  lam[:] = 1.e4/data['col1'][1:-1]

  # loop over temperature points and store data
  for i in range(Ntemp):
    temp[i]  = data['col'+str(i+2)][0]
    cia[i,:] = data['col'+str(i+2)][1:-1]

  return temp,lam,cia
