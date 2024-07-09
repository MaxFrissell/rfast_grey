# rfast_grey

Modified version of the [rfast](https://github.com/hablabx/rfast) radiative transfer code. Currently reports retrieved albedo instead of flux ratios.

To run rfast retrievals, you need to download the hires_opacities folder [here](drive.google.com/drive/folders/1FzznH6nwhBrCZ99O5PwsEJ7JylJ3W9hc?usp=sharing)
and move the folder into the rfast_grey directory

Current features include:
- Sample of spectra from 1,000 mcmc steps (spectra_gen_pp.py)
- Constant error at all wavelengths of continuum value/SNR
- Only reading cloud spectra once instead of every mcmc step
