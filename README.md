# rfast_grey

Modified version of the [rfast](https://github.com/hablabx/rfast) radiative transfer code. Currently reports retrieved albedo instead of flux ratios.

Current features include:
- Sample of spectra from 1,000 mcmc steps (spectra_gen_pp.py)
- Constant error at all wavelengths of continuum value/SNR
- Only reading cloud spectra once instead of every mcmc step
