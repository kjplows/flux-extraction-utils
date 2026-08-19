import matplotlib.pyplot as plt
import uproot
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic, WhiteKernel

#fuin = uproot.open("1M_nominal.root")
fuin = uproot.open("extraction_script_noPionMuonDAR.root")
fkey = "detector"
#fuin = uproot.open("baz.root")
#fkey = "z000"

flav = "h503" # 501, 2, 3, 4 --> nue, ebar, mu, mubar

def rebin_hist(values, sumw2s, factor=10):

    if factor == 1.0:
        return values, sumw2s
    
    len_keep = factor * (len(values) // factor)
    len_new  = len(values) // factor
    values, sumw2s = values[:len_keep], sumw2s[:len_keep]
    new_values = values.reshape(len_new, factor).sum(axis=1)
    new_sumw2s = sumw2s.reshape(len_new, factor).sum(axis=1)

    return new_values, new_sumw2s

# Rebin to per 5 MeV, i.e. 10 times denser.
# Keep up to 3 GeV for numu
# Cutoff is in the original bin widths of 0.5 MeV
cutoffs = {
    "h501": 500,
    "h502": 500,
    "h503": 500,
    "h504": 500
}
#cutoff = 220 if flav == "h503" else 110 # bins of 0.05 GeV with m_mu = 105.66 --> cut at 110 MeV or 55 MeV
cutoff = cutoffs[flav]

colours = {
    "h501": ('xkcd:irish green', 'xkcd:irish green'),
    "h502": ('xkcd:goldenrod', 'xkcd:pumpkin'),
    "h503": ('xkcd:cornflower', 'xkcd:petrol'),
    "h504": ('xkcd:scarlet', 'xkcd:brick'),
}

bc = fuin[fkey][flav].axis('x').centers()[:cutoff]
be = fuin[fkey][flav].axis('x').edges()[:cutoff]
bw = np.diff(be)[0]
area = fuin[fkey]["hArea"].axis('x').centers()[0]
POT = fuin[fkey]["hPOT"].axis('x').centers()[0]
#area=1.0
#POT=1.0

# DO NOT bin-width normalise yet. The GPR wants unnormalised values.
# Errors do not scale straightforwardly...
# For Wednesday and Thursday, demonstrate what happens when you forget this!
vals = fuin[fkey][flav].values()[:cutoff]
errs = fuin[fkey][flav].errors()[:cutoff]

# Remove two-body decay spikes.
if flav == "h503": #spikes at 2-body pion (29.7 MeV) and kaon (236 MeV) decays
    max1=np.argmin(1.01*np.max(vals)-vals)
    vals[max1] = 0.5 * (vals[max1-1] + vals[max1+1])
    errs[max1] = 0.5 * (errs[max1-1] + errs[max1+1])
    max2 = np.argmin(1.01*np.max(vals[bc<0.24])-vals[bc<0.24])
    vals[max2] = 0.5 * (vals[max2-1] + vals[max2+1])
    errs[max2] = 0.5 * (errs[max2-1] + errs[max2+1])
elif flav == "h501": #once again, spike at 2-body decay, much smaller but visible at 69.5 MeV
    max1=np.argmin(1.01*np.max(vals[bc>0.06])-vals[bc>0.06]) + len(bc[bc<=0.06])
    vals[max1] = 0.5 * (vals[max1-1] + vals[max1+1])
    errs[max1] = 0.5 * (errs[max1-1] + errs[max1+1])

factor = 1
vals, sigs = rebin_hist(vals, errs**2, factor=factor)
errs = np.sqrt(sigs)
len_new = len(be) // factor
be = np.arange(0.0, be[factor * len_new - 1] + 1.01 * factor * bw, factor * bw)
be = be[:len(vals)+1]
bw *= factor
bc = 0.5 * (be[1:] + be[:-1])
dense_bc = np.linspace(0.0, be[-1], num=factor*len(bc))

# THIS GIVES YOU BIAS
bias = False
if bias:
    vals /= bw
    errs /= bw

cgp = GPR(
    ConstantKernel(1.0) * RBF(length_scale=2.0*bw, length_scale_bounds=(bw, 5.0*bw)) + \
    WhiteKernel(noise_level=np.mean(errs**2), noise_level_bounds=(0.1*np.min(errs**2), 5.0 * np.max(errs**2))),
    alpha=errs**2
)
cgp.fit(bc.reshape(-1, 1), vals)
pred, epred = cgp.predict(dense_bc.reshape(-1, 1), return_std=True)

# NOTA BENE! This is a DENSITY.
# For example, numu (see small_voxel_sum_noPionMuonDAT.root) at factor=1 gives you a peak of
# 1.6e-8 numu / cm2 / POT / bin.
#
# Multiplied back out with the 0.05 GeV bins (axis unit == GeV), that peak becomes 8e-10 which
# matches DocDB 37816 (== 8e+0 numu / m2 / 1e6POT / 0.05GeV).
if not bias:
    plt.fill_between(dense_bc, (pred-epred)/(bw*area*POT), (pred+epred)/(bw*area*POT), alpha=0.6, color=colours[flav][0], label="Gaussian process estimate")
    plt.errorbar(bc, vals/(bw*area*POT), xerr=0.5*bw, yerr=errs/(bw*area*POT), fmt='None', color=colours[flav][1], label="G4BNB")
else:
    plt.fill_between(dense_bc, (pred-epred)/(area*POT), (pred+epred)/(area*POT), alpha=0.6, color=colours[flav][0], label="Gaussian process estimate")
    plt.errorbar(bc, vals/(area*POT), xerr=0.5*bw, yerr=errs/(area*POT), fmt='None', color=colours[flav][1], label="G4BNB")
plt.xlabel(r"$E_{{\nu}}$ [GeV]")
plt.ylabel(r"Density $\Phi [\nu / \mathrm{{cm}}^{{2}} / \mathrm{{P.O.T}} / \mathrm{{bin}}]$")
plt.legend(loc='best')
plt.show()
