import matplotlib.pyplot as plt
import uproot
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic, WhiteKernel

#fuin = uproot.open("1M_nominal.root")
fuin1 = uproot.open("extraction_script_yesPionMuonDAR.root")
fuin2 = uproot.open("extraction_script_noPionMuonDAR.root")
fkey = "detector"
#fuin = uproot.open("baz.root")
#fkey = "z000"

flav = "h502" # 501, 2, 3, 4 --> nue, ebar, mu, mubar

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
    "h501": 250,
    "h502": 250,
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

alt_colours = {
    "h501": ('xkcd:goldenrod', 'xkcd:pumpkin'),
    "h502": ('xkcd:irish green', 'xkcd:irish green'),
    "h503": ('xkcd:scarlet', 'xkcd:brick'),
    "h504": ('xkcd:cornflower', 'xkcd:petrol'),
}

bc1 = fuin1[fkey][flav].axis('x').centers()[:cutoff]
be1 = fuin1[fkey][flav].axis('x').edges()[:cutoff]
bw1 = np.diff(be1)[0]
area1 = fuin1[fkey]["hArea"].axis('x').centers()[0]
POT1 = fuin1[fkey]["hPOT"].axis('x').centers()[0]

bc2 = fuin2[fkey][flav].axis('x').centers()[:cutoff]
be2 = fuin2[fkey][flav].axis('x').edges()[:cutoff]
bw2 = np.diff(be2)[0]
area2 = fuin2[fkey]["hArea"].axis('x').centers()[0]
POT2 = fuin2[fkey]["hPOT"].axis('x').centers()[0]

# DO NOT bin-width normalise yet. The GPR wants unnormalised values.
# Errors do not scale straightforwardly...
# For Wednesday and Thursday, demonstrate what happens when you forget this!
vals1 = fuin1[fkey][flav].values()[:cutoff]
errs1 = fuin1[fkey][flav].errors()[:cutoff]

vals2 = fuin2[fkey][flav].values()[:cutoff]
errs2 = fuin2[fkey][flav].errors()[:cutoff]

#cpval = np.array([v for v in vals])
#cperr = np.array([e for e in errs])

# Remove two-body decay spikes.
if flav == "h503": #spikes at 2-body pion (29.7 MeV) and kaon (236 MeV) decays
    max11=np.argmin(1.01*np.max(vals1)-vals1)
    vals1[max11] = 0.5 * (vals1[max11-1] + vals1[max11+1])
    errs1[max11] = 0.5 * (errs1[max11-1] + errs1[max11+1])
    max21 = np.argmin(1.01*np.max(vals1[bc1<0.24])-vals1[bc1<0.24])
    vals1[max21] = 0.5 * (vals1[max21-1] + vals1[max21+1])
    errs1[max21] = 0.5 * (errs1[max21-1] + errs1[max21+1])

    max12=np.argmin(1.01*np.max(vals2)-vals2)
    vals2[max12] = 0.5 * (vals2[max12-1] + vals2[max12+1])
    errs2[max12] = 0.5 * (errs2[max12-1] + errs2[max12+1])
    max22 = np.argmin(1.01*np.max(vals2[bc2<0.24])-vals2[bc2<0.24])
    vals2[max22] = 0.5 * (vals2[max22-1] + vals2[max22+1])
    errs2[max22] = 0.5 * (errs2[max22-1] + errs2[max22+1])
elif flav == "h501": #once again, spike at 2-body decay, much smaller but visible at 69.5 MeV
    max11=np.argmin(1.01*np.max(vals1[bc1>0.06])-vals1[bc1>0.06]) + len(bc1[bc1<=0.06])
    vals1[max11] = 0.5 * (vals1[max11-1] + vals1[max11+1])
    errs1[max11] = 0.5 * (errs1[max11-1] + errs1[max11+1])

    max12=np.argmin(1.01*np.max(vals2[bc2>0.06])-vals2[bc2>0.06]) + len(bc2[bc2<=0.06])
    vals2[max12] = 0.5 * (vals2[max12-1] + vals2[max12+1])
    errs2[max12] = 0.5 * (errs2[max12-1] + errs2[max12+1])

factor = 1

vals1, sigs1 = rebin_hist(vals1, errs1**2, factor=factor)
errs1 = np.sqrt(sigs1)
len_new1 = len(be1) // factor
be1 = np.arange(0.0, be1[factor * len_new1 - 1] + 1.01 * factor * bw1, factor * bw1)
be1 = be1[:len(vals1)+1]
bw1 *= factor
bc1 = 0.5 * (be1[1:] + be1[:-1])
dense_bc1 = np.linspace(0.0, be1[-1], num=factor*len(bc1))

vals2, sigs2 = rebin_hist(vals2, errs2**2, factor=factor)
errs2 = np.sqrt(sigs2)
len_new2 = len(be2) // factor
be2 = np.arange(0.0, be2[factor * len_new2 - 1] + 1.01 * factor * bw2, factor * bw2)
be2 = be2[:len(vals2)+1]
bw2 *= factor
bc2 = 0.5 * (be2[1:] + be2[:-1])
dense_bc2 = np.linspace(0.0, be2[-1], num=factor*len(bc2))

cgp1 = GPR(
    ConstantKernel(1.0) * RBF(length_scale=2.0*bw1, length_scale_bounds=(bw1, 5.0*bw1)) + \
    WhiteKernel(noise_level=np.mean(errs1**2), noise_level_bounds=(0.1*np.min(errs1**2), 5.0 * np.max(errs1**2))),
    alpha=errs1**2
)
cgp1.fit(bc1.reshape(-1, 1), vals1)
pred1, epred1 = cgp1.predict(dense_bc1.reshape(-1, 1), return_std=True)

cgp2 = GPR(
    ConstantKernel(1.0) * RBF(length_scale=2.0*bw2, length_scale_bounds=(bw2, 5.0*bw2)) + \
    WhiteKernel(noise_level=np.mean(errs2**2), noise_level_bounds=(0.1*np.min(errs2**2), 5.0 * np.max(errs2**2))),
    alpha=errs1**2
)
cgp2.fit(bc2.reshape(-1, 1), vals2)
pred2, epred2 = cgp2.predict(dense_bc2.reshape(-1, 1), return_std=True)

# NOTA BENE! This is a DENSITY.
# For example, numu (see small_voxel_sum_noPionMuonDAT.root) at factor=1 gives you a peak of
# 1.6e-8 numu / cm2 / POT / bin.
#
# Multiplied back out with the 0.05 GeV bins (axis unit == GeV), that peak becomes 8e-10 which
# matches DocDB 37816 (== 8e+0 numu / m2 / 1e6POT / 0.05GeV).
plt.fill_between(dense_bc1, (pred1-epred1)/(bw1*area1*POT1), (pred1+epred1)/(bw1*area1*POT1), alpha=0.6, color=colours[flav][0], label="With DAR, GPR")
plt.errorbar(bc1, vals1/(bw1*area1*POT1), xerr=0.5*bw1, yerr=errs1/(bw1*area1*POT1), fmt='None', color=colours[flav][1], label="G4BNB with DAR")
plt.fill_between(dense_bc2, (pred2-epred2)/(bw2*area2*POT2), (pred2+epred2)/(bw2*area2*POT2), alpha=0.6, color=alt_colours[flav][0], label="Without DAR, GPR")
plt.errorbar(bc2, vals2/(bw2*area2*POT2), xerr=0.5*bw2, yerr=errs2/(bw2*area2*POT2), fmt='None', color=alt_colours[flav][1], label="G4BNB without DAR")
plt.xlabel(r"$E_{{\nu}}$ [GeV]")
plt.ylabel(r"Density $\Phi [\nu / \mathrm{{cm}}^{{2}} / \mathrm{{P.O.T}} / \mathrm{{bin}}]$")
plt.legend(loc='best')
plt.show()
