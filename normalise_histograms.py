import uproot
from uproot.writing.identify import to_TH1x, to_TAxis
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, RationalQuadratic, WhiteKernel
import argparse
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from colorama import Fore
from tqdm import tqdm

import warnings
from sklearn.exceptions import ConvergenceWarning

# Don't keep track of convergence or invalid warnings
warnings.filterwarnings("ignore",category=ConvergenceWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)

# A wrapper around uproot.writing.identify.to_TH1x
# note that to_TH1x expects first and last bin under/overflow
def construct_uproot_hist(data, bins=None, entries=None, sumw2s=None, name=None, title=None, axis_titles=None):

    uname="foo" if name is None else name
    uentries = 0 if entries is None else entries
    xcentres = 0.5 * (bins[1:] + bins[:-1])
    '''
    xcentres = np.concat([ [bins[0] - 0.5 * np.diff(bins)[0]],
                           xcentres,
                           [bins[-1] + 0.5 * np.diff(bins)[0]] ])
    '''
    
    return to_TH1x(
        fName    = uname,
        fTitle   = title,
        data     = np.concat([[0.0], data, [0.0]]),
        fEntries = uentries,
        fSumw2   = np.concat([[0.0], sumw2s, [0.0]]),
        fTsumw   = np.sum(data),
        fTsumw2  = np.sum(sumw2s),
        fXaxis   = to_TAxis(fName  = "xaxis",   fTitle = axis_titles[0],
                            fNbins = data.shape[0],
                            fXmin  = bins[0], fXmax  = bins[-1]),
        fYaxis   = to_TAxis(fName  = "yaxis",   fTitle = axis_titles[1],
                            fNbins = 1,
                            fXmin  = max(0.0, 0.95 * np.min(data)), fXmax = 1.05 * np.max(data)),
        fTsumwx  = np.sum(data * xcentres),
        fTsumwx2 = np.sum(data * xcentres ** 2)
    )

def normalise(bins, vals, errs):
    bc = 0.5 * (bins[1:] + bins[:-1])
    bw = np.diff(bins)[0]
    
    cgp = GPR(
        ConstantKernel(1.0) * RBF(length_scale=2.0*bw, length_scale_bounds=(bw, 5.0*bw)) + \
        WhiteKernel(noise_level=np.mean(errs**2),
                    noise_level_bounds=(0.1*np.min(errs**2), 5.0 * np.max(errs**2))),
        alpha=errs**2
    )
    cgp.fit(bc.reshape(-1, 1), vals)
    # Amd get new values that are normalised..
    normvals, normerrs = cgp.predict(bc.reshape(-1, 1), return_std=True)
    #_, cov = cgp.predict(bc.reshape(-1, 1), return_cov=True)
    return normvals, normerrs#, cov


def main(args):
    flavours = ["numu", "numubar", "nue", "nuebar"]
    parents = ["pion", "kaon", "kzero", "muon"]
    secondaries = ["pimu", "pinomu", "kaon", "kzero", "nucleon"]
    sec_titles  = ["pion->...->muon", "pion->...->(not-muon)",
                   "kzero->...", "kaon->...", "nucleon->..."]

    with uproot.open(Path(args.input).resolve()) as fuin, \
         uproot.recreate(Path(args.output).resolve()) as fuout:

        zpos = sorted(set([f.split('/')[0].split(';')[0] for f in fuin.keys()]))
        ebins = deepcopy(fuin[zpos[0]]["numu"]["Flux"].axis('x').edges())

        area = fuin[zpos[0]]["hArea"].axis('x').centers()[0]
        x, y = fuin[zpos[0]]["x"].axis('x').centers()[0], fuin[zpos[0]]["y"].axis('x').centers()[0]
        nPOT = fuin[zpos[0]]["hPOT"].axis('x').centers()[0]

        for z in tqdm(zpos, desc="Looping over z...",
                      bar_format=Fore.MAGENTA + "{l_bar}{bar}{r_bar}" + Fore.RESET):
            
            zdir = fuout.mkdir(z)
            zdir["hPOT"]  = np.histogram([nPOT], bins=1)
            zdir["hArea"] = np.histogram([area], bins=1)
            zdir["x"]     = np.histogram([x],    bins=1)
            zdir["y"]     = np.histogram([y],    bins=1)
            
            for flav in flavours:
                dflav = zdir.mkdir(flav)
                vals    = fuin[z][flav]["Flux"].values()
                errs    = fuin[z][flav]["Flux"].errors()
                entries = fuin[z][flav]["Flux"].member("fEntries")

                # Find the KDAR peak and yeet it out for training
                if(flav == "numu"):
                    kpidx = 4 # [200, 250] contains 236...
                    kdar_peak, err_kdar_peak = vals[kpidx], errs[kpidx]
                    vals[kpidx] = 0.5 * (vals[kpidx-1] + vals[kpidx+1])
                    errs[kpidx] = 0.5 * (errs[kpidx-1] + errs[kpidx+1])
                    
                    res_kdar_peak = kdar_peak - vals[kpidx]
                    kdar_ratio = res_kdar_peak / kdar_peak
                    res_err_kdar_peak = err_kdar_peak * (1 - kdar_ratio)
                elif(flav == "nue"): # Bins at [100, 200) containing vicinity of 3-body KDAR peak
                    kpi1, kpi2 = 2, 3
                    kpk1, kpk2, ekpk1, ekpk2 = vals[kpi1], vals[kpi2], errs[kpi1], errs[kpi2]
                    vals[kpi1] = 0.75 * vals[kpi1-1] + 0.25 * vals[kpi2+1]
                    vals[kpi2] = 0.25 * vals[kpi1-1] + 0.75 * vals[kpi2+1]
                    # idk what to do about the errors so underreport these
                    errs[kpi1] = 0.75 * errs[kpi1-1] + 0.25 * errs[kpi2+1]
                    errs[kpi2] = 0.25 * errs[kpi1-1] + 0.75 * errs[kpi2+1]

                    res1, res2 = kpk1 - vals[kpi1], kpk2 - vals[kpi2]
                    krat1, krat2 = res1/kpk1, res2/kpk2

                norm_vals, norm_errs = normalise(ebins, vals, errs)
                if(flav == "numu"):
                    norm_vals[kpidx] += res_kdar_peak
                    norm_errs[kpidx] *= err_kdar_peak * (1 / (1-kdar_ratio))
                elif(flav == "nue"):
                    norm_vals[kpi1] += res1
                    norm_vals[kpi2] += res2
                #norm_vals, norm_errs, cov = normalise(ebins, vals, errs)
                #corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
                dflav["Flux"] = construct_uproot_hist(
                    norm_vals,
                    bins = ebins,
                    sumw2s  = norm_errs**2,
                    entries = entries,
                    name = "Flux", title = f"{flav} (total, RBF-normalised)",
                    axis_titles = ("Energy (GeV)", f"Flux ({flav} / 50MeV)")
                )
                #dflav["Covariance"] = (cov, ebins, ebins)
                #dflav["Correlation"] = (corr, ebins, ebins)


if __name__ == "__main__":
    print(Fore.YELLOW + "Hello world!" + Fore.RESET)
    
    parser = argparse.ArgumentParser(description='''Run a GPR on each histogram in an input file and save the output, along with the covariances.''')
    parser.add_argument('-i', '--input', type=str, required=True, help="Input file.")
    parser.add_argument('-o', '--output', type=str, default='normalised.root', help="Output file.")
    args = parser.parse_args()

    main(args)

    print(Fore.YELLOW + "Goodbye world!" + Fore.RESET)
