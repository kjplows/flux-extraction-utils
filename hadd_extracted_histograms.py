import uproot
from uproot.writing.identify import to_TH1x, to_TAxis
import numpy as np
import argparse
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from colorama import Fore
from tqdm import tqdm

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
        
def main(args):
    files = [Path(f).resolve() for f in args.input]
        
    
    flavours = ["numu", "numubar", "nue", "nuebar"]
    parents = ["pion", "kaon", "kzero", "muon"]
    secondaries = ["pimu", "pinomu", "kaon", "kzero", "nucleon"]
    sec_titles  = ["pion->...->muon", "pion->...->(not-muon)",
                   "kzero->...", "kaon->...", "nucleon->..."]

    with uproot.open(files[0]) as fuin:
        zpos = sorted(set([f.split('/')[0].split(';')[0] for f in fuin.keys()]))
        reference = np.empty_like(fuin[zpos[0]]["numu"]["Flux"].values()) # to get the shape
        ebins = deepcopy(fuin[zpos[0]]["numu"]["Flux"].axis('x').edges())

        area = fuin[zpos[0]]["hArea"].axis('x').centers()[0]
        x, y = fuin[zpos[0]]["x"].axis('x').centers()[0], fuin[zpos[0]]["y"].axis('x').centers()[0]

        print(Fore.GREEN + f"Adding {len(files)} histograms at x = {x} cm and y = {y} cm of area = {area} cm2" + Fore.RESET)


    # Write an output file
    with uproot.recreate(args.output) as fuout:
        for z in tqdm(zpos, desc="Looping over z...",
                      bar_format=Fore.MAGENTA + "{l_bar}{bar}{r_bar}" + Fore.RESET):
            nPOT = 0.0
            
            hists = {}
            entries = defaultdict(int)
            for flav in flavours:
                hists[f"{flav}"] = np.zeros_like(reference)
                hists[f"{flav} sw2"] = np.zeros_like(reference)
                entries[f"{flav}"] = 0
                for par in parents:
                    hists[f"{flav}-parent-{par}"] = np.zeros_like(reference)
                    hists[f"{flav}-parent-{par} sw2"] = np.zeros_like(reference)
                    entries[f"{flav}-parent-{par}"] = 0
                for sec in secondaries:
                    hists[f"{flav}-secondary-{sec}"] = np.zeros_like(reference)
                    hists[f"{flav}-secondary-{sec} sw2"] = np.zeros_like(reference)
                    entries[f"{flav}-secondary-{sec}"] = 0

            # Now we'll start iterating over all the files
            for f in files:
                with uproot.open(f) as fuin:
                    nPOT += fuin[z]["hPOT"].axis('x').centers()[0]
                    if(z == zpos[0]):
                        print(Fore.GREEN + f'[x = {x} cm and y = {y} cm] ' + Fore.CYAN + f'Now at {nPOT:.2e} POT...' + Fore.RESET)

                    for flav in flavours:
                        hists[f"{flav}"] = np.add(hists[f"{flav}"], fuin[z][flav]["Flux"].values())
                        hists[f"{flav} sw2"] = np.add(hists[f"{flav} sw2"],
                                                      fuin[z][flav]["Flux"].variances())
                        entries[f"{flav}"] += fuin[z][flav]["Flux"].member("fEntries")
                        for par in parents:
                            hists[f"{flav}-parent-{par}"] = np.add(hists[f"{flav}-parent-{par}"], fuin[z][flav]["Flux by parent"][f"{flav} from {par}"].values())
                            hists[f"{flav}-parent-{par} sw2"] = np.add(hists[f"{flav}-parent-{par} sw2"],
                                                                       fuin[z][flav]["Flux by parent"][f"{flav} from {par}"].variances())
                            entries[f"{flav}-parent-{par}"] += fuin[z][flav]["Flux by parent"][f"{flav} from {par}"].member("fEntries")
                        for sec in secondaries:
                            hists[f"{flav}-secondary-{sec}"] = np.add(hists[f"{flav}-secondary-{sec}"], fuin[z][flav]["Flux by secondary"][f"{flav} from {sec}"].values())
                            hists[f"{flav}-secondary-{sec} sw2"] = np.add(hists[f"{flav}-secondary-{sec} sw2"],
                                                                          fuin[z][flav]["Flux by secondary"][f"{flav} from {sec}"].variances())
                            entries[f"{flav}-secondary-{sec}"] += fuin[z][flav]["Flux by secondary"][f"{flav} from {sec}"].member("fEntries")
                        
            zdir = fuout.mkdir(z)
            zdir["hPOT"]  = np.histogram([nPOT], bins=1)
            zdir["hArea"] = np.histogram([area], bins=1)
            zdir["x"]     = np.histogram([x],    bins=1)
            zdir["y"]     = np.histogram([y],    bins=1)
            
            for flav in flavours:
                dflav = zdir.mkdir(flav)
                pdflav = dflav.mkdir("Flux by parent")
                sdflav = dflav.mkdir("Flux by secondary")
                
                # Cumulative fluxes.
                dflav["Flux"] = construct_uproot_hist(
                    hists[f"{flav}"],
                    bins = ebins,
                    sumw2s  = hists[f"{flav} sw2"],
                    entries = entries[f"{flav}"],
                    name = "Flux", title = f"{flav} (total)",
                    axis_titles = ("Energy (GeV)", f"Flux ({flav} / 50MeV)")
                )
                
                # Flux by parent.
                for par in parents:
                    pdflav[f"{flav} from {par}"] = construct_uproot_hist(
                        hists[f"{flav}-parent-{par}"],
                        bins = ebins,
                        sumw2s  = hists[f"{flav}-parent-{par} sw2"],
                        entries = entries[f"{flav}-parent-{par}"],
                        name = f"{flav} from {par}", title = f"...->{par}->{flav}",
                        axis_titles = ("Energy (GeV)", f"Flux ({flav} / 50MeV)")
                    )
                    
                # Flux by secondary.
                for sec, psec in zip( secondaries, sec_titles ):
                    sdflav[f"{flav} from {sec}"] = construct_uproot_hist(
                        hists[f"{flav}-secondary-{sec}"],
                        bins = ebins,
                        sumw2s  = hists[f"{flav}-secondary-{sec} sw2"],
                        entries = entries[f"{flav}-secondary-{sec}"],
                        name = f"{flav} from {sec}", title = f"primary->{psec}->{flav}",
                        axis_titles = ("Energy (GeV)", f"Flux ({flav} / 50MeV)")
                    )

if __name__ == "__main__":
    print(Fore.YELLOW + "Hello world!" + Fore.RESET)
    
    parser = argparse.ArgumentParser(description='''Hadd flux files extracted on the same voxel but run over different input files.''')
    parser.add_argument('-i', '--input', nargs='+', type=str, required=True, help="Input files.")
    parser.add_argument('-o', '--output', type=str, default='hadded.root', help="Output file.")
    args = parser.parse_args()

    main(args)

    print(Fore.YELLOW + "Goodbye world!" + Fore.RESET)
