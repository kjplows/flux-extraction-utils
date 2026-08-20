import uproot
from uproot.writing.identify import to_TH1x, to_TAxis
import h5py
import numpy as np
import argparse
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from colorama import Fore
from tqdm import tqdm
from scipy.linalg import solve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from itertools import product

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

flavours = ["numu", "numubar", "nue", "nuebar"]
parents = ["pion", "kaon", "kzero", "muon"]
secondaries = ["pimu", "pinomu", "kaon", "kzero", "nucleon"]
sec_titles  = ["pion->...->muon", "pion->...->(not-muon)",
               "kzero->...", "kaon->...", "nucleon->..."]

def main(args):
    files = sorted([Path(f).resolve() for f in Path(args.input).glob("*.root")])

    print(Fore.GREEN + "Interpolating from " + str(len(files)) + " files..." + Fore.RESET)

    # For now, let's pick the 800 MeV bin == value 16
    # Run the interpolant over all three directions simultaneously]
    with uproot.open(files[0]) as fuin:
        znames = sorted(set([f.split('/')[0].split(';')[0] for f in fuin.keys()]))
        ebins = fuin[znames[0]]["numu"]["Flux"].axis('x').edges()
        bc = 0.5 * (ebins[1:] + ebins[:-1])
        bw = np.diff(ebins)[0]
    xlist, ylist = [], []
    for f in files:
        with uproot.open(f) as fuin:
            xlist.append(fuin[znames[0]]['x'].axis('x').centers()[0] - 73.78) # correct for beam->det
            ylist.append(fuin[znames[0]]['y'].axis('x').centers()[0])
    xpos = sorted(set(xlist))
    ypos = sorted(set(ylist))
    zpos = sorted(set([float(z[1:]) for z in znames]))

    # We'll need to make combinations of these parameters and map to the files
    print(Fore.MAGENTA + "Mapping files..." + Fore.RESET)
    coordinates = np.empty( (len(xpos), len(ypos), len(zpos), 3), dtype=np.float64 )
    indices     = np.empty( (len(xpos), len(ypos)), dtype=int )
    # coordinates shape: (NX, NY, NZ, 3) - indices shape: (NX, NY)

    for (x, y) in product(xpos, ypos):
        ix, iy = xpos.index(x), ypos.index(y)
        for i, f in enumerate(files):
            ftok = f.name.split('_')
            if( int(x+73.78) == int(ftok[1][1:]) and
                int(y) == int(ftok[2][1:-5])):
            #if f"x_{int(x+73.78)}_y_{int(y)}" in f.name:
                indices[ix, iy] = i
        for iz, z in enumerate(zpos):
            coordinates[ix, iy, iz] = np.array([x, y, z])

    poly = PolynomialFeatures(degree=4)
    design_matrix = poly.fit_transform(coordinates.reshape(len(xpos)*len(ypos)*len(zpos), 3))
    # shape: ( NX*NY*NZ, comb(3+deg, deg) ) = (12584, 35) for 3 parameters, deg = 4,
    # NX = NY = 22, NZ = 26

    # Load in data: go from structured array to flat one
    print(Fore.MAGENTA + "Filling data..." + Fore.RESET)
    data = np.empty( (len(xpos), len(ypos), len(zpos)), dtype=np.float64 )
    for ix in range(len(xpos)):
        for iy in range(len(ypos)):
            with uproot.open(files[indices[ix, iy]]) as fuin:
                for iz, z in enumerate(znames):
                    area  = fuin[z]["hArea"].axis('x').centers()[0]
                    POT   = fuin[z]["hPOT"].axis('x').centers()[0]
                    scval = fuin[z]["numu"]["Flux"].values()[16] / (bw*area*POT)
                    data[ix, iy, iz] = scval

    # Flatten.
    data = data.reshape( len(xpos)*len(ypos)*len(zpos) )

    # Fit the data.
    print(Fore.MAGENTA + "Fitting model to data..." + Fore.RESET)
    model = LinearRegression()
    model.fit(design_matrix, data)

    # Get the prediction
    print(Fore.MAGENTA + "Predicting..." + Fore.RESET)
    ypred = model.predict(design_matrix) # shape: (NX*NY*NZ), might need the intercept added
    ypred = ypred.reshape( (len(xpos), len(ypos), len(zpos)) )
    
    mco = model.coef_ # shape: comb(3+deg, deg)
    icp = model.intercept_ # scalar

    data = data.reshape( (len(xpos), len(ypos), len(zpos)) )
    with h5py.File(Path(args.output).resolve(), 'w') as fhout:
        fhout.create_dataset("Coordinates-x", data=coordinates[...,0])
        fhout.create_dataset("Coordinates-y", data=coordinates[...,1])
        fhout.create_dataset("Coordinates-z", data=coordinates[...,2])
        fhout.create_dataset("Data", data=data)
        fhout.create_dataset("Prediction", data=ypred)
        fhout.create_dataset("Deviation", data=data-ypred)
        fhout.create_dataset("Fractional deviation", data=(data-ypred)/data)

    print(Fore.GREEN + "Saved file to " + str(Path(args.output).resolve()) + Fore.RESET)

if __name__ == "__main__":
    print(Fore.YELLOW + "Hello world!" + Fore.RESET)
    
    parser = argparse.ArgumentParser(description='''On every histogram, read in x and y, get the density out, and interpolate that.''')
    parser.add_argument('-i', '--input', type=str, required=True, help="Input directory. I will use all the files in this dir.")
    parser.add_argument('-o', '--output', type=str, default='interpolated.h5', help="Output HDF5 file.")
    args = parser.parse_args()

    main(args)

    print(Fore.YELLOW + "Goodbye world!" + Fore.RESET)
