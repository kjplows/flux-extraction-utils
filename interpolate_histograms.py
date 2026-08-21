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
from math import comb

flavours = ["numu", "numubar", "nue", "nuebar"]
parents = ["pion", "kaon", "kzero", "muon"]
secondaries = ["pimu", "pinomu", "kaon", "kzero", "nucleon"]
sec_titles  = ["pion->...->muon", "pion->...->(not-muon)",
               "kzero->...", "kaon->...", "nucleon->..."]

def main(args):
    files = sorted([Path(f).resolve() for f in Path(args.input).glob("*.root")])

    print(Fore.GREEN + "Interpolating from " + str(len(files)) + " files..." + Fore.RESET)

    # Run the interpolant over all three directions simultaneously
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

    # For every bin in the flux, get the map of fractional deviations (relative to error)
    data   = np.zeros( (len(ebins[:5]), len(xpos), len(ypos), len(zpos)) )
    errors = np.zeros_like( data )
    ypred  = np.zeros_like( data )
    frac_devs  = np.zeros_like( data )
    frac_preds = np.zeros_like( data )
    coeffs = np.zeros( (data.shape[0], comb(3 + args.degree, args.degree)) ) # To save coefficients
    powers = None # To save representations of powers
    with tqdm(total = data.shape[0], desc="Fitting bins...",
              bar_format = Fore.RED + "{l_bar}{bar}{r_bar}" + Fore.RESET) as pbar:
        for ibin in range(data.shape[0]):
            poly = PolynomialFeatures(degree=args.degree)
            design_matrix = poly.fit_transform(coordinates.reshape(len(xpos)*len(ypos)*len(zpos), 3))
            # shape: ( NX*NY*NZ, comb(3+deg, deg) ) = (12584, 35) for 3 parameters, deg = 4,
            # NX = NY = 22, NZ = 26

            if powers is None:
                powers = np.empty( comb(3 + args.degree, args.degree), dtype=object )
                terms = poly.get_feature_names_out(['x', 'y', 'z'])
                for i in range(len(terms)):
                    powers[i] = terms[i]

            # Load in data: go from structured array to flat one
            
            for ix in range(len(xpos)):
                for iy in range(len(ypos)):
                    with uproot.open(files[indices[ix, iy]]) as fuin:
                        for iz, z in enumerate(znames):
                            area  = fuin[z]["hArea"].axis('x').centers()[0]
                            POT   = fuin[z]["hPOT"].axis('x').centers()[0]
                            scval = fuin[z][args.flavour]["Flux"].values()[ibin] / (bw*area*POT)
                            erval = fuin[z][args.flavour]["Flux"].errors()[ibin] / (bw*area*POT)
                            data[ibin, ix, iy, iz]   = scval
                            errors[ibin, ix, iy, iz] = erval

            # Flatten.
            nudata = data[ibin, ...].reshape( len(xpos)*len(ypos)*len(zpos) )
            #errors = errors.reshape( len(xpos)*len(ypos)*len(zpos) )
            errors[ibin, ...][errors[ibin, ...]==0.0] = 1.0 # No data, no deviation

            # Fit the data.
            model = LinearRegression()
            model.fit(design_matrix, nudata)

            # Get the prediction
            nuypred = model.predict(design_matrix) # shape: (NX*NY*NZ)

            # Reshape back to structured
            nudata  =  nudata.reshape( (len(xpos), len(ypos), len(zpos)) )
            nuypred = nuypred.reshape( (len(xpos), len(ypos), len(zpos)) )
            ypred[ibin, ...] = nuypred
    
            mco = model.coef_ # shape: comb(3+deg, deg)
            icp = model.intercept_ # scalar that needs to be added to mco[0]

            coeffs[ibin, 0] = mco[0] + icp # include the intercept!
            coeffs[ibin, 1:] = mco[1:]

            pbar.update(1)

    # Now, calculate the fractional deviations
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_devs = np.abs((data - ypred)/data)
        frac_devs = np.abs((data - ypred)/ypred)

    with h5py.File(Path(args.output).resolve(), 'w') as fhout:
        fhout.create_dataset("Energy bins", data=ebins)
        fhout.create_dataset("Term expansion", data=powers)
        
        gflav = fhout.require_group(args.flavour)
        gflav.create_dataset("Data", data=data)
        gflav.create_dataset("Prediction", data=ypred)
        gflav.create_dataset("Deviation", data=data-ypred)
        gflav.create_dataset("Fractional deviation (over obs)", data=frac_devs)
        gflav.create_dataset("Fractional deviation (over pred)", data=frac_preds)
        gflav.create_dataset("Model coefficients", data=coeffs)

    print(Fore.GREEN + "Saved file to " + str(Path(args.output).resolve()) + Fore.RESET)

if __name__ == "__main__":
    print(Fore.YELLOW + "Hello world!" + Fore.RESET)
    
    parser = argparse.ArgumentParser(description='''On every histogram, read in x and y, get the density out, and interpolate that.''')
    parser.add_argument('-i', '--input', type=str, required=True, help="Input directory. I will use all the files in this dir.")
    parser.add_argument('-o', '--output', type=str, default='interpolated.h5', help="Output HDF5 file.")
    parser.add_argument('-d', '--degree', type=int, default=4, help='Polynomial degree.')
    parser.add_argument('--flavour', type=str, default='numu',
                        choices=['numu', 'nue', 'numubar', 'nuebar'], help="Which flux to interpolate.")
    args = parser.parse_args()

    main(args)

    print(Fore.YELLOW + "Goodbye world!" + Fore.RESET)
