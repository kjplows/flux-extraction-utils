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
from itertools import product
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

flavours = ["numu", "numubar", "nue", "nuebar"]
parents = ["pion", "kaon", "kzero", "muon"]
secondaries = ["pimu", "pinomu", "kaon", "kzero", "nucleon"]
sec_titles  = ["pion->...->muon", "pion->...->(not-muon)",
               "kaon->...", "kzero->...", "nucleon->..."]

def main(args):
    files = sorted([Path(f).resolve() for f in Path(args.input).glob("*.root")])
    val_files = sorted([Path(f).resolve() for f in Path(args.validation).glob("*.root")]) \
        if args.validation is not None else None

    print(Fore.GREEN + "Interpolating from " + str(len(files)) + " files..." + Fore.RESET)
    if args.validation is not None:
        print(Fore.GREEN + "Validating from " + str(len(val_files)) + " files..." + Fore.RESET)

    # Get binning and z information
    with uproot.open(files[0]) as fuin:
        znames = sorted(set([f.split('/')[0].split(';')[0] for f in fuin.keys()]))
        ebins = fuin[znames[0]]["numu"]["Flux"].axis('x').edges()
        bc = 0.5 * (ebins[1:] + ebins[:-1])
        bw = np.diff(ebins)[0]

    # Run the interpolant over just two directions simultaneously, leaving out z
    
    xlist, ylist = [], []
    val_xlist, val_ylist = [], []
    
    for f in files:
        with uproot.open(f) as fuin:
            xlist.append(fuin[znames[0]]['x'].axis('x').centers()[0] - 73.78) # correct for beam->det
            ylist.append(fuin[znames[0]]['y'].axis('x').centers()[0])
    xpos, val_xpos = sorted(set(xlist)), None
    ypos, val_ypos = sorted(set(ylist)), None
    xdense = np.linspace(xpos[0], xpos[-1], num=100*len(xpos))
    ydense = np.linspace(ypos[0], ypos[-1], num=100*len(ypos))
    xw, yw = 0.5 * np.diff(xpos)[0], 0.5 * np.diff(ypos)[0]

    if args.validation is not None:
        for f in val_files:
            with uproot.open(f) as fuin:
                val_xlist.append(
                    fuin[znames[0]]['x'].axis('x').centers()[0] - 73.78) # correct for beam->det
                val_ylist.append(fuin[znames[0]]['y'].axis('x').centers()[0])

        val_xpos, val_ypos = sorted(set(val_xlist)), sorted(set(val_ylist))

    # We'll need to make combinations of these parameters and map to the files
    print(Fore.MAGENTA + "Mapping files..." + Fore.RESET)
    coordinates, val_coordinates = np.empty( (len(xpos), len(ypos), 2), dtype=np.float64 ), None
    indices,     val_indices     = np.empty( (len(xpos), len(ypos)), dtype=int ), None
    # coordinates shape: (NX, NY, 2) - indices shape: (NX, NY)

    for (x, y) in product(xpos, ypos):
        ix, iy = xpos.index(x), ypos.index(y)
        for i, f in enumerate(files):
            ftok = f.name.split('_')
            if( int(x+73.78) == int(ftok[1][1:]) and
                int(y) == int(ftok[2][1:-5]) ):
            #if f"x_{int(x+73.78)}_y_{int(y)}" in f.name:
                indices[ix, iy] = i
        coordinates[ix, iy] = np.array([x, y])

    if args.validation is not None:
        val_coordinates = np.empty( (len(val_xpos), len(val_ypos), 2), dtype=np.float64 )
        val_indices     = np.empty( (len(val_xpos), len(val_ypos)), dtype=int )

        for (x, y) in product(val_xpos, val_ypos):
            ix, iy = val_xpos.index(x), val_ypos.index(y)
            for i, f in enumerate(val_files):
                ftok = f.name.split('_')
                if( int(x+73.78) == int(ftok[2][1:]) and
                    int(y) == int(ftok[3][1:-5]) ):
                    val_indices[ix, iy] = i
            val_coordinates[ix, iy] = np.array([x, y])

    # For every bin in the flux, get the map of fractional deviations (relative to error)
    # Cutoff at 7 GeV --> bin 140
    cutoff = int(5.0 / bw)
    data   = np.zeros( (len(ebins[:cutoff]), len(xpos), len(ypos)) )
    errors = np.zeros_like( data )

    # How do we estimate systematic uncertainty?
    # Compare spline to simulated validation data.
    # Do the same for a bilinear interpolation and keep the spline <-> bilinear agreement as a diagnostic
    ypred  = np.zeros_like( data ) # to hold spline predictions at the voxel points
    val_data = np.zeros( (len(ebins[:cutoff]), len(val_xpos), len(val_ypos)) ) \
        if args.validation is not None else None
    val_pred = np.zeros_like(val_data) if args.validation is not None else None
    val_bil_pred = np.zeros_like(val_data) if args.validation is not None else None
    frac_wig = np.zeros( (len(ebins[:cutoff]), len(xpos)-1, len(ypos)-1) )
    
    with tqdm(total = data.shape[0], desc="Fitting bins...",
              bar_format = Fore.RED + "{l_bar}{bar}{r_bar}" + Fore.RESET) as pbar:
        for ibin in range(data.shape[0]): 
            
            # Load in data
            
            for ix in range(len(xpos)):
                for iy in range(len(ypos)):
                    with uproot.open(files[indices[ix, iy]]) as fuin:
                        area  = fuin[znames[0]]["hArea"].axis('x').centers()[0]
                        POT   = fuin[znames[0]]["hPOT"].axis('x').centers()[0]
                        scval = fuin[znames[0]][args.flavour]["Flux"].values()[ibin] / (bw*area*POT)
                        erval = fuin[znames[0]][args.flavour]["Flux"].errors()[ibin] / (bw*area*POT)
                        data[ibin, ix, iy]   = scval
                        errors[ibin, ix, iy] = erval

            spline = RectBivariateSpline( xpos, ypos, data[ibin, ...], kx=3, ky=3, s=0 )
            ypred[ibin] = spline(xpos, ypos)

            bilinear = RegularGridInterpolator( (xpos, ypos), data[ibin, ...], method="linear" )

            # Evaluate the agreement behind bilinear interpolation (a "mesh") and spline interpolation
            # densely inside a grid, and take the maximum difference as a "wiggle" systematic
            for ix in range(len(xpos)-1):
                for iy in range(len(ypos)-1):

                    xlocal = np.linspace(xpos[ix], xpos[ix+1], num=100)
                    ylocal = np.linspace(ypos[iy], ypos[iy+1], num=100)
                    cell_spline = spline(xlocal, ylocal)
                    VCX, VCY = np.meshgrid(xlocal, ylocal, indexing='ij')
                    VCP = np.column_stack( (VCX.ravel(), VCY.ravel()) )
                    cell_bilinear = bilinear(VCP).reshape(VCX.shape)

                    dphi = np.nanmax(np.abs(cell_spline - cell_bilinear))
                    xc, yc = np.array([0.5 * (xpos[ix] + xpos[ix+1])]), \
                        np.array([0.5 * (ypos[iy] + ypos[iy+1])])
                    spred = spline( xc, yc )[0,0]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        frac_wig[ibin, ix, iy] = dphi / spred

            if args.validation is not None:
                for ix in range(len(val_xpos)):
                    for iy in range(len(val_ypos)):
                        with uproot.open(val_files[val_indices[ix, iy]]) as fuin:
                            area  = fuin[znames[0]]["hArea"].axis('x').centers()[0]
                            POT   = fuin[znames[0]]["hPOT"].axis('x').centers()[0]
                            scval = fuin[znames[0]][args.flavour]["Flux"].values()[ibin] / (bw*area*POT)
                            val_data[ibin, ix, iy]   = scval
                val_pred[ibin] = spline( val_xpos, val_ypos )
                VX, VY = np.meshgrid(val_xpos, val_ypos, indexing='ij')
                VP = np.column_stack( (VX.ravel(), VY.ravel()) )
                val_bil_pred[ibin] = bilinear(VP).reshape(VX.shape)
            
            pbar.update(1)

    # If we have a validation set, make the comparison
    val_dev = val_data - val_pred if args.validation is not None else None
    val_bil_dev = val_data - val_bil_pred if args.validation is not None else None
    with np.errstate(divide='ignore', invalid='ignore'):
        val_frac_dev = val_dev / val_data if args.validation is not None else None
        val_bil_frac_dev = val_bil_dev / val_data if args.validation is not None else None

    with h5py.File(Path(args.output).resolve(), 'w') as fhout:
        fhout.create_dataset("Energy bins", data=ebins[:cutoff])
        
        gflav = fhout.require_group(args.flavour)
        gflav.create_dataset("Data", data=data)
        gflav.create_dataset("Prediction", data=ypred)
        if args.validation is not None:
            gflav.create_dataset("Validation data", data=val_data)
            gflav.create_dataset("Fractional deviation (spline)", data=val_frac_dev)
            gflav.create_dataset("Fractional deviation (bilinear)", data=val_bil_frac_dev)
            gflav.create_dataset("Wiggle systematic", data=frac_wig)

    print(Fore.GREEN + "Saved file to " + str(Path(args.output).resolve()) + Fore.RESET)

if __name__ == "__main__":
    print(Fore.YELLOW + "Hello world!" + Fore.RESET)
    
    parser = argparse.ArgumentParser(description='''On every histogram, read in x and y, get the density out, and interpolate that ~~using splines!!! Not polynomials~~.''')
    parser.add_argument('-i', '--input', type=str, required=True, help="Input directory. I will use all the files in this dir.")
    parser.add_argument('-v', '--validation', type=str, help="If passed, I will use all the files in this dir to validate the spline agreement.")
    parser.add_argument('-o', '--output', type=str, default='interpolated.h5', help="Output HDF5 file.")
    parser.add_argument('--flavour', type=str, default='numu',
                        choices=['numu', 'nue', 'numubar', 'nuebar'], help="Which flux to interpolate.")
    args = parser.parse_args()

    main(args)

    print(Fore.YELLOW + "Goodbye world!" + Fore.RESET)
