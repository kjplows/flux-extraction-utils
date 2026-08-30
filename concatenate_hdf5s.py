import h5py
import numpy as np
import argparse
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from colorama import Fore
from tqdm import tqdm

ZMULT=20.0 # I made these histograms one every 20 cm along z from 0 cm to 500 cm

def main(args):
    flavour_inputs = {}
    shapes = {} # Different flavours have different energy cutoffs.
    val_shapes = {} # Validation datasets have different shapes
    ebins, xvals, yvals, zvals = None, None, None, None

    for flinp in args.input:
        antinu = True if "bar" in Path(flinp).stem else False
        muon   = True if "mu"  in Path(flinp).stem else False
        if muon:
            flavour = "numu" if not antinu else "numubar"
        else:
            flavour = "nue" if not antinu else "nuebar"

        flavour_inputs[flavour] = sorted([Path(f).resolve() for f in Path(flinp).glob("*.h5")])

    # Make structure. numu has the highest cutoff, so use it to get energy bins
    with h5py.File(flavour_inputs['numu'][0]) as fhin:
        ebins = fhin["Energy bins"][:]
        xvals = fhin["x coordinates"][:]
        yvals = fhin["y coordinates"][:]
    zvals = np.array([ZMULT * i for i in range(len(flavour_inputs['numu']))])

    # And now start to read the shapes
    for flav, flinp in flavour_inputs.items():
        with h5py.File(flinp[0]) as fhin:
            shapes[flav] = fhin[flav]['Data'].shape
            val_shapes[flav] = fhin[flav]['Validation data'].shape

    # Promote the output data to shape x len(zvals)
    full_shapes  = { key: shape + (len(zvals),) for key, shape in shapes.items() }
    full_vshapes = { key: shape + (len(zvals),) for key, shape in val_shapes.items() }
    data     = { key: np.zeros(shape) for key, shape in full_shapes.items() }
    errors   = { key: np.zeros(shape) for key, shape in full_shapes.items() }
    val_data = { key: np.zeros(shape) for key, shape in full_vshapes.items() }
    val_errs = { key: np.zeros(shape) for key, shape in full_vshapes.items() }
    preds    = { key: np.zeros(shape) for key, shape in full_shapes.items() }
    val_pred = { key: np.zeros(shape) for key, shape in full_vshapes.items() }
    val_pred_bil = { key: np.zeros(shape) for key, shape in full_vshapes.items() }
    frac_dev = { key: np.zeros(shape) for key, shape in full_vshapes.items() }
    frac_dev_bil = { key: np.zeros(shape) for key, shape in full_vshapes.items() }
    wiggle   = { key: np.zeros(shape) for key, shape in full_vshapes.items() }

    for flav, flinp in flavour_inputs.items():
        with tqdm(total=len(flinp), desc=f"Getting inputs from {flav}...",
                  bar_format = "{l_bar}{bar}{r_bar}") as pbar:
            for i, fin in enumerate(flinp):
                with h5py.File(fin) as fhin:
                    data[flav][...,i]     = fhin[flav]["Data"][:]
                    errors[flav][...,i]   = fhin[flav]["Errors on data"][:]
                    val_data[flav][...,i] = fhin[flav]["Validation data"][:]
                    val_errs[flav][...,i] = fhin[flav]["Errors on validation data"][:]
                    preds[flav][...,i]    = fhin[flav]["Prediction"][:]
                    val_pred[flav][...,i] = fhin[flav]["Validation prediction (spline)"][:]
                    val_pred_bil[flav][...,i] = fhin[flav]["Validation prediction (bilinear)"][:]
                    frac_dev[flav][...,i] = fhin[flav]["Fractional deviation (spline)"][:]
                    frac_dev_bil[flav][...,i] = fhin[flav]["Fractional deviation (bilinear)"][:]
                    wiggle[flav][...,i]   = fhin[flav]["Wiggle systematic"][:]

                pbar.update(1)

    compression_kwargs = {'compression': 'gzip', 'compression_opts': 9}
    with h5py.File(Path(args.output).resolve(), 'w') as fhout:
        fhout.create_dataset('Energy bins', data=ebins)
        fhout.create_dataset('x coordinates', data=xvals)
        fhout.create_dataset('y coordinates', data=yvals)
        fhout.create_dataset('z coordinates', data=zvals)

        for flav in flavour_inputs:
            gflav = fhout.require_group(flav)
            gflav.create_dataset("Data", data=data[flav], **compression_kwargs)
            gflav.create_dataset("Errors on data", data=errors[flav], **compression_kwargs)
            gflav.create_dataset("Validation data", data=val_data[flav], **compression_kwargs)
            gflav.create_dataset("Errors on validation data", data=val_errs[flav], **compression_kwargs)
            gflav.create_dataset("Prediction", data=preds[flav], **compression_kwargs)
            gflav.create_dataset("Validation prediction (spline)", data=val_pred[flav], **compression_kwargs)
            gflav.create_dataset("Validation prediction (bilinear)", data=val_pred_bil[flav], **compression_kwargs)
            gflav.create_dataset("Fractional deviation (spline)", data=frac_dev[flav], **compression_kwargs)
            gflav.create_dataset("Fractional deviation (bilinear)", data=frac_dev_bil[flav], **compression_kwargs)
            gflav.create_dataset("Wiggle systematic", data=wiggle[flav], **compression_kwargs)

    print(Fore.GREEN + f"Wrote file: {Path(args.output).resolve()}" + Fore.RESET)

if __name__ == "__main__":
    print(Fore.YELLOW + "Hello world!" + Fore.RESET)
    
    parser = argparse.ArgumentParser(description='''Take the many individual hdf5s from `interpolate_histograms_2D.py' and turn them into one big one with all the z positions and flavour to distribute.''')
    parser.add_argument('-i', '--input', type=str, nargs=4, required=True, help="Input directories. 1 per flavour, each having the same number of HDF5s.")
    parser.add_argument('-o', '--output', type=str, default='interpolated.h5', help="Output HDF5 file.")
    args = parser.parse_args()

    main(args)

    print(Fore.YELLOW + "Goodbye world!" + Fore.RESET)
