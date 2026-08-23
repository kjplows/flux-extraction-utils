import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
from colorama import Fore

# data is a (N, K, K) array of N energy bins and K x K knots
# coords is a (2, K) array of the evaluation coordinates
# returns the averaged flux over [xmin, xmax] x [ymin, ymax] as an array (N,)
def average_flux(data, coords, xmin, xmax, ymin, ymax):

    avg_flux = np.empty(data.shape[0])
    for ibin in range(data.shape[0]):
        spline = RectBivariateSpline( coords[0,:], coords[1,:],
                                      data[ibin, ...], kx=3, ky=3, s=0 )
        avg_flux[ibin] = spline.integral(xmin, xmax, ymin, ymax) / ((xmax-xmin) * (ymax-ymin))

    return avg_flux

def main(args):

    xpos = np.arange(-210, 211, 20)
    ypos = np.arange(-210, 211, 20)

    FV1 = np.array([[-190, -10], [-190, 190]]) # East TPC
    FV2 = np.array([[10, 190],   [-190, 190]]) # West TPC

    with h5py.File(args.input) as fhin:
        data = fhin[args.flavour]["Data"][:]
    
    avg_flux_1 = average_flux(data, np.array([xpos, ypos]),
                              -190.0, -10.0, -190.0, 190.0)
    avg_flux_2 = average_flux(data, np.array([xpos, ypos]),
                              10.0, 190.0, -190.0, 190.0)
    avg_flux_3 = average_flux(data, np.array([xpos, ypos]),
                              -190.0, 190.0, -190.0, 190.0)
    

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.stairs(avg_flux_1[:60], np.arange(0.0, 3.01, 0.05))
    ax.stairs(avg_flux_2[:60], np.arange(0.0, 3.01, 0.05))
    ax.stairs(avg_flux_3[:60], np.arange(0.0, 3.01, 0.05))
    plt.show()

if __name__ == "__main__":
    print(Fore.CYAN + "Hello world!" + Fore.RESET)
    
    parser = argparse.ArgumentParser(description='''Average yon fluxes.''')
    parser.add_argument('-i', '--input', type=str, required=True, help="Input HDF5")
    parser.add_argument('-f', '--flavour', type=str, default='numu',
                        choices=['numu', 'nue', 'numubar', 'nuebar'], help="Which flux to grab.")
    args = parser.parse_args()

    main(args)

    print(Fore.CYAN + "Goodbye world!" + Fore.RESET)
