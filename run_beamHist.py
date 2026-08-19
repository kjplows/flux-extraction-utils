import uproot
from uproot.writing.identify import to_TH1x, to_TAxis
import numpy as np
import pandas as pd
#import awkward as ak
import multiprocessing
import argparse
from argparse import RawTextHelpFormatter # for newlines in help, see SO 3853722
from pathlib import Path
from colorama import Fore
from tqdm import tqdm
import time
from copy import deepcopy

#import pydk2nu as pk

NPROCESS_MAX = 50 # No more threads than this.
#enu_bins = np.arange(0.0, 10.05, 0.05) # Standard beamHist binning, 200 bins
#enu_bedges = np.arange(0.0, 10.05, 0.05) # To write edges
# This guy has finer binning.
enu_bins = np.arange(0.0, 10.0005, 0.0005)
enu_bedges = np.arange(0.0, 10.0005, 0.0005) # To write edges

grammar = ["st", "nd", "rd", "th"]

masses = { # From dk2nu code
    'pion'   : 0.1395701,
    'kaon'   : 0.493677,
    'kzero'  : 0.497614,
    'muon'   : 0.1056583715,
    'neutron': 0.93956536
}
pdgs = {
    'nue'        :    12,
    'nuebar'     : -  12,
    'numu'       :    14,
    'numubar'    : -  14,
    'muplus'     : -  13,
    'muminus'    :    13,
    'piplus'     :   211,
    'piminus'    : - 211,
    'k0l'        :   130,
    'k0s'        :   310,
    'k0mix'      :   311,
    'kplus'      :   321,
    'kminus'     : - 321,
    'neutron'    :  2112,
    'antineutron': -2112,
    'proton'     :  2212
}

colours = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.MAGENTA]

Erange = (enu_bins[0], enu_bins[-2])
Nbins = len(enu_bins)-1
dE = (Erange[1]-Erange[0])/Nbins
#xbins = np.arange(Erange[0], Erange[1], dE)
xbins = enu_bins
xcentres = 0.5 * (xbins[1:] + xbins[:-1])

# A wrapper around uproot.writing.identify.to_TH1x
# note that to_TH1x expects first and last bin under/overflow
def construct_uproot_hist(data, entries=None, sumw2s=None, name=None, title=None, axis_titles=None):

    uname="foo" if name is None else name
    uentries = 0 if entries is None else entries

    ldata, lsums = [0.0], [0.0]
    for d, s in zip(data, sumw2s):
        ldata.append(d)
        lsums.append(s)
    ldata.append(0.0); lsums.append(0.0)
    
    return to_TH1x(
        fName    = uname,
        fTitle   = title,
        data     = np.array(ldata),
        fEntries = uentries,
        fSumw2   = np.array(lsums),
        fTsumw   = np.sum(data),
        fTsumw2  = np.sum(sumw2s),
        fXaxis   = to_TAxis(fName  = "xaxis",   fTitle = axis_titles[0],
                            fNbins = data.shape[0],
                            fXmin  = Erange[0], fXmax  = Erange[1]),
        fYaxis   = to_TAxis(fName  = "yaxis",   fTitle = axis_titles[1],
                            fNbins = 1,
                            fXmin  = max(0.0, 0.95 * np.min(data)), fXmax = 1.05 * np.max(data)),
        fTsumwx  = np.sum(data * xcentres),
        fTsumwx2 = np.sum(data * xcentres ** 2)
    )

class BeamHistClass:
    """A container for histograms"""
    
    def __init__(self):
        self.data = {
            'POT': 0,
            'histFlux': { # For cumulative fluxes at the face
                'numu'   : np.zeros( Nbins ),
                'numubar': np.zeros( Nbins ),
                'nue'    : np.zeros( Nbins ),
                'nuebar' : np.zeros( Nbins )
            },
            'sumw2Flux': { # For sumw2 on the cumulative fluxes
                'numu'   : np.zeros( Nbins ),
                'numubar': np.zeros( Nbins ),
                'nue'    : np.zeros( Nbins ),
                'nuebar' : np.zeros( Nbins )
            },
            'entries': { # Track N entries
                'numu'   : 0,
                'numubar': 0,
                'nue'    : 0,
                'nuebar' : 0
            }
        }

    # Simple repr method to show POT
    def __repr__(self):
        pot = self.data['POT']
        return f"POT: {pot}"

    # Utility method to add results from a processed file into a tally
    def Fill(self, hist):
        for name, item in hist.items():
            if isinstance(item, float):
                self.data[name] += item
            elif isinstance(item, np.ndarray):
                self.data[name] = np.add(self.data[name], item)
            elif isinstance(item, dict):
                for hname, hitem in item.items():
                    if isinstance(hitem, np.ndarray):
                        self.data[name][hname] = np.add(self.data[name][hname], hitem)
                    elif isinstance(hitem, dict):
                        for h2name, h2item in hitem.items():
                            if isinstance(h2item, np.ndarray):
                                self.data[name][hname][h2name] = \
                                    np.add(self.data[name][hname][h2name], h2item)
                    elif np.isscalar(hitem):
                        self.data[name][hname] += hitem

    # Deep copy and return
    def GetHists(self):
        return deepcopy(self.data)

# A clone of `bsim::calcEnuWgt` from dk2nugenie. This is suboptimal, but defeats the need to have a full
# bsim::Decay object. Plus, it accepts vectorised numpy operations.
# decay is a dictionary from N entries, xyz is a numpy array of shape (3, N)
def calc_enu_wgt( decay, xyz ):

    # For convenience, just name the decay variables
    ptype = decay['dk2nu/decay/decay.ptype'].to_numpy() # (N,)
    necm  = decay['dk2nu/decay/decay.necm'].to_numpy()  # (N,)
    pdp   = np.array([ decay['dk2nu/decay/decay.pdpx'].to_numpy(), 
                       decay['dk2nu/decay/decay.pdpy'].to_numpy(), 
                       decay['dk2nu/decay/decay.pdpz'].to_numpy() ]) # (3, N)
    vtx   = np.array([ decay['dk2nu/decay/decay.vx'].to_numpy(),   
                       decay['dk2nu/decay/decay.vy'].to_numpy(),   
                       decay['dk2nu/decay/decay.vz'].to_numpy() ]) # (3, N)


    # Let's normalise this flux to a circle of area 1 cm2, not a circle of radius 100 cm.
    # The radius of this circle is about 5.6 mm
    kRDET = np.full_like(xyz[0, :], 1.0 / np.sqrt(np.pi))

    # Define some masks
    mask_pion_parent    = (ptype == pdgs['piplus'])  | (ptype == pdgs['piminus'])
    mask_kaon_parent    = (ptype == pdgs['kplus'])   | (ptype == pdgs['kminus'])
    mask_kzero_parent   = (ptype == pdgs['k0l'])     | (ptype == pdgs['k0s'])         | (ptype == pdgs['k0mix'])
    mask_muon_parent    = (ptype == pdgs['muminus']) | (ptype == pdgs['muplus'])
    mask_neutron_parent = (ptype == pdgs['neutron']) | (ptype == pdgs['antineutron'])
    mask_unknown_parent = (~mask_pion_parent) & (~mask_kaon_parent) & (~mask_kzero_parent) & \
        (~mask_muon_parent) & (~mask_neutron_parent)

    # Parent masses
    parent_masses                      = np.zeros_like(ptype, dtype=np.float32) # You need dtype, else m = 0[.blah]
    parent_masses[mask_pion_parent]    = masses['pion']
    parent_masses[mask_kaon_parent]    = masses['kaon']
    parent_masses[mask_kzero_parent]   = masses['kzero']
    parent_masses[mask_muon_parent]    = masses['muon']
    parent_masses[mask_neutron_parent] = masses['neutron']

    # Parent momentum magnitudes
    parent_p3 = np.linalg.norm(pdp, axis=0)
    parent_E  = np.sqrt(parent_p3**2 + parent_masses**2)

    with np.errstate(divide='ignore', invalid='ignore'):
        gamma = parent_E / parent_masses
        gamma[np.isnan(gamma)] = 1.0
        gamma[gamma < 1.0] = 1.0
        
    beta3 = np.sqrt( (gamma**2 - 1)/(gamma**2) )

    # A mask for stopped / invalid parents
    mask_stopped = mask_unknown_parent | (parent_p3 == 0.0)

    # Angle from parent line of flight to chosen point in beam frame
    rad = np.linalg.norm( (xyz - vtx), axis=0 ) # = sanddetcomp
    emrat = np.ones_like(rad)
    with np.errstate(divide='ignore', invalid='ignore'):
        costh_pardet = np.einsum('ij,ij->j', pdp, (xyz - vtx)) / (parent_p3 * rad)
        costh_pardet[np.isnan(costh_pardet)] = -1.0
        costh_pardet = np.clip(costh_pardet, -1.0, 1.0)

    emrat[~mask_stopped] = 1.0 / (gamma[~mask_stopped] * (1.0 - beta3[~mask_stopped] * costh_pardet[~mask_stopped]))

    enus = necm * emrat

    sangdet = 0.5 * ( 1.0 - np.cos( np.arctan2( kRDET, rad ) ) )

    wgts_xy = sangdet * emrat**2

    # TODO add muon polarisation

    return enus, wgts_xy

# queue and result are instances of `multiprocessing.Queue()`.
# queue  <-- input files for this thread to process.
# result <-- histograms coming out of this thread. List with 0th element inclusive, others PRISM
# index <-- index of the process
# det <-- array of (det origin (x, y, z) [cm], half side [cm])
def individual_thread(queue, result, index=0, det=None):

    file_list, clean_list = [], []
    while True:
        qitem = queue.get()
        if qitem is None:
            break
        file_list.append( qitem )

    # Open the dkmetaTree and read in the number of POT
    NPOT = 0
    for q in file_list :
        with uproot.open(q) as fuin:
            with fuin["dkmetaTree"] as meta_tree:
                try:
                    apots  = meta_tree["dkmeta/pots"].array(library='np')
                    NAPOT  = np.sum(apots)
                    NPOT  += NAPOT
                    clean_list.append(q)

                except Exception as e: # we do not want to loop over bad files
                    print(Fore.RED + f"Skipping file {q}, exception {e}" + Fore.RESET)

    file_list = clean_list

    # We now know how many POT we used, and will loop over only those files to get histograms
    # No redecays for now

    # The output will be a dictionary of histograms + the POT used.
    # Histograms = numpy arrays, which are lighter than the ROOT types - we'll construct them in main
    
    needed_arrays = [
        "dk2nu/decay/decay.ntype", "dk2nu/decay/decay.ptype", "dk2nu/decay/decay.ndecay",
        "dk2nu/decay/decay.necm", "dk2nu/decay/decay.nimpwt",
        "dk2nu/decay/decay.pdpx", "dk2nu/decay/decay.pdpy", "dk2nu/decay/decay.pdpz", 
        "dk2nu/decay/decay.vx", "dk2nu/decay/decay.vy", "dk2nu/decay/decay.vz",
        "dk2nu/ancestor/ancestor.proc", "dk2nu/ancestor/ancestor.pdg"
    ]
    ret_class  = BeamHistClass()
    
    icol = index % len(colours)
    pbar = tqdm(total=len(file_list), position=index, desc=f"Thread {index}",
                bar_format = colours[icol] + "{l_bar}{bar}{r_bar}" + Fore.RESET)
    for iq, q in enumerate(file_list):
        #file_class = BeamHistClass()
        with uproot.open(q) as fuin:
            with fuin["dk2nuTree"] as tree, fuin["dkmetaTree"] as meta_tree:
                apots  = meta_tree["dkmeta/pots"].array(library='np')
                NAPOT  = np.sum(apots)
                #file_class.data['POT'] = NAPOT
                ret_class.data['POT'] += NAPOT
                
                try:
                    for branches in tree.iterate(needed_arrays,
                                                 step_size="2 GB"):
                        # Histograms
                        nimpwt   = branches["dk2nu/decay/decay.nimpwt"].to_numpy()
                        n_events = nimpwt.shape[0]
                        ntype    = branches["dk2nu/decay/decay.ntype"].to_numpy()
                        ptype    = branches["dk2nu/decay/decay.ptype"].to_numpy()
                        proc     = branches["dk2nu/ancestor/ancestor.proc"]
                        apdg     = branches["dk2nu/ancestor/ancestor.pdg"]
                         
                        # Generate positions distributed in detector
                        if( det is not None ):
                            positions = np.random.uniform(-det[3], det[3], size=(n_events, 2))
                            enus, wgts = calc_enu_wgt(branches, np.array([
                                (positions[:, 0] + np.full(n_events, det[0])),
                                (positions[:, 1] + np.full(n_events, det[1])),
                                np.full(n_events, det[2])
                            ]))
                            full_wgts = nimpwt * wgts * ((2.0*det[3])**2) # accounting for detector area

                            # Make masks to fill out the necessary histograms
                            mask_type_numu    = (ntype == pdgs["numu"])
                            mask_type_numubar = (ntype == pdgs["numubar"])
                            mask_type_nue     = (ntype == pdgs["nue"])
                            mask_type_nuebar  = (ntype == pdgs["nuebar"])

                            # Build a dictionary of histogram type <-- mask to use
                            mask_dict = {
                                'histFlux': {
                                    'numu'   : mask_type_numu,
                                    'numubar': mask_type_numubar,
                                    'nue'    : mask_type_nue,
                                    'nuebar' : mask_type_nuebar
                                }
                            }

                            # Get number of entries
                            nent = np.ones_like(enus)
                            ret_class.data['entries']['numu'] += np.sum(nent[mask_type_numu])
                            ret_class.data['entries']['numubar'] += np.sum(nent[mask_type_numubar])
                            ret_class.data['entries']['nue'] += np.sum(nent[mask_type_nue])
                            ret_class.data['entries']['nuebar'] += np.sum(nent[mask_type_nuebar])

                            # Fill histograms.
                            for htype, hitem in mask_dict.items():
                                for flav, fitem in hitem.items(): # fitem is a mask, or dict of masks.
                                    if htype == "histFlux":
                                        ret_class.data["histFlux"][flav] += (np.histogram(
                                            enus[fitem], bins=enu_bins, weights=full_wgts[fitem]
                                        ))[0]
                                        ret_class.data["sumw2Flux"][flav] += (np.histogram(
                                            enus[fitem], bins=enu_bins, weights=(full_wgts[fitem]**2)
                                        ))[0]

                    #ret_class.Fill(file_class.data)
                    
                except Exception as e:
                    print(Fore.RED + f"Skipping file {q}, exception {e}" + Fore.RESET)

                    # remove this file's POT
                    NPOT  -= NAPOT
                    ret_class.data['POT'] -= NAPOT

                pbar.update(1)

        piq = min(3, iq)

    full_classes = [ret_class]
        
    result.put(full_classes, block=False)
    return

def main(args):
    NTHREADS = args.threads
    if( NTHREADS > NPROCESS_MAX ):
        print(Fore.RED + f"""You have asked for {NTHREADS} threads which is over the max limit.
This is bad for performance (especially on a gpvm...). Reducing down to {NPROCESS_MAX} threads.""" + Fore.RESET)
        NTHREADS = NPROCESS_MAX

    input_files = [ p for p in Path(args.inpath).glob("*.dk2nu.root") ]
    if len(input_files) == 0:
        input_files = [ p for p in Path(args.inpath).glob("*/*.dk2nu.root") ]
    NFILES_TOTAL = len( input_files )
        
    print(Fore.GREEN + f"Processing {NFILES_TOTAL} dk2nu files over {NTHREADS} threads" + Fore.RESET)
    NFILE_PER_THREAD = int(NFILES_TOTAL / NTHREADS)
    NFILE_REMAINDER  = NFILES_TOTAL - NFILE_PER_THREAD * NTHREADS

    det = np.array([ args.detector[0], args.detector[1], args.detector[2], args.size ])
    print(Fore.GREEN + f"Using detector with origin ({det[0]}, {det[1]}, {det[2]}) cm and half side {det[3]} cm" + Fore.RESET)

    # Put these in their queues and pass them
    # We'll offload any remaining files 1 per thread
    processes = []
    NFILES_PROCESSED, NTHREADS_PROCESSED = 0, 1
    NFILES_TARGET = NFILE_PER_THREAD+1 if NTHREADS_PROCESSED < NFILE_REMAINDER else NFILE_PER_THREAD
    file_queue, hist_queue = multiprocessing.Queue(), multiprocessing.Queue()
    for p in input_files:
        NFILES_PROCESSED += 1
        file_queue.put(p)
        
        if(NFILES_PROCESSED == NFILES_TARGET): # Construct a process and reset counters
            processes.append(multiprocessing.Process( target=individual_thread,
                                                      args=(file_queue, hist_queue,
                                                            NTHREADS_PROCESSED, det) ))
            file_queue.put(None) # signal end
            file_queue = multiprocessing.Queue()
            
            grammar_postfix = grammar[min(3, NTHREADS_PROCESSED-1)]
            print(Fore.GREEN + f"Adding {NFILES_PROCESSED} files to the {NTHREADS_PROCESSED}" +\
                  f"{grammar_postfix} thread..." + Fore.RESET)

            NFILES_PROCESSED = 0
            NTHREADS_PROCESSED += 1
            NFILES_TARGET = NFILE_PER_THREAD+1 if NTHREADS_PROCESSED < NFILE_REMAINDER else NFILE_PER_THREAD

    # Start spawning the processes.
    print(Fore.YELLOW + "Spawning processes with 1s delay.." + Fore.RESET)
    for ip, proc in enumerate(processes):
        grammar_postfix = grammar[min(3, ip)]
        time.sleep(1)
        pip = ip+1
        #print(Fore.CYAN + f"Spawning {pip}{grammar_postfix} process." + Fore.RESET)
        proc.start()

    # Wait till they're all done
    results = []
    for _ in processes:
        results.append(hist_queue.get())
    for proc in processes:
        proc.join()

    # Get the histograms from all the queue objects, hadd them together
    final_classes = []

    for thread_result in results:
        for i, cc in enumerate(thread_result):
            if( len(final_classes) == i ):
                final_classes.append(BeamHistClass()) # Explicitly append so Python behaves...

            final_classes[i].Fill(cc.data)

    print(Fore.GREEN + f"Finally, {final_classes[0]}" + Fore.RESET)

    # Prepare some sugar for writing
    dirnames = ["detector"]
    suffixes = [f" (parameter {args.name} at variation {args.sigma} sigma)"]
    
    # Write the output into a directory.
    with uproot.recreate(args.output) as fuout:

        for ia, (fc, dname, sfx) in enumerate(zip(final_classes, dirnames, suffixes)):
            det = fuout.mkdir(dname)

            det["hPOT"]  = np.histogram([fc.data['POT']], bins=1)
            area = (2.0 * args.size)**2
            det["hArea"] = np.histogram([area], bins=1)
            
            det["h501"] = construct_uproot_hist(
                fc.data["histFlux"]["nue"],
                sumw2s  = fc.data["sumw2Flux"]["nue"],
                entries = fc.data["entries"]["nue"],
                name = "h501", title = f"nue (all){sfx}",
                axis_titles = ("Energy (GeV)", "Flux (nue / 0.5MeV)")
            )
            det["h502"] = construct_uproot_hist(
                fc.data["histFlux"]["nuebar"],
                sumw2s  = fc.data["sumw2Flux"]["nuebar"],
                entries = fc.data["entries"]["nuebar"],
                name = "h502", title = f"nuebar (all){sfx}",
                axis_titles = ("Energy (GeV)", "Flux (nuebar / 0.5MeV)")
            )
            det["h503"] = construct_uproot_hist(
                fc.data["histFlux"]["numu"],
                sumw2s  = fc.data["sumw2Flux"]["numu"],
                entries = fc.data["entries"]["numu"],
                name = "h503", title = f"numu (all){sfx}",
                axis_titles = ("Energy (GeV)", "Flux (numu / 0.5MeV)")
            )
            det["h504"] = construct_uproot_hist(
                fc.data["histFlux"]["numubar"],
                sumw2s  = fc.data["sumw2Flux"]["numubar"],
                entries = fc.data["entries"]["numubar"],
                name = "h504", title = f"numubar (all){sfx}",
                axis_titles = ("Energy (GeV)", "Flux (numubar / 0.5MeV)")
            )
            
if __name__ == "__main__":
    print(Fore.YELLOW + "Hello world!" + Fore.RESET)

    parser = argparse.ArgumentParser(description=Fore.CYAN+"A script to extract fluxes from dk2nu files."+Fore.RESET, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--inpath', type=str, required=True, help=Fore.RED+"I will read all the files in this directory."+Fore.RESET)
    parser.add_argument('-o', "--output", type=str, default="./out.root", help=Fore.YELLOW+"Output file"+Fore.RESET)
    parser.add_argument('-t', '--threads', type=int, default=8, help=Fore.YELLOW+f"Number of concurrent threads. Maximum {NPROCESS_MAX}"+Fore.RESET)
    parser.add_argument("--detector", type=float, nargs=3, required=True, help=Fore.RED+"Detector origin in beamline coordinates (cm)"+Fore.RESET)
    parser.add_argument('-s', "--size", type=float, required=True, help=Fore.RED+"Half side of the square detector face (cm)"+Fore.RESET)
    parser.add_argument('--name', type=str, default="UNKNOWN", help=Fore.YELLOW+"Name of the varied parameter."+Fore.RESET)
    parser.add_argument('--sigma', type=str, default="UNKNOWN", help=Fore.YELLOW+"Variation of the parameter in sigma" + Fore.RESET)
    
    args = parser.parse_args()

    main(args)
    
    print(Fore.YELLOW + "Goodbye world!" + Fore.RESET)

