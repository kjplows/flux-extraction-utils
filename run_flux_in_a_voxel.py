import uproot
from uproot.writing.identify import to_TH1x, to_TAxis
import numpy as np
import pandas as pd
import awkward as ak
import multiprocessing
import argparse
from argparse import RawTextHelpFormatter # for newlines in help, see SO 3853722
from pathlib import Path
from colorama import Fore
from tqdm import tqdm
import time
from copy import deepcopy

NPROCESS_MAX = 50 # No more threads than this.
enu_bins = np.arange(0.0, 10.05, 0.05) # Standard beamHist binning, 200 bins
enu_bedges = np.arange(0.0, 10.05, 0.05) # To write edges
Nbins = enu_bins.shape[0]-1

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

class BeamHistClass:
    """A container for histograms exactly like what is written in beamHist"""
    # Edit, August 17th: Introduce lists of [counts, sumw2s, entries].

    '''
            'sFlux': { # For cumulative fluxes at the face, sumw2
                'numu'   : np.zeros( Nbins ),
                'numubar': np.zeros( Nbins ),
                'nue'    : np.zeros( Nbins ),
                'nuebar' : np.zeros( Nbins )
            },
            'eFlux': { # Track N entries
                'numu'   : 0,
                'numubar': 0,
                'nue'    : 0,
                'nuebar' : 0
            },
    '''
    
    def __init__(self):
        self.data = {
            'POT': 0,
            'hFlux': { # For cumulative fluxes at the face
                'numu'   : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                'numubar': [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                'nue'    : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                'nuebar' : [np.zeros( Nbins ), np.zeros( Nbins ), 0]
            },
            'hparent': { # For fluxes broken down by parent
                'numu'   : {
                    'muon' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'pion' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kaon' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kzero': [np.zeros( Nbins ), np.zeros( Nbins ), 0]
                },
                'numubar': {
                    'muon' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'pion' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kaon' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kzero': [np.zeros( Nbins ), np.zeros( Nbins ), 0]
                } ,
                'nue'    : {
                    'muon' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'pion' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kaon' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kzero': [np.zeros( Nbins ), np.zeros( Nbins ), 0]
                } ,
                'nuebar' : {
                    'muon' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'pion' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kaon' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kzero': [np.zeros( Nbins ), np.zeros( Nbins ), 0]
                }
            },
            'hsec': { # For fluxes broken down by secondary in the pBe interaction
                'numu'   : {
                    'pimu'   : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'pinomu' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kzero'  : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kaon'   : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'nucleon': [np.zeros( Nbins ), np.zeros( Nbins ), 0]
                } ,
                'numubar': {
                    'pimu'   : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'pinomu' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kzero'  : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kaon'   : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'nucleon': [np.zeros( Nbins ), np.zeros( Nbins ), 0]
                } ,
                'nue'    : {
                    'pimu'   : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'pinomu' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kzero'  : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kaon'   : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'nucleon': [np.zeros( Nbins ), np.zeros( Nbins ), 0]
                },
                'nuebar' : {
                    'pimu'   : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'pinomu' : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kzero'  : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'kaon'   : [np.zeros( Nbins ), np.zeros( Nbins ), 0],
                    'nucleon': [np.zeros( Nbins ), np.zeros( Nbins ), 0]
                }
            }
        }

    # Simple repr method to show POT
    def __repr__(self):
        pot = self.data['POT']
        return f"POT: {pot}"

    # Utility method to add results from a processed file into a tally
    def Fill(self, hist):
        for name, item in hist.items():
            if isinstance(item, float) or isinstance(item, int):
                self.data[name] += item
            elif isinstance(item, np.ndarray):
                self.data[name] = np.add(self.data[name], item)
            elif isinstance(item, dict):
                for hname, hitem in item.items():
                    if isinstance(hitem, float) or isinstance(item, int):
                        self.data[name][hname] += hitem
                    elif isinstance(hitem, np.ndarray):
                        self.data[name][hname] = np.add(self.data[name][hname], hitem)
                    elif isinstance(hitem, list):
                        for i, entry in enumerate(hitem):
                            if isinstance(entry, float) or isinstance(entry, int):
                                self.data[name][hname][i] += entry
                            elif isinstance(entry, np.ndarray):
                                self.data[name][hname][i] = np.add(self.data[name][hname][i], entry)
                    elif isinstance(hitem, dict):
                        for h2name, h2item in hitem.items():
                            if isinstance(hitem, float) or isinstance(item, int):
                                self.data[name][hname][h2name] += h2item
                            elif isinstance(h2item, np.ndarray):
                                self.data[name][hname][h2name] = \
                                    np.add(self.data[name][hname][h2name], h2item)
                            elif isinstance(h2item, list):
                                for i, entry in enumerate(h2item):
                                    if isinstance(entry, float) or isinstance(entry, int):
                                        self.data[name][hname][h2name][i] += entry
                                    elif isinstance(entry, np.ndarray):
                                        self.data[name][hname][h2name][i] = np.add(self.data[name][hname][h2name][i], entry)

    # Deep copy and return
    def GetHists(self):
        return deepcopy(self.data)

# A clone of `bsim::calcEnuWgt` from dk2nugenie. This is suboptimal, but defeats the need to have a full
# bsim::Decay object. Plus, it accepts vectorised numpy operations.
#
# Edit, August 9th: Extrude the calculation along z.
# # decay is a dictionary from N eFlux, xy and z is a numpy array of shape (2, N) and (M, N)
def calc_enu_wgt( decay, xy, z ):

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
    kRDET = np.full_like(xy[0], 1.0 / np.sqrt(np.pi)) # shape: (N,)
    # And broadcast to (M, N)
    kRDETs = np.broadcast_to(kRDET[None, :], (z.shape[0], kRDET.shape[0]))

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
    # Broadcast it to be reused across z
    masks  = np.broadcast_to(mask_stopped[None, :], (z.shape[0], parent_p3.shape[0]))
    gammas = np.broadcast_to(gamma[None, :], (z.shape[0], parent_p3.shape[0]))
    beta3s = np.broadcast_to(beta3[None, :], (z.shape[0], parent_p3.shape[0]))
    necms  = np.broadcast_to(necm[None, :], (z.shape[0], parent_p3.shape[0]))

    # We deal with M extruded positions of z
    # Build an array (M, N) of the parent lines of flight to the points in beam frame.
    # Index 0 (M) is the various z positions with x, y identical, and index 1 (N) is the events.

    # First, split out transverse and longitudinal components of everything
    vtx_xy, vtx_z = vtx[:2, :], vtx[2, :] # (2, N), (N,)
    pdp_xy, pdp_z = pdp[:2, :], pdp[2, :] # (2, N), (N,)
    # Careful with the neutrino displacements...
    disp_xy = xy - vtx_xy # (2, N)
    disp_z  = z - vtx_z[None, :] # (M, N)

    # Partial terms in the inner product
    prod_xy = np.einsum('ij,ij->j', pdp_xy, disp_xy) # (N,)
    prod_z  = pdp_z[None, :] * disp_z # (M, N)
    product = prod_xy[None, :] + prod_z # (M, N)
    
    # Magnitudes of the displacements. Shape (M, N)
    rad = np.sqrt(
        np.sum(disp_xy**2, axis=0)[None, :] + \
        disp_z**2 
    )

    # Find the boost factors, emrat. Note that mask_stopped is (N,) so broadcast to (M, N)
    # To do that, find the costheta (which is now (M, N) as one per point == one per z)
    emrat = np.ones_like(rad)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        costh_pardet = product / (parent_p3[None, :] * rad)
        costh_pardet[np.isnan(costh_pardet)] = -1.0
        costh_pardet = np.clip(costh_pardet, -1.0, 1.0)

    # All (N,) arrays broadcast to (M, N)
    emrat[~masks] = 1.0 / (gammas[~masks] * \
                           (1.0 - beta3s[~masks] * costh_pardet[~masks]) )

    enus = necms * emrat

    sangdet = 0.5 * ( 1.0 - np.cos( np.arctan2( kRDETs, rad ) ) )

    wgts_xy = sangdet * emrat**2

    # TODO add muon polarisation

    return enus, wgts_xy # both shape (M, N)

# queue and result are instances of `multiprocessing.Queue()`.
# queue  <-- input files for this thread to process.
# result <-- histograms coming out of this thread. List with 0th element inclusive, others PRISM
# index <-- index of the process
# All coordinates are BEAM
# xy = 2-tuple of centre [cm], side = square half-side [cm]
# z = array of shape (M,) [cm]
def individual_thread(queue, result, index=0, xy=None, side=None, z=None):

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
    full_classes = [BeamHistClass() for _ in range(len(z))] # len = M
    
    icol = index % len(colours)
    pbar = tqdm(total=len(file_list), position=index, desc=f"Thread {index}",
                bar_format = colours[icol] + "{l_bar}{bar}{r_bar}" + Fore.RESET)
    for iq, q in enumerate(file_list):
        with uproot.open(q) as fuin:
            with fuin["dk2nuTree"] as tree, fuin["dkmetaTree"] as meta_tree:
                apots  = meta_tree["dkmeta/pots"].array(library='np')
                NAPOT  = np.sum(apots)
                for ret_class in full_classes:
                    ret_class.data['POT'] += NAPOT                
                try:
                    for branches in tree.iterate(needed_arrays,
                                                 step_size="1 GB"):
                        # Histograms
                        nimpwt   = branches["dk2nu/decay/decay.nimpwt"].to_numpy()
                        n_events = nimpwt.shape[0]
                        ntype    = branches["dk2nu/decay/decay.ntype"].to_numpy()
                        ptype    = branches["dk2nu/decay/decay.ptype"].to_numpy()
                        proc     = branches["dk2nu/ancestor/ancestor.proc"]
                        apdg     = branches["dk2nu/ancestor/ancestor.pdg"]
                         
                        # Generate positions distributed in detector
                        # We reuse the transverse positions
                        positions = np.random.uniform(-side, side, size=(n_events, 2))
                        xys = np.array([
                            (positions[:, 0] + np.full(n_events, xy[0])),
                            (positions[:, 1] + np.full(n_events, xy[1]))
                        ])
                        # We broadcast z (shape: (M,)) to be reused across the N events
                        zs = np.broadcast_to(z[:, None], (len(z), n_events))

                        # These objects are enu and wgt for every event in N, for the M z positions
                        enus, wgts = calc_enu_wgt(branches, xys, zs) # shape: (M, N)
                        full_wgts = nimpwt * wgts * ((2.0*side)**2) # accounting for detector area

                        # Make masks to fill out the necessary histograms
                        mask_type_numu    = (ntype == pdgs["numu"])
                        mask_type_numubar = (ntype == pdgs["numubar"])
                        mask_type_nue     = (ntype == pdgs["nue"])
                        mask_type_nuebar  = (ntype == pdgs["nuebar"])
                        
                        mask_parent_pion  = (np.abs(ptype) == pdgs["piplus"])
                        mask_parent_kaon  = (np.abs(ptype) == pdgs["kplus"])
                        mask_parent_muon  = (np.abs(ptype) == pdgs["muminus"])
                        mask_parent_kzero = (np.abs(ptype) == pdgs["k0l"])

                        # Some magic to get the first instance of "HadronInelastic" from awkward array
                        # The amount of setup needed to get pyarrow, so we can call find_substring...
                        awkward_first_inelastic = ak.str.find_substring(proc, "HadronInelastic")
                        # argmax takes the first _ancestor_ index where _string_ index from find_substring > -1
                        first_inelastic = (ak.argmax(awkward_first_inelastic, axis=-1)).to_numpy()
                        
                        # And now, extract the PDGs from the ancestor arrays
                        first_inelastic_pdg = np.array([
                            row[i] for row, i in zip(apdg, first_inelastic)
                        ])
                        
                        mask_sec_pimu     = \
                            (abs(first_inelastic_pdg) == pdgs["piplus"]) & (abs(ptype) == pdgs["muminus"] )
                        mask_sec_pinomu   = \
                            (abs(first_inelastic_pdg) == pdgs["piplus"]) & (abs(ptype) != pdgs["muminus"] )
                        mask_sec_kzero    = \
                            (abs(first_inelastic_pdg) == pdgs["k0l"])
                        mask_sec_kaon     = \
                            (abs(first_inelastic_pdg) == pdgs["kplus"])
                        mask_sec_nucleon  = (first_inelastic_pdg == pdgs["neutron"]) | \
                            (first_inelastic_pdg == pdgs["proton"])
                        
                        # Build a dictionary of histogram type <-- mask to use
                        mask_dict = {
                            'hFlux': {
                                'numu'   : mask_type_numu,
                                'numubar': mask_type_numubar,
                                'nue'    : mask_type_nue,
                                'nuebar' : mask_type_nuebar
                            } ,
                            'hparent': {
                                'numu'   : {
                                    'muon' : mask_type_numu    & mask_parent_muon,
                                    'pion' : mask_type_numu    & mask_parent_pion,
                                    'kaon' : mask_type_numu    & mask_parent_kaon,
                                    'kzero': mask_type_numu    & mask_parent_kzero,
                                },
                                'numubar': {
                                    'muon' : mask_type_numubar & mask_parent_muon,
                                    'pion' : mask_type_numubar & mask_parent_pion,
                                    'kaon' : mask_type_numubar & mask_parent_kaon,
                                    'kzero': mask_type_numubar & mask_parent_kzero,
                                },
                                'nue'    : {
                                    'muon' : mask_type_nue     & mask_parent_muon,
                                    'pion' : mask_type_nue     & mask_parent_pion,
                                    'kaon' : mask_type_nue     & mask_parent_kaon,
                                    'kzero': mask_type_nue     & mask_parent_kzero,
                                },
                                'nuebar' : {
                                    'muon' : mask_type_nuebar  & mask_parent_muon,
                                    'pion' : mask_type_nuebar  & mask_parent_pion,
                                    'kaon' : mask_type_nuebar  & mask_parent_kaon,
                                    'kzero': mask_type_nuebar  & mask_parent_kzero,
                                }
                            } ,
                            'hsec': {
                                'numu'   : {
                                    'pimu'   : mask_type_numu    & mask_sec_pimu,
                                    'pinomu' : mask_type_numu    & mask_sec_pinomu,
                                    'kzero'  : mask_type_numu    & mask_sec_kzero,
                                    'kaon'   : mask_type_numu    & mask_sec_kaon,
                                    'nucleon': mask_type_numu    & mask_sec_nucleon
                                },
                                'numubar': {
                                    'pimu'   : mask_type_numubar & mask_sec_pimu,
                                    'pinomu' : mask_type_numubar & mask_sec_pinomu,
                                    'kzero'  : mask_type_numubar & mask_sec_kzero,
                                    'kaon'   : mask_type_numubar & mask_sec_kaon,
                                    'nucleon': mask_type_numubar & mask_sec_nucleon
                                },
                                'nue'    : {
                                    'pimu'   : mask_type_nue     & mask_sec_pimu,
                                    'pinomu' : mask_type_nue     & mask_sec_pinomu,
                                    'kzero'  : mask_type_nue     & mask_sec_kzero,
                                    'kaon'   : mask_type_nue     & mask_sec_kaon,
                                    'nucleon': mask_type_nue     & mask_sec_nucleon
                                },
                                'nuebar' : {
                                    'pimu'   : mask_type_nuebar  & mask_sec_pimu,
                                    'pinomu' : mask_type_nuebar  & mask_sec_pinomu,
                                    'kzero'  : mask_type_nuebar  & mask_sec_kzero,
                                    'kaon'   : mask_type_nuebar  & mask_sec_kaon,
                                    'nucleon': mask_type_nuebar  & mask_sec_nucleon
                                }
                            }
                        }

                        # Get number of eFlux. N entries == how many elements in masked array of ones == sum of this masked array
                        nent = np.ones(enus.shape[1])
                        
                        # Fill histograms.
                        for htype, hitem in mask_dict.items():
                            for flav, fitem in hitem.items(): # fitem is a mask, or dict of masks.
                                if isinstance(fitem, np.ndarray):
                                    for ret_class, enu, wgt in zip(full_classes,
                                                                   enus, full_wgts):
                                        ret_class.data[htype][flav][0] += (np.histogram(
                                            enu[fitem], bins=enu_bins, weights=wgt[fitem]
                                        ))[0]
                                        ret_class.data[htype][flav][1] += (np.histogram(
                                            enu[fitem], bins=enu_bins, weights=wgt[fitem]**2
                                        ))[0]
                                        ret_class.data[htype][flav][2] += np.sum(nent[fitem])
                                                
                                elif isinstance(fitem, dict):
                                    for anc, pitem in fitem.items():
                                        for ret_class, enu, wgt in zip(full_classes,
                                                                       enus, full_wgts):
                                            ret_class.data[htype][flav][anc][0] += (np.histogram(
                                                enu[pitem], bins=enu_bins, weights=wgt[pitem]
                                            ))[0]
                                            ret_class.data[htype][flav][anc][1] += (np.histogram(
                                                enu[pitem], bins=enu_bins, weights=wgt[pitem]**2
                                            ))[0]
                                            ret_class.data[htype][flav][anc][2] += np.sum(nent[pitem])
                    
                except Exception as e:
                    print(Fore.RED + f"Skipping file {q}, exception {e}" + Fore.RESET)

                    # remove this file's POT
                    NPOT  -= NAPOT

                pbar.update(1)
    
    result.put(full_classes, block=False)
    return

def main(args):
    NTHREADS = args.threads
    if( NTHREADS > NPROCESS_MAX ):
        print(Fore.RED + f"""You have asked for {NTHREADS} threads which is over the max limit.
This is bad for performance (especially on a gpvm...). Reducing down to {NPROCESS_MAX} threads.""" + Fore.RESET)
        NTHREADS = NPROCESS_MAX

    input_files = [ p for p in Path(args.inpath).glob("*.dk2nu.root") ]
    NFILES_TOTAL = len( input_files )
        
    print(Fore.GREEN + f"Processing {NFILES_TOTAL} dk2nu files over {NTHREADS} threads" + Fore.RESET)
    NFILE_PER_THREAD = int(NFILES_TOTAL / NTHREADS)
    NFILE_REMAINDER  = NFILES_TOTAL - NFILE_PER_THREAD * NTHREADS

    det = np.array([ args.detector[0], args.detector[1], args.detector[2] ])
    print(Fore.GREEN + f"Using detector with origin ({det[0]}, {det[1]}, {det[2]}) cm and half side {args.size} cm" + Fore.RESET)
    zpoints = np.array([det[2] + z for z in args.zpos])

    print(Fore.GREEN + f"Evaluating the following z positions (in detector coords, cm):" + Fore.RESET)
    print(Fore.CYAN + f"{args.zpos}" + Fore.RESET)
    print(Fore.GREEN + f"which translates to the z positions (in beam coords, cm):" + Fore.RESET)
    print(Fore.CYAN + f"{zpoints}" + Fore.RESET)

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
                                                            NTHREADS_PROCESSED,
                                                            det[:2], args.size,
                                                            zpoints)
                                                     ))
            file_queue.put(None) # signal end
            file_queue = multiprocessing.Queue()

            gidx = (NTHREADS_PROCESSED-1) % 10
            if NTHREADS_PROCESSED < 14 and NTHREADS_PROCESSED > 10:
                gidx = 3
            gidx = min(gidx, 3)
            grammar_postfix = grammar[gidx]
            
            print(Fore.GREEN + f"Adding {NFILES_PROCESSED} files to the {NTHREADS_PROCESSED}" +\
                  f"{grammar_postfix} thread..." + Fore.RESET)

            NFILES_PROCESSED = 0
            NTHREADS_PROCESSED += 1
            NFILES_TARGET = NFILE_PER_THREAD+1 if NTHREADS_PROCESSED < NFILE_REMAINDER else NFILE_PER_THREAD

    # Start spawning the processes.
    print(Fore.YELLOW + "Spawning processes with 1s delay.." + Fore.RESET)
    for ip, proc in enumerate(processes):
        gidx = ip % 10
        pip = ip+1
        if pip < 14 and pip > 10:
            gidx = 3
        gidx = min(gidx, 3)
        grammar_postfix = grammar[gidx]
        time.sleep(1)
        #print(Fore.CYAN + f"Spawning {pip}{grammar_postfix} process." + Fore.RESET)
        proc.start()

    # Wait till they're all done
    results = []
    for _ in processes:
        results.append(hist_queue.get())
    for proc in processes:
        proc.join()

    # Get the histograms from all the queue objects, hadd them together
    final_classes = [BeamHistClass() for _ in range(len(args.zpos))]

    for thread_result in results:
        for i in range(len(args.zpos)):
            final_classes[i].Fill(thread_result[i].data)

    # Prepare some sugar for writing
    dirnames = ["z{0:03d}".format(int(zp)) for zp in args.zpos]
    suffixes = [", z_{{det}} = {0} cm".format(int(zp)) for zp in args.zpos]

    # Write the output into a directory.
    with uproot.recreate(args.output) as fuout:

        for (fc, dname, sfx) in zip(final_classes, dirnames, suffixes):
            det = fuout.mkdir(dname)

            # Metadata.
            det["hPOT"]  = np.histogram([fc.data['POT']], bins=1)
            area = (2.0 * args.size)**2
            det["hArea"] = np.histogram([area], bins=1)

            det["x"] = np.histogram([args.detector[0]], bins=1)
            det["y"] = np.histogram([args.detector[1]], bins=1)

            for flav in ["nue", "nuebar", "numu", "numubar"]:
                dflav = det.mkdir(flav)
                pdflav = dflav.mkdir("Flux by parent")
                sdflav = dflav.mkdir("Flux by secondary")
                
                # Cumulative fluxes.
                dflav["Flux"] = construct_uproot_hist(
                    fc.data["hFlux"][flav][0],
                    bins = enu_bins,
                    sumw2s  = fc.data["hFlux"][flav][1],
                    entries = fc.data["hFlux"][flav][2],
                    name = "Flux", title = f"{flav} (total){sfx}",
                    axis_titles = ("Energy (GeV)", f"Flux ({flav} / 50MeV)")
                )

                # Flux by parent.
                for par in ["muon", "pion", "kzero", "kaon"]:
                    pdflav[f"{flav} from {par}"] = construct_uproot_hist(
                        fc.data["hparent"][flav][par][0],
                        bins = enu_bins,
                        sumw2s  = fc.data["hparent"][flav][par][1],
                        entries = fc.data["hparent"][flav][par][2],
                        name = f"{flav} from {par}", title = f"...->{par}->{flav}{sfx}",
                        axis_titles = ("Energy (GeV)", f"Flux ({flav} / 50MeV)")
                    )

                # Flux by secondary.
                for sec, psec in zip(
                        ["pimu", "pinomu", "kzero", "kaon", "nucleon"],
                        ["pion->...->muon", "pion->...->(not-muon)",
                         "kzero->...", "kaon->...", "nucleon->..."]
                ):
                    sdflav[f"{flav} from {sec}"] = construct_uproot_hist(
                        fc.data["hsec"][flav][sec][0],
                        bins = enu_bins,
                        sumw2s  = fc.data["hsec"][flav][sec][1],
                        entries = fc.data["hsec"][flav][sec][2],
                        name = f"{flav} from {sec}", title = f"primary->{psec}->{flav}{sfx}",
                        axis_titles = ("Energy (GeV)", f"Flux ({flav} / 50MeV)")
                    )

if __name__ == "__main__":
    print(Fore.YELLOW + "Hello world!" + Fore.RESET)

    parser = argparse.ArgumentParser(description=Fore.CYAN+"A script to extract fluxes from dk2nu files."+Fore.RESET, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-i', '--inpath', type=str, required=True, help=Fore.RED+"I will read all the files in this directory."+Fore.RESET)
    parser.add_argument('-o', "--output", type=str, default="./out.root", help=Fore.YELLOW+"Output file"+Fore.RESET)
    parser.add_argument('-t', '--threads', type=int, default=8, help=Fore.YELLOW+f"Number of concurrent threads. Maximum {NPROCESS_MAX}"+Fore.RESET)
    parser.add_argument("--detector", type=float, nargs=3, required=True, help=Fore.RED+"Detector origin in beamline coordinates (cm)"+Fore.RESET)
    parser.add_argument('-s', "--size", type=float, required=True, help=Fore.RED+"Half side of the square detector face (cm)"+Fore.RESET)
    parser.add_argument('-z', "--zpos", type=float, nargs='+', required=True, help=Fore.YELLOW+"An array of z positions (cm, detector coordinates) to evaluate fluxes at."+Fore.RESET)
    args = parser.parse_args()

    main(args)
    
    print(Fore.YELLOW + "Goodbye world!" + Fore.RESET)
