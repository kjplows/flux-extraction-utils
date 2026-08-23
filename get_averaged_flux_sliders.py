import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse

from scipy.interpolate import RectBivariateSpline
from matplotlib.widgets import Slider, TextBox, Button

import uproot
from uproot.writing.identify import to_TH1x, to_TAxis

# A wrapper around uproot.writing.identify.to_TH1x
# note that to_TH1x expects first and last bin under/overflow
def construct_uproot_hist(data, bins=None, entries=None, sumw2s=None, name=None, title=None, axis_titles=None):

    uname="foo" if name is None else name
    uentries = 0 if entries is None else entries
    xcentres = 0.5 * (bins[1:] + bins[:-1])
    nsw2s = sumw2s if sumw2s is not None else data**2
    
    return to_TH1x(
        fName    = uname,
        fTitle   = title,
        data     = np.concat([[0.0], data, [0.0]]),
        fEntries = uentries,
        fSumw2   = np.concat([[0.0], nsw2s, [0.0]]),
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


# ============================================================
# Configuration
# ============================================================

parser = argparse.ArgumentParser(description='''I am an interactive display of the SBND flux. Just vgive me a file to grab the splines from.''')
parser.add_argument('-i', '--input', type=str, required=True, help="Input HDF5")
parser.add_argument('-f', '--flavour', type=str, default='numu',
                    choices=['numu', 'nue', 'numubar', 'nuebar'], help="Which flux to interpolate.")
args = parser.parse_args()

H5FILE = Path(args.input).resolve()
FLAVOUR = args.flavour


# ============================================================
# Load data and construct splines
# ============================================================

with h5py.File(H5FILE, "r") as fhin:

    E_edges = fhin["Energy bins"][:][:61]

    # These should ideally be stored in the HDF5 too
    #xpos = fhin["x positions"][:] # (K,)
    #ypos = fhin["y positions"][:] # (K,)
    xpos = np.arange(-210, 211, 20)
    ypos = np.arange(-210, 211, 20)

    data = fhin[FLAVOUR]["Data"][:] # shape: (N, K, K)


print(f"Loaded {data.shape[0]} energy bins")
print(f"Spatial grid: {len(xpos)} x {len(ypos)}")


splines = []

for ibin in range(data.shape[0]):

    spline = RectBivariateSpline(
        xpos,
        ypos,
        data[ibin],
        kx=3,
        ky=3,
        s=0
    )

    splines.append(spline)


# ============================================================
# Flux averaging
# ============================================================

def average_flux(xmin, xmax, ymin, ymax):

    area = (xmax - xmin) * (ymax - ymin)

    if area <= 0:
        return np.zeros(len(splines))

    return np.array([
        spline.integral(xmin, xmax, ymin, ymax) / area
        for spline in splines
    ])


# ============================================================
# Initial FV
# ============================================================

initial = {
    "xmin": -190,
    "xmax": -10,
    "ymin": -190,
    "ymax": 190,
}

phi = average_flux(**initial)


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

plt.subplots_adjust(bottom=0.32)

stairs = ax.stairs(
    phi[:60],
    E_edges,
    color='xkcd:royal blue',
    lw=2.0
)

ax.set_xlabel(r"$E_\nu$ [GeV]")
ax.set_ylabel(
    r"Average $\Phi_\nu$ / POT / cm$^2$ / GeV"
)

ax.set_title("Fiducial-volume averaged flux")


# ============================================================
# Sliders + text boxes
# ============================================================

ax_xmin = fig.add_axes([0.15, 0.22, 0.55, 0.03])
ax_xmin_text = fig.add_axes([0.73, 0.215, 0.10, 0.04])

ax_xmax = fig.add_axes([0.15, 0.17, 0.55, 0.03])
ax_xmax_text = fig.add_axes([0.73, 0.165, 0.10, 0.04])

ax_ymin = fig.add_axes([0.15, 0.12, 0.55, 0.03])
ax_ymin_text = fig.add_axes([0.73, 0.115, 0.10, 0.04])

ax_ymax = fig.add_axes([0.15, 0.07, 0.55, 0.03])
ax_ymax_text = fig.add_axes([0.73, 0.065, 0.10, 0.04])


slider_xmin = Slider(
    ax_xmin, "xmin",
    xpos[0], xpos[-1],
    valinit=initial["xmin"],
    valstep=1.0,
)

slider_xmax = Slider(
    ax_xmax, "xmax",
    xpos[0], xpos[-1],
    valinit=initial["xmax"],
    valstep=1.0,
)

slider_ymin = Slider(
    ax_ymin, "ymin",
    ypos[0], ypos[-1],
    valinit=initial["ymin"],
    valstep=1.0,
)

slider_ymax = Slider(
    ax_ymax, "ymax",
    ypos[0], ypos[-1],
    valinit=initial["ymax"],
    valstep=1.0,
)


box_xmin = TextBox(ax_xmin_text, "", initial=str(initial["xmin"]))
box_xmax = TextBox(ax_xmax_text, "", initial=str(initial["xmax"]))
box_ymin = TextBox(ax_ymin_text, "", initial=str(initial["ymin"]))
box_ymax = TextBox(ax_ymax_text, "", initial=str(initial["ymax"]))

# ============================================================
# Export button for saving histograms!
# ============================================================
ax_export = fig.add_axes([0.73, 0.01, 0.12, 0.04])
button_export = Button(ax_export, "Export")

def export_flux(event):
    xmin, xmax = slider_xmin.val, slider_xmax.val
    ymin, ymax = slider_ymin.val, slider_ymax.val
    phi = average_flux(xmin, xmax, ymin, ymax)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    timestamp = timestamp.replace(":", "-")

    filename = f"flux_export_{timestamp}.root"
    # There is absolutely no reason for 2% other than that that's what my numu uncertainties are
    hflux = construct_uproot_hist(phi[:60], bins=E_edges, name="Flux", sumw2s = (0.02 * phi[:60])**2,
                                  title = f"(xmin, xmax), (ymin, ymax) = ({xmin},{xmax}), ({ymin},{ymax})",
                                  axis_titles=("Enu [GeV]", "Flux density (nu / POT / cm2 / bin)"))

    with uproot.recreate(filename) as fuout:
        fuout["Flux"] = hflux
    print(f"Exported flux to ROOT file {filename}")

button_export.on_clicked(export_flux)

# ============================================================
# Updating
# ============================================================

updating = False


def update(val=None):
    global stairs, updating

    if updating:
        return

    xmin = slider_xmin.val
    xmax = slider_xmax.val
    ymin = slider_ymin.val
    ymax = slider_ymax.val

    if xmin >= xmax or ymin >= ymax:
        return

    updating = True

    phi = average_flux(xmin, xmax, ymin, ymax)

    #stairs.remove()
    #stairs = ax.stairs(phi, E_edges)
    #line.set_ydata(phi)

    box_xmin.set_val(f"{xmin:.1f}")
    box_xmax.set_val(f"{xmax:.1f}")
    box_ymin.set_val(f"{ymin:.1f}")
    box_ymax.set_val(f"{ymax:.1f}")

    ax.set_title(
        f"x = [{xmin:.1f}, {xmax:.1f}] cm, "
        f"y = [{ymin:.1f}, {ymax:.1f}] cm"
    )

    updating = False
    fig.canvas.draw_idle()


# Slider -> update
slider_xmin.on_changed(update)
slider_xmax.on_changed(update)
slider_ymin.on_changed(update)
slider_ymax.on_changed(update)


# Text box -> slider
def set_xmin(text):
    try:
        slider_xmin.set_val(float(text))
    except ValueError:
        pass

def set_xmax(text):
    try:
        slider_xmax.set_val(float(text))
    except ValueError:
        pass

def set_ymin(text):
    try:
        slider_ymin.set_val(float(text))
    except ValueError:
        pass

def set_ymax(text):
    try:
        slider_ymax.set_val(float(text))
    except ValueError:
        pass

box_xmin.on_submit(set_xmin)
box_xmax.on_submit(set_xmax)
box_ymin.on_submit(set_ymin)
box_ymax.on_submit(set_ymax)
        
# ============================================================
# Update callback
# ============================================================

def update(val):

    global stairs

    xmin = slider_xmin.val
    xmax = slider_xmax.val

    ymin = slider_ymin.val
    ymax = slider_ymax.val

    if xmin >= xmax or ymin >= ymax:
        return

    phi = average_flux(
        xmin,
        xmax,
        ymin,
        ymax,
    )

    # Update the stairs artist
    #stairs.set_data(phi, E_edges)
    stairs.remove()
    stairs = ax.stairs(
        phi[:60],
        E_edges,
        color='xkcd:royal blue',
        lw=2.0
    )

    ax.set_title(
        f"x = [{xmin:.0f}, {xmax:.0f}] cm, "
        f"y = [{ymin:.0f}, {ymax:.0f}] cm"
    )

    fig.canvas.draw_idle()


slider_xmin.on_changed(update)
slider_xmax.on_changed(update)

slider_ymin.on_changed(update)
slider_ymax.on_changed(update)


plt.show()
