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

cutoffs = {
    'numu': 100,
    'numubar': 60,
    'nue': 80,
    'nuebar': 80
}
colours = {
    'numu' : ('xkcd:royal blue', 'xkcd:cornflower'),
    'numubar': ('xkcd:brick', 'xkcd:scarlet'),
    'nue': ('xkcd:forest green', 'xkcd:irish green'),
    'nuebar': ('xkcd:pumpkin', 'xkcd:goldenrod')
}

# ============================================================
# ROOT histogram helper
# ============================================================

# A wrapper around uproot.writing.identify.to_TH1x
# Note that to_TH1x expects first and last bin under/overflow.
def construct_uproot_hist(
    data,
    bins=None,
    entries=None,
    sumw2s=None,
    name=None,
    title=None,
    axis_titles=None,
):
    uname = "foo" if name is None else name
    uentries = 0 if entries is None else entries

    xcentres = 0.5 * (bins[1:] + bins[:-1])
    nsw2s = sumw2s if sumw2s is not None else data**2

    return to_TH1x(
        fName=uname,
        fTitle=title,
        data=np.concat([[0.0], data, [0.0]]),
        fEntries=uentries,
        fSumw2=np.concat([[0.0], nsw2s, [0.0]]),
        fTsumw=np.sum(data),
        fTsumw2=np.sum(nsw2s),
        fXaxis=to_TAxis(
            fName="xaxis",
            fTitle=axis_titles[0],
            fNbins=data.shape[0],
            fXmin=bins[0],
            fXmax=bins[-1],
        ),
        fYaxis=to_TAxis(
            fName="yaxis",
            fTitle=axis_titles[1],
            fNbins=1,
            fXmin=max(0.0, 0.95 * np.min(data)),
            fXmax=1.05 * np.max(data),
        ),
        fTsumwx=np.sum(data * xcentres),
        fTsumwx2=np.sum(data * xcentres**2),
    )


# ============================================================
# Configuration
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Interactive display of the SBND flux. "
        "Reads spline predictions from an HDF5 file."
    )
)

parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Input HDF5",
)

parser.add_argument(
    "-f",
    "--flavour",
    type=str,
    default="numu",
    choices=["numu", "nue", "numubar", "nuebar"],
    help="Which flux to interpolate.",
)

args = parser.parse_args()

H5FILE = Path(args.input).resolve()
FLAVOUR = args.flavour


# ============================================================
# Load data
# ============================================================

with h5py.File(H5FILE, "r") as fhin:
    
    E_edges = fhin["Energy bins"][:(cutoffs[FLAVOUR]+1)]

    # Spatial grid.
    #
    xpos = fhin["x coordinates"][:]
    ypos = fhin["y coordinates"][:]
    zpos = fhin["z coordinates"][:]

    # Expected shape:
    #
    #   (N_energy, Nx, Ny, Nz)
    #
    pred = fhin[FLAVOUR]["Prediction"][:]

    wig = fhin[FLAVOUR]["Wiggle systematic"][:]
    res = fhin[FLAVOUR]["Fractional deviation (spline)"][:]

    # Sanity checks.
    if pred.ndim != 4:
        raise ValueError(
            f"Expected Prediction to have 4 dimensions "
            f"(E, x, y, z), got shape {pred.shape}"
        )

    if pred.shape[-1] != len(zpos):
        raise ValueError(
            f"Prediction has {pred.shape[-1]} z points, "
            f"but zpos contains {len(zpos)} points."
        )

    if pred.shape[1] != len(xpos) or pred.shape[2] != len(ypos):
        raise ValueError(
            f"Prediction spatial shape is "
            f"{pred.shape[1:3]}, expected "
            f"({len(xpos)}, {len(ypos)})."
        )

    # --------------------------------------------------------
    # Systematic
    #
    # Combine residual + wiggle in quadrature and take the
    # maximum over spatial voxels.
    #
    # The resulting frac_syst is one number per energy bin.
    # --------------------------------------------------------

    both = np.sqrt(wig**2 + res**2)

    # Collapse x, y, and z.
    both_flat = both.reshape(
        (both.shape[0], -1)
    )

    frac_syst = np.max(
        both_flat,
        axis=1,
    )


print(f"Loaded {pred.shape[0]} energy bins")
print(
    f"Spatial grid: "
    f"{len(xpos)} x {len(ypos)} x {len(zpos)}"
)
print(
    f"z range: "
    f"{zpos[0]:.0f} -- {zpos[-1]:.0f} cm"
)


# ============================================================
# Construct 2D splines
#
# One spline for every (energy, z) point.
#
# splines[ibin][iz]
# ============================================================

splines = []
low_splines = []
high_splines = []

with np.errstate(divide='ignore', invalid='ignore'):
    for ibin in range(pred.shape[0]):

        energy_splines = []
        energy_low_splines = []
        energy_high_splines = []
        
        for iz in range(len(zpos)):
            
            spline = RectBivariateSpline(
                xpos,
                ypos,
                pred[ibin, :, :, iz],
                kx=3,
                ky=3,
                s=0,
            )
            
            energy_splines.append(spline)
            
            lspline = RectBivariateSpline(
                xpos,
                ypos,
                pred[ibin, :, :, iz] * (1 - frac_syst[ibin]),
                kx=3,
                ky=3,
                s=0,
            )
            
            energy_low_splines.append(lspline)
            
            hspline = RectBivariateSpline(
                xpos,
                ypos,
                pred[ibin, :, :, iz] * (1 + frac_syst[ibin]),
                kx=3,
                ky=3,
                s=0,
            )
            
            energy_high_splines.append(hspline)
            
        splines.append(energy_splines)
        low_splines.append(energy_low_splines)
        high_splines.append(energy_high_splines)


# ============================================================
# Helper: integrate one stack of 2D splines over z
# ============================================================

def integrate_z(
    spline_stack,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
):
    """
    Integrate a stack of 2D splines over:

        xmin <= x <= xmax
        ymin <= y <= ymax
        zmin <= z <= zmax

    spline_stack contains one 2D spline per z grid point.

    The z dependence is assumed sufficiently linear that
    trapezoidal integration is appropriate.

    The requested z endpoints need not coincide with the
    stored z grid; they are linearly interpolated.
    """

    # --------------------------------------------------------
    # Find the z grid points surrounding zmin and zmax.
    # --------------------------------------------------------

    if zmin >= zmax:
        return 0.0

    if zmin < zpos[0] or zmax > zpos[-1]:
        raise ValueError(
            f"Requested z range [{zmin}, {zmax}] lies "
            f"outside available range "
            f"[{zpos[0]}, {zpos[-1]}]."
        )

    # Index of the grid point immediately below each endpoint.
    izmin = np.searchsorted(zpos, zmin, side="right") - 1
    izmax = np.searchsorted(zpos, zmax, side="right") - 1

    # If the endpoint is exactly at z=500, there is no point
    # above it, so handle that case explicitly.
    izmin = min(izmin, len(zpos) - 1)
    izmax = min(izmax, len(zpos) - 1)

    # --------------------------------------------------------
    # Evaluate the XY integral at all relevant z grid points.
    # --------------------------------------------------------

    z_values = []
    xy_integrals = []

    # If zmin is exactly on a grid point, use that point.
    if np.isclose(zmin, zpos[izmin]):

        z_values.append(zpos[izmin])

        xy_integrals.append(
            spline_stack[izmin].integral(
                xmin,
                xmax,
                ymin,
                ymax,
            )
        )

        first_interior = izmin + 1

    else:

        # Interpolate the XY integral between the two
        # surrounding z points.
        z0 = zpos[izmin]
        z1 = zpos[izmin + 1]

        I0 = spline_stack[izmin].integral(
            xmin,
            xmax,
            ymin,
            ymax,
        )

        I1 = spline_stack[izmin + 1].integral(
            xmin,
            xmax,
            ymin,
            ymax,
        )

        t = (zmin - z0) / (z1 - z0)

        Imin = I0 * (1 - t) + I1 * t

        z_values.append(zmin)
        xy_integrals.append(Imin)

        first_interior = izmin + 1

    # --------------------------------------------------------
    # Interior grid points.
    # --------------------------------------------------------

    for iz in range(first_interior, izmax + 1):

        # Don't add zmax here if it is exactly a grid point;
        # that gets handled below.
        if iz < izmax or not np.isclose(zmax, zpos[iz]):

            z_values.append(zpos[iz])

            xy_integrals.append(
                spline_stack[iz].integral(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                )
            )

    # --------------------------------------------------------
    # zmax endpoint.
    # --------------------------------------------------------

    if np.isclose(zmax, zpos[izmax]):

        # Avoid duplicating the point if it was just added.
        if not z_values or not np.isclose(z_values[-1], zmax):

            z_values.append(zmax)

            xy_integrals.append(
                spline_stack[izmax].integral(
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                )
            )

    else:

        z0 = zpos[izmax]
        z1 = zpos[izmax + 1]

        I0 = spline_stack[izmax].integral(
            xmin,
            xmax,
            ymin,
            ymax,
        )

        I1 = spline_stack[izmax + 1].integral(
            xmin,
            xmax,
            ymin,
            ymax,
        )

        t = (zmax - z0) / (z1 - z0)

        Imax = I0 * (1 - t) + I1 * t

        z_values.append(zmax)
        xy_integrals.append(Imax)

    z_values = np.asarray(z_values)
    xy_integrals = np.asarray(xy_integrals)

    # Trapezoidal integration along z.
    return np.trapezoid(
        xy_integrals,
        z_values,
    )


# ============================================================
# Flux averaging
# ============================================================

def average_flux(
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
):

    volume = (
        (xmax - xmin)
        * (ymax - ymin)
        * (zmax - zmin)
    )

    if volume <= 0:
        n = len(splines)

        return (
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
        )

    phi = np.array([
        integrate_z(
            splines[ibin],
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        ) / volume
        for ibin in range(len(splines))
    ])

    phi_low = np.array([
        integrate_z(
            low_splines[ibin],
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        ) / volume
        for ibin in range(len(low_splines))
    ])

    phi_high = np.array([
        integrate_z(
            high_splines[ibin],
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        ) / volume
        for ibin in range(len(high_splines))
    ])

    return phi, phi_low, phi_high


# ============================================================
# Initial fiducial volume
# ============================================================

initial = {
    "xmin": -190,
    "xmax": -5,
    "ymin": -190,
    "ymax": 190,
    "zmin": 10,
    "zmax": 450,
}

phi, phi_low, phi_high = average_flux(**initial)


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

plt.subplots_adjust(bottom=0.40)
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(left=0.05)
plt.subplots_adjust(right=0.95)

# Draw systematic band first.
band = ax.fill_between(
    E_edges[:-1],
    phi_low[:cutoffs[FLAVOUR]],
    phi_high[:cutoffs[FLAVOUR]],
    step="post",
    color=colours[FLAVOUR][1],
    alpha=0.3,
)

# Draw central prediction on top.
stairs = ax.stairs(
    phi[:cutoffs[FLAVOUR]],
    E_edges,
    color=colours[FLAVOUR][0],
    lw=2.0,
)

ax.set_xlabel(r"$E_\nu$ [GeV]")

ax.set_ylabel(
    r"Average $\Phi_\nu$ / POT / cm$^2$ / GeV"
)

ax.set_title("Fiducial-volume averaged flux")


# ============================================================
# Sliders
# ============================================================

ax_xmin = fig.add_axes([0.15, 0.28, 0.55, 0.03])
ax_xmax = fig.add_axes([0.15, 0.23, 0.55, 0.03])
ax_ymin = fig.add_axes([0.15, 0.18, 0.55, 0.03])
ax_ymax = fig.add_axes([0.15, 0.13, 0.55, 0.03])
ax_zmin = fig.add_axes([0.15, 0.08, 0.55, 0.03])
ax_zmax = fig.add_axes([0.15, 0.03, 0.55, 0.03])


slider_xmin = Slider(
    ax_xmin,
    "xmin",
    xpos[0],
    xpos[-1],
    valinit=initial["xmin"],
    valstep=1.0,
)

slider_xmax = Slider(
    ax_xmax,
    "xmax",
    xpos[0],
    xpos[-1],
    valinit=initial["xmax"],
    valstep=1.0,
)

slider_ymin = Slider(
    ax_ymin,
    "ymin",
    ypos[0],
    ypos[-1],
    valinit=initial["ymin"],
    valstep=1.0,
)

slider_ymax = Slider(
    ax_ymax,
    "ymax",
    ypos[0],
    ypos[-1],
    valinit=initial["ymax"],
    valstep=1.0,
)

slider_zmin = Slider(
    ax_zmin,
    "zmin",
    zpos[0],
    zpos[-1],
    valinit=initial["zmin"],
    valstep=1.0,
)

slider_zmax = Slider(
    ax_zmax,
    "zmax",
    zpos[0],
    zpos[-1],
    valinit=initial["zmax"],
    valstep=1.0,
)


# ============================================================
# Text boxes
# ============================================================

ax_xmin_text = fig.add_axes([0.75, 0.275, 0.10, 0.04])
ax_xmax_text = fig.add_axes([0.75, 0.225, 0.10, 0.04])
ax_ymin_text = fig.add_axes([0.75, 0.175, 0.10, 0.04])
ax_ymax_text = fig.add_axes([0.75, 0.125, 0.10, 0.04])
ax_zmin_text = fig.add_axes([0.75, 0.075, 0.10, 0.04])
ax_zmax_text = fig.add_axes([0.75, 0.025, 0.10, 0.04])


box_xmin = TextBox(
    ax_xmin_text,
    "",
    initial=str(initial["xmin"]),
)

box_xmax = TextBox(
    ax_xmax_text,
    "",
    initial=str(initial["xmax"]),
)

box_ymin = TextBox(
    ax_ymin_text,
    "",
    initial=str(initial["ymin"]),
)

box_ymax = TextBox(
    ax_ymax_text,
    "",
    initial=str(initial["ymax"]),
)

box_zmin = TextBox(
    ax_zmin_text,
    "",
    initial=str(initial["zmin"]),
)

box_zmax = TextBox(
    ax_zmax_text,
    "",
    initial=str(initial["zmax"]),
)


# ============================================================
# Export button
# ============================================================

ax_export = fig.add_axes(
    [0.73, 0.87, 0.12, 0.04]
)
ax_export_path = fig.add_axes(
    [0.73, 0.83, 0.12, 0.04]
)

button_export = Button(
    ax_export,
    "Export",
)
box_export_path = TextBox(
    ax_export_path,
    "Enter path/to/filename...",
    initial=""
)

def export_flux(event):

    xmin = slider_xmin.val
    xmax = slider_xmax.val

    ymin = slider_ymin.val
    ymax = slider_ymax.val

    zmin = slider_zmin.val
    zmax = slider_zmax.val

    phi, phi_low, phi_high = average_flux(
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
    )

    filename = None
    user_path = box_export_path.text.strip()
    if user_path:
        filename = Path(user_path).expanduser()
    else:
        timestamp = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        # Avoid ':' in filenames.
        timestamp = timestamp.replace(":", "-")
        
        filename = f"flux_export_{timestamp}.root"

    # Symmetric uncertainty corresponding to the
    # low/high systematic band.
    errors = 0.5 * (phi_high - phi_low)

    hflux = construct_uproot_hist(
        phi[:cutoffs[FLAVOUR]],
        bins=E_edges,
        name="Flux",
        sumw2s=errors[:cutoffs[FLAVOUR]] ** 2,
        title=(
            f"(x,y,z) = "
            f"([{xmin},{xmax}], "
            f"[{ymin},{ymax}], "
            f"[{zmin},{zmax}]) cm"
        ),
        axis_titles=(
            "Enu [GeV]",
            "Flux density (nu / POT / cm2 / GeV)",
        ),
    )

    with uproot.recreate(filename) as fuout:
        fuout["Flux"] = hflux
        fuout["Volume"] = np.histogram([(xmax-xmin)*(ymax-ymin)*(zmax-zmin)], bins=1)

    print(
        f"Exported flux to ROOT file {filename}"
    )


button_export.on_clicked(export_flux)


# ============================================================
# Updating
# ============================================================

updating = False


def update(val=None):

    global band, stairs, updating

    if updating:
        return

    xmin = slider_xmin.val
    xmax = slider_xmax.val

    ymin = slider_ymin.val
    ymax = slider_ymax.val

    zmin = slider_zmin.val
    zmax = slider_zmax.val

    if (
        xmin >= xmax
        or ymin >= ymax
        or zmin >= zmax
    ):
        return

    updating = True

    # Calculate current FV-averaged flux.
    phi, phi_low, phi_high = average_flux(
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
    )

    # Remove old artists.
    stairs.remove()
    band.remove()

    # Draw systematic band first.
    band = ax.fill_between(
        E_edges[:-1],
        phi_low[:cutoffs[FLAVOUR]],
        phi_high[:cutoffs[FLAVOUR]],
        step="post",
        color=colours[FLAVOUR][1],
        alpha=0.3,
    )

    # Draw central prediction on top.
    stairs = ax.stairs(
        phi[:cutoffs[FLAVOUR]],
        E_edges,
        color=colours[FLAVOUR][0],
        lw=2.0,
    )

    # Keep text boxes synchronized with sliders.
    box_xmin.set_val(f"{xmin:.1f}")
    box_xmax.set_val(f"{xmax:.1f}")

    box_ymin.set_val(f"{ymin:.1f}")
    box_ymax.set_val(f"{ymax:.1f}")

    box_zmin.set_val(f"{zmin:.1f}")
    box_zmax.set_val(f"{zmax:.1f}")

    ax.set_title(
        f"x = [{xmin:.0f}, {xmax:.0f}] cm, "
        f"y = [{ymin:.0f}, {ymax:.0f}] cm, "
        f"z = [{zmin:.0f}, {zmax:.0f}] cm"
    )

    updating = False

    fig.canvas.draw_idle()


# Slider -> update
slider_xmin.on_changed(update)
slider_xmax.on_changed(update)

slider_ymin.on_changed(update)
slider_ymax.on_changed(update)

slider_zmin.on_changed(update)
slider_zmax.on_changed(update)


# ============================================================
# Text box -> slider
# ============================================================

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


def set_zmin(text):
    try:
        slider_zmin.set_val(float(text))
    except ValueError:
        pass


def set_zmax(text):
    try:
        slider_zmax.set_val(float(text))
    except ValueError:
        pass


box_xmin.on_submit(set_xmin)
box_xmax.on_submit(set_xmax)

box_ymin.on_submit(set_ymin)
box_ymax.on_submit(set_ymax)

box_zmin.on_submit(set_zmin)
box_zmax.on_submit(set_zmax)


# ============================================================
# Show
# ============================================================

plt.show()
