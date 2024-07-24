import os
import sys
import shutil
import scipy
import time
import glob
import ipympl

import numpy as np
import xarray as xr


from IPython.display import clear_output
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

# importing matplotlib for plots.
import matplotlib.pyplot as plt

# unit conversion functions

def twotheta_to_q(twotheta, wavelength):
    """
    Converts two-theta value to units of q, based on the wavelength

    Args:
        twotheta (array_like): float or float array of two-theta values
        wavelength (float): X-ray beam wavelength value

    Returns:
        (array_like): twotheta converted to units of q
    """
    twotheta = np.asarray(twotheta)
    wavelength = float(wavelength)
    pre_factor = ((4 * np.pi) / wavelength)
    return pre_factor * np.sin(twotheta / 2)
def q_to_twotheta(q, wavelength):
    """
    Converts q value to units of two-theta, based on the wavelength

    Args:
        q (array_like):  float or float array of q values
        wavelength (float): X-ray beam wavelength value

    Returns:
        (array_like): q converted to units of two-theta
    """
    q = np.asarray(q)
    wavelength = float(wavelength)
    pre_factor = wavelength / (4 * np.pi)
    return 2 * np.arcsin(q * pre_factor)
def q_to_d(q):
    """
    Converts q to units of d

    Args:
        q (array_like): float or float array of q values

    Returns:
        array_like: q converted to units of d
    """
    return (2 * np.pi) / np.asarray(q)
def d_to_q(d):
    """
    Converts d to units of q

    Args:
        d (array_like): float or float array of d values

    Returns:
        array_like: d converted to units of q
    """
    return (2 * np.pi) / np.asarray(d)
def twotheta_to_d(twotheta, wavelength):
    """
    Converts two-theta value to units of d, based on the wavelength

    Args:
        twotheta (array_like): float or float array of two-theta values
        wavelength (float): X-ray beam wavelength value

    Returns:
        (array_like): twotheta converted to units of d
    """
    th = np.asarray(twotheta)/2
    rad = np.radians(th)
    t = 2*np.sin(rad)
    d = (wavelength)/t
    return d
def tth_wl1_to_wl2(tth1,wl1=0.187,wl2=0.4592):
    """
    Converts two-theta value with one wavelength, to two-theta value 
    with another wavelength.

    Args:
        tth1 (array_like): float or float array of two-theta value
        wl1 (float, optional): initial wavelength. Defaults to 0.187.
        wl2 (float, optional): new wavelength. Defaults to 0.4592.

    Returns:
        (array_like): two-theta converted to correspond to new wavelength
    """
    q = twotheta_to_q(np.deg2rad(tth1), wl1)
    return np.rad2deg(q_to_twotheta(q,wl2))

# plotting tools for plotting from cif files

def hkl_plotter(
    line_axes=None,
    stem_axes=None,
    mp_id=None,
    final=False,
    structure=None,
    str_file=None,
    label=None,
    color="C0",
    label_x=0.9,
    label_y=0.9,
    unit="2th_deg",
    radial_range=(1, 16),
    wl=0.77,
    scale=1,
    scale_a=1,
    scale_b=1,
    scale_c=1,
    export_cif_as=None,
    stem=True,
):
    """
    Method for plotting an intensity vs two-theta plot on top of a theoretical 
    hkl theoretical bragg peak stem plot. 

    Args:
        line_axes (Axes, optional): matplotlib.axes object for top plot. Defaults to None.
        stem_axes (Axes, optional): matplotlib.axes object for bottom plot. Defaults to None.
        mp_id (str, optional): material project API id. Defaults to None.
        final (bool, optional): whether to get final structure from mp API. Defaults to False.
        structure (Structure, optional): object containing structure information. Defaults to None.
        str_file (str, optional): CIF or other structure containing file. Defaults to None.
        label (str, optional): plot label. Defaults to None.
        color (str, optional): color for the plot. Defaults to "C0".
        label_x (float, optional): x label for stem plot. Defaults to 0.9.
        label_y (float, optional): y label for stem plot. Defaults to 0.9.           
        unit (str, optional): the unit for x axis. Defaults to "2th_deg".
        radial_range (tuple, optional): range for the x axis. Defaults to (1, 16).
        
        bottom (float, optional): _description_. Defaults to -0.2.
        wl (float, optional): _description_. Defaults to 0.77.
        scale (int, optional): _description_. Defaults to 1.
        scale_a (int, optional): _description_. Defaults to 1.
        scale_b (int, optional): _description_. Defaults to 1.
        scale_c (int, optional): _description_. Defaults to 1.
        export_cif_as (_type_, optional): _description_. Defaults to None.
        stem (bool, optional): _description_. Defaults to True.
        stem_logscale (bool, optional): _description_. Defaults to True.
        da_visible (_type_, optional): _description_. Defaults to None.
    """

    if mp_id is not None:
        from mp_api.client import MPRester

        mpr = MPRester("dHgNQRNYSpuizBPZYYab75iJNMJYCklB")  ###
        structure = mpr.get_structure_by_material_id(mp_id, final=final)[0]
    elif structure is None:
        structure = Structure.from_file(str_file)

    structure.lattice = Lattice.from_parameters(
        a=structure.lattice.abc[0] * scale * scale_a,
        b=structure.lattice.abc[1] * scale * scale_b,
        c=structure.lattice.abc[2] * scale * scale_c,
        alpha=structure.lattice.angles[0],
        beta=structure.lattice.angles[1],
        gamma=structure.lattice.angles[2],
    )

    xrdc = XRDCalculator(wavelength=wl)  ###computes xrd pattern given wavelength , debye scherrer rings, and symmetry precision

    if unit == "q_A^-1":
        ps = xrdc.get_pattern(
            structure,
            scaled=True,
            two_theta_range=np.rad2deg(q_to_twotheta(radial_range, wl)),
        )
        X, Y = twotheta_to_q(np.deg2rad(ps.x), wl), ps.y
    elif unit == "2th_deg":
        ps = xrdc.get_pattern(structure, scaled=True, two_theta_range=radial_range)
        X, Y = ps.x, ps.y
    else:
        ps = xrdc.get_pattern(structure, scaled=True, two_theta_range=radial_range)
        X, Y = ps.x, ps.y

    for axl in line_axes:
        for i in X:
            axl.axvline(x=i, lw=0.6, color=color)
            axl.set_xlim([radial_range[0], radial_range[1]])

    for axs in stem_axes:
        axs_stem = axs.twinx()
        if stem:
            markerline, stemlines, baseline = axs_stem.stem(X, Y, markerfmt=".")
            plt.setp(stemlines, linewidth=0.5, color=color)
            plt.setp(markerline, color=color)
        axs_stem.set_xlim([radial_range[0], radial_range[1]])
        axs_stem.set_yticks([])
        axs_stem.set_ylim(bottom=0.1)
        axs_stem.text(
            label_x, label_y, label, color=color, transform=axs_stem.transAxes
        )

    if export_cif_as is not None:
        structure.to(fmt="cif", filename=export_cif_as)
        
def phase_plotter(
        wl,
        line_axes=[],
        stem_axes=[],
        radial_range=(1, 16),
        stem=True,
        y_shift=0.1,
        phases=[],
        unit="2th_deg",
        mp_ids=[],
):
    """
    Method for plotting an array of phases using the hkl_plotter() method.

    Args:
        wl (float): wavelength
        line_axes (Axes, optional): matplotlib.axes object for top plot. Defaults to [].
        stem_axes (Axes, optional): matplotlib.axes object for bottom plot. Defaults to [].
        radial_range (tuple, optional): range for the x axis. Defaults to (1, 16).
        stem (bool, optional): whether to add stem plot. Defaults to True.
        y_shift (float, optional): y shift between phase labels. Defaults to 0.1.
        phases (array_like, optional): list of dictionaries each representing a phase, in the format `[{"cif": '_cifs/CeO2.cif', "label": "CeO$_2$", "scale": 1}]`. Defaults to [].
        unit (str, optional): the unit for x axis. Defaults to "2th_deg".
        mp_ids (array_like, optional): list of mp_id strings, to be passed to the hkl_plotter method; if for a given phase there is a ciff provided, set corresponding mp_id=None in the mp_id array
    """
    for ep, phase in enumerate(phases):
        hkl_plotter(
            line_axes=line_axes,
            stem_axes=stem_axes,
            str_file=phase["cif_abs_path"],
            label=phase["phase_name"],
            scale=phase["scale"],
            marker="o",
            color="C%d" % ep,
            label_x=1.02,
            label_y=ep * y_shift,
            unit=unit,
            radial_range=radial_range,
            bottom=-0.2,
            wl=wl,
            stem=stem,
            stem_logscale=False,
            mp_id=mp_ids[ep]
        )

# Tools for navigating the gpx object revcieved from GSAS-II analysis

def print_dict_as_tree(struct, indent=0, depth=0):
    """
    Recursively navigate a dictionary and print it as a tree structure.

    Args:
        struct (dict_like): dictionary-like structure to be printed
        indent (int): current level of indentation, use as a "memory" of recursive depth
        depth (int): current depth in the dictionary, 0 represents the first level
    """
    for key, value in struct.items():
        print(' ' * indent + str(depth) + ". " + str(key))
        if isinstance(value, dict):
            print_dict_as_tree(value, indent + 4, depth + 1)
        else:
            print(' ' * (indent + 4) + str(depth) + ". " + str(value))
            
def find_key(struct, key):
    """
    Recursively navigate a dictionary-like structure and search for a key. This
    doesn't work great if the key is redundant, but is ideal for keys which occur 
    only once in the gpx structure.

    Args:
        struct (dict_like): any dictionary, works with GPX      
        key (str): the key to find
        
    Returns:
        result (unknown): the value of the key 
    """
    if isinstance(struct, dict):
        if key in struct:
            return struct[key]
        else:
            for k, v in struct.items():
                result = find_key(v, key)
                if result is not None:
                    return result
    elif isinstance(struct, list):
        for item in struct:
            result = find_key(item, key)
            if result is not None:
                return result
            
def find_allof_key(d, key, results=None, depth=0, min_depth=0, max_depth=np.inf):
    """
    Recursively navigate a dictionary-like structure and search for each instance of a
    key. This returns an array of all the structures found stemming from the key. If 
    it reaches a structure at the lowest level (ie, no subdictionaries are in the 
    structure), containing the key, returns the structure containing the key. The 
    min_depth and max_depth thresholds are useful if there are many results, and you 
    need to filter them.

    Args:
        d (dict_like): any dictionary, works with GPX       
        key (str): the key to find
        results (array_like): tuple list which grows with each instance of key
        depth (int): depth of a given appended structure
        min_depth (int): sets a minimum depth for collected structures
        min_depth (int): analogous to min depth         
        
    Returns:        
        results (array_like): list of tuples, where each first index represents tells the depths, and the second index is the structure found; to access an output structue structure at a given index, do `find_allof_key(...)[index][1]`
    """
    # instantiates results list
    if results is None:
        results = []

    # checks if we have a dict; if so looks for target key in dict, and continues recursively searching dict 
    if isinstance(d, dict):
        if key in d and depth >= min_depth and depth <= max_depth:
            results.append(("Depth " + str(depth), d))
        for key, value in d.items():
            if isinstance(value, dict):
                find_allof_key(value, key, results, depth=depth+1, min_depth=min_depth, max_depth=max_depth)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        find_allof_key(item, key, results, min_depth=min_depth, max_depth=max_depth)

    # checks if we have a list; if so looks for dictionaries in the list and searches them
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, dict):
                find_key(item, key, results, min_depth=min_depth, max_depth=max_depth)

    # trys to perform search on dictionary-like object or list
    else:
        try:
            if key in d and depth >= min_depth and depth <= max_depth:
                results.append(("Depth " + str(depth), d))
            for key, value in d.items():
                if isinstance(value, dict):
                    find_key(value, key, results, depth=depth+1, min_depth=min_depth, max_depth=max_depth)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            find_key(item, key, results, depth=depth+1, min_depth=min_depth, max_depth=max_depth)
        except:
            print("Not a valid dictionary.")
            return 0
    return results

def get_valid_phases(gpx):
    """
    Args:
        gpx (dict_like): any dictionary, but meant for GPX  
        
    Returns:        
        phases (array_like): list of strs represent the keys corresponding to each phase fitted in the GSAS-II refinement
    """
    try:
        phases = gpx['Phases']
        return list(phases.keys())[1:]
    except:
        print("Invalid gpx object.")
        return 0

def get_cell_consts(gpx, phase, include_angles=False):
    """
    Navigates a gpx data output to search for a phase, then creates and
    returns a dictionary containing lattice constant information.

    Args:
        gpx (dict_like): any dictionary, but meant for GPX  
        phase (str): the phase tag
        include_angles (bool, optional): flag which specified whether to include angle constants. Defaults to False.

    Returns:
        const_dict (dict_like): dictionary with lattice constants as keys 
    """
    try:
        phase_info = gpx['Phases'][phase]
        constants = find_key(phase_info, 'Cell')
    except:
        print("Invalid gpx object.")
        return 0
    const_dict = {}
    const_dict['a'] = constants[1]
    const_dict['b'] = constants[2]
    const_dict['c'] = constants[3]
    if include_angles:
        const_dict['alpha'] = constants[4]
        const_dict['beta'] = constants[5]
        const_dict['gamma'] = constants[6]
    return const_dict

def print_all_cell_consts(gpx):
    """
        Prints out each phase, and its respective lattice constants.

        gpx (dict_like): any dictionary, but meant for GPX       
    """
    valid_phases = get_valid_phases(gpx)
    for phase in valid_phases:
        print(phase)
        cell_constants = get_cell_consts(gpx, phase)
        for const in cell_constants:
            print(' ' * 4 + const + ':')
            print(' ' * 8 + str(cell_constants[const]))


def gpx_plotter(
    gpx,
    line_axes,
    stem_axes,
    phases,
    radial_range,
    marker="o",
    stem=True,
    unit="d",
    plot_unit="q_A^-1",
    y_shift=0.1
):
    """
    Methods which plots intensity points at the given d location (X) with 
    normalized intensity (Y). It gets these values from the gpx object
    created by the GSAS-II extender functions.

    Args:
        gpx (dict_like): GSAS project object, has dictionary structure
        line_axes (_type_): axes where to plot points representing peaks
        stem_axes (_type_): axes where to plot vertical lines representing peaks
        phases (_type_): string array of phases (can get from get_valid_phases(GPX))
        radial_range (_type_): range for the x axis.
        marker (str, optional): marker style. Defaults to "o".
        stem (bool, optional): whether to add stem plot. Defaults to True.
        unit (str, optional): the unit of the peak positions in gpx file. Defaults to "d".
        plot_unit (str, optional): the unit for x axis. Defaults to "q_A^-1".
        y_shift (float, optional): y shift between phase labels. Defaults to 0.1.
    """
    for ep, phase in enumerate(phases):
        # getting intensity peak positions (X) and relative magnitudes (Y)
        X = np.array([])
        Y = np.array([])
        for value in gpx['PWDR data.xy']['Reflection Lists'][phase]['RefList']:
            # takes the d column from the Reflection Lists dataframe-like array
            X = np.append(X, value[4])
        # Calculates the intensity maxes and normalized with 100 being the max
        peaks = gpx['PWDR data.xy']['Reflection Lists'][phase]
        Y = peaks['RefList'].T[8]*np.array([refl[11] for refl in peaks['RefList']])
        Imax = np.max(Y)
        if Imax:
            Y *= 100.0/Imax

        if unit == plot_unit:
            continue
        elif unit=="2th_deg" and plot_unit=="q_A^-1":
            X = twotheta_to_q(X, gpx['PWDR data.xy']['Instrument Parameters'][0]['Lam'][ep])
        elif unit=="d" and plot_unit=="q_A^-1":
            X = d_to_q(X)
            
        label=phase
        label_x=1.02
        color="C%d" % ep
        label_y=ep * y_shift
        
        for axl in line_axes:
            for i in X:
                axl.axvline(x=i, lw=0.6, color=color)
                axl.set_xlim([radial_range[0], radial_range[1]])

        for axs in stem_axes:
            axs_stem = axs.twinx()
            if stem:
                markerline, stemlines, baseline = axs_stem.stem(X, Y, markerfmt=".")
                plt.setp(stemlines, linewidth=0.5, color=color)
                plt.setp(markerline, color=color)
            axs_stem.set_xlim([radial_range[0], radial_range[1]])
            axs_stem.set_yticks([])
            axs_stem.set_ylim(bottom=0.1)
            axs_stem.text(
                label_x, label_y, label, color=color, transform=axs_stem.transAxes
            )
