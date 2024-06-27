import numpy as np
"""
Fit functions
"""
models={'exp': 'exponential',
        1: 'polynomialOrder1',
        2: 'polynomialOrder2',
        3: 'polynomialOrder3',}

def exponential(x, *params):
    if isinstance(x, list):
        x = np.array(x)
    return params[1]*np.exp(-x*params[0])

def guess(x, y):
    y0_guess = y[x==min(x)]
    sign = np.sign(y0_guess)
    
    y /= sign    
    logged = np.log(y[y>0])
    rough_slope = np.nanmedian(y[y>0]/x[y>0])
    
    return [rough_slope, y[x==min(x)][0]] 

def get_poly_model(order):
    import math
    factorials = np.array([math.factorial(n) for n in range(order+1)])
    def poly_model(x, *params):
        if isinstance(x, list):
            x = np.array(x)
        return np.sum((params[1]/factorials) * (-params[0]*x[:,None])**np.arange(order+1)[None,:],axis=1)
    return poly_model

def polynomialOrder1(x, *params):
    if isinstance(x, list):
        x = np.array(x)
    return params[0]*(x)+params[1]

def polynomialOrder2(x, *params):
    if isinstance(x, list):
        x = np.array(x)
    return params[0]*(x**2)+params[1]*(x)+params[2]

def polynomialOrder3(x, *params):
    if isinstance(x, list):
        x = np.array(x)
    return params[0]*(x**3)+params[1]*(x**2)+params[2]*(x)+params[3]

def fit(x, y, y_errs=None, order='exp', absolute_sigma=False, guesses=None):
    from scipy import optimize
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(y_errs, list):
        y_errs = np.array(y_errs)
    
    if order == 'exp':
        model=models[order]
        #guesses = [1, 1]
        if guesses != None:
            guesses = guesses
        else:
            guesses = guess(x, y) 
        lo_bounds = [0, -np.inf]
        #lo_bounds = [-np.inf, -np.inf]
        hi_bounds = [np.inf, np.inf]
        popt, pcov = optimize.curve_fit(eval(model), x, y, sigma=y_errs, absolute_sigma=absolute_sigma, p0=guesses, bounds=[lo_bounds, hi_bounds])
    else:
        model=models[order]
        guesses = [1]*(order+1) # this could instead be a smart guess computed from the data.
        popt, pcov = optimize.curve_fit(eval(model), x, y, sigma=y_errs, absolute_sigma=absolute_sigma, p0=guesses)

    unctty = np.sqrt(pcov.diagonal())
#     print('uncertainty of each parameter:',unctty)
    return popt, unctty

def fit_eval(x, *popt, order):
    model=models[order]
    return eval(model)(x, *popt)

"""
plot device map
"""
qubit_coordinates_map={}
qubit_coordinates_map[127] = [
    [0, 0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [0, 5],
    [0, 6],
    [0, 7],
    [0, 8],
    [0, 9],
    [0, 10],
    [0, 11],
    [0, 12],
    [0, 13],
    [1, 0],
    [1, 4],
    [1, 8],
    [1, 12],
    [2, 0],
    [2, 1],
    [2, 2],
    [2, 3],
    [2, 4],
    [2, 5],
    [2, 6],
    [2, 7],
    [2, 8],
    [2, 9],
    [2, 10],
    [2, 11],
    [2, 12],
    [2, 13],
    [2, 14],
    [3, 2],
    [3, 6],
    [3, 10],
    [3, 14],
    [4, 0],
    [4, 1],
    [4, 2],
    [4, 3],
    [4, 4],
    [4, 5],
    [4, 6],
    [4, 7],
    [4, 8],
    [4, 9],
    [4, 10],
    [4, 11],
    [4, 12],
    [4, 13],
    [4, 14],
    [5, 0],
    [5, 4],
    [5, 8],
    [5, 12],
    [6, 0],
    [6, 1],
    [6, 2],
    [6, 3],
    [6, 4],
    [6, 5],
    [6, 6],
    [6, 7],
    [6, 8],
    [6, 9],
    [6, 10],
    [6, 11],
    [6, 12],
    [6, 13],
    [6, 14],
    [7, 2],
    [7, 6],
    [7, 10],
    [7, 14],
    [8, 0],
    [8, 1],
    [8, 2],
    [8, 3],
    [8, 4],
    [8, 5],
    [8, 6],
    [8, 7],
    [8, 8],
    [8, 9],
    [8, 10],
    [8, 11],
    [8, 12],
    [8, 13],
    [8, 14],
    [9, 0],
    [9, 4],
    [9, 8],
    [9, 12],
    [10, 0],
    [10, 1],
    [10, 2],
    [10, 3],
    [10, 4],
    [10, 5],
    [10, 6],
    [10, 7],
    [10, 8],
    [10, 9],
    [10, 10],
    [10, 11],
    [10, 12],
    [10, 13],
    [10, 14],
    [11, 2],
    [11, 6],
    [11, 10],
    [11, 14],
    [12, 1],
    [12, 2],
    [12, 3],
    [12, 4],
    [12, 5],
    [12, 6],
    [12, 7],
    [12, 8],
    [12, 9],
    [12, 10],
    [12, 11],
    [12, 12],
    [12, 13],
    [12, 14],
]

def plot_qubits(qubit_lists, nqubits, coupling_map, 
                colors=None, folder_plot=None, label_qubits=False):
    """
    layernames (str): pre-defined pair names 
    nqubits (int): number of qubits of the device
    
    nqubits; this needs to be defined in definedpairs.py above
    """
    import matplotlib
    from matplotlib import pyplot as plt
    from qiskit.transpiler.coupling import CouplingMap
    #from qiskit.visualization import plot_coupling_map
    matplotlib.use("Agg")
    from matplotlib import cm
    import numpy as np
    import os
    import copy

    if nqubits not in qubit_coordinates_map.keys():
        print(f"{nqubits} qubit device is not defined")
        return
    
    #ncol=len(qubit_lists); nrow=1;
    ncol=1; nrow=1;
    fig, ax = plt.subplots(nrow,ncol, figsize=(8*ncol,8*nrow))
    if colors == None:
        colors = cm.rainbow(np.linspace(0.2, 0.8, len(qubit_lists)))
    qubit_color = ["k"] * nqubits
    
    line_color = ['k'] * len(coupling_map)
    coupling_map=CouplingMap(coupling_map)

    for i, (qubits, color) in enumerate(zip(qubit_lists, colors)):
        for q in qubits:
            qubit_color[q]=color
        
    plot_coupling_map(nqubits, qubit_coordinates_map[nqubits], coupling_map.get_edges(), 
                      qubit_color=qubit_color, ax=ax, line_color=line_color, label_qubits=label_qubits)
    ax.axis('off')

    if folder_plot is not None:
        plt.savefig(folder_plot)
    plt.show()
    #plt.close()

def plot_multiple_qubits(qubit_lists, nqubits, coupling_map, 
                         colors=None, folder_plot=None, label_qubits=False):
    """
    layernames (str): pre-defined pair names 
    nqubits (int): number of qubits of the device
    
    nqubits; this needs to be defined in definedpairs.py above
    """
    import matplotlib
    from matplotlib import pyplot as plt
    from qiskit.transpiler.coupling import CouplingMap
    #from qiskit.visualization import plot_coupling_map
    matplotlib.use("Agg")
    from matplotlib import cm
    import numpy as np
    import os
    import copy

    if nqubits not in qubit_coordinates_map.keys():
        print(f"{nqubits} qubit device is not defined")
        return
    
    ncol=len(qubit_lists); nrow=1;
    fig, axes = plt.subplots(nrow,ncol, figsize=(6*ncol,6*nrow))
    if colors == None:
        npaulis=1
        for key in qubit_lists.keys():
            npaulis = np.max([npaulis, len(qubit_lists[key])])
        colors = cm.rainbow(np.linspace(0.2, 0.8, npaulis))
    basecolor="lightgrey"
    qubit_color = [basecolor] * nqubits
    
    line_color = ['grey'] * len(coupling_map)
    coupling_map=CouplingMap(coupling_map)
    
    for i, (key, qubit_list) in enumerate(qubit_lists.items()):
        if len(qubit_lists) > 1:
            ax = axes[i]
        else:
            ax = axes

        for j, qubits in enumerate(qubit_list):
            for q in qubits:
                if isinstance(qubit_color[q], list):
                    qubit_color[q].append(colors[j])
                elif qubit_color[q] == basecolor:
                    qubit_color[q]=colors[j]
                elif isinstance(qubit_color[q], str):
                    qubit_color[q]=[qubit_color[q], colors[j]]
                else:
                    qubit_color[q]=colors[j]
        
        plot_coupling_map(nqubits, qubit_coordinates_map[nqubits], coupling_map.get_edges(), 
                          qubit_color=qubit_color, ax=ax, line_color=line_color, label_qubits=label_qubits)
        ax.axis('off')

    plt.tight_layout()
    if folder_plot is not None:
        plt.savefig(folder_plot)
    plt.show()
    plt.close()

    
from typing import List

def plot_coupling_map(
    num_qubits: int,
    qubit_coordinates: List[List[int]],
    coupling_map: List[List[int]],
    figsize=None,
    plot_directed=False,
    label_qubits=True,
    qubit_size=None,
    line_width=4,
    font_size=None,
    qubit_color=None,
    qubit_labels=None,
    line_color=None,
    font_color="w",
    ax=None,
    filename=None,
):
    """Plots an arbitrary coupling map of qubits (embedded in a plane).

    Args:
        num_qubits (int): The number of qubits defined and plotted.
        qubit_coordinates (List[List[int]]): A list of two-element lists, with entries of each nested
            list being the planar coordinates in a 0-based square grid where each qubit is located.
        coupling_map (List[List[int]]): A list of two-element lists, with entries of each nested
            list being the qubit numbers of the bonds to be plotted.
        figsize (tuple): Output figure size (wxh) in inches.
        plot_directed (bool): Plot directed coupling map.
        label_qubits (bool): Label the qubits.
        qubit_size (float): Size of qubit marker.
        line_width (float): Width of lines.
        font_size (int): Font size of qubit labels.
        qubit_color (list): A list of colors for the qubits
        qubit_labels (list): A list of qubit labels
        line_color (list): A list of colors for each line from coupling_map.
        font_color (str): The font color for the qubit labels.
        ax (Axes): A Matplotlib axes instance.
        filename (str): file path to save image to.

    Returns:
        Figure: A Matplotlib figure instance.

    Raises:
        MissingOptionalLibraryError: if matplotlib not installed.
        QiskitError: If length of qubit labels does not match number of qubits.

    Example:

        .. jupyter-execute::

            from qiskit.visualization import plot_coupling_map
            %matplotlib inline

            num_qubits = 8
            coupling_map = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 5], [5, 6], [2, 4], [6, 7]]
            qubit_coordinates = [[0, 1], [1, 1], [1, 0], [1, 2], [2, 0], [2, 2], [2, 1], [3, 1]]
            plot_coupling_map(num_qubits, coupling_map, qubit_coordinates)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    input_axes = False
    if ax:
        input_axes = True

    if font_size is None:
        font_size = 12

    if qubit_size is None:
        qubit_size = 24
    if num_qubits > 20:
        qubit_size = 28
        font_size = 10

    if qubit_labels is None:
        qubit_labels = list(range(num_qubits))
    else:
        if len(qubit_labels) != num_qubits:
            raise QiskitError("Length of qubit labels does not equal number of qubits.")

    if qubit_coordinates is not None:
        grid_data = qubit_coordinates
    else:
        if not input_axes:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.axis("off")
            if filename:
                fig.savefig(filename)
            return fig

    x_max = max(d[1] for d in grid_data)
    y_max = max(d[0] for d in grid_data)
    max_dim = max(x_max, y_max)

    if figsize is None:
        if num_qubits == 1 or (x_max / max_dim > 0.33 and y_max / max_dim > 0.33):
            figsize = (5, 5)
        else:
            figsize = (9, 3)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

    # set coloring
    if qubit_color is None:
        qubit_color = ["#648fff"] * num_qubits
    if line_color is None:
        line_color = ["#648fff"] * len(coupling_map) if coupling_map else []

    # Add lines for couplings
    if num_qubits != 1:
        for ind, edge in enumerate(coupling_map):
            is_symmetric = False
            if edge[::-1] in coupling_map:
                is_symmetric = True
            y_start = grid_data[edge[0]][0]
            x_start = grid_data[edge[0]][1]
            y_end = grid_data[edge[1]][0]
            x_end = grid_data[edge[1]][1]

            if is_symmetric:
                if y_start == y_end:
                    x_end = (x_end - x_start) / 2 + x_start

                elif x_start == x_end:
                    y_end = (y_end - y_start) / 2 + y_start

                else:
                    x_end = (x_end - x_start) / 2 + x_start
                    y_end = (y_end - y_start) / 2 + y_start
            ax.add_artist(
                plt.Line2D(
                    [x_start, x_end],
                    [-y_start, -y_end],
                    color=line_color[ind],
                    linewidth=line_width,
                    zorder=0,
                )
            )
            if plot_directed:
                dx = x_end - x_start
                dy = y_end - y_start
                if is_symmetric:
                    x_arrow = x_start + dx * 0.95
                    y_arrow = -y_start - dy * 0.95
                    dx_arrow = dx * 0.01
                    dy_arrow = -dy * 0.01
                    head_width = 0.15
                else:
                    x_arrow = x_start + dx * 0.5
                    y_arrow = -y_start - dy * 0.5
                    dx_arrow = dx * 0.2
                    dy_arrow = -dy * 0.2
                    head_width = 0.2
                ax.add_patch(
                    mpatches.FancyArrow(
                        x_arrow,
                        y_arrow,
                        dx_arrow,
                        dy_arrow,
                        head_width=head_width,
                        length_includes_head=True,
                        edgecolor=None,
                        linewidth=0,
                        facecolor=line_color[ind],
                        zorder=1,
                    )
                )

    # Add circles for qubits
    for var, idx in enumerate(grid_data):
        _idx = [idx[1], -idx[0]]
        if isinstance(qubit_color[var], list):
            N = len(qubit_color[var])
            for j, color in enumerate(qubit_color[var]):
                add_half_circle(ax, color, _idx[0], _idx[1], qubit_size / (48*2), j, N)
        else:
            ax.add_artist(
                mpatches.Ellipse(
                    _idx,
                    qubit_size / 48,
                    qubit_size / 48,  # This is here so that the changes
                    color=qubit_color[var],
                    zorder=1,
                    alpha=1.0,
                )
            )  # to how qubits are plotted does
        if label_qubits:  # not affect qubit size kwarg.
            ax.text(
                *_idx,
                s=qubit_labels[var],
                horizontalalignment="center",
                verticalalignment="center",
                color=font_color,
                size=font_size,
                weight="bold",
            )
    ax.set_xlim([-1, x_max + 1])
    ax.set_ylim([-(y_max + 1), 1])
    ax.set_aspect("equal")

    if not input_axes:
        matplotlib_close_if_inline(fig)
        if filename:
            fig.savefig(filename)
        return fig
    return None

def add_half_circle(ax, color, x,y,radius, j, N):
    from matplotlib.patches import Polygon
    alpha = np.linspace(0,2*np.pi/N,101) + np.pi/2 + j*2*np.pi/N
    xy = np.zeros((len(alpha)+1,2))
    xy[0,0] = x
    xy[0,1] = y
    xy[1:,0] = x + radius * np.cos(alpha)
    xy[1:,1] = y + radius * np.sin(alpha)

    ax.add_patch(Polygon(xy, closed=True, facecolor=color, edgecolor=None, fill=True, zorder=0, alpha=1))