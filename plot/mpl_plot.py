# MPL Boilerplate
import matplotlib

matplotlib.use("Agg")
# pylint: disable=unused-import
import matplotlib as mpl

# pylint: disable=unused-import
import matplotlib.cm as cm
import matplotlib.patches

# pylint: disable=unused-import
import matplotlib.pyplot as plt

# pylint: disable=unused-import
import numpy as np

# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

# Plot colors
white = "#ffffff"
black = "#000000"

grey = "#929591"
dark_grey = "#363737"

light_blue = "#95d0fc"
blue = "#047495"
dark_blue = "#00035b"

light_teal = "#82cbb2"
teal = "#029386"
dark_teal = "#014d4e"
deep_teal = "#00555a"

light_green = "#96f97b"
green = "#007A00"
dark_green = "#033500"

yellow = "#FFC857"

light_orange = "#F77F00"
orange = "#f97306"
dark_orange = "#C65102"

light_pink = "#ffd1df"
pink = "#CF6275"
dark_pink = "#cb416b"
crimson = "#8C000F"

purple = "#571F4E"

colors = [black, blue, teal, dark_green, dark_orange, purple, grey, green, orange, yellow, pink]
mrk = ["o", "D", "^", "s"]

#   luminance channel sweeps from dark to light, (for ordered comparisons)
#fig_size = (5, 4)
fig_size = (8, 6)
rcParams["figure.figsize"] = fig_size  # (w,h)
rcParams["figure.dpi"] = 150
# !$%ing matplotlib broke the interface. Why would you *replace* this!? >:(
try:
    from cycler import cycler

    rcParams["axes.prop_cycle"] = cycler("color", colors)
except ImportError:
    rcParams["axes.color_cycle"] = colors
rcParams["lines.linewidth"] = 2
rcParams["lines.marker"] = ""
rcParams["lines.markeredgewidth"] = 0
rcParams["axes.facecolor"] = "white"
rcParams["font.size"] = 16
rcParams["patch.edgecolor"] = "black"
rcParams["patch.facecolor"] = colors[0]
rcParams["xtick.major.pad"] = 8
rcParams["xtick.minor.pad"] = 8
rcParams["ytick.major.pad"] = 8
rcParams["ytick.minor.pad"] = 8
#rcParams['font.family'] = 'Helvetica'
rcParams['font.family'] = 'DejaVu Sans'
rcParams["font.weight"] = 100
rcParams.update({"figure.max_open_warning": 0})