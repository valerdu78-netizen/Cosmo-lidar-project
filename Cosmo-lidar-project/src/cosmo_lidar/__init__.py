from .io import (
    fetch_html,
    extract_ut_column_dat_links,
    read_radiosonde_dat,
    read_many_radiosonde,
    save_table,
    load_table,
    download_some
)

# import the actual symbol from atm_tools (correct capitalization)
from .atm_tools import Calcul_T_ant_1_el

# optional alias for older code expecting the lowercase name
calcul_t_ant_1_el = Calcul_T_ant_1_el

# import other exports (example placeholder; add real ones as needed)
#from .mc_tools import monte_carlo_t_ant

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "fetch_html",
    "extract_ut_column_dat_links",
    "read_radiosonde_dat",
    "read_many_radiosonde",
    "save_table",
    "load_table",
    "Calcul_T_ant_1_el",
    "calcul_t_ant_1_el",
    "download_some",
    #"monte_carlo_t_ant",
    "plt",
    "np"
]
__version__ = "0.1.0"
__author__ = "Valérian LE LOREC"