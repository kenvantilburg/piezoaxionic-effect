import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
import random
from time import time as tictoc
from scipy.optimize import fmin
from scipy.optimize import minimize
from scipy.optimize import brentq

from IPython.core.display import display, HTML
from IPython.display import display, clear_output
display(HTML("<style>.container { width:100% !important; }</style>"))
np.set_printoptions(edgeitems=3, linewidth=200) 
pd.set_option('display.max_columns', None)
pd.set_option('max_rows',200) and pandas.set_option('max_columns',20)

plt.rcdefaults()
fontsize = 12
from matplotlib import font_manager
from matplotlib import rcParams
from matplotlib import rc
rcParams['font.family'] = 'sans-serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':fontsize})
#rc('text', usetex=True)