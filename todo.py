""" 
================
LIBRERIAS USADAS
================
"""
import pandas as pd
import mpmath
import numpy as np
from scipy.fft import fft, ifft
from latex2sympy2 import latex2sympy, latex2latex
from manim import *
import calendar
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import image
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import warnings
warnings.filterwarnings("ignore")
config.media_width = "100%"
config.verbosity = "WARNING"
import compass_functions as cf
import sympy as sp
from sympy import symbols, Eq, latex, sqrt,solve ,Function 
from sympy import   I, simplify, expand , print_latex,init_printing
from sympy import Symbol, sin,cos,arg,exp ,integrate,Derivative,Integral  
from sympy.utilities.lambdify import lambdify
sympy.init_printing()
from IPython.display import Markdown as md
#DSP con placa de audio 
import sounddevice as sd
# Ajustes del fondo de plot para matplotlib 
plt.rcParams.update({
        "lines.color": "black",
        "patch.edgecolor": "black",
        "text.color": "white",
        "axes.facecolor": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "gray",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"})

""" 
===================
SÍMBOLOS PARA SYMPY
===================
"""



 
alpha, beta, gamma, delta = symbols('alpha, beta, gamma, delta')
epsilon, zeta, eta, theta = symbols('epsilon, zeta, eta, theta')
iota, kappa, lamda, mu = symbols('iota, kappa, lamda, mu')
nu, xi, omicron, pi = symbols('nu, xi, omicron, pi')
rho, sigma, tau, upsilon = symbols('rho, sigma, tau, upsilon')
phi, chi, psi, omega = symbols('phi, chi, psi, omega')

Alpha, Beta, Gamma, Delta = symbols('Alpha, Beta, Gamma, Delta')
Epsilon, Zeta, Eta, Theta = symbols('Epsilon, Zeta, Eta, Theta')
Iota, Kappa, Lamda, Mu = symbols('Iota, Kappa, Lamda, Mu')
Nu, Xi, Omicron, Pi = symbols('Nu, Xi, Omicron, Pi')
Rho, Sigma, Tau, Upsilon = symbols('Rho, Sigma, Tau, Upsilon')
Phi, Chi, Psi, Omega = symbols('Phi, Chi, Psi, Omega')


""" 
================
COLECCIÓN SCRIPTS 
================
"""
