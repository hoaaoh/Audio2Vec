#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
def main():
    AE_small_list = [ 0.730, 0.685, 0.737, 0.693, 0.881, 0.713 ]
    AE_large_list = [ 0.234, 0.307, 0.400, 0.323, 0.317, 0.233 ]
    ### m = [ 3, 6, 10, 15, 21, 26 ] ###
    NE_small_list = [ 0.390, 0.490, 0.484, 0.460, 0.351,   ]
    NE_large_list = [ 0.100, 0.158, 0.169, 0.150, 0.092,  ] 
    dim = [100, 200, 400, 600, 800, 1000 ]
    small_dim = [117, 234, 390, 585, 819, 1014 ]

    #dim_new = np.linspace( min(dim), max(dim),300) 
    #AE_small_smooth = spline(dim, AE_small_list, dim_new)
    #plt.plot(dim_new, AE_small_smooth , label = 'AE_small_smooth')
    plt.plot(dim, AE_small_list, '-o', label='SA_small')

    plt.plot(dim, AE_large_list, '-o', label='SA_large') 
    plt.plot(small_dim, NE_small_list, '-o', label='NE_small')
    plt.plot(small_dim, NE_large_list,'-o', label='NE_large')
    plt.xlabel('Representation Dimension', fontsize=12)
    plt.ylabel('MAP', fontsize=12)
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    main()
