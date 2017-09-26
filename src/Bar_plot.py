#!/usr/bin/env python3
"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

N = 2

zero_means = (0.267, 0.297)
zero_std = (0.039, 0.033)

one_means = ( 0.172, 0.164)
one_std = (0.034, 0.039)

two_means = (0.090, 0.055)
two_std = ( 0.032, 0.039)

three_means = (0.059, 0.009)
three_std = ( 0.027, 0.039)

four_means = ( 0.045 ,-0.004)
four_std = (0.024, 0.037)

#zero_means = (0.314,0.293,0.267,0.195,0.211)
#zero_std = (0.056, 0.047, 0.039, 0.042,0.039)
#
#one_means = (0.198, 0.186, 0.172, 0.096,0.109)
#one_std = (0.051, 0.041, 0.034, 0.038, 0.035)
#
#two_means = (0.092, 0.088, 0.090,0.026,0.030)
#two_std = ( 0.051, 0.040, 0.032, 0.032, 0.031)
#
#three_means = (0.071, 0.067, 0.059, 0.005, 0.005)
#three_std = (0.044, 0.034, 0.027, 0.026, 0.026)
#
#four_means = (0.057, 0.052, 0.045 , -0.004, -0.005)
#four_std = (0.042, 0.031, 0.024,  0.023, 0.023)
#
#German_means = (0.43, 0.33, 0.18, 0.08, -0.01)
#German_std = (0.03, 0.05, 0.06, 0.06, 0.05)


#zero_means = (0.317,0.289,0.297,0.260,0.339)
#zero_std = (0.048, 0.060, 0.033, 0.034,0.050)
#
#one_means = (0.175, 0.144, 0.164, 0.141,0.167)
#one_std = (0.057, 0.056, 0.039, 0.035, 0.063)
#
#two_means = (0.069, 0.069, 0.055,0.050,0.038)
#two_std = ( 0.053, 0.055, 0.039, 0.031, 0.061)
#
#three_means = (0.037, 0.034, 0.009, 0.015, 0.005)
#three_std = (0.046, 0.056, 0.039, 0.029, 0.050)
#
#four_means = (-0.005, 0.012, -0.004 , -0.009, 0.002)
#four_std = (0.042, 0.053, 0.037,  0.028, 0.041)
#
ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, zero_means, width, color='forestgreen', yerr=zero_std)

French_means = (0.42, 0.31, 0.17, 0.07, 0.0)
French_std = (0.04, 0.05, 0.06, 0.06, 0.04)
rects2 = ax.bar(ind + width, one_means, width, color='blueviolet',
    yerr=one_std)


Czech_means = (0.378, 0.211, 0.061, 0.021, 0.012)
Czech_std = (0.042, 0.050, 0.048, 0.042, 0.043)
rects3 = ax.bar(ind + 2*width, two_means, width, color='orangered',
    yerr=two_std)

Spanish_means = (0.348, 0.168, 0.028, 0.017, 0.005)
Spanish_std = (0.056, 0.073, 0.064, 0.053, 0.050)
rects4 = ax.bar(ind + 3*width, three_means, width, color='olive',
    yerr=three_std)

Mandarine_means = (0.186, 0.082, 0.029, 0.019, 0.009)
Mandarine_std = (0.039, 0.032, 0.030, 0.029, 0.027)
rects5= ax.bar(ind + 4*width, four_means, width, color='royalblue',
    yerr= four_std)
# add some text for labels, title and axes ticks



ax.set_ylabel('Cosine Similarity')
# ax.set_title('Cosine similarity in different phoneme edit distance')
#ax.set_xlabel('Dimension')
ax.set_xticks(ind + 1.8*width )
ax.set_xticklabels(('NE with m=10', 'SA with dim=400'))

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]),
    ('PSED=0','PSED=1',
    'PSED=2', 'PSED=3', 'PSED=4'), loc='upper right')


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,
            0.01 +height,
            '%.2f' % float(height),
            ha='center',
            va='bottom')

#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)
#autolabel(rects4)
#autolabel(rects5)

plt.show()
