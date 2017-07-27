#!/usr/bin/env python3
"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

N = 4

zero_means = (0.297,0.43,0.42,0.348)
zero_std = (0.033, 0.03, 0.04, 0.056)

one_means = (0.164, 0.33, 0.31, 0.168)
one_std = (0.039, 0.05, 0.05, 0.073)

two_means = (0.055, 0.18, 0.17,0.028)
two_std = ( 0.039, 0.06, 0.06, 0.064)

three_means = (0.009, 0.08, 0.07, 0.017)
three_std = (0.039, 0.06, 0.06, 0.053)

four_means = (-0.004, -0.01, 0.0 , 0.005)
four_std = (0.039, 0.05, 0.04,  0.05)

German_means = (0.43, 0.33, 0.18, 0.08, -0.01)
German_std = (0.03, 0.05, 0.06, 0.06, 0.05)

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
ax.set_xlabel('Language')
ax.set_xticks(ind + 1.8*width )
ax.set_xticklabels(('English','German', 'French', 'Spanish'))

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0]), ('0','1',
    '2', '3', 'more than 4'), loc='upper right')


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
