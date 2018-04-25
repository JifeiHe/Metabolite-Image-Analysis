#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:19:29 2018

@author: jifeihe
"""

from vgg_1000_simi import simi_loop
from matplotlib.backends.backend_pdf import PdfPages
#for i in range(20):
#    simi_loop(i)
data_dir = ''
os.chdir(data_dir)    
    
pp = PdfPages('multipage.pdf')
for i in range(20):
    fig = simi_loop(i)
    fig.savefig(pp, format='pdf')
pp.close()
