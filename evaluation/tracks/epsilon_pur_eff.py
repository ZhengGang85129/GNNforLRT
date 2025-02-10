#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import multiprocessing

import numpy as np
import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots import tracks

from track_reconstruction import reconstruct_tracks
from track_reconstruction.algorithm import DBSCANTrackReco
from track_matching import match_tracks

from workflows import reconstruct_and_match_tracks

import sys, os 



def plot_pur_eff(
    epsilon_sample_points,
    efficiencies,
    purities,
    save=None
):
    plt.rcParams.update({'font.size': 12})

    # Request 1x2 figure for purity and efficiency plot.
    fig, ax = plt.subplots(
        1, 2, figsize=(16, 8), tight_layout=True
    )
    print(efficiencies)
    ax[0].plot(epsilon_sample_points, efficiencies)
    ax[0].set_xlabel(r'DBSCAN $\epsilon$')
    ax[0].set_ylabel('Efficiency')

    ax[1].plot(epsilon_sample_points, purities)
    ax[1].set_xlabel(r'DBSCAN $\epsilon$')
    ax[1].set_ylabel('Purity')

    if save is not None:
        plt.savefig(save/'DBSCAN_eff_pur.pdf')
    else:
        plt.show()


if __name__ == '__main__':
    
    output_path = Path('.')
    output_path.mkdir(exist_ok=True, parents=True)
    filename_template = './../metrics-PU200//epsilon_scan/epsilon_EPSILON/summary.txt'
    
    epsilon_sample_points=["0.05", "0.15", "0.20","0.21", "0.22", "0.23", "0.24", "0.25", "0.26", "0.27", "0.28", "0.29", "0.30", "0.31", "0.32", "0.33", "0.34","0.35","0.45","0.55","0.65","0.75","0.85"]
    epsilon_sample_points=["0.15", "0.25","0.35","0.45","0.55","0.65","0.75","0.85"]
    
    epsilon_sample_points=["0.05","0.15","0.25","0.35","0.36", "0.37", "0.38", "0.39", "0.40", "0.41", "0.42", "0.43", "0.44", "0.45","0.55","0.56","0.57","0.58","0.59","0.60","0.61","0.62","0.63","0.64","0.65","0.66","0.67","0.68","0.69","0.70","0.75","0.85"]
    epsilon_sample_points=("0.05","0.06","0.07","0.08","0.09" ,"0.10" ,"0.11", "0.12" ,"0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.20" ,"0.25", "0.30", "0.35", "0.40", "0.45" ,"0.50", "0.55", "0.60" ,"0.65", "0.70" ,"0.75", "0.80" ,"0.85", "0.90", "0.95") 
    epsilon_sample_points_float = [float(point) for point in epsilon_sample_points] 
    
    efficiencies = []
    purities = [] 
    for point in epsilon_sample_points:
        filename = filename_template.replace('EPSILON', str(point))
        with open(filename, 'r') as f:
            #print(f.readlines())
            for line in f.readlines():
                line = line.split('              ')
                if 'Eff.' in line[0]:
                    efficiencies.append(float(line[1].replace('\n', '')))
                if 'Pur.' in line[0]:
                    purities.append(float(line[1].replace('\n', '')))
    print(efficiencies, purities)
    plot_pur_eff(
        epsilon_sample_points_float,
        efficiencies,
        purities,
        output_path
    )
    