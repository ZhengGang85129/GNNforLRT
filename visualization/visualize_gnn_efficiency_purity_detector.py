#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import multiprocessing

import numpy as np
import pandas as pd
import math
from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig
from ExaTrkXPlots import tracks

import sys, os 
import itertools


def make_detector_plot(hits, fig, ax):
    
    # draw less dots to reduce file size
    hits = hits.sample(frac=1).reset_index(drop=True)
    r = hits[0:hits.shape[0]].r
    z = hits[0:hits.shape[0]].z
    #x_to_draw = x_to_draw[:len(event_file.x)//10]
    ax.scatter(z/1000, r/1000, s=1, color='lightgrey')
    ax.set_xlabel("z [m]")
    ax.set_ylabel("r [m]")

    return fig, ax
    

def make_eff_pur_detector_map(hits = None, edge_cut:float = 0.1, ax = None, prec = 2):
    def edges_in_radius(hits, r_range, z_range):
        '''
        Consider the edge is single direction.
        '''
        x_idxs = np.nonzero(np.logical_and.reduce((
        hits.r/1000 > r_range[0],
        hits.r/1000 <= r_range[1],
        hits.z/1000 > z_range[0],
        hits.z/1000 <= z_range[1])))
        return len(x_idxs[0])


    # Go over all parts of the detector
    r_ranges = []
    z_ranges = []


    # Pixel
    for z_range in [(-2.0, -0.5), (-0.5, 0.5), (0.5, 2.0)]:
        r_ranges.append((0.0,0.2))
        z_ranges.append(z_range)


    # SStrip, LStrip
    for r_range in [(0.2,0.7), (0.7,1.2)]:
        for z_range in [(-3.1, -1.2), (-1.2,1.2), (1.2, 3.1)]:
            r_ranges.append(r_range)
            z_ranges.append(z_range)


    colors = ['r', 'g', 'b', 'y', 'c']
    for r_range, z_range, color in zip(r_ranges, z_ranges, itertools.cycle(colors)):
        #n_true_edges = edges_in_radius(hits[hits['truth']], r_range, z_range)
        n_constructed_edges_in_graph =edges_in_radius(hits[hits['score'] >  0.334], r_range, z_range)
        n_truth_edges_in_graph = edges_in_radius(hits[hits['truth']], r_range, z_range)
        n_intersection_edges_in_graph = edges_in_radius(hits[hits['truth']][hits['score'] >  0.334], r_range, z_range)
        # print(r_range, z_range)


        #eff, pur = eff_pur_cantor(all_edges_selected, true_edges_selected)
        fmt_str = "eff: {:." + str(prec) + "f} \npur: {:." + str(prec) + "f}"
        #fmt_str = "true: {} \nall: {}"
        ax.text(
            np.mean(z_range)*1.1 - 0.5,
            np.mean(r_range) - 0.05,
            #fmt_str.format(n_truth_edges_in_graph, n_all_edges_in_graph)
            fmt_str.format(n_intersection_edges_in_graph/(max(n_truth_edges_in_graph, 1e-20)),n_intersection_edges_in_graph/(max(n_constructed_edges_in_graph, 1e-20)))
            #fmt_str.format(n_truth_edges_in_graph, n_all_edges_in_graph)
            , fontsize = 8
            )
        rectangle_args = {
            "xy": (z_range[0], r_range[0]),
            "width": (z_range[1]-z_range[0]),
            "height": (r_range[1]-r_range[0]),
        }
        ax.add_patch(plt.Rectangle(**rectangle_args, alpha=0.1, color=color, linewidth=0)) 
    return ax

   


 
def plot_tracks_2D(hits: pd.DataFrame , edge_cut:float = 0.334, save: str = './') -> None:
    plt.rcParams.update({'font.size': 12})
    
    
    fig, ax = plt.subplots()
    length = 4.5
    etas = [0.5, 1.5, 2.5, 3.0, 3.5]
    x_texts = [1., 2.8, 3.5, 3.5, 3.5]
    y_texts = [1.5, 3, 4, 4, 4]
    offset = -0.75
    for eta, x_text, y_text in zip(etas,x_texts, y_texts):
        theta = math.atan(math.exp(-eta)) * 2
        x = [0, length * np.cos(theta)]
        y = [0, length * np.sin(theta)]
        ax.plot(x, y, c = 'black', alpha=0.1)
        ax.text(x = x_text*np.cos(theta), y = y_text * np.sin(theta), s = rf'$\eta = {eta}$', fontsize = 8)
        ax.text(x = -x_text*np.cos(theta)+offset, y = y_text * np.sin(theta), s = rf'$\eta = -{eta}$', fontsize = 8)
        x = [0, -length * np.cos(theta)]
        ax.plot(x, y, c = 'black', alpha=0.1)

    fig, ax = make_detector_plot(hits, fig = fig, ax = ax)
    ax = make_eff_pur_detector_map(hits = hits, edge_cut = edge_cut, ax = ax)
    ax.set_title("Metrics for smeared training")
    ax.set_xlim([-length, length])
    ax.set_ylim([-0.05, 1.5])
    fig.savefig("eff_pur_2D.png")
    fig.savefig("eff_pur_2D.pdf")

  
if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        raise RuntimeError('usage: python3 ./tracks/plot_eff_pur_eff.py <configuration file>')
    
    
    edge_cut =  0.333
    path = Path(f'{sys.argv[1]}')
    base_dir = Path(f'{os.path.basename(sys.argv[1])}')
    event_ids = range(0, 10000)
    
    reader = DataReader(
        config_path=path,
        base_dir=base_dir
    )
    
    for event_id in event_ids:
        data = reader.read_one(evtid = event_id)
        if data is None: continue
        hits = data['hits'].reset_index().rename(columns = {'index': 'h_id'}) 
        edges = data['edges']
        edges = edges
        edges = pd.concat([edges, edges.rename(columns = {'sender': 'receiver_tmp', 'receiver': 'sender_tmp'}).rename(columns = {'receiver_tmp': 'h_id', 'sender_tmp': 'sender'})], ignore_index=True)
        
        hits = edges.merge(hits, how = 'left', on = 'h_id')
        
        plot_tracks_2D(hits = hits, edge_cut = edge_cut)
        break