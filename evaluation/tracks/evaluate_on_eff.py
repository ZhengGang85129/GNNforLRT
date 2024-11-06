import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from ExaTrkXDataIO import DataReader

import matplotlib.pyplot as plt
from ExaTrkXPlotting import Plotter, PlotConfig

from workflows import reconstruct_and_match_tracks
from plot_configurations import (
    particle_filters,
)

import sys, os
import argparse
def myParse() -> argparse.ArgumentParser:
    '''
    usage: 
    (1) Extract the generated/reconstructable/matched particles from GNN and DBSCAN stage
        -> python3 ./tracks/evaluate_on_eff.py --config <PathToYourConfig> --output metrics/final<or Your specified Path> --mode extract --lepton <prompt/displaced/all> --fname <Specify Whatever you like>
    (2) Evaluate on the model
        -> python3 ./tracks/evaluate_on_eff.py --config <PathToYourConfig> --output metrics/final<or Your specified Path> --mode evaluate --lepton <prompt/displaced/all> --fname <Specify Whatever you like>
    '''
    
    parser = argparse.ArgumentParser(description = 'parser for evaluation_of_eff')
    parser.add_argument('--config', type = str, default = None)
    parser.add_argument('-d', '--output', type = str, default = 'metrics/final')
    parser.add_argument('-o', '--fname', type = str, default = 'test')
    parser.add_argument('-m', '--mode', type = str, default = 'extract', choices = ['extract', 'evaluate']) 
    parser.add_argument('--lepton', choices = ['prompt', 'displaced', 'all'], default = 'prompt')
    
    args = parser.parse_args()
    #assert args.config is not None
    return args

def put_text(ax, sample_name = 'ttbar', pileup = '200'):
    ax.text(0.05, 0.9, r'ExatrkX', transform = ax.transAxes, fontsize = 25 )
    if pileup == 'flat':
        ax.text(0.05, 0.85, r'Evalutated on $t\bar{t}$ events', transform = ax.transAxes, fontsize = 15)
    else:
        ax.text(0.05, 0.85, rf"Evalutated on $\mathbf{{\mu}}^{{\pm}}$ from $t\bar{{t}}$ events with $\mu = {pileup}$", transform = ax.transAxes, fontsize = 16)
    ax.text(0.05, 0.81, r'$\mathbf{Open \  data \ detector} (Fastras sim.)$', transform = ax.transAxes, fontsize = 14 )
    ax.text(0.75, 0.79, r'$\sqrt{s} = XX \  \mathbf{TeV}$', transform = ax.transAxes, fontsize = 15) 
    ax.text(0.05, 0.77, r'DBScan with $\varepsilon = 0.XX$', transform = ax.transAxes, fontsize = 14 )

    #.legend(title='abc xyz')
def plot_eff(data, plt_config, var_name = 'pt')->None:
    var_names = ['pt', 'eta', 'd0', 'z0', 'vr', 'npileup']
    if var_name not in var_names: raise RuntimeError(f'No corresponding plot setting for this variable name: {var_name}') 
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 8), tight_layout = True)
    #ax.hist(data['gen'][var_name], bins = plt_config['bins'], edgecolor = 'black')
    
    Types = ['gen', 'reco', 'matched']
    counts = {}
    bin_edges = {}
    bin_centers = {}
    bin_width = {}  
    for Type in Types:
        counts[Type], bin_edges[Type], _ = ax.hist(data[Type][var_name], bins = plt_config['bins'])
        bin_centers[Type] = (bin_edges[Type][:-1] + bin_edges[Type][1:])/2
        bin_width[Type] = (bin_edges[Type][1:] - bin_edges[Type][:-1])/2
    ratio = {'tech': None, 'phys': None}
    error = {'tech': None, 'phys': None}
    
    ratio['phys'] = np.divide(counts['matched'], counts['gen'],  where = (counts['gen']!=0))
    ratio['tech'] = np.divide(counts['matched'], counts['reco'],  where = (counts['reco']!=0))
    ratio['tech'][counts['reco'] == 0] = 0
    ratio['phys'][counts['gen'] == 0] = 0
    
    print(sum(counts['reco']), sum(counts['gen']))
     
    error['phys'] = np.sqrt(ratio['phys'] * (1 - ratio['phys'])/counts['gen'])
    error['phys'] = np.nan_to_num(error['phys'], nan = 0, posinf = 0, neginf = 0)
    error['tech'] = np.sqrt(ratio['tech'] * (1 - ratio['tech'])/counts['reco']) 
    error['tech'] = np.nan_to_num(error['tech'], nan = 0, posinf = 0, neginf = 0)
    
    ax.clear()
    _, ymax = ax.get_ylim()
    ax.set_xscale(plt_config['ax_opts']['xscale'])
    ax.set_ylim(0, 1.5 * ymax)
    ax.legend() 
    put_text(ax)
    ax.errorbar(bin_centers['gen'], ratio['phys'], yerr = error['phys'], fmt = 'o', label = 'Physical efficacy', xerr = bin_width['gen'])
    
    ax.errorbar(bin_centers['reco'], ratio['tech'], yerr = error['tech'], fmt = 'o', label = 'Techincal efficacy', xerr = bin_width['reco'])
    ax.legend() 
    ax.set_xlabel(plt_config['x_label'])
    ax.grid(True, linestyle = '--', alpha = 0.7)
    
    plt_name = f'{args.output}/{var_name}-{args.lepton}_{args.fname}_eff.png' 
    fig.savefig(plt_name) 
    print(f'Check: {plt_name}') 
    
    return 


def plot(data, plt_config, var_name = 'pt')->None:
    var_names = ['pt', 'eta', 'd0', 'z0', 'vr', 'npileup']
    if var_name not in var_names: raise RuntimeError(f'No corresponding plot setting for this variable name: {var_name}') 
    
    fig, ax = plt.subplots(1, 1, figsize = (8, 8), tight_layout = True)
    #ax.hist(data['gen'][var_name], bins = plt_config['bins'], edgecolor = 'black')
    
    Types = ['gen', 'reco', 'matched']
    counts = {}
    bin_edges = {}
    bin_centers = {}
    bin_width = {}  
    label_names = ['Generated', 'Reconstructable', 'Matched']
    #<tab>
    transparencies = [0.0, 0., 0.]
    colors = ['red', 'blue', 'green'] 
    #plt.margins(y=0.25) 
    for Type, label_name, alpha, color in zip(Types, label_names, transparencies, colors):
        ax.hist(data[Type][var_name], bins = plt_config['bins'], label = label_name, facecolor = 'none', edgecolor = color)

    _, ymax = ax.get_ylim()
    #ax.set_ylim(0, 1.5 * ymax)
    ax.set_xscale(plt_config['ax_opts']['xscale'])
    ax.set_yscale(plt_config['ax_opts']['yscale'])
    ax.legend() 
    ax.set_xlabel(plt_config['x_label'])
    ax.grid(True, linestyle = '--', alpha = 0.7)
    
    plt_name = f'{args.output}/{var_name}-{args.lepton}_{args.fname}.png' 
    fig.savefig(plt_name) 
    print(f'Check: {plt_name}') 
    
    return 



def extract():
    with multiprocessing.Pool(processes = 8) as pool:
        reader = DataReader(
            config_path = path,
            base_dir = base_dir
        )
        particles = pd.concat(
            pool.map(reconstruct_and_match_tracks, reader.read())
        )
    
    particles = particles[particle_filters[args.lepton](particles)]
    particles.to_csv(os.path.join(args.output, args.fname+f'_gen-{args.lepton}.csv'), index = False)
    particles[particles.is_trackable].to_csv(os.path.join(args.output, args.fname+f'_reco-{args.lepton}.csv'), index = False)
    particles[particles.is_trackable & particles.is_matched].to_csv(os.path.join(args.output, args.fname+f'_match-{args.lepton}.csv'), index = False)

    
    return
    
def evaluate():
    gen_particles = pd.read_csv(os.path.join(args.output, args.fname+f'_gen-{args.lepton}.csv'))
    reco_particles = pd.read_csv(os.path.join(args.output, args.fname+f'_reco-{args.lepton}.csv'))
    match_particles = pd.read_csv(os.path.join(args.output, args.fname+f'_match-{args.lepton}.csv'))
    
    data = {
        'gen': gen_particles,
        'reco': reco_particles,
        'matched': match_particles
    }
    #print(data['gen']) 
    plt_configs = {
        'pt':{
        'x_label': r'$p_{T}$ [GeV]',
        'bins': np.array([0,15, 30, 60, 90, 120, 150]),
        'ax_opts': {
            'xscale': 'linear',
            'yscale': 'log'
        }
    },
        'eta':{
            'x_label': r'$\eta$',
            'bins': np.array([-4, -3, -1.5, 1.5, 3, 4]),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
        },
        'd0':{
            'x_label': r'$d0 [m]$',
            'bins': np.array([0, 0.005, 0.01,0.015, 0.02, 0.025, 0.03, 0.04, 0.05]),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'log'
            }
            
        },
        'z0':{
            'x_label': r'$z0 [mm]$',
            'bins': np.array([-150,-120,-75,-50, -25, -12, 12, 25, 50, 75, 120, 150]),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
            
        },
        'vr':{
            'x_label': r'$v_{r} [cm]$',
            'bins': np.array([0, 12, 25, 37, 50, 65, 100, 150]),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
            
        },
        'npileup':{
            'x_label': r'$N_{PU}$',
            'bins': 20,
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
            
        }
    }
     
    print(f"#generated: {len(data['gen'])}")
    print(f"#reconstructable: {len(data['reco'])}")
    print(f"#matched: {len(data['matched'])}")
    for plt_key in plt_configs.keys():
        print(plt_key)
        plot(data, var_name = plt_key, plt_config = plt_configs[plt_key] )
        plot_eff(data, var_name = plt_key, plt_config = plt_configs[plt_key] )
    return  
    
     
def main():
    global path, base_dir, args 
    args = myParse()
    if args.mode == 'extract':
        path = Path(args.config)
        base_dir = Path(os.path.basename(args.config))
    save = Path(args.output)
    save.mkdir(parents = True, exist_ok = True)
    
        #raise ValueError  

    exec(args.mode+'()')
    
    
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 15})
    
    import matplotlib.pyplot as plt

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    AXES_SIZE = 17
    TICK_SIZE = 15 
    LEGEND_SIZE = 14
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=AXES_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=AXES_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
     
    main()

 