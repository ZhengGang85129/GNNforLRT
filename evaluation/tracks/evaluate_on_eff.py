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
    error['phys'] = np.sqrt(ratio['phys'] * (1 - ratio['phys'])/counts['gen'])
    error['phys'] = np.nan_to_num(error['phys'], nan = 0, posinf = 0, neginf = 0)
    error['tech'] = np.sqrt(ratio['tech'] * (1 - ratio['tech'])/counts['reco']) 
    error['tech'] = np.nan_to_num(error['tech'], nan = 0, posinf = 0, neginf = 0)
    
    ax.clear() 
    ax.set_ylim(0.0, 1.05)
    ax.set_xscale(plt_config['ax_opts']['xscale'])
    ax.set_yscale(plt_config['ax_opts']['yscale'])
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
    for Type, label_name, alpha, color in zip(Types, label_names, transparencies, colors):
        ax.hist(data[Type][var_name], bins = plt_config['bins'], label = label_name, facecolor = 'none', edgecolor = color)

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
        'bins': 10 ** np.arange(0, 2.1, 0.1),
        'ax_opts': {
            'xscale': 'log',
            'yscale': 'linear'
        }
    },
        'eta':{
            'x_label': r'$\eta$',
            'bins': np.arange(-4.0, 4.1, 0.4),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
        },
        'd0':{
            'x_label': r'$d0$',
            'bins': np.arange(0, 0.1, 0.01),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
            
        },
        'z0':{
            'x_label': r'$z0$',
            'bins': np.arange(-200, 201, 20),
            'ax_opts': {
                'xscale': 'linear',
                'yscale': 'linear'
            }
            
        },
        'vr':{
            'x_label': r'$v_{r}$',
            'bins': np.arange(0, 220, 20),
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
    main()

 