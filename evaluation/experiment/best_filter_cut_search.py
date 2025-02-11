import pandas as pd
import os 
import math
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import seaborn as sns

def main(filter_cut: float, walk_min: float, walk_max: float)->None:
    fname = FNAME.replace('PARAM1', f'{filter_cut:.2f}'.replace('.', 'p'))
    fname = fname.replace('PARAM2', f'{walk_min:.2f}'.replace('.', 'p'))
    fname = fname.replace('PARAM3', f'{walk_max:.2f}'.replace('.', 'p'))
    gen_file = os.path.join(OUTPUT_DIR, fname+f'_gen-{lepton_type}.csv')
    rec_file = os.path.join(OUTPUT_DIR, fname+f'_reco-{lepton_type}.csv')
    mac_file = os.path.join(OUTPUT_DIR, fname+f'_match-{lepton_type}.csv')
    if not os.path.isfile(gen_file): return
    if not os.path.isfile(rec_file): return
    if not os.path.isfile(mac_file): return
     
    gen_particles = pd.read_csv(gen_file)
    reco_particles = pd.read_csv(rec_file)
    match_particles = pd.read_csv(mac_file)
    
    data = {
        'gen': len(gen_particles),
        'reco': len(reco_particles),
        'matched': len(match_particles),
        'eff': len(match_particles)/len(gen_particles),
        'eff-error': math.sqrt(((1 - len(match_particles)/len(gen_particles)) *len(match_particles)/len(gen_particles))/ len(gen_particles)) 
    }
    return data 
def read_parameter_file(filename):
    parameters = []
    if not os.path.isfile(filename):return
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Skip header and separator
        for line in lines[2:]:
            filter_cut, walk_min, walk_max = map(float, line.strip().split())
            parameters.append({
                'filter_cut': filter_cut,
                'walk_min': walk_min,
                'walk_max': walk_max
            })
    return parameters

if __name__ == "__main__":
    
    global OUTPUT_DIR, FNAME, lepton_type 
    
    OUTPUT_DIR = 'metrics/final/experiments_on_filter_cut'    
    FNAME = 'filtercutPARAM1_wrangercut-minPARAM2_wrangercut-maxPARAM3'  
    lepton_type = 'prompt' #!CHANGE ME!
    
    
    
    efficiency = [np.zeros((11, 11)) for _ in range(11)]
    eff_error = [np.zeros((11, 11)) for _ in range(11)]
    index = {
        0.05: 0,
        0.10: 1,
        0.20: 2,
        0.30: 3,
        0.40: 4,
        0.50: 5,
        0.60: 6,
        0.70: 7,
        0.80: 8,
        0.90: 9,
        0.95: 10,
    }
    convert_index = {
        0: 0.05,
        1: 0.10,
        2: 0.20,
        3: 0.30,
        4: 0.40,
        5: 0.50,
        6: 0.60,
        7: 0.70,
        8: 0.80,
        9: 0.90,
        10: 0.95,
    }
    
    
    for raw in read_parameter_file('experiment/grid_list.txt'):
        results = main(raw['filter_cut'], raw['walk_min'], raw['walk_max'])
        if results is None: continue
        eff, error = results['eff'], results['eff-error']
        print(f'filter_cut: {raw["filter_cut"]:.2f}, walk_min: {raw["walk_min"]:.2f}, walk_max: {raw["walk_max"]:.2f} => effciency : {eff:.3f}'f"(+- {error:.3f})")
        efficiency[index[raw["filter_cut"]]][index[raw["walk_min"]]][index[raw["walk_max"]]] = eff 
        eff_error[index[raw["filter_cut"]]][index[raw["walk_min"]]][index[raw["walk_max"]]] = error 
    
    for index, cut in convert_index.items():
        max_idx = np.unravel_index(np.argmax(efficiency[index]), efficiency[index].shape) 
        max_value = efficiency[index][max_idx]
        plt.clf()
        plt.close()

        masked_array = np.ma.masked_where(efficiency[index] == 0, efficiency[index])
        x = np.arange(efficiency[index].shape[1])
        y = np.arange(efficiency[index].shape[0])
        plt.imshow(efficiency[index], origin = "lower", interpolation = "gaussian",extent=[-1,11,-1,11]) 
        plt.colorbar()
        
        plt.plot(max_idx[1], max_idx[0], 'r.', markersize=15, label=f'Max: {max_value:.2f}')
        plt.annotate(rf'Best efficacy: {max_value:.3f} ($\pm$ {eff_error[index][max_idx]:.3f})',
                xy=(max_idx[1], max_idx[0]),  # point to annotate
                xytext=(max_idx[1]-6, max_idx[0]+2),  # text position
                arrowprops=dict(facecolor='red', shrink=0.05),
                )
        plt.annotate(rf'at ({convert_index[max_idx[1]]}, {convert_index[max_idx[0]]})',
                xy=(max_idx[1], max_idx[0]-1),  # point to annotate
                xytext=(max_idx[1], max_idx[0]),  # text position
                #arrowprops=dict(facecolor='red', shrink=0.05),
                )
        #plt.pcolormesh(x, y, masked_array, shading='auto', )
        plt.title(f'Reconstruction efficiency (filter cut = {cut:.2f})')
        plt.xlabel('walk max cut')
        plt.ylabel('walk min cut')
        
        x_ticks = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])  # or your custom values
        y_ticks = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])  # or your custom values
        x_ticks_index = np.arange(efficiency[index].shape[1])  # or your custom values
        y_ticks_index = np.arange(efficiency[index].shape[0]) 
        plt.xticks(x_ticks_index, [str(i) for i in x_ticks], rotation=45)
        plt.yticks(y_ticks_index, [str(i) for i in y_ticks])
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlim(-2, 12)
        plt.ylim(-2, 12)
        plt.tight_layout() 
        plt.savefig(f'experiment/grid_search/experiment-filter_cut{index}.png')
        plt.cla()
        plt.clf()