import os
import sys 
def read_parameter_file(filename):
    parameters = []
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
    
    grid_filename = sys.argv[1] 
    for raw in read_parameter_file(grid_filename):
        os.system(f'python3 experiment/grid_search_algorithm_params.py --filter_cut {raw["filter_cut"]:.2f} --walk_min {raw["walk_min"]:.2f} --walk_max {raw["walk_max"]:.2f}')