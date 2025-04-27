import os
def format_parameter_file(input_file, output_file):
    """
    Read parameter combinations from input file and write them in a formatted way
    """
    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Extract headers and data
    headers = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:]]
    
    # Find the maximum width needed for each column
    max_widths = [max(len(header), max(len(row[i]) for row in data))
                 for i, header in enumerate(headers)]
    
    # Create the formatted header
    formatted_header = '  '.join(header.ljust(width) for header, width in zip(headers, max_widths))
    separator = '-' * len(formatted_header)
    os.system(f"rm {input_file}") 
    print(f"rm {input_file}") 
    # Write the formatted output
    with open(output_file, 'w') as f:
        f.write(formatted_header + '\n')
        f.write(separator + '\n')
        for row in data:
            formatted_row = '  '.join(val.ljust(width) for val, width in zip(row, max_widths))
            f.write(formatted_row + '\n')
    print(output_file)
cuts = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

def write_file(in_name, rows, out_name):
    with open(in_name, 'w') as f:
        f.write('filter_cut,walk_min,walk_max\n')
        for row in rows:
            f.write(f'{row[0]:.2f},{row[1]:.2f},{row[1]:.2f}\n')
    format_parameter_file(in_name, out_name)

ncount = 0
n_grids = 0
rows = {index: [] for index in range(20)}
for filter_cut in cuts:
    if filter_cut == 0: continue
    for walk_min in cuts:
        if walk_min < filter_cut:continue
        for walk_max in cuts:
            if walk_max < walk_min:continue
            ncount += 1
            rows[ncount].append([filter_cut, walk_min, walk_max]) 
            if ncount % 20 == 19:
                ncount = 0
            n_grids += 1
            print(n_grids,' => ', filter_cut, walk_min, walk_max)

n_grids = 0
count = 10
print(rows)
for key in rows.keys():
    print(key)
    if len(rows[key]) == 0: continue
    #print(rows)
    write_file(in_name = f'./experiment/grid_list{key}-raw.txt', rows = rows[key], out_name=f'./experiment/grid_list{key}.txt')