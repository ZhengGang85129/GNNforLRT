import yaml
import matplotlib.pyplot as plt
import sys


def plt_rate(rates, fname):
    stage = ['Embedding', 'Filter', 'GNN']
    
    plt.figure(figsize = (10, 6))
    plt.plot(stage, rates, marker = 's')
    
    for _, (i, j) in enumerate(zip([0, 1, 2], rates)):
        if i != 2:
            plt.annotate(f"{j:.3f}", xy=(i + 0.1, j))
        else:
            plt.annotate(f"{j:.3f}", xy=(i - 0.05, j + 0.001))
        #plt.text(xi, yi, f'{yi:.2f}', 
        #        xytext=(0, 10),  # Offset the text 10 points above the point
        #        textcoords='offset points',
        #        ha='center',  # Horizontal alignment
        #        va='bottom',  # Vertical alignment
        #        fontsize=9,
        #        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.7))
    
    plt.grid(True)
    plt.xlabel('Stages')
    plt.savefig(fname)

def main(data):
    with open(data, 'r') as file:
        data = yaml.safe_load(file)
    #print(data)
    eff = [data['emb_eff'], data['fil_eff'], data['gnn_eff']]
    pur = [data['emb_pur'], data['fil_pur'], data['gnn_pur']]
    plt_rate(eff, 'eff.png') 
    plt_rate(pur, 'pur.png') 
    
    return 

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError()
    main(sys.argv[1])   