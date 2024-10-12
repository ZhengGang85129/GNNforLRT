import yaml
import os
#Embed
##Hyperparams = ['emb_hidden','emb_dim', 'weight', 'r_test', 'knn', 'margin',]
##performance = ['emb_eff', 'emb_pur']
#Filter
Hyperparams = ['filter_cut', 'hidden', 'nb_layer', 'weight']
Hyperparams = ['edgecut_cut',
        'hidden',
        'n_graph_iters',
        'nb_node_layer',
        'nb_edge_layer',
        'weight',
        'delta_eta']


performance = ['emb_eff', 'emb_pur'] #FIXME
performance = ['gnn_eff', 'gnn_pur'] #FIXME
#performance = ['fil_eff', 'fil_pur'] #FIXME 
result_template = f'HyperOptim/results/GNN_optim_for_MIX/tmp-GNN-POSTFIX' #FIXME

with open('hyper_optim.txt', 'w') as out:
    out.write('tag,')
    out.write(','.join(performance)+'\n')
    #out.write(','.join(Hyperparams)+'\n')
    for i in range(100):
        out.write(f'{i:04d}'+',')
        if not os.path.isfile(result_template.replace('POSTFIX', f'{i:04d}')+'.yaml'):continue
        with open(result_template.replace('POSTFIX', f'{i:04d}')+'.yaml', 'r') as file:
            yy = yaml.safe_load(file)
            #if yy.get('emb_eff', None) is None:
            #    print(result_template.replace('POSTFIX', f'{i:04d}')+'.yaml'+ '-> skip.')
            #    continue
            if yy.get("gnn_eff"):
                out.write(f'{yy["gnn_eff"]:.3f},{yy["gnn_pur"]:.3f}\n') #FIXME
             
        #with open(f'./HyperOptim/GNN/filter-grid_search_trial-set1-{i:04d}.yaml', 'r') as file:
        #    yy = yaml.safe_load(file)
        #    values = []
        #    for key in Hyperparams:
        #        if type(yy[key]) is float:
        #            values.append(f'{yy[key]:.03f}')
        #        elif type(yy[key]) is list:
        #            emb_hidden = ':'.join(map(str,yy[key][0]))
        #            values.append(emb_hidden) 
                    #values.append(' '.join(map(str, yy[key])))
        #        else:
        #            values.append(str(yy[key]))
        #    out.write(','.join(values)+'\n')