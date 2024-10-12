import yaml
import os 
import ruamel.yaml



def main() -> None:
    global example_file, stages, input_folder, SAMPLENAME, config_file, GroupInFolder
    SAMPLENAME = 'PU40' #FIXME
    GroupInFolder = '/global/cfs/cdirs/m3443/data/GNNforLRT/trackPt1GeV' #FIXME
    stages = ['Embedding', 'Filter', 'GNN']
    
    SampleInFolder = {
        'noPU': os.path.join(GroupInFolder, 'TTbar_DiLep_output_noPU_npz'),
        'PU200': os.path.join(GroupInFolder, 'TTbar_DiLep_output_PU200_npz'),
    }
    example_file = 'HyperOptim/train-STAGE_template.yaml' #FIXME
    config_file = 'HyperOptim/STAGE/train-SAMPLENAME.yaml' #FIXME
    
    input_folder = SampleInFolder[SAMPLENAME] #Input Sample for first stage: Embedding 
        
    Prepare_stages_train_config() #FIXME 

def Prepare_stages_train_config() -> None:
    output_dirs = []
    yaml = ruamel.yaml.YAML(typ='rt')
    for idx, stage in enumerate(stages):
        if stage == 'Embedding':
            input_dir = input_folder
        else:
            input_dir = output_dirs[idx - 1]      
        output_dir =  os.path.join(GroupInFolder, f'TTbar_{SAMPLENAME}_{stage}_output')

        output_dirs.append(output_dir)
        file = open(example_file.replace('STAGE', stage), 'r')
        train_config = yaml.load(file)
        ####### Edit #######
        train_config['input_dir'] = input_dir #FIXME
        train_config['output_dir'] = output_dir #FIXME
        new_file = open(config_file.replace('STAGE', stage).replace('SAMPLENAME', SAMPLENAME), 'w') 
        yaml.dump(train_config, new_file)
        #################### 
        file.close()
        new_file.close()
            

if __name__ == '__main__':
    main()          
    
    
    


 
     
    
    
    
    
    





