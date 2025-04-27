import os 
import argparse
import yaml
#python3 ./run_model.py --sample noPU --task_name knn4 --embedding ./HyperOptim/Embedding/knn4.yaml --gnn ./HyperOptim/GNN/knn4.yaml ; sbatch tmp.sh

"""
This script is aiming to provide the config file for specific modified parameters in each stage, and providing the tmp.sh. 
Always refer to 


"""


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = 'train parser')
    parser.add_argument('--stage', choices = ['Embedding', 'Filter', 'GNN'], default = 'all', type = str)
    parser.add_argument('--sample', choices = ['noPU', 'PU40', 'PU200', 'MIX'], required = True, type = str)
    parser.add_argument('--task_name', type = str, required = True)
    parser.add_argument('--embedding', default = None, type = str) 
    parser.add_argument('--filter', default = None, type = str) 
    parser.add_argument('--gnn', default = None, type = str) 
    parser.add_argument('--only', default = 'Embedding:Filter:GNN', choices = ['Embedding', 'Filter', 'GNN', 'Embedding:Filter', 'Embedding:GNN', 'Filter:GNN'])
    args = parser.parse_args()
    return args

def main():
    global args, pipeline_task, pipeline_template
    args = parser()
    pipeline_task = f'configs/pipeline-{args.sample}_{args.task_name}.yaml'  
    pipeline_template = f'configs/pipeline-{args.sample}_template.yaml'
    prepare_pipeline()
    write_script()


def prepare_pipeline() -> None:
    #pipeline_task = f'configs/pipeline-{args.sample}_{args.task_name}.yaml' 
    
    #print(pipeline_example_file, pipeline_sample_file)
    
    command_for_hyperconfig = 'python3 ./HyperOptim/config_hyperparam.py' 
    
    for stage in ['embedding', 'filter', 'gnn']:
        if getattr(args, stage) is not None:
            command_for_hyperconfig += f' --{stage} {getattr(args, stage)}'
        
    command_for_hyperconfig += f' --task_name {args.task_name} --sample {args.sample} --only {args.only}' 
    print(command_for_hyperconfig) 
    os.system(command_for_hyperconfig)  
    
    file = open(pipeline_template, 'r')
    pipeline = yaml.safe_load(file)
    stages = ['Embedding', 'Filter', 'GNN']
    
    pipeline_tmp = {}
    for stage, line in zip(stages, pipeline['stage_list']):
        pipeline_tmp[stage] = line 
    
    pipeline_to_be_write = {'stage_list': []} 
    for Idx, (condition, stage) in enumerate(zip(pipeline['stage_list'], stages)):
        if stage in args.only:
            condition['config'] = f'optim_{args.task_name}.yaml'
            pipeline_to_be_write['stage_list'].append(condition)

        
    new_file = open(pipeline_task, 'w')
    yaml.dump(pipeline_to_be_write, new_file)
    file.close()
    new_file.close() 
    
    print(f'Prepare pipeline config: {new_file}') 

def write_script()->None:
    
    with open('tmp.sh', 'w') as f:
        f.write('#!/usr/bin/bash\n')
        f.write('#SBATCH -q regular\n')
        f.write('#SBATCH --nodes=1\n')
        f.write('#SBATCH --ntasks-per-node=1\n')
        f.write('#SBATCH --constraint=gpu\n')
        f.write('#SBATCH --gpus-per-node=1\n')
        f.write('#SBATCH -c 32\n')
        f.write('#SBATCH -Am3443\n')
        f.write('#SBATCH --time 24:0:0\n') 
        # Load any required modules
        f.write('#SBATCH --mail-type=END,FAIL,BEGIN\n')
        f.write('#SBATCH --mail-user=disney85129@gmail.com\n')
        f.write('#SBATCH -o ./sinfo/output.%J.out\n')
        f.write('export SLURM_CPU_BIND="cores"\n')

# Change to your working directory
        f.write('cd /global/homes/z/zhenggan/workspace/Project\n')
        f.write('pwd\n')
        f.write('bash\n')
        f.write('nvidia-smi\n')
        f.write('which bash\n')
        f.write('source /global/homes/z/zhenggan/miniconda3/etc/profile.d/conda.sh\n')
        f.write('conda activate exatrkx-gpu\n')

        pipeline_task = f'configs/pipeline-{args.sample}_{args.task_name}.yaml'
        command_for_traintrack = f'traintrack {pipeline_task} #> output-{args.sample}_{args.task_name}.log 2>&1 \n' 
        
        
        f.write(command_for_traintrack)
        if "GNN" in args.only:
            with open(f'./LightningModules/GNN/optim_{args.task_name}.yaml', 'r') as f0:
                config = yaml.safe_load(f0)
            #f.write(f'mv {config["output_dir"]} $Group/results\n')
            f.write(f'mv stage_performance.png stage_performance-{args.sample}-{args.task_name}.png\n')
            f.write(f'mv stage_performance.pdf stage_performance-{args.sample}-{args.task_name}.pdf\n')
    
    print('Do: sbatch tmp.sh') 

if __name__ == "__main__":
    main()