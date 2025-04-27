import os 
import ruamel.yaml
import argparse


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = 'train parser')
    parser.add_argument('--sample', choices = ['noPU', 'PU40', 'PU200'], required = True, type = str)
    parser.add_argument('--task_name', type = str, required = True)
    parser.add_argument('--embedding', default = None, type = str) 
    parser.add_argument('--filter', default = None, type = str) 
    parser.add_argument('--gnn', default = None, type = str) 
    args = parser.parse_args()
    return args

def main() -> None:
    global pipeline_template, args, pipeline_task
    
    args = parser()
     
    SAMPLE = args.sample
    TASK_NAME = args.task_name
    #pipeline_example_file = 'configs/pipeline.yaml'    
    pipeline_template = f'configs/pipeline-{args.sample}_template.yaml'
    
    pipeline_task = f'configs/pipeline-{args.sample}_{args.task_name}.yaml' 
    
    #print(pipeline_example_file, pipeline_sample_file)
    prepare_pipeline() 
    
    command_for_hyperconfig = 'python3 ./config_generate_hyperparam.py' 
    
    for stage in ['embedding', 'filter', 'gnn']:
        if getattr(args, stage) is not None:
            command_for_hyperconfig += f' --{stage} {getattr(args, stage)}'
        
        command_for_hyperconfig += f' --task_name {args.task_name} --sample {args.sample}' 
    print(command_for_hyperconfig) 
    command_for_traintrack = f'traintrack {pipeline_task} > output-{SAMPLE}.log 2>&1' 
    print(command_for_traintrack)
    os.system(command_for_hyperconfig) 
    os.system(command_for_traintrack)





def prepare_pipeline() -> None:
    
    #train-noPU-{args.task_name}.yaml    
    yaml = ruamel.yaml.YAML(typ='rt')
    
    file = open(pipeline_template, 'r')
    pipeline = yaml.load(file)
    for Idx, stage in enumerate(pipeline['stage_list']):
        pipeline['stage_list'][Idx]['config'] = f'train-{args.sample}-{args.task_name}.yaml'
    
    new_file = open(pipeline_task, 'w')
    yaml.dump(pipeline, new_file)
    file.close()
    new_file.close() 
    
    print(f'Prepare pipeline config: {new_file}') 
    

    

    
    
     
if __name__ == '__main__':
    main()



