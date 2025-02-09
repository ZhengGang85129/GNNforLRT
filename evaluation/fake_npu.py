import pandas
import os

file_template = '/global/cfs/cdirs/m3443/data/GNNforLRT/raw_dataset/raw_HNL_FlatPU/eventEVENTNUMBER-particles.csv'
out_template = '/global/cfs/cdirs/m3443/data/GNNforLRT/trackPt1GeV-smeared/HNL_output_FlatPU_RAW/eventEVENTNUMBER-particles_fake.csv'
for i in range(20000):
    filename = file_template.replace('EVENTNUMBER', f'{i:09}')
    if not os.path.isfile(filename): 
        print(f'No such file: {filename} (skip)')
        continue 
    df = pandas.read_csv(filename)
    #print(df)
    if len(df[df['npileup'] != 1]) == 0: 
        continue
    condition = (df['npileup'] != 1) 
    df['npileup'] = df[condition]['npileup'].iloc[0] 
    
    #df['group'] = condition.astype(int) 
    #most_common_value = df.groupby('group')['npileup'].transform(lambda x: x.mode().iloc[0])
    #df.loc[condition, 'npileup'] = most_common_value[condition] 
    #df = df.drop('group', axis = 1)
    df.to_csv(out_template.replace('EVENTNUMBER', f'{i:09}')
    , index = False)
    if i % 100 == 0:
        print(f'event: {i:05}')
    