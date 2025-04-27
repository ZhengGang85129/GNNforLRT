
for i in {0000..0001}
do
cp tmp-Smeared-best-MIX.yaml ./HyperOptim/results/GNN_optim_for_MIX/tmp-GNN-$i.yaml #FIXME -> for Filter & GNN
python3 ./HyperOptim/run_model.py --sample PU200 --task_name Embedding-$i --embed ./LightningModules/Embedding/optim_Embedding-$i.yaml --only Embedding # FIXME 
#sbatch tmp.sh
done
