
for i in {0000..0099}
do
cp tmp-Smeared-best-MIX.yaml ./HyperOptim/results/GNN_optim_for_MIX/tmp-GNN-$i.yaml #FIXME
python3 ./HyperOptim/run_model.py --sample MIX --task_name GNN-$i --gnn ./LightningModules/GNN/optim_GNN-$i.yaml --only GNN # FIXME 
#sbatch tmp.sh
done
