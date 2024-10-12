import sys
import os
import copy
import logging
import tracemalloc
import gc
from memory_profiler import profile
from pytorch_lightning import LightningModule, Trainer

from pytorch_lightning.callbacks import Callback
import torch.nn.functional as F
import sklearn.metrics
import matplotlib.pyplot as plt
import torch
import numpy as np
import yaml

class Plotting(Callback):
    def __init__(self):
        super().__init__()
        print("Plotting")
    
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.effs = []
        self.purs = []

class GNNEffPur(Callback):

    def __init__(self):
        super().__init__()
        print("Calculating pur and eff")

    def on_test_start(self, trainer, pl_module) -> None:
        """
        This hook is automatically called when the model is tested after training. The best checkpoint is automatically loaded
        """
        self.preds = []
        self.truth = []
        self._true = 0

        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        """
        Get the relevant outputs from each batch
        """
        
        
        self.preds.append(outputs["preds"].cpu())
        self.truth.append(outputs["truth"].cpu())
        self._true += batch.layerless_true_edges.shape[1]

    def on_test_end(self, trainer, pl_module):

        self.truth = torch.cat(self.truth)
        self.preds = torch.cat(self.preds)
        
        score_cut = pl_module.hparams["edge_cut"]
        
        positives = (self.preds > score_cut).sum()
        true_positives = ((self.preds > score_cut).to(dtype=torch.bool) & self.truth.to(dtype=torch.bool)).sum()
                
        eff = true_positives / self._true
        pur = true_positives / positives
        
        
        print("\n\n=====================================================================")
        print("eff dominator", self._true)
        print("eff", eff.item())
        print("pur", pur.item())
        data = {"gnn_eff": eff.item(), "gnn_pur": pur.item()}
        if pl_module.hparams['stage_dir'] is None:
            stage_dir = './'
        else:
            stage_dir =  pl_module.hparams['stage_dir']
        with open(f"{stage_dir}/tmp-{pl_module.hparams['TAG']}.yaml", 'a') as file:
            yaml.dump(data, file)
        print("=====================================================================\n\n")

        # =================================================================================
        ##############################  Start Plotting ####################################
        # =================================================================================

        print("plotting...")
        print(f'save in {stage_dir}/tmp-{pl_module.hparams["TAG"]}.yaml')
        with open(f"{stage_dir}/tmp-{pl_module.hparams['TAG']}.yaml", 'r') as file:
            datas = yaml.load(file, yaml.FullLoader)

        effs = [
            datas["emb_eff"],
            datas["fil_eff"],
            datas['gnn_eff'],
        ]

        purs = [
            datas["emb_pur"],
            datas["fil_pur"],
            datas["gnn_pur"],
        ]

        x = np.arange(3)

        fig, ax = plt.subplots(nrows=2, figsize=(9, 12))
        ax1, ax2 = ax
        ax1.set_title("Stage Efficiency")
        ax1.plot(effs)
        ax1.set_ylabel("Efficieny")
        ax1.set_xticks(x, ["embedding", "filtering", "GNN"])
        for i, j in zip(x, effs):
            ax1.annotate(f"{j:.3f}", xy=(i, j))

        ax2.set_title("Stage Purity")
        ax2.plot(purs)
        ax2.set_ylabel("Purity")
        ax2.set_xticks(x, ["embedding", "filtering", "GNN"])
        for i, j in zip(x, purs):
            ax2.annotate(f"{j:.3f}", xy=(i, j))

        fig.savefig("stage_performance.png")
        fig.savefig("stage_performance.pdf")
        #os.remove("tmp.yaml")



"""
Class-based Callback inference for integration with Lightning
"""


class GNNTelemetry(Callback):

    """
    This callback contains standardised tests of the performance of a GNN
    """

    def __init__(self):
        super().__init__()
        logging.info("Constructing telemetry callback")

    def on_test_start(self, trainer, pl_module):

        """
        This hook is automatically called when the model is tested after training. The best checkpoint is automatically loaded
        """
        self.preds = []
        self.truth = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        """
        Get the relevant outputs from each batch
        """
        
        self.preds.append(outputs["preds"].cpu())
        self.truth.append(outputs["truth"].cpu())
                
        
    def on_test_end(self, trainer, pl_module):

        """
        1. Aggregate all outputs,
        2. Calculate the ROC curve,
        3. Plot ROC curve,
        4. Save plots to PDF 'metrics.pdf'
        """
        
        metrics = self.calculate_metrics()

        metrics_plots = self.plot_metrics(metrics)
        
        self.save_metrics(metrics_plots, pl_module.hparams.output_dir)
        
    
    def get_eff_pur_metrics(self):
                        
        self.truth = torch.cat(self.truth)
        self.preds = torch.cat(self.preds)
        
        score_cuts = np.arange(0., 1., 0.05)
        
        positives = np.array([(self.preds > score_cut).sum() for score_cut in score_cuts])
        print(type(self.truth))
        true_positives = np.array([((self.preds > score_cut).to(dtype=torch.bool) & self.truth.to(dtype=torch.bool)).sum() for score_cut in score_cuts])
                
        eff = true_positives / self.truth.sum()
        pur = true_positives / positives
        
        return eff, pur, score_cuts
        

    def calculate_metrics(self):
        
        eff, pur, score_cuts = self.get_eff_pur_metrics()
        
        return {"eff_plot": {"eff": eff, "score_cuts": score_cuts}, 
                "pur_plot": {"pur": pur, "score_cuts": score_cuts}}
    
    def make_plot(self, x_val, y_val, x_lab, y_lab, title):
        
        # Update this to dynamically adapt to number of metrics
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
        axs = axs.flatten() if type(axs) is list else [axs]

        axs[0].plot(x_val, y_val)
        axs[0].set_xlabel(x_lab)
        axs[0].set_ylabel(y_lab)
        axs[0].set_title(title)
        plt.tight_layout()
        
        return fig, axs
    
    def plot_metrics(self, metrics):
                
        eff_fig, eff_axs = self.make_plot(metrics["eff_plot"]["score_cuts"], metrics["eff_plot"]["eff"], "cut", "Eff", "Efficiency vs. cut")
        pur_fig, pur_axs = self.make_plot(metrics["pur_plot"]["score_cuts"], metrics["pur_plot"]["pur"], "cut", "Pur", "Purity vs. cut")
        
        return {"eff_plot": [eff_fig, eff_axs], "pur_plot": [pur_fig, pur_axs]}
    
    def save_metrics(self, metrics_plots, output_dir):
        
        os.makedirs(output_dir, exist_ok=True)
        
        for metric, (fig, axs) in metrics_plots.items():
            fig.savefig(
                os.path.join(output_dir, f"metrics_{metric}.pdf"), format="pdf"
            )
        
class GNNBuilder(Callback):        
    """Callback handling filter inference for later stages.

    This callback is used to apply a trained filter model to the dataset of a LightningModule. 
    The data structure is preloaded in the model, as training, validation and testing sets.
    Intended usage: run training and examine the telemetry to decide on the hyperparameters (e.g. r_test) that
    lead to desired efficiency-purity tradeoff. Then set these hyperparameters in the pipeline configuration and run
    with the --inference flag. Otherwise, to just run straight through automatically, train with this callback included.

    """
    
    def __init__(self):
        self.output_dir = None
        self.overwrite = False

    def on_test_end(self, trainer, pl_module):
        
        print("Testing finished, running inference to build graphs...")
        
        datasets = self.prepare_datastructure(pl_module)
        
        total_length = sum([len(dataset) for dataset in datasets.values()])
        
        pl_module.eval()
        with torch.no_grad():
            batch_incr = 0
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                for batch_idx, batch in enumerate(dataset):
                    percent = (batch_incr / total_length) * 100
                    sys.stdout.flush()
                    sys.stdout.write(f"{percent:.01f}% inference complete \r")
                    if (
                        not os.path.exists(
                            os.path.join(
                                self.output_dir, datatype, batch.event_file[-4:]
                            )
                        )
                    ) or self.overwrite:
                        batch_to_save = copy.deepcopy(batch)
                        batch_to_save = batch_to_save.to(pl_module.device)
                        self.construct_downstream(batch_to_save, pl_module, datatype)

                    batch_incr += 1

    def prepare_datastructure(self, pl_module):
        # Prep the directory to produce inference data to
        self.output_dir = pl_module.hparams.output_dir
        self.datatypes = ["train", "val", "test"]
        
        os.makedirs(self.output_dir, exist_ok=True)
        [
            os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True)
            for datatype in self.datatypes
        ]

        # Set overwrite setting if it is in config
        self.overwrite = (
            pl_module.hparams.overwrite if "overwrite" in pl_module.hparams else False
        )

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {
            "train": pl_module.trainset,
            "val": pl_module.valset,
            "test": pl_module.testset,
        }
        
        return datasets
                    
    def construct_downstream(self, batch, pl_module, datatype):

        output = (
            pl_module(
                torch.cat([batch.cell_data, batch.x], axis=-1), batch.edge_index
            ).squeeze()
            if ("ci" in pl_module.hparams["regime"])
            else pl_module(batch.x, batch.edge_index).squeeze()
        )

        truth = (
            # edge_index: the list of the indices of the two nodes
            # pid[edge_index[0]]: the list of pid of the first node (a[0, 2, 1, 1, 2] -> [a[0], a[2], a[1], .....])
            # this line actually return [1, 1, 0, 1, 0, 0, 1, 1, ...], a boolean list of whether the first node and the second node are the same
            (batch.pid[batch.edge_index[0]] == batch.pid[batch.edge_index[1]]).float()
            if "pid" in pl_module.hparams["regime"]
            else batch.y
        )

        batch['score'] = F.sigmoid(output)
        batch['truth'] = truth

        self.save_downstream(batch, pl_module, datatype)
        

    def save_downstream(self, batch, pl_module, datatype):

        with open(
            os.path.join(self.output_dir, datatype, batch.event_file[-4:]), "wb"
        ) as pickle_file:
            torch.save(batch, pickle_file)

        logging.info("Saved event {}".format(batch.event_file[-4:]))
