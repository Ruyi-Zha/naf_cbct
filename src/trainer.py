import os
import os.path as osp
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from shutil import copyfile
import numpy as np

from .dataset import TIGREDataset as Dataset
from .network import get_network
from .encoder import get_encoder


class Trainer:
    def __init__(self, cfg, device="cuda"):

        # Args
        self.global_step = 0
        self.conf = cfg
        self.n_fine = cfg["render"]["n_fine"]
        self.epochs = cfg["train"]["epoch"]
        self.i_eval = cfg["log"]["i_eval"]
        self.i_save = cfg["log"]["i_save"]
        self.netchunk = cfg["render"]["netchunk"]
        self.n_rays = cfg["train"]["n_rays"]
  
        # Log direcotry
        self.expdir = osp.join(cfg["exp"]["expdir"], cfg["exp"]["expname"])
        self.ckptdir = osp.join(self.expdir, "ckpt.tar")
        self.ckptdir_backup = osp.join(self.expdir, "ckpt_backup.tar")
        self.evaldir = osp.join(self.expdir, "eval")
        os.makedirs(self.evaldir, exist_ok=True)

        # Dataset
        train_dset = Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "train", device)
        self.eval_dset = Dataset(cfg["exp"]["datadir"], cfg["train"]["n_rays"], "val", device) if self.i_eval > 0 else None
        self.train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=cfg["train"]["n_batch"])
        self.voxels = self.eval_dset.voxels if self.i_eval > 0 else None
    
        # Network
        network = get_network(cfg["network"]["net_type"])
        cfg["network"].pop("net_type", None)
        encoder = get_encoder(**cfg["encoder"])
        self.net = network(encoder, **cfg["network"]).to(device)
        grad_vars = list(self.net.parameters())
        self.net_fine = None
        if self.n_fine > 0:
            self.net_fine = network(encoder, **cfg["network"]).to(device)
            grad_vars += list(self.net_fine.parameters())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=cfg["train"]["lrate"], betas=(0.9, 0.999))
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer, gamma=cfg["train"]["lrate_gamma"])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=cfg["train"]["lrate_step"], gamma=cfg["train"]["lrate_gamma"])

        # Load checkpoints
        self.epoch_start = 0
        if cfg["train"]["resume"] and osp.exists(self.ckptdir):
            print(f"Load checkpoints from {self.ckptdir}.")
            ckpt = torch.load(self.ckptdir)
            self.epoch_start = ckpt["epoch"] + 1
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.global_step = self.epoch_start * len(self.train_dloader)
            self.net.load_state_dict(ckpt["network"])
            if self.n_fine > 0:
                self.net_fine.load_state_dict(ckpt["network_fine"])

        # Summary writer
        self.writer = SummaryWriter(self.expdir)
        self.writer.add_text("parameters", self.args2string(cfg), global_step=0)


    def args2string(self, hp):
        """
        Transfer args to string.
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))

    def start(self):
        """
        Main loop.
        """
        def fmt_loss_str(losses):
            return "".join(", " + k + ": " + f"{losses[k].item():.3g}" for k in losses)
        iter_per_epoch = len(self.train_dloader)
        pbar = tqdm(total= iter_per_epoch * self.epochs, leave=True)
        if self.epoch_start > 0:
            pbar.update(self.epoch_start*iter_per_epoch)

        for idx_epoch in range(self.epoch_start, self.epochs):

            # Evaluate
            if idx_epoch % self.i_eval == 0 and self.i_eval > 0:
                self.net.eval()
                with torch.no_grad():
                    loss_test = self.eval_step(global_step=self.global_step)
                self.net.train()
                tqdm.write(f"[EVAL] epoch: {idx_epoch}/{self.epochs}{fmt_loss_str(loss_test)}")
            
            # Train
            for data in self.train_dloader:
                self.global_step += 1
                # Train
                self.net.train()
                loss_train = self.train_step(data, global_step=self.global_step, idx_epoch=idx_epoch)
                pbar.set_description(f"epoch={idx_epoch}/{self.epochs}, loss={loss_train:.3g}, lr={self.optimizer.param_groups[0]['lr']:.3g}")
                pbar.update(1)
            
            # Save
            if (idx_epoch+1) % self.i_save == 0 or idx_epoch == self.epochs-1:
                if osp.exists(self.ckptdir):
                    copyfile(self.ckptdir, self.ckptdir_backup)
                tqdm.write(f"[SAVE] epoch: {idx_epoch}/{self.epochs}, path: {self.ckptdir}")
                torch.save(
                    {
                        "epoch": idx_epoch,
                        "network": self.net.state_dict(),
                        "network_fine": self.net_fine.state_dict() if self.n_fine > 0 else None,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.ckptdir,
                )

            # Update lrate
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.lr_scheduler.step()

        print(f"Training complete! {self.expdir}")

    def train_step(self, data, global_step, idx_epoch):
        """
        Training step
        """
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, global_step, idx_epoch)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def compute_loss(self, data, global_step, idx_epoch):
        """
        Training step
        """
        raise NotImplementedError()


    def eval_step(self, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()
        