import os
import os.path as osp
import torch
import imageio.v2 as iio
import numpy as np
import argparse

from src.config.configloading import load_config
from src.render import render, run_network
from src.trainer import Trainer
from src.loss import calc_mse_loss
from src.utils import get_psnr, get_mse, get_psnr_3d, get_ssim_3d, cast_to_image


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/abdomen_50.yaml",
                        help="configs file path")
    return parser

parser = config_parser()
args = parser.parse_args()

cfg = load_config(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicTrainer(Trainer):
    def __init__(self):
        """
        Basic network trainer.
        """
        super().__init__(cfg, device)
        print(f"[Start] exp: {cfg['exp']['expname']}, net: Basic network")

    def compute_loss(self, data, global_step, idx_epoch):
        rays = data["rays"].reshape(-1, 8)
        projs = data["projs"].reshape(-1)
        ret = render(rays, self.net, self.net_fine, **self.conf["render"])
        projs_pred = ret["acc"]

        loss = {"loss": 0.}
        calc_mse_loss(loss, projs, projs_pred)

        # Log
        for ls in loss.keys():
            self.writer.add_scalar(f"train/{ls}", loss[ls].item(), global_step)

        return loss["loss"]

    def eval_step(self, global_step, idx_epoch):
        """
        Evaluation step
        """
        # Evaluate projection
        select_ind = np.random.choice(len(self.eval_dset))
        projs = self.eval_dset.projs[select_ind]
        rays = self.eval_dset.rays[select_ind].reshape(-1, 8)
        H, W = projs.shape
        projs_pred = []
        for i in range(0, rays.shape[0], self.n_rays):
            projs_pred.append(render(rays[i:i+self.n_rays], self.net, self.net_fine, **self.conf["render"])["acc"])
        projs_pred = torch.cat(projs_pred, 0).reshape(H, W)

        # Evaluate density
        image = self.eval_dset.image
        image_pred = run_network(self.eval_dset.voxels, self.net_fine if self.net_fine is not None else self.net, self.netchunk)
        image_pred = image_pred.squeeze()
        loss = {
            "proj_mse": get_mse(projs_pred, projs),
            "proj_psnr": get_psnr(projs_pred, projs),
            "psnr_3d": get_psnr_3d(image_pred, image),
            "ssim_3d": get_ssim_3d(image_pred, image),
        }

        # Logging
        show_slice = 5
        show_step = image.shape[-1]//show_slice
        show_image = image[...,::show_step]
        show_image_pred = image_pred[...,::show_step]
        show = []
        for i_show in range(show_slice):
            show.append(torch.concat([show_image[..., i_show], show_image_pred[..., i_show]], dim=0))
        show_density = torch.concat(show, dim=1)
        show_proj = torch.concat([projs, projs_pred], dim=1)

        self.writer.add_image("eval/density (row1: gt, row2: pred)", cast_to_image(show_density), global_step, dataformats="HWC")
        self.writer.add_image("eval/projection (left: gt, right: pred)", cast_to_image(show_proj), global_step, dataformats="HWC")

        for ls in loss.keys():
            self.writer.add_scalar(f"eval/{ls}", loss[ls], global_step)
            
        # Save
        eval_save_dir = osp.join(self.evaldir, f"epoch_{idx_epoch:05d}")
        os.makedirs(eval_save_dir, exist_ok=True)
        np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred.cpu().detach().numpy())
        np.save(osp.join(eval_save_dir, "image_gt.npy"), image.cpu().detach().numpy())
        iio.imwrite(osp.join(eval_save_dir, "slice_show_row1_gt_row2_pred.png"), (cast_to_image(show_density)*255).astype(np.uint8))
        iio.imwrite(osp.join(eval_save_dir, "proj_show_left_gt_right_pred.png"), (cast_to_image(show_proj)*255).astype(np.uint8))
        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
            for key, value in loss.items(): 
                f.write("%s: %f\n" % (key, value.item()))

        return loss


trainer = BasicTrainer()
trainer.start()
        
