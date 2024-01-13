import coinpp.conversion as conversion
import coinpp.losses as losses
import coinpp.metalearning as metalearning
import torch
import wandb
import numpy as np
import utils
import os
from math import ceil
import logging
import copy
import sys
from accelerate import Accelerator
from tqdm import tqdm, trange

class Trainer:
    def __init__(
        self,
        func_rep,
        converter,
        args,
        train_dataset,
        test_dataset,
        initial_dataset,
        patcher=None,
        model_path="",
    ):
        """Module to handle meta-learning of COIN++ model.

        Args:
            func_rep (models.ModulatedSiren):
            converter (conversion.Converter):
            args: Training arguments (see main.py).
            train_dataset:
            test_dataset:
            patcher: If not None, patcher that is used to create random patches during
                training and to partition data into patches during validation.
            model_path: If not empty, wandb path where best (validation) model
                will be saved.
        """
        self.func_rep = func_rep
        self.converter = converter
        self.args = args
        self.patcher = patcher

        params = [param for name, param in self.func_rep.named_parameters()]
        # self.outer_optimizer = torch.optim.Adam(
        #     self.func_rep.parameters(), lr=args.outer_lr
        # )

        self.outer_optimizer1 = torch.optim.Adam(
            params[:], lr=args.outer_lr
        )
        self.outer_optimizer2 = torch.optim.Adam(
            params[22:], lr=args.outer_lr
        )
        self.outer_optimizer = self.outer_optimizer1
        self.initial_optimizer = torch.optim.Adam(
            params[:], lr=args.initial_lr, weight_decay=0.0001
        )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.initial_dataset = initial_dataset
        self._process_datasets()

        self.model_path = model_path
        self.step = 0
        self.best_val_psnr = 0.0
        self.num_accumulation = 20
        self.initial_epoch = args.initial_epochs
        
        count = 0
        self.base_dir = os.path.join(self.args.work_space,'imgs',f'base_net_width_{self.args.dim_hidden}_height_{self.args.num_layers}_use_latent_{self.args.use_latent}_accumulated_{self.args.accumulated}')
        self.dir = os.path.join(self.base_dir, f'_{count}')
        while os.path.exists(self.dir):
            count += 1
            self.dir = os.path.join(self.base_dir, f'_{count}')
        os.makedirs(self.dir, exist_ok=True)
        logfile = os.path.join(self.dir,'logfile')
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        logging.basicConfig(filename=logfile, level=logging.DEBUG, format=LOG_FORMAT)
        file_handler = logging.FileHandler(logfile, mode="w")
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(file_handler)

    def _process_datasets(self):
        """Create dataloaders for datasets based on self.args."""
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=0,
            pin_memory=self.args.num_workers > 0,
        )

        # If we are using patching, require data loader to have a batch size of 1,
        # since we can potentially have different sized outputs which cannot be batched
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=1 if self.patcher else self.args.batch_size,
            num_workers=self.args.num_workers,
        )

        self.initial_dataloader = torch.utils.data.DataLoader(
            self.initial_dataset,
            shuffle=False,
            batch_size=16 if self.patcher else self.args.batch_size,
            num_workers=self.args.num_workers,
        )

    def train_epoch(self):
        """Train model for a single epoch."""
        block_num = 0
        losses = []
        psnrs = []
        for data in self.train_dataloader:
            data = data.to(self.args.device)
            coordinates, features = self.converter.to_coordinates_and_features(data)
            block_num += features.shape[0]
            
            # Optionally subsample points
            if self.args.subsample_num_points != -1:
                # Coordinates have shape (batch_size, *, coordinate_dim)
                # Features have shape (batch_size, *, feature_dim)
                # Flatten both along spatial dimension and randomly select points
                coordinates = coordinates.reshape(
                    coordinates.shape[0], -1, coordinates.shape[-1]
                )
                features = features.reshape(features.shape[0], -1, features.shape[-1])
                # Compute random indices (no good pytorch function to do this,
                # so do it this slightly hacky way)
                permutation = torch.randperm(coordinates.shape[1])
                idx = permutation[: self.args.subsample_num_points]
                coordinates = coordinates[:, idx, :]
                features = features[:, idx, :]

            outputs = metalearning.outer_step(
                self.func_rep,
                coordinates,
                features,
                inner_steps=self.args.inner_steps,
                inner_lr=self.args.inner_lr,
                is_train=True,
                return_reconstructions=False,
                gradient_checkpointing=self.args.gradient_checkpointing,
            )

            if self.args.accumulated:
                loss = outputs["loss"]/self.num_accumulation*features.shape[0]
                losses.append(outputs["loss"])
                psnrs.append(outputs["psnr"])
                loss.backward(create_graph=False)
                # Update parameters of base network
                if block_num == self.num_accumulation:
                    mean_loss = sum(losses)/len(losses)
                    mean_psnr = sum(psnrs)/len(psnrs)
                    self.outer_optimizer.step()
                    block_num = 0
                    self.outer_optimizer.zero_grad()

                    # if self.step % self.args.validate_every == 0 and self.step != 0:
                    #     self.validation()
                    
                    log_dict = {"loss": mean_loss, "psnr": mean_psnr}
                    self.step += 1
                    print(f'Step {self.step}, Loss {mean_loss:.4f}, PSNR {mean_psnr:.4f}')
                    # if mean_psnr >= 20:
                    #     self.outer_optimizer = self.outer_optimizer2
                    losses=[]
                    psnrs=[]
                    self.logger.info(f'{log_dict}')
                    if self.args.use_wandb:
                        wandb.log(log_dict, step=self.step)
            else:
                self.outer_optimizer.zero_grad()
                outputs["loss"].backward(create_graph=False)
                # accelerator.backward(outputs["loss"])
                self.outer_optimizer.step()

                # if self.step % self.args.validate_every == 0 and self.step != 0:
                    # self.validation()

                log_dict = {"loss": outputs["loss"].item(), "psnr": outputs["psnr"]}
                self.step += 1
                print(f'Step {self.step}, Loss {log_dict["loss"]:.3f}, PSNR {log_dict["psnr"]:.3f}')

                self.logger.info(f'{log_dict}')
                if self.args.use_wandb:
                    wandb.log(log_dict, step=self.step)
            
            if self.step % self.args.warm_up ==0 and self.step != 0:
                new_learning_rate = min(8e-5, self.args.outer_lr + 3e-6)
                for param_group in self.outer_optimizer.param_groups:
                    param_group['lr'] = new_learning_rate

    def validation(self):
        """Run trained model on validation dataset."""
        print(f"\nValidation, Step {self.step}:")
        

        # If num_validation_points is -1, validate on entire validation dataset,
        # otherwise validate on a subsample of points
        full_validation = self.args.num_validation_points == -1
        num_validation_batches = self.args.num_validation_points // self.args.batch_size

        # Initialize validation logging dict
        log_dict = {}
        log_dict[f"\nValidation, Step"] = self.step
        
        # Evaluate model for different numbers of inner loop steps
        for inner_steps in self.args.validation_inner_steps:
            log_dict[f"val_psnr_{inner_steps}_steps"] = 0.0
            log_dict[f"val_loss_{inner_steps}_steps"] = 0.0

            # Fit modulations for each validation datapoint
            for i, data in enumerate(self.test_dataloader):
                data = data.to(self.args.device)
                if self.patcher:
                    # If using patching, test data will have a batch size of 1.
                    # Remove batch dimension and instead convert data into
                    # patches, with patch dimension acting as batch size
                    patches, spatial_shape = self.patcher.patch(data[0])
                    coordinates, features = self.converter.to_coordinates_and_features(
                        patches
                    )

                    # As num_patches may be much larger than args.batch_size,
                    # split the fitting of patches into batch_size chunks to
                    # reduce memory
                    outputs = metalearning.outer_step_chunked(
                        self.func_rep,
                        coordinates,
                        features,
                        inner_steps=inner_steps,
                        inner_lr=self.args.inner_lr,
                        chunk_size=self.args.batch_size,
                        gradient_checkpointing=self.args.gradient_checkpointing,
                    )

                    # Shape (num_patches, *patch_shape, feature_dim)
                    patch_features = outputs["reconstructions"]

                    # When using patches, we cannot directly use psnr and loss
                    # output by outer step, since these are calculated on the
                    # padded patches. Therefore we need to reconstruct the data
                    # in its original unpadded form and manually calculate mse
                    # and psnr
                    # Shape (num_patches, *patch_shape, feature_dim) ->
                    # (num_patches, feature_dim, *patch_shape)
                    patch_data = conversion.features2data(patch_features, batched=True)
                    # Shape (feature_dim, *spatial_shape)
                    data_recon = self.patcher.unpatch(patch_data, spatial_shape)
                    # Calculate MSE and PSNR values and log them
                    mse = losses.mse_fn(data_recon, data[0])
                    psnr = losses.mse2psnr(mse)
                    log_dict[f"val_psnr_{inner_steps}_steps"] += psnr.item()
                    log_dict[f"val_loss_{inner_steps}_steps"] += mse.item()
                else:
                    func_rep = copy.deepcopy(self.func_rep)
                    coordinates, features = self.converter.to_coordinates_and_features(
                        data
                    )
                    
                    for outer_step in range(5):
                        outputs = metalearning.outer_step(
                            self.func_rep,
                            coordinates,
                            features,
                            inner_steps=inner_steps,
                            inner_lr=self.args.inner_lr,
                            is_train=True,
                            return_reconstructions=True,
                            gradient_checkpointing=self.args.gradient_checkpointing,
                        )
                        
                        # Update parameters of base network
                        self.outer_optimizer.zero_grad()
                        outputs["loss"].backward(create_graph=False)
                        self.outer_optimizer.step()
                    self.func_rep = func_rep
                    log_dict[f"val_psnr_{inner_steps}_steps"] += outputs["psnr"]
                    log_dict[f"val_loss_{inner_steps}_steps"] += outputs["loss"].item()
                    
                    # draw blocks
                    # os.makedirs(os.path.join(self.args.work_space,'imgs',f'base_net_width_{self.args.dim_hidden}_height_{self.args.num_layers}_use_latent_{self.args.use_latent}'), exist_ok=True)
                    file_name1 = os.path.join(self.dir,f'step_{self.step}.png')
                    file_name2 = os.path.join(self.dir,f'step_{self.step}_recon.png')
                    features_recon = outputs["reconstructions"]
                    utils.vtk_draw_blocks(features.reshape(features.shape[0],int(ceil(features.shape[1]**(1.0/3))),int(ceil(features.shape[1]**(1.0/3))),int(ceil(features.shape[1]**(1.0/3)))).cpu(), off_screen=True, file_name=file_name1)
                    utils.vtk_draw_blocks(features_recon.reshape(features.shape[0],int(ceil(features.shape[1]**(1.0/3))),int(ceil(features.shape[1]**(1.0/3))),int(ceil(features.shape[1]**(1.0/3)))).cpu().detach().numpy(), off_screen=True, file_name=file_name2)

                if not full_validation and i >= num_validation_batches - 1:
                    break

            # Calculate average PSNR and loss by dividing by number of batches
            log_dict[f"val_psnr_{inner_steps}_steps"] /= i + 1
            log_dict[f"val_loss_{inner_steps}_steps"] /= i + 1

            mean_psnr, mean_loss = (
                log_dict[f"val_psnr_{inner_steps}_steps"],
                log_dict[f"val_loss_{inner_steps}_steps"],
            )
            print(
                f"Inner steps {inner_steps}, Loss {mean_loss:.5f}, PSNR {mean_psnr:.3f}"
            )

            # Use first setting of inner steps for best validation PSNR
            if inner_steps == self.args.validation_inner_steps[0]:
                if mean_psnr > self.best_val_psnr:
                    self.best_val_psnr = mean_psnr
                    # Optionally save new best model
                    if self.args.use_wandb and self.model_path:
                        torch.save(
                            {
                                "args": self.args,
                                "state_dict": self.func_rep.state_dict(),
                            },
                            self.model_path,
                        )

            if self.args.use_wandb:
                # Store final batch of reconstructions to visually inspect model
                # Shape (batch_size, channels, *spatial_dims)
                reconstruction = self.converter.to_data(
                    None, outputs["reconstructions"]
                )
                if self.patcher:
                    # If using patches, unpatch the reconstruction
                    # Shape (channels, *spatial_dims)
                    reconstruction = self.patcher.unpatch(reconstruction, spatial_shape)
                if self.converter.data_type == "mri":
                    # To store an image, slice MRI data along a single dimension
                    # Shape (1, depth, height, width) -> (1, height, width)
                    reconstruction = reconstruction[:, reconstruction.shape[1] // 2]

                if self.converter.data_type == "audio":
                    # Currently only support audio saving when using patches
                    if self.patcher:
                        # Unnormalize data from [0, 1] to [-1, 1] as expected by wandb
                        if self.test_dataloader.dataset.normalize:
                            reconstruction = 2 * reconstruction - 1
                        # Saved audio sample needs shape (num_samples, num_channels),
                        # so transpose
                        log_dict[
                            f"val_reconstruction_{inner_steps}_steps"
                        ] = wandb.Audio(
                            reconstruction.T.cpu(),
                            sample_rate=self.test_dataloader.dataset.sample_rate,
                        )
                else:
                    log_dict[f"val_reconstruction_{inner_steps}_steps"] = wandb.Image(
                        reconstruction
                    )

                wandb.log(log_dict, step=self.step)
        self.logger.info(f'{log_dict}')
        print("\n")

    def initial_model(self):
        for i in trange(self.initial_epoch):
            for data in self.initial_dataloader:
                data = data.to(self.args.device)
                coordinates, features = self.converter.to_coordinates_and_features(data)

                outputs = metalearning.outer_step(
                    self.func_rep,
                    coordinates,
                    features,
                    inner_steps=self.args.inner_steps,
                    inner_lr=self.args.inner_lr,
                    is_train=True,
                    return_reconstructions=True,
                    gradient_checkpointing=self.args.gradient_checkpointing,
                )

                self.initial_optimizer.zero_grad()
                outputs["loss"].backward(create_graph=False)
                self.initial_optimizer.step()
                log_dict = {"loss": outputs["loss"].item(), "psnr": outputs["psnr"]}
                self.step += 1
                print(f'Step {self.step}, Loss {log_dict["loss"]:.3f}, PSNR {log_dict["psnr"]:.3f}')

                # if self.step % self.args.warm_up ==0 and self.step != 0:
                #     new_learning_rate = min(8e-5, self.args.outer_lr + 3e-6)
                #     for param_group in self.outer_optimizer.param_groups:
                #         param_group['lr'] = new_learning_rate
            if self.step % 100 == 0 and self.step != 0:
                features_recon = outputs["reconstructions"]
                OriginVsReconsturct = [features.reshape(int(ceil(features.shape[1]**(1.0/3))),int(ceil(features.shape[1]**(1.0/3))),int(ceil(features.shape[1]**(1.0/3)))).cpu(),
                                    features_recon.reshape(int(ceil(features.shape[1]**(1.0/3))),int(ceil(features.shape[1]**(1.0/3))),int(ceil(features.shape[1]**(1.0/3)))).cpu().detach().numpy()]
                utils.vtk_draw_blocks(OriginVsReconsturct)


        self.step = 0
        
        # utils.vtk_draw_blocks(features_recon.reshape(features.shape[0],int(ceil(features.shape[1]**(1.0/3))),int(ceil(features.shape[1]**(1.0/3))),int(ceil(features.shape[1]**(1.0/3)))).cpu().detach().numpy())