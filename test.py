import data.Asteroid as Asteroid
import vtk
import numpy as np
from vtkmodules.util import numpy_support
import utils
import coinpp.conversion as conversion
import helpers
import coinpp.models as models
import torch
from coinpp.training import Trainer
import argparse
from main import add_arguments
import coinpp.metalearning as metalearning
from tqdm import tqdm, trange
import coinpp.losses as losses
from datetime import datetime
from helpers import get_dataset_root
from itertools import islice
from collections import OrderedDict

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
add_arguments(parser)
args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.GPU)

train_dataset, test_dataset, converter, initial_dataset = helpers.get_datasets_and_converter(args)

asteroid = Asteroid.Asteroids(root=get_dataset_root("Asteroid_test"), attr_name='v02')
asteroid.read_data()
blocks = Asteroid.Blocks(asteroid.get_volume_data(),asteroid.get_volume_res())
pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), [50,50,50]).reshape([-1, 3])
pos1 = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), [100,100,100]).reshape([-1, 3])
block1 = Asteroid.Block(blocks.volume_data[0][100:150,100:150,100:150], [50,50,50], pos)
block2 = Asteroid.Block(blocks.volume_data[0][120:170,120:170,120:170], [50,50,50], pos)
block3 = Asteroid.Block(blocks.volume_data[0][150:200,150:200,150:200], [50,50,50], pos)
block4 = Asteroid.Block(blocks.volume_data[0][100:200,100:200,100:200], [100,100,100], pos1)
test_dataset = Asteroid.MetaDataset([block4], args.maml_chunk_num)
initial_dataset = Asteroid.MetaDataset([block4], args.maml_chunk_num)

utils.vtk_draw_blocks([block4.v])
utils.vtk_draw_blocks([block1.v, block2.v, block3.v])

test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=args.num_workers,
        )
initial_dataloader = torch.utils.data.DataLoader(
            initial_dataset,
            shuffle=False,
            batch_size=1,
            num_workers=args.num_workers,
        )

model = helpers.get_model(args)
params = [param for name, param in model.named_parameters()]
optimizer1 = torch.optim.Adam(params[22:], lr=args.outer_lr)
optimizer2 = torch.optim.Adam(params[:22], lr=args.outer_lr)

# for i in trange(5000):
#     for data in initial_dataloader:
#         data = data.to(args.device)
#         coordinates, features = converter.to_coordinates_and_features(data)
#         modulations = torch.zeros(1, model.modulation_net.latent_dim, device=coordinates.device).requires_grad_()

#         features_recon = model.forward(coordinates, modulations)
#         per_example_loss = losses.batch_mse_fn(features_recon, features)
#         loss = per_example_loss.mean()

#         optimizer2.zero_grad()
#         loss.backward()
#         optimizer2.step()
#         print(f'Loss {loss}, PSNR {-10.0 * torch.log10(loss)}')

#         if i % 500 == 0 and i != 0:
#             # 保存模型
#             save_path = 'coinpp/models/'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#             torch.save(model.state_dict(), save_path)

state_dict = torch.load('/home/WeiTeng/coinpp/coinpp/models/2024-01-12_22-31-28')
# state_dict = OrderedDict(islice(state_dict.items(), 0, 22))
# current_state_dict = model.state_dict()
# current_state_dict.update(state_dict)
# model.load_state_dict(current_state_dict)

model.load_state_dict(state_dict)

for j, data in enumerate(tqdm(test_dataloader)):
    data = data.to(args.device)
    coordinates, features = converter.to_coordinates_and_features(data)
    modulations = torch.zeros(1, model.modulation_net.latent_dim, device=coordinates.device).requires_grad_()
    for i in range(10000):
        features_recon = model.modulated_forward(coordinates, modulations)
        per_example_loss = losses.batch_mse_fn(features_recon, features)
        loss = per_example_loss.mean()

        # Perform single gradient descent step
        grad =torch.autograd.grad(
            loss,
            modulations,
            retain_graph=True)
        modulations = modulations - args.inner_lr * grad[0]
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        print(f'Block {j} Loss {loss} PSNR {-10.0 * torch.log10(loss)}')
        
