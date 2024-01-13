import numpy as np
import torch
import renderer

def get_grid_xyz(minlim, maxlim, cube_res):
    x = np.linspace(minlim[0], maxlim[0], cube_res[0])
    y = np.linspace(minlim[1], maxlim[1], cube_res[1])
    z = np.linspace(minlim[2], maxlim[2], cube_res[2])
    return x, y, z

def vtk_draw_blocks(block_arrays, off_screen=False, file_name=None):
    paths = []
    arrs = []
    for array in block_arrays:
        arrs.append(array)
        paths.append(None)
    res = block_arrays[-1].shape
    ren = renderer.VolumeRender(paths, arrs, res, len(block_arrays))
    # ren = renderer.VolumeRender(paths, arrs, res, 5)
    # ren.render(off_screen, file_name, len(block_arrays))
    ren.render(off_screen, file_name)
    del ren

def get_query_coords(minlim, maxlim, cube_res):
    """
        Get regular coordinates for querying the block implicit representation
    """
    x, y, z = get_grid_xyz(minlim, maxlim, cube_res)

    X, Y, Z = np.meshgrid(x, y, z)
    coords_gen = np.hstack((X.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            Z.reshape(-1, 1)))

    return coords_gen

# 从全局点序号转换为[domain_min,domain_max]之间的坐标
def index_to_domain_xyz(index, domain_min, domain_max, res):
        z_index = index // (res[0]*res[1])
        y_index = index % (res[0]*res[1]) // res[1]
        x_index = index % (res[0]*res[1]) % res[1]
        z = z_index * (domain_max[2]-domain_min[2]) / res[2]
        y = y_index * (domain_max[1]-domain_min[1]) / res[1]
        x = x_index * (domain_max[0]-domain_min[0]) / res[0]
        return np.array([x,y,z])
    
#一维点序号转为三维点序号
def index_to_domain_xyz_index(index, res):
    z_index = index // (res[0]*res[1])
    y_index = index % (res[0]*res[1]) // res[0]
    x_index = index % (res[0]*res[1]) % res[0]
    return np.array([x_index, y_index, z_index])

def vec3f(x, y=None, z=None):
    if y is None:
        return torch.tensor([x, x, x])
    else:
        return torch.tensor([x, y, z])
    
def generate_shuffle_number(num):
    return np.random.permutation(num)