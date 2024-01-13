import vtk
import os
import random
import torch
import numpy as np
import utils
from vtkmodules.util import numpy_support
from pathlib import Path
from typing import Callable, List, Optional, Union
import re

class AsteroidDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, name, batch_points_num=1024*2, datatype='PointData', timestamp=None):
        super(AsteroidDataset, self).__init__()
        self.file_path = file_path
        self.name = name
        self.datatype = datatype
        self.timestamp = timestamp
        self.res = None
        self.v = None
        self.batch_points_num = batch_points_num

    def read_volume_data(self):
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(self.file_path)
        reader.Update()
        vtkimage = reader.GetOutput()

        print("using" + " volume data from data source " + self.file_path)
        if self.datatype == 'PointData':
            temp = vtkimage.GetPointData().GetScalars(self.name)
            self.res = np.array(vtkimage.GetDimensions())
            raw_array = numpy_support.vtk_to_numpy(temp)

            # self.v = normalize(raw_array, full_normalize=True)
            # TODO 先不做数据的归一化
            # self.v = standardization(raw_array)
            self.v = raw_array
            # print(np.min(self.v), np.max(self.v))
            self.v = self.v.reshape(self.res)

    
    def __getitem__(self, item):
        # 插值出来的batch_size
        # TODO 随机点三线性插值得到训练数据
        # points = get_random_points_inside_domain(num_points=self.batch_points_num,
        #                                          domain_min=vec3f(-1.), domain_max=vec3f(1.))
        # x, y, z = get_grid_xyz(vec3f(-1.), vec3f(1.), self.res)
        #
        # res = interp3(x, y, z, self.v, points)
        # res = np.expand_dims(res, axis=1)
        # return points, res'
        # 直接取值
        # TODO 根据item输出某个坐标
        point = utils.index_to_domain_xyz(item, utils.vec3f(0.0), utils.vec3f(1.0), self.res)
        xyz = utils.index_to_domain_xyz_index(item, self.res)
        val = self.v[xyz[0], xyz[1], xyz[2]]
        return point, np.array([val])
    
    def get_volume_res(self):
        return self.res

class Asteroids():

    def __init__(self, root, attr_name, limit = 100):
        self.root = root
        self.files = os.listdir(self.root)
        self.files = sorted(self.files, key=lambda x: x)
        self.data = []
        self.attr_name = attr_name
        count = 0
        for file in self.files:
            if count > limit:
                break
            data_pattern = r'.*?\.vti$'
            if re.match(data_pattern, file):
                self.data.append(AsteroidDataset(os.path.join(self.root,file), attr_name))
                count += 1

    def read_data(self):
        for data in self.data:
            data.read_volume_data()

    def get_volume_data(self):
        data = []
        for dat in self.data:
            data.append(dat.v)
        return np.array(data)
        
    def get_volume_res(self):
        time = len(self.data)
        res = self.data[0].get_volume_res()
        return np.append(time, res)

class Blocks:
    def __init__(self, volume_data, res, chunk=1024*512):
        self.res = np.array(res)
        self.chunk = chunk
        self.volume_data = volume_data
        if res.size == 4:
            self.has_timestamp = True
        else:
            self.has_timestamp = False

    # res为volume的大小, size为block的大小
    def uniform_part(self, size):
        pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), size).reshape([-1, 3])
        data_block_array = []
        if self.has_timestamp:
            [w, h, d] = self.res[1:] // size
            t = self.res[0]
            delta = size
            for time in range(t):
                for k in range(d):
                    for j in range(h):
                        for i in range(w):
                            data_block = self.volume_data[time,
                                         i * delta[0]:(i + 1) * delta[0],
                                         j * delta[1]:(j + 1) * delta[1],
                                         k * delta[2]:(k + 1) * delta[2]]
                            data_block = Block(data_block, size, pos, self.chunk)
                            data_block_array.append(data_block)
        else:
            [w, h, d] = self.res // size
            delta = size
            for k in range(d):
                for j in range(h):
                    for i in range(w):
                        data_block = self.volume_data[i * delta[0]:(i + 1) * delta[0],
                                                      j * delta[1]:(j + 1) * delta[1],
                                                      k * delta[2]:(k + 1) * delta[2]]
                        data_block = Block(data_block, size, pos, self.chunk)
                        data_block_array.append(data_block)
        return data_block_array

    def random_sample(self, size, block_num=None):
        pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), size).reshape([-1, 3])
        data_block_array = []
        if self.has_timestamp is False:
            if block_num is None:
                [w, h, d] = self.res // size
                block_num = w*h*d
            # 此处有bug
            num = (self.res - size).prod()
            xyz = utils.generate_shuffle_number(num)

            for i in range(block_num):
                left = utils.index_to_domain_xyz_index(xyz[i], self.res-size)
                right = left + size
                data_block = self.volume_data[left[0]:right[0],
                                              left[1]:right[1],
                                              left[2]:right[2]]
                data_block = Block(data_block, size, pos, self.chunk)
                data_block_array.append(data_block)
        else:
            times = self.res[0]
            if block_num is None:
                [w, h, d] = self.res[1:] // size
                block_num = w * h * d
            num = (self.res[1:] - size).prod()
            for time in range(times):
                xyz = utils.generate_shuffle_number(num)
                for i in range(block_num):
                    left = utils.index_to_domain_xyz_index(xyz[i], self.res[1:] - size)
                    right = left + size
                    data_block = self.volume_data[time,
                                                 left[0]:right[0],
                                                 left[1]:right[1],
                                                 left[2]:right[2]]
                    data_block = Block(data_block, size, pos, self.chunk)
                    data_block_array.append(data_block)
        return data_block_array
    
    def near_sample(self, block_size, block_num):
        pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), block_size).reshape([-1, 3])
        data_block_array = []
        
        t = np.random.random((block_num, 3))
        t = t-[0.5,0.5,0.5]
        # print(t)
        if self.has_timestamp is False:
            center = np.array([self.res[0]//2-block_size[0]//2, self.res[1]//2-block_size[1]//2, self.res[2]//2-block_size[2]//2])
            for i in range(block_num):
                offset = t[i]*block_size[0]//2
                data_block = self.volume_data[int(center[0]+offset[0]):int(center[0]+offset[0]+block_size[0]),
                            int(center[1]+offset[1]):int(center[1]+offset[1]+block_size[1]),
                            int(center[2]+offset[2]):int(center[2]+offset[2]+block_size[2])]
                data_block = Block(data_block, block_size, pos, self.chunk)
                data_block_array.append(data_block)
        else:
            center = np.array([self.res[1]//2-block_size[0]//2, self.res[2]//2-block_size[1]//2, self.res[3 ]//2-block_size[2]//2])
            # for time in range(self.res[0]):
            for time in range(1):
                for i in range(block_num):
                    offset = t[i]*block_size[0]//2
                    data_block = self.volume_data[time,
                                int(center[0]+offset[0]):int(center[0]+offset[0]+block_size[0]),
                                int(center[1]+offset[1]):int(center[1]+offset[1]+block_size[1]),
                                int(center[2]+offset[2]):int(center[2]+offset[2]+block_size[2])]
                    data_block = Block(data_block, block_size, pos, self.chunk)
                    data_block_array.append(data_block)
            # utils.vtk_draw_blocks([block.v for block in data_block_array])
        return data_block_array

    def same_sample(self, block_size, interval):
        pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), block_size).reshape([-1, 3])
        data_block_array = []
        center = np.array([self.res[1]//2-block_size[0]//2, self.res[2]//2-block_size[1]//2, self.res[3 ]//2-block_size[2]//2])
        # for time in range(self.res[0]):
        for time in range(1):
            data_block = self.volume_data[time,
                                int(center[0]):int(center[0]+block_size[0]),
                                int(center[1]):int(center[1]+block_size[1]),
                                int(center[2]):int(center[2]+block_size[2])]
            data_block = Block(data_block, block_size, pos, self.chunk)
            data_block_array.append(data_block)
        
        return data_block_array

    def generate_data_block(self, block_size, method='uniform', block_num=None):
        block_size = np.array(block_size)
        if method == 'uniform':
            return self.uniform_part(block_size)
        elif method == 'random':
            return self.random_sample(block_size, block_num)
        elif method == 'near':
            return self.near_sample(block_size, block_num)
        elif method == 'same':
            interval = 1
            return self.same_sample(block_size, interval)
        else:
            raise NotImplementedError

class Block(torch.utils.data.Dataset):
    # TODO 降低内存消耗 利于多个数据块的训练，可能cache也会更好一些
    def __init__(self, block_volume, block_size, pos, chunk=1024*256):
        self.v = block_volume
        self.res = np.array(block_size)
        self.chunk = chunk
        self.pos = pos

    def __getitem__(self, item):
        # 直接取值
        # point = index_to_domain_xyz(item, vec3f(0.0), vec3f(1.0), self.res)
        # xyz = index_to_domain_xyz_index(item, self.res)
        # val = self.v[xyz[0], xyz[1], xyz[2]]
        # return point, np.array([val])

        pos_ = self.pos[item*self.chunk:(item+1)*self.chunk, ...]
        vol = self.v.reshape([-1,1])
        vol = vol[item*self.chunk:(item+1)*self.chunk, ...]
        num = utils.generate_shuffle_number(vol.shape[0])
        # shuffle
        return pos_[num, ...], vol[num, ...]

    def __len__(self):
        num_points = self.res.prod()
        return int(np.ceil(num_points/self.chunk))
    
# # TODO 分解这个数据集,降低显存的使用
# class MetaDataset(torch.utils.data.Dataset):
#     def __init__(self, data_block_array, maml_chunk_num=10):
#         super(MetaDataset, self).__init__()
#         self.data_block_array = data_block_array
#         self.chunk = maml_chunk_num
#         vol = []
#         num = len(self.data_block_array)
#         for data_block in self.data_block_array:
#             vol.append(data_block.v.flatten())
#         vol = np.expand_dims(np.array(vol), axis=-1)
#         pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), self.data_block_array[0].res).reshape([-1, 3])
#         pos = np.expand_dims(pos, 0).repeat(num, axis=0)
#         # TODO 分 batch_size
#         self.res = np.concatenate((pos, vol), axis=-1)

#     def __len__(self):
#         return self.chunk

#     def __getitem__(self, item):
#         # block层级之间 shuffle
#         # np.random.shuffle(self.res[:, ...])
#         num_points = self.data_block_array[0].res.prod()
#         chunk_num = int(num_points/self.chunk)
#         return self.res[:, item*chunk_num:(item+1)*chunk_num, ...]

class MetaDataset(torch.utils.data.Dataset):
    def __init__(self, data_block_array, maml_chunk_num=10):
        super(MetaDataset, self).__init__()
        self.data_block_array = data_block_array
        self.chunk = maml_chunk_num
        vol = []
        num = len(self.data_block_array)
        for data_block in self.data_block_array:
            vol.append(data_block.v.flatten())
        vol = np.expand_dims(np.array(vol), axis=-1)
        pos = utils.get_query_coords(utils.vec3f(-1), utils.vec3f(1), self.data_block_array[0].res).reshape([-1, 3])
        pos = np.expand_dims(pos, 0).repeat(num, axis=0)
        # TODO 分 batch_size
        self.res = np.concatenate((pos, vol), axis=-1)
        self.index = np.arange(len(self.data_block_array))
        np.random.shuffle(self.index)

    def __len__(self):
        return len(self.data_block_array)

    def __getitem__(self, item):
        # block层级之间 shuffle
        # np.random.shuffle(self.res[:, ...])
        
        # num_points = self.data_block_array[0].res.prod()
        # chunk_num = int(num_points/self.chunk)
        # return self.res[:, item*chunk_num:(item+1)*chunk_num, ...]
        
        return self.res[self.index[item], ...]
        