# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import namedtuple

import torch
import torch.nn as nn

from smplx.lbs import transform_mat


PerspParams = namedtuple('ModelOutput',
                         ['rotation', 'translation', 'center',
                          'focal_length'])

def extract_cam_param_xml(xml_path='', dtype=torch.float32):
    from bs4 import BeautifulSoup
    
    # pip install beautifulsoup4
    # pip install lxml
    # Reading the data inside the xml
    # file to a variable under the name 
    # data
    with open(xml_path, 'r') as f:
        data = f.read()
    
    # Passing the stored data inside
    # the beautifulsoup parser, storing
    # the returned object 
    Bs_data = BeautifulSoup(data, "xml")

    extrinsics_mat = [float(s) for s in Bs_data.find_all('CameraMatrix')[0].find_all('data')[0].text.split()]
    intrinsics_mat = [float(s) for s in Bs_data.find_all('Intrinsics')[0].find_all('data')[0].text.split()]
    distortion_vec = [float(s) for s in Bs_data.find_all('Distortion')[0].find_all('data')[0].text.split()]

    focal_length_x = intrinsics_mat[0]
    focal_length_y = intrinsics_mat[4]
    center = torch.tensor([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
    
    rotation = torch.tensor([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]], 
                            [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]], 
                            [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

    translation = torch.tensor([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

    # t = -Rc --> c = -R^Tt
    cam_center = [  -extrinsics_mat[0]*extrinsics_mat[3] - extrinsics_mat[4]*extrinsics_mat[7] - extrinsics_mat[8]*extrinsics_mat[11],
                    -extrinsics_mat[1]*extrinsics_mat[3] - extrinsics_mat[5]*extrinsics_mat[7] - extrinsics_mat[9]*extrinsics_mat[11], 
                    -extrinsics_mat[2]*extrinsics_mat[3] - extrinsics_mat[6]*extrinsics_mat[7] - extrinsics_mat[10]*extrinsics_mat[11]]

    cam_center = torch.tensor([cam_center], dtype=dtype)

    k1 = torch.tensor([distortion_vec[0]], dtype=dtype)
    k2 = torch.tensor([distortion_vec[1]], dtype=dtype)

    return focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2

# def extract_cam_param_xml(xml_path='', dtype=torch.float32):
    
#     import xml.etree.ElementTree as ET
#     tree = ET.parse(xml_path)

#     extrinsics_mat = [float(s) for s in tree.find('./CameraMatrix/data').text.split()]
#     intrinsics_mat = [float(s) for s in tree.find('./Intrinsics/data').text.split()]
#     distortion_vec = [float(s) for s in tree.find('./Distortion/data').text.split()]

#     focal_length_x = intrinsics_mat[0]
#     focal_length_y = intrinsics_mat[4]
#     center = torch.tensor([[intrinsics_mat[2], intrinsics_mat[5]]], dtype=dtype)
    
#     rotation = torch.tensor([[extrinsics_mat[0], extrinsics_mat[1], extrinsics_mat[2]], 
#                             [extrinsics_mat[4], extrinsics_mat[5], extrinsics_mat[6]], 
#                             [extrinsics_mat[8], extrinsics_mat[9], extrinsics_mat[10]]], dtype=dtype)

#     translation = torch.tensor([[extrinsics_mat[3], extrinsics_mat[7], extrinsics_mat[11]]], dtype=dtype)

#     # t = -Rc --> c = -R^Tt
#     cam_center = [  -extrinsics_mat[0]*extrinsics_mat[3] - extrinsics_mat[4]*extrinsics_mat[7] - extrinsics_mat[8]*extrinsics_mat[11],
#                     -extrinsics_mat[1]*extrinsics_mat[3] - extrinsics_mat[5]*extrinsics_mat[7] - extrinsics_mat[9]*extrinsics_mat[11], 
#                     -extrinsics_mat[2]*extrinsics_mat[3] - extrinsics_mat[6]*extrinsics_mat[7] - extrinsics_mat[10]*extrinsics_mat[11]]

#     cam_center = torch.tensor([cam_center], dtype=dtype)

#     k1 = torch.tensor([distortion_vec[0]], dtype=dtype)
#     k2 = torch.tensor([distortion_vec[1]], dtype=dtype)

#     return focal_length_x, focal_length_y, center, rotation, translation, cam_center, k1, k2

def create_camera(camera_type='persp', **kwargs):
    if camera_type.lower() == 'persp':
        return PerspectiveCamera(**kwargs)
    elif camera_type.lower() == 'user':
        return CalibratedUserCamera(**kwargs)
    else:
        raise ValueError('Uknown camera type: {}'.format(camera_type))

def create_multicameras(xml_folder='', **kwargs):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = sorted([join(xml_folder, f) for f in listdir(xml_folder) if isfile(join(xml_folder, f))])       # take only files
    
    return [CalibratedUserCamera(**{**kwargs, 'calib_path':f}) for f in onlyfiles]
    


class PerspectiveCamera(nn.Module):

    FOCAL_LENGTH = 5000

    def __init__(self, rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None,
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(PerspectiveCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_x is None else
                focal_length_x,
                dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_y is None else
                focal_length_y,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        if rotation is None:
            rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)

        rotation = nn.Parameter(rotation, requires_grad=True)
        self.register_parameter('rotation', rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)

        translation = nn.Parameter(translation,
                                   requires_grad=True)
        self.register_parameter('translation', translation)

    def forward(self, points):
        device = points.device

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points

class CalibratedUserCamera(nn.Module):

    def __init__(self, calib_path='', rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None, 
                 batch_size=1,
                 center=None, dtype=torch.float32, **kwargs):
        super(CalibratedUserCamera, self).__init__()
        # TODO: camera batch always=1
        self.batch_size = batch_size
        # self.batch_size = 1
        self.dtype = dtype
        self.calib_path = calib_path
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([self.batch_size], dtype=dtype))

        import os.path as osp
        if not osp.exists(calib_path):
            raise FileNotFoundError('Could''t find {}.'.format(calib_path))
        else:
            focal_length_x, focal_length_y, center, rotation, translation, cam_center, _, _ \
                    = extract_cam_param_xml(xml_path=calib_path, dtype=dtype)
        
        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [self.batch_size],               
                focal_length_x,
                dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [self.batch_size],                
                focal_length_y,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([self.batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)

        rotation = rotation.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        rotation = nn.Parameter(rotation, requires_grad=False)
    
        self.register_parameter('rotation', rotation)

        if translation is None:
            translation = torch.zeros([self.batch_size, 3], dtype=dtype)

        translation = nn.Parameter(translation, requires_grad=False)
        self.register_parameter('translation', translation)
        
        cam_center = nn.Parameter(cam_center, requires_grad=False)
        self.register_parameter('cam_center', cam_center)

    def forward(self, points):
        # import pdb;pdb.set_trace()
        device = points.device

        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1).repeat(self.batch_size, 1, 1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points