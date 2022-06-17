import torch
import torch.nn as nn
import numpy as np

from sdf import SDF

class SDFLossObjs(nn.Module):

    def __init__(self, grid_size=32, robustifier=None, debugging=False):
        super(SDFLossObjs, self).__init__()
        self.sdf = SDF()
        self.grid_size = grid_size
        self.robustifier = robustifier
        self.debugging = debugging

    @torch.no_grad()
    def get_bounding_boxes(self, vertices): 
        # Args: input list of Tensor, n x v x 3
        # import pdb;pdb.set_trace()
        num_objs = len(vertices)
        device = vertices[0].device
        boxes = torch.zeros(num_objs, 2, 3, device=device)
        for i in range(num_objs):
            boxes[i, 0, :] = vertices[i].min(dim=0)[0]
            boxes[i, 1, :] = vertices[i].max(dim=0)[0]
        return boxes

    @torch.no_grad()
    def check_overlap(self, bbox1, bbox2):
        # import pdb;pdb.set_trace()
        # check x
        if bbox1[0,0] > bbox2[1,0] or bbox2[0,0] > bbox1[1,0]:
            return False
        #check y
        if bbox1[0,1] > bbox2[1,1] or bbox2[0,1] > bbox1[1,1]:
            return False
        #check z
        if bbox1[0,2] > bbox2[1,2] or bbox2[0,2] > bbox1[1,2]:
            return False
        return True

    def filter_isolated_boxes(self, boxes):

        num_objs = boxes.shape[0]
        isolated = torch.zeros(num_objs, device=boxes.device, dtype=torch.uint8)
        for i in range(num_objs):
            isolated_i = False
            for j in range(num_objs):
                if j != i and self.check_overlap(boxes[i], boxes[j]):
                    isolated_i = True
            isolated[i] = isolated_i
        return isolated

    def forward(self, vertices, faces, scale_factor=0.2):
        num_objs = len(vertices)
        # If only one person in the scene, return 0
        loss = torch.tensor(0., device=vertices[0].device)
        if num_objs == 1:
            return loss
        
        boxes = self.get_bounding_boxes(vertices)
        # print(self.filter_isolated_boxes(boxes))
        # overlapping_boxes = [not one for one in self.filter_isolated_boxes(boxes)]
        overlapping_boxes = self.filter_isolated_boxes(boxes)

        # import pdb; pdb.set_trace()
        # If no overlapping voxels return 0
        if overlapping_boxes.sum() == 0:
            return loss
        
        # Filter out the isolated boxes
        vertices = [val.unsqueeze(0) for is_filter, val in zip(overlapping_boxes, vertices) if is_filter]
        boxes = boxes[overlapping_boxes]
        boxes_center = boxes.mean(dim=1).unsqueeze(dim=1)
        boxes_scale = (1+scale_factor) * 0.5*(boxes[:,1] - boxes[:,0]).max(dim=-1)[0][:,None,None]
        
        sdf_list = []
        for idx, vertice in enumerate(vertices):
            with torch.no_grad():
                vertice_centered = vertice - boxes_center[idx]
                vertice_centered_scaled = vertice_centered / boxes_scale[idx]
                assert(vertice_centered_scaled.min() >= -1)
                assert(vertice_centered_scaled.max() <= 1)
                phi = self.sdf(faces[idx], vertice_centered_scaled)
                assert(phi.min() >= 0)
                sdf_list.append(phi)

        valid_people = len(vertices)
        # Convert vertices to the format expected by grid_sample
        for i in range(valid_people):
            # Change coordinate system to local coordinate system of each person
            vertices_exclude_i = torch.cat([v for idx, v in enumerate(vertices) if idx != i], dim=1)
            vertices_local_exclude_i = (vertices_exclude_i - boxes_center[i].unsqueeze(dim=0)) / boxes_scale[i].unsqueeze(dim=0) #->[-1,1]
            vertices_grid = vertices_local_exclude_i.view(1,-1,1,1,3)
            # Sample from the phi grid
            phi_val = nn.functional.grid_sample(sdf_list[i][0][None, None], vertices_grid).view(-1)
            # ignore the phi values for the i-th shape
            cur_loss = phi_val
            if self.debugging:
                import ipdb;ipdb.set_trace()
            # robustifier
            if self.robustifier:
                frac = (cur_loss / self.robustifier) ** 2
                cur_loss = frac / (frac + 1)
            loss += cur_loss.sum() / valid_people ** 2
        return loss
