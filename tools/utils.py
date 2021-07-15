import numpy as np
import torch
from torch.nn import functional as F
import json

#################### IO #############################
def json_load(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def json_save(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def write_obj(verts, faces, file_name):
    with open(file_name, 'w') as fp:
        for i in range(verts.shape[0]):
            v = verts[i]
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        for i in range(faces.shape[0]):
            f = faces[i]
            fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

def load_pose(path):
    # LabelType2Openpose
    joint_mapping = [1,6,7,8,9,10,11,12,13,14,15,16,20,21,22,2,4,3,5,24,25,23,18,19,17]
    joint_mapping = (np.array(joint_mapping)-1).tolist()

    with open(path, 'r') as fid:
        line = fid.readline()

        # resolution
        strs = line.split(' ')
        resolution = [float(strs[1]), float(strs[0])]

        # bbox
        line = fid.readline()
        line = fid.readline()
        strs = line.split(' ')
        try:
            bbox = [float(strs[0]), float(strs[1]), float(strs[2]), float(strs[3])]
        except:
            print('errr '+path)
            return None, None
        minX = min(bbox[0],bbox[2]);
        minY = min(bbox[1],bbox[3]);
        deltaX = abs(bbox[0]-bbox[2]);
        deltaY = abs(bbox[1]-bbox[3]);
        bbox = [minX, minY, deltaX, deltaY]
        # bbox = [float(strs[0]), resolution(2) - float(strs[3]), ...
        #    float(strs[2]) - float(strs[0]), float(strs[3]) - float(strs[1])];

        # subActionNum
        subAction = strs[4]

        # pofloats
        line = fid.readline()
        strs = line.split(' ')
        line = fid.readline()
        strs_state = line.split(' ')
        if 2 * (len(strs_state)- 1) != len(strs) - 1:
            print('errr  '+path)
            return None, None

        if len(strs_state) -1 != 25:
            print('errr  '+path)
            return None, None
        skes = np.zeros([25,3])
        for i in range(0, int((len(strs)-1)/2)):
            
            x = float(strs[2*i])
            y = float(strs[2*i+1])
            z = float(strs_state[i])
            if z == 2: # 2 represent can't infer
                x = -1
                y = -1
            skes[i,:] = [x, y, z]
        skes = skes[joint_mapping,:]

    #plot([bbox[0] bbox[0]+bbox[2]], [bbox[1] bbox[1]+bbox[3]], 'r+')
    #plot(skes(:,1), skes(:,2), 'b*')
    return skes, bbox, subAction

def crop_img_skes(img, skes, bbox=None):
    img_hw = img.shape[0:2]
    if bbox is None:
        bbox = cal_body_bbox(skes, img_hw, margin_rate=0.15, is_square=True)
    bbox = np.array(bbox, dtype=np.int)
    # to-do change to squre
    
    min_xy = bbox[0:2]
    len_xy = bbox[2:4]
    max_xy = min_xy + len_xy
    img_crop = img[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :]
    skes_crop = np.copy(skes[:,0:2]) - min_xy
    return img_crop, skes_crop

#################### Vis #############################
def BuildConnectionMap():
    cxn_map = [[16,18],[1,16],[1,17],[17,19],[1,2],[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],
               [10,11],[11,12],[9,13],[13,14],[14,15],[12,25],[12,23],[23,24],[15,22],[15,20],[20,21]]
    cxn_map = np.array(cxn_map, dtype = np.int);
    cxn_map = cxn_map - 1
    return cxn_map

def BuildColorPattern():
    color_pattern = [(196,0,255),(216,18,2),(214,71,0),(214,163,0),(153,153,0),(102,153,0),(51,153,0),(0,153,0),(153,0,0),(0,153,51),(0,153,102),(0,153,153),(0,102,153),
                     (0,51,153),(0,0,153),(153,0,102),(102,0,153),(153,0,153),(51,0,153),(0,0,153),(0,0,153),(0,0,153),(0,153,153),(0,153,153),(0,153,153)]
    color_pattern = np.array(color_pattern)/255.
    return color_pattern

###################### Projection ####################
def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

###################### Math ####################

def rot6d_to_rotmat(x):
    x = x.view(-1,3,2)

    # Normalize the first vector
    b1 = F.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = F.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats

def cal_body_area(skes):
    # extract valid value
    skes = skes[skes[:,2]!=0, :] 
    
    min_x = np.min(skes[:,0])
    max_x = np.max(skes[:,0])
    min_y = np.min(skes[:,1])
    max_y = np.max(skes[:,1])
    area = (max_x-min_x)*(max_y-min_y)
    return area

def cal_body_bbox(skes_orig, img_hw, mask_valid=None, margin_rate=0.05, is_square=False):
    skes = np.copy(skes_orig) if mask_valid is None else np.copy(skes_orig)[mask_valid,:]

    min_x = np.min(skes[:,0])
    max_x = np.max(skes[:,0])
    min_y = np.min(skes[:,1])
    max_y = np.max(skes[:,1])
    len_x = max_x-min_x
    len_y = max_y-min_y
    
    min_x = int(min_x - margin_rate*len_x)
    min_y = int(min_y - margin_rate*len_y)
    len_x = int(len_x*(1+margin_rate*2))
    len_y = int(len_y*(1+margin_rate*2))
    bbox = [min_x, min_y, len_x, len_y]

    if is_square:
        bbox = np.array(bbox, dtype=np.int)
        center = bbox[0:2] + 0.5*bbox[2:4]
        bbox[2:4] = max(bbox[2:4]) # set max length
        bbox[0:2] = center - 0.5*bbox[2:4]
        # no padding now
        bbox = bbox.tolist()     
    min_x, min_y, len_x, len_y = bbox
    
    # check img boundary
    min_x = max(0, min_x)
    min_y = max(0, min_y)    
    len_x = min(min_x+len_x, img_hw[1] - 1) - min_x # width
    len_y = min(min_y+len_y, img_hw[0] - 1) - min_y # height    
    bbox = [min_x, min_y, len_x, len_y]
    return bbox

def cal_ske_norm(skes, bbox, mask_valid=None):
    bbox = np.array(bbox, dtype=np.float)
    
    ## not keep width-heigth rate
    #skes_norm = np.copy(skes)
    #skes_norm[:,0:2] = (skes[:,0:2] - bbox[0:2])/bbox[2:4]
    
    # normalize to [0, 1]
    # keep width-heigth rate
    skes_norm = np.copy(skes)
    center = bbox[0:2] + 0.5*bbox[2:4]
    skes_norm[:,0:2] = (skes[:,0:2] - center)/np.max(bbox[2:4]) # move to center and scale
    skes_norm[:,0:2] = skes_norm[:,0:2] + [0.5, 0.5] # move back to new center

    # move invalid to center [0.5, 0.5]
    skes_norm[~mask_valid,:] = 0.5
    return skes_norm    

def cal_ske_recover(skes, bbox):
    bbox = np.array(bbox, dtype=np.float)
    
    ## recover from not keeping width-heigth rate
    
    # reciver from to [0, 1]
    # keep width-heigth rate
    center = bbox[0:2] + 0.5*bbox[2:4]

    skes_rec = np.copy(skes)
    skes_rec = skes_rec - [0.5, 0.5]
    skes_rec = skes_rec*np.max(bbox[2:4]) + center

    # invalid point in [0,0]
    return skes_rec

def cal_body_bbox_3d(skes):
    min_anchor = np.amin(skes,axis=0)
    max_anchor = np.amax(skes,axis=0)
    len_sides = max_anchor - min_anchor
    
    # expand bbox
    margin_rate = 0.05
    min_anchor = (min_anchor - margin_rate *len_sides).astype(np.int)
    len_sides = (len_sides*(1+margin_rate*2)).astype(np.int)
    
    # [min_x, min_y, min_z, len_x, len_y, len_z]
    return np.concatenate([min_anchor, len_sides])

def cal_ske_norm_3d(skes, bbox):
    # to-do 
    ## not keep width heigth rate
    # skes_norm = np.copy(skes)
    # skes_norm[:,0:2] = (skes[:,0:2] - bbox[0:2])/bbox[2:4]
    
    # keep width heigth rate
    skes_norm = np.copy(skes)
    center = bbox[0:3] + 0.5*bbox[3:6]
    skes_norm = (skes - center)/np.amax(bbox[3:6]) # move to center and scale
    skes_norm = skes_norm + [0.5, 0.5, 0.5] # move back to new center    
    return skes_norm