import torch
from torchvision import ops
import torch.nn.functional as F
import numpy as np
import matplotlib.patches as patches
import json
from skimage import io
from skimage.transform import resize
import pandas as pd
from tqdm import tqdm
from matplotlib.pyplot import cm

def filter_aspect_ratio(path, aspect_max, aspect_min):

    f = open(path)
    data = json.load(f)
    df_images = pd.json_normalize(data['images'])
    df_images['aspect_ratio'] = df_images['width'] / df_images['height']
    return df_images.loc[(df_images['aspect_ratio'] < aspect_max) & (df_images['aspect_ratio'] > aspect_min)]

def resize_image(img_path, bboxes, img_dim, max_objects):

    height_resize, width_resize = img_dim
    image = io.imread(img_path)  
    x_ = image.shape[1]
    y_ = image.shape[0]
    x_scale = width_resize / x_
    y_scale = height_resize / y_

    image = resize(image, img_dim)
    img_tensor = torch.from_numpy(image).permute(2, 0, 1)
    bboxes_resized = []

    for bbox in bboxes[0:max_objects]:
        (origLeft, origTop, origRight, origBottom) = bbox

        x = int(np.round(origLeft * x_scale))
        y = int(np.round(origTop * y_scale))
        xmax = int(np.round(origRight * x_scale))
        ymax = int(np.round(origBottom * y_scale))
        bbox_resized = [x, y, xmax, ymax]
        bboxes_resized.append(bbox_resized)

    return img_tensor, bboxes_resized

def load_dataset(df_images, path, img_dir, max_objects, img_dim):#is_train=True

    f = open(path)
    data = json.load(f)
    df_annot =  pd.json_normalize(data['annotations'])
    image_list =  df_images['file_name'].values

    gt_boxes_all = [] #shape (B, n_images, max_objects, 4]
    gt_classes_all = [] #shape(B, n_images, max_objects) (all filled with ones)
    img_data_all = [] #shape [B, n_images]

    for image in tqdm(image_list[0:200]):
        
        img_path = img_dir  + image
        img_id = df_images.loc[df_images['file_name'] == image, 'id'].item()
        bboxes = df_annot.loc[(df_annot['image_id'] == img_id) & (df_annot['category_id'] == 0), 'bbox'].values
        
        if len(bboxes) <= max_objects:
            img_tensor, bboxes_resized = resize_image(img_path, bboxes, img_dim, max_objects)

            gt_classes = np.ones((len(bboxes), )) #since its binary
            gt_classes_all.append(torch.Tensor(gt_classes))
            gt_boxes_all.append(torch.Tensor(bboxes_resized))
            img_data_all.append(img_tensor)

    
    return  gt_boxes_all, gt_classes_all, img_data_all

def display_img(img_data, fig, axes):

    for i, img in enumerate(img_data):
        if type(img) == torch.Tensor:
            img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
    
    return fig, axes

def display_bbox(bboxes, fig, ax, classes=None, in_format='xyxy', color='y', line_width=3):


    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if classes:
        assert len(bboxes) == len(classes)

    if in_format == 'xyxy':
        #convert boxes to xywh format
        bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')

    color_spectrum = cm.rainbow(np.linspace(0, 1, len(bboxes)))
    for index, box in enumerate(bboxes):
        
        x, y, w, h = map(int, box)

        if color == 'vary':
            rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color_spectrum[index], facecolor='none')
        else:
            rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none') 
        ax.add_patch(rect)

    return fig, ax

def display_grid(x_points, y_points, fig, ax, special_point=None):
    # plot grid
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, color="w", marker='+')
            
    # plot a special point we want to emphasize on the grid
    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')
        
    return fig, ax

def gen_anc_centers(out_size):
    out_h, out_w = out_size
    
    anc_pts_x = torch.arange(0, out_w) +0.5
    anc_pts_y = torch.arange(0, out_h)+0.5
    
    return anc_pts_x, anc_pts_y

def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    n_anc_boxes = len(anc_scales) * len(anc_ratios)
    anc_base = torch.zeros(1, anc_pts_x.size(dim=0) \
                              , anc_pts_y.size(dim=0), n_anc_boxes, 4) # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]
    
    for ix, xc in enumerate(anc_pts_x):
        for jx, yc in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            c = 0
            for i, scale in enumerate(anc_scales):
                for j, ratio in enumerate(anc_ratios):
                    w = scale * ratio
                    h = scale
                    
                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1

            anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)
            
    return anc_base

def bboxes_to_nearest_anchors(bboxes, anc_pts_x_proj, anc_pts_y_proj, in_format='xywh'):
    proj_bboxes = bboxes.clone()
    invalid_bbox_mask = (proj_bboxes == -1) 

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    #currently, bboxes values are as x,y,w,h while it needs to be in format xmin,ymin,xmax,ymax
    if in_format=='xywh':
        proj_bboxes[:, :, [2]] += proj_bboxes[:, :, [0]]
        proj_bboxes[:, :, [3]] += proj_bboxes[:, :, [1]]

    for n_image, image in enumerate(proj_bboxes):
        for n_bbox, box in enumerate(image):
            box =  box.numpy()
            anc_x = anc_pts_x_proj.numpy()
            anc_y = anc_pts_y_proj.numpy()
            box_projected = torch.from_numpy(np.asarray([find_nearest(anc_x, box[0]), find_nearest(anc_y, box[1]), find_nearest(anc_x, box[2]), find_nearest(anc_y, box[3])]))
            proj_bboxes[n_image, n_bbox, :] =  box_projected

    proj_bboxes.masked_fill_(invalid_bbox_mask, -1)

    return proj_bboxes


def project_bboxes(bboxes, w_scale, h_scale, in_format='xywh', mode='a2p'):
    'scales bboxes '
    proj_bboxes = bboxes.clone().reshape(bboxes.size(dim=0), -1, 4) #reshaping to format (B, n_ground_truth_bbox, 4)
    invalid_bbox_mask = (proj_bboxes == -1) 

    #currently, bboxes values are as x,y,w,h while it needs to be in format xmin,ymin,xmax,ymax
    if in_format=='xywh':
        proj_bboxes[:, :, [2]] += proj_bboxes[:, :, [0]]
        proj_bboxes[:, :, [3]] += proj_bboxes[:, :, [1]]

    #then change by scale factor
    if mode == 'a2p':
        # activation map to pixel image
        proj_bboxes[:, :, [0, 2]] *= w_scale
        proj_bboxes[:, :, [1, 3]] *= h_scale
    else:
        # pixel image to activation map
        proj_bboxes[:, :, [0, 2]] /= w_scale
        proj_bboxes[:, :, [1, 3]] /= h_scale


    proj_bboxes.masked_fill_(invalid_bbox_mask, -1) # fill padded bboxes back with -1
    proj_bboxes.resize_as_(bboxes)

    return proj_bboxes

def get_IoU_matrix(n_images, anc_boxes_all, gt_bboxes_all, tot_anc_boxes):
    #gt_bboxes_al are all of the corrected anchor boxes
    #anc_boxes_all is every single anchor box ever

    anc_boxes_flat = anc_boxes_all.reshape(n_images, tot_anc_boxes, 4)
    max_objects = gt_bboxes_all.size(dim=1)
    IoU_mat = torch.zeros((n_images, tot_anc_boxes, max_objects))

    for i in range(n_images):
        bbox = gt_bboxes_all[i]
        anc_box = anc_boxes_flat[i]
        IoU_mat[i, :] = ops.box_iou(anc_box, bbox)

    return IoU_mat

def IoU_matrix_conditions(IoU_matrix, threshold, max_iou_per_box,  type='positive'):


    if type=='positive':
        anc_mask = torch.logical_and(IoU_matrix == max_iou_per_box, max_iou_per_box > 0) #make sure max anchor is included
        # condition 2: anchor boxes with iou above a threshold with any of the gt bboxes
        anc_mask = torch.logical_or(anc_mask, IoU_matrix > threshold) #and that you keep anything over positive threshold
    elif type=='negative':
        anc_mask = (max_iou_per_box < threshold)
    else:
        raise ValueError('Invalid input `type`. Options are `positive` or `negative`.')

    return anc_mask

def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2):

    #gt_bboxes_al are all of the corrected anchor boxes
    #anc_boxes_all is every single anchor box ever
    B, w_amap, h_amap, A, _ = anc_boxes_all.shape

    N = gt_bboxes_all.shape[1] # max number of groundtruth bboxes in a batch

    tot_anc_boxes = A * w_amap * h_amap

    anc_boxes_all_flat = anc_boxes_all.reshape(-1, 4)

    IoU_mat =  get_IoU_matrix(B, anc_boxes_all, gt_bboxes_all, tot_anc_boxes)

    #finds the max of any given anchor in all anchors for a given image
    max_iou_per_gt_box, _ = IoU_mat.max(dim=1, keepdim=True) #finds the max of any given anchor in all anchors for a given image
    max_iou_per_anc, max_iou_per_anc_ind = IoU_mat.max(dim=-1) 
    max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)

    positive_anc_mask = IoU_matrix_conditions(IoU_mat, pos_thresh, max_iou_per_gt_box, type='positive')

    positive_anc_ind_sep = torch.where(positive_anc_mask)[0] #to know which image the positive sample comes from
    positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)   # combine all the batches and get the idxs of the +ve anchor boxes
    positive_anc_ind = torch.where(positive_anc_mask)[0]#to know which index/row an achor of positive value came from
    positive_anc_col = torch.where(positive_anc_mask)[1]#to know which column an achor of positive value came from i.e. which ground truth box it is associated with
    positive_anc_coords = torch.stack([torch.FloatTensor(anc_boxes_all_flat[i,:]) for i in positive_anc_ind])

    negative_anc_mask = IoU_matrix_conditions(IoU_mat, pos_thresh, max_iou_per_anc, type='negative')

    negative_anc_ind = torch.where(negative_anc_mask)[0]
    negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (positive_anc_ind.shape[0],))]
    negative_anc_coords = torch.stack([torch.FloatTensor(anc_boxes_all_flat[i,:]) for i in negative_anc_ind])
    

    GT_conf_scores = max_iou_per_anc[positive_anc_ind]
    GT_class_pos = [1]*len(GT_conf_scores) #binary classification so can leave as [1]
    gt_bboxes_all_flat = gt_bboxes_all.reshape(-1, 4)
    GT_bboxes_pos = torch.stack([torch.FloatTensor(gt_bboxes_all_flat[i,:]) for i in positive_anc_col])

    return positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_class_pos, positive_anc_coords, negative_anc_coords, positive_anc_ind_sep, GT_bboxes_pos