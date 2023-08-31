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
import matplotlib.pyplot as plt

def filter_aspect_ratio(path, aspect_max, aspect_min):
    '''filter for images with a particular aspect ratio
    Input
    ------
    path: (str) path to coco format images details - has keys images, annotations and categories where images json contains image file, width and height
    aspect_max: (float) max aspect ratio to filter for
    aspect_min: (float) min aspect ratio to filter for
    
    Returns
    -------
    dataframe object from the coco formated data for the images key with the filtered aspect ratio'''

    f = open(path)
    data = json.load(f)
    df_images = pd.json_normalize(data['images'])
    df_images['aspect_ratio'] = df_images['width'] / df_images['height']
    return df_images.loc[(df_images['aspect_ratio'] < aspect_max) & (df_images['aspect_ratio'] > aspect_min)]

def resize_image(img_path, bboxes, img_dim, max_objects):
    ''''resizes image to a given dimensions alonf with their respective bounding boxes
    Input
    -----
    img_path: (str) path to the image file
    bboxes: (arr) (N, 4) array of bounding boxes for a given image
    img_dim: (tuple) new dimensions of image
    max_objects: (int) max objects/bounding boxes for a given image

    Returns
    -------
    img_tensor: (Torch.tensor) tensor of the image which been resized - channels are first
    bboxes_resized: (arr) (N, 4) array which has been resized according to the image
    
    '''

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

def load_dataset(path, img_dir, max_objects, img_dim, df_images=None):
    '''loads in crowd human dataset - gets the classes, ground truth bounding boxes and images
    Input
    -----
    path: (str) path to the annotations in coco format
    img_dir: (str) path to the image directory
    max_objects: (int) max bounding boxes per image (for padding)
    img_dim: (tuple) image dimensions
    df_images: (dataframe, optional) pandas dataframe of the images json in the coco dataset.
                contains file_name, id, height, width 
    Returns
    -------
    gt_boxes_all: (Torch.tensor) (batch_size, N, max_objects, 4) tensor which contains the ground truth bounding 
                   boxes for the image dataste
    gt_classes_all: (Torch.tensor) (batch_size, N, max_objects) tensor which contains the ground truth classes 
                    for the image dataset bounding boxes
    img_data_all: (arr) (batch-size, n_images, img channels, img height, img width) array of image tensors in a batch
    '''

    f = open(path)
    data = json.load(f)
    df_annot =  pd.json_normalize(data['annotations'])

    if not isinstance(df_images, pd.DataFrame):
        df_images =  pd.json_normalize(data['images'])
    
    image_list =  df_images['file_name'].values

    gt_boxes_all = [] #shape (B, n_images, max_objects, 4]
    gt_classes_all = [] #shape(B, n_images, max_objects) (all filled with ones)
    img_data_all = [] #shape [B, n_images, img_dimensions]

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
    '''plots a batch of images
    Input
    -----
    img_data: (arr) array of images tensors with channels first
    fig: (matplotlib.figure) figure used for plotting
    axes: (matplotlib.axes) axes used for plotting
    
    Returns
    -------
    fig: (matplotlib.figure) figure used for plotting
    axes: (matplotlib.axes) axes used for plotting'''

    for i, img in enumerate(img_data):
        if type(img) == torch.Tensor:
            img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)
    
    return fig, axes

def display_bbox(bboxes, fig, ax, in_format='xyxy', color='y', line_width=3):
    '''plots bounding boxes

    Input
    -----
    bboxes: (arr) (B, N, 4) array of bounding boxes
    fig: (matplotlib.figure) figure used for plotting
    axes: (matplotlib.axes) axes used for plotting
    in_format: (str) options xyxy or xywh to convert between the two for plotting
    color: (str) color code or 'vary' which creates a range of colors
    line_width: (int) line width for plotting
      
    Returns
    -------
    fig: (matplotlib.figure) figure used for plotting
    axes: (matplotlib.axes) axes used for plotting'''


    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if in_format:
        assert in_format == 'xyxy' or in_format == 'xywh'

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
    '''plots a grid along an axes
    Input
    -----
    x_points: (arr) array of x values for the grid to plot along
    y_points: (arr) array of y values for the grid to be plot along
    fig: (matplotlib.figure) figure used for plotting
    axes: (matplotlib.axes) axes used for plotting
    special_point: (tuple) (x,y) value for a particular grid point to be highlighted

    Returns
    -------
    fig: (matplotlib.figure) figure used for plotting
    axes: (matplotlib.axes) axes used for plotting
    '''
    # plot grid
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, color="w", marker='+')
            
    # plot a special point we want to emphasize on the grid
    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')
        
    return fig, ax

def verify_dataset_classes(proj_anchors, anc_boxes_all, img_data_all, feature_extracted_data, positive_anc_ind):
    '''function that plots the anchor and class associated with it based on IoU scores to verify a good dataset 
    has been made based on the generated anchors
    Input
    -----
    proj_anchors: (torch.Tensor) tensor of the projected generated anchors
    anc_boxes_all: (torch.Tensor) tensor of the generated anchors
    img_data_all: (arr) array of image data that has channels first
    feature_extracted_data: (arr) arr of image data that has had its features extracted by a resent model
    positive_anc_ind: (torch.Tensor) a tensor of the indexes of anc_boxes_all which have sufficient IoU score
    Returns
    -------
    plotted figures
    '''

    anc_boxes_all_flat = anc_boxes_all.reshape(len(img_data_all), -1, 4)
    proj_boxes_all_flat = proj_anchors.reshape(len(img_data_all), -1, 4)

    classes_all = np.zeros((anc_boxes_all_flat.shape[0] * anc_boxes_all_flat.shape[1],))
    classes_all[positive_anc_ind] = 1
    classes_all =  classes_all.reshape(2, -1)
    class_dict = {1.0: 'person', 0.0: 'background'}

    for index, (gt_image, feature_extracted_img) in enumerate(zip(img_data_all, feature_extracted_data)):
        for (anchor, proj_anchor, category) in zip(anc_boxes_all_flat[index][290:305], proj_boxes_all_flat[index][290:305], classes_all[index][290:305]):
            #plot
            nrows, ncols = (1, 2)
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
            #put ground truth and feature extracted next to each other
            images = [gt_image, feature_extracted_img]
            fig, axes = display_img(images, fig, axes)
            axes[0].set_title('class: ' + class_dict[category])
            #display anchors on each image
            fig, _ = display_bbox(np.array([proj_anchor]), fig, axes[0], color='vary', in_format='xyxy')
            fig, _ = display_bbox(np.array([anchor]), fig, axes[1], color='vary', in_format='xyxy')

def gen_anc_centers(out_size):
    '''generates anchor points to create a grid on an image
    Inputs
    ------
    out_size: (tuple) dimensions of height and width

    Returns
    -------
    anc_pts_x: (arr) array of anchor points in x axis
    anc_pts_y: (arr) array of anchor points in y axis
    '''
    out_h, out_w = out_size
    
    anc_pts_x = torch.arange(0, out_w) +0.5
    anc_pts_y = torch.arange(0, out_h)+0.5
    
    return anc_pts_x, anc_pts_y

def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    '''generates anchors for each anchor point
    Input
    -----
    anc_pts_x: (arr) array of values for which are the grid/anchor points along x axis
    anc_pts_y: (arr) array of values for which are the grid/anchor points along y axis
    anc_scales: (arr) the scale of the anchor/generated bounding box size
    anc_ratios: (arr) the apect ratio of the anchor/generated bounding box
    out_size: (tuple) out height and width

    Returns
    -------
    anc_base: (arr) (1, X, Y, N, 4) array of every single generated bounding box for an image
    
    '''
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
    '''moves groun truth bounding boxes to the closest anchor points
    Input
    ------
    bboxes: (arr) array of bounding boxes (max_objects, N, 4)
    anc_pts_x_proj: (arr) anchor points projected to ground truth image dimensions in x
    anc_pts_y_proj: (arr) anchor points projected to ground truth image dimensions in y
    in_format: (str) either xywh or xyxy - transforms bboxes coordiates to xyxy if in xywh

    Returns
    -------
    proj_bboses: (arr) (max_objects, N, 4) array of bounding boxes that have been adjusted to the 
                  nearest grid points'''
    proj_bboxes = bboxes.clone()
    invalid_bbox_mask = (proj_bboxes == -1) 

    def find_nearest(array, value):
        '''finds nearest value in an array'''
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
    ''' scales bboxes up or down and converts them to [xmin ymin xmax ymax] format
    Input
    -----
    bboxes: (arr) array of bounding boxes
    w_scale: (int) scale for width
    h_scale: (int) scale for height
    in_format: (str) either xywh or xyxy to denote how the bounding boxes are described
    mode: (str) either a2p or p2a where a2s scales up and p2a scales down

    Returns
    -------
    proj_bboxes: (arr) array of projected bounding boxes'''
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
    '''calculates IoU matrix between ground truth bounding boxes and all anchor boxes
    Input
    -----
    n_images: (int) batch size
    anc_boxes_all: (Torch.tensor) tensor of all anchor boxes (B, w_amap, h_amap, max_objects, 4) 
    gt_bboxes_all: (Torch.tensor) tensor of all ground truth bounding boxes
    tot_anc_boxes: (int) total number of generated bounding boxes

    Returns
    -------
    IoU_mat: (arr) calculates IoU matrix
    '''
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
    '''conditions that returns anchor boxes that sufficiently overlap (or don't) with ground truth bboxes
    Input
    -----
    IoU_matrix: (arr) IoU_matrix
    threshold: (int) threshold for IoU score
    max_iou_per_box: (arr) array which has the index of the max IoU per ground truth box
    type: (str) whether its a positive or negative box (i.e. a box that overlaps sufficiently or doesn't overlap sufficiently)

    Returns
    -------
    anc_max: mask of IoU box for values which fit the conditions'''


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
    ''' gets the required anchors that have the generated bounding boxes that either are sufficiently
    overlapping or not overlapping
    Input
    -----
    anc_boxes_all: (Torch.tensor) tensor of all generated bounding boxes across an image (B, w_amap, h_amap, max_objects, 4) 
    gt_bboxes_all: (Torch.tensor) of all ground truth bounding boxes (B, max_objects, 4) 
    gt_clases_all: (Torch.tensor) of classes for respective bounding boxes (B, max_objects)
    pos_thresh: (int) positive threshold for IoU overlap
    neg_thesh: (int) negative threshold for IoU overlap
    
    Returns
    -------
    positive_anc_ind: (torch.Tensor) of shape (n_pos,)
        flattened positive indices for all the images in the batch
    negative_anc_ind: torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    GT_conf_scores: torch.Tensor of shape (n_pos,), IoU scores of +ve anchors
    GT_class_pos: torch.Tensor of shape (n_pos,)
        mapped classes of +ve anchors
    positive_anc_coords: (n_pos, 4) coords of +ve anchors (for visualization)
    negative_anc_coords: (n_pos, 4) coords of -ve anchors (for visualization)
    positive_anc_ind_sep: list of indices to keep track of +ve anchors'''

    #gt_bboxes_al are all of the corrected anchor boxes
    #anc_boxes_all is every single anchor box ever
    B, w_amap, h_amap, A, _ = anc_boxes_all.shape

    N = gt_bboxes_all.shape[1] # max number of groundtruth bboxes in a batch

    tot_anc_boxes = A * w_amap * h_amap

    anc_boxes_all_flat = anc_boxes_all.reshape(-1, 4)

    IoU_mat =  get_IoU_matrix(B, anc_boxes_all, gt_bboxes_all, tot_anc_boxes)

    #finds the max of any given anchor in all anchors for a given image
    max_iou_per_gt_box, _ = IoU_mat.max(dim=1, keepdim=True) 
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