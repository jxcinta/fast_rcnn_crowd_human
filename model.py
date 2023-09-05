import torch
import torchvision
from torchvision import ops
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import *
import itertools

class FeatureExtractor(nn.Module):
    '''resnet pretrained feature extractor for images'''
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        req_layers = list(model.children())[:8]
        self.backbone = nn.Sequential(*req_layers)
        for param in self.backbone.named_parameters():
            param[1].requires_grad = True
        
    def forward(self, img_data):
        return self.backbone(img_data)
    
class ClassificationModule(nn.Module):
    '''classifies images of variable input size'''
    def __init__(self, out_channels, n_classes, roi_size, hidden_dim=512, p_dropout=0.3):
        super().__init__()       
       
        self.roi_size = roi_size
        # hidden network
        self.avg_pool = nn.AvgPool2d(self.roi_size)
        self.fc = nn.Linear(out_channels, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)
        # define classification head
        self.cls_head = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, feature_map, proposals_list, gt_classes=None):

        if gt_classes is None:
            mode = 'eval'
         
        else:
            mode = 'train'
         
      
        # apply roi pooling on proposals followed by avg pooling (varibale input size)
    
        roi_out = ops.roi_pool(feature_map, proposals_list, self.roi_size)
 
        roi_out = self.avg_pool(roi_out)
        
        roi_out = roi_out.squeeze(-1).squeeze(-1)

        out = self.fc(roi_out)

        out = F.relu(self.dropout(out))

        cls_scores = self.cls_head(out)
        
        if mode == 'eval':
            return cls_scores
 
        # compute cross entropy loss with logits as its just binary class i.e. apply sigmoid 
        cls_loss = F.binary_cross_entropy_with_logits(torch.Tensor(cls_scores), gt_classes.float())
   

        return cls_loss
    
class Detector(nn.Module):
    '''whole model. Generates anchors and feeds them into a classifer for classification. 
    Returns classes of all generated anchors'''
    def __init__(self, img_size, out_size, out_channels, n_classes, roi_size):
        super().__init__()
        
        self.img_height, self.img_width = img_size
        self.out_h, self.out_w = out_size
        
        # downsampling scale factor 
        self.width_scale_factor = self.img_width // self.out_w
        self.height_scale_factor = self.img_height // self.out_h 
        
        # scales and ratios for anchor boxes
        self.anc_scales = [2, 6, 10]
        self.anc_ratios = [0.5, 1, 1.5]
        self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios)
        
        # IoU thresholds for +ve and -ve anchors
        self.pos_thresh = 0.7
        self.neg_thresh = 0.02

        self.feature_extractor = FeatureExtractor()
        self.classifier = ClassificationModule(out_channels, n_classes, roi_size)
    
        
    def forward(self, images, gt_bboxes, gt_classes):
        batch_size = images.size(dim=0)
        feature_map = self.feature_extractor(images)
        batch_size = images.size(dim=0)
        
        # generate anchors
        anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))
        anc_pts_x_proj = anc_pts_x.clone() * self.width_scale_factor 
        anc_pts_y_proj = anc_pts_y.clone() * self.height_scale_factor

        anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
        anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
        anc_boxes_all_flat = anc_boxes_all.reshape(-1, 4)
        anc_boxes_sep = anc_boxes_all.reshape(batch_size, anc_boxes_all.size(dim=1)*anc_boxes_all.size(dim=2)*anc_boxes_all.size(dim=3), 4)

        #project anchor boxes to image size, and to nearest anchor points
        anc_boxes_proj = project_bboxes(anc_boxes_all, self.width_scale_factor, self.height_scale_factor, in_format='xyxy')
        gt_bboxes_proj = bboxes_to_nearest_anchors(gt_bboxes, anc_pts_x_proj, anc_pts_y_proj, in_format='xywh') #converetd into xyxy format

 
        #get positive and negative anchors and indices that seperate them by batch
        positive_anc_ind, negative_anc_ind, GT_conf_scores,\
        GT_class_pos, positive_anc_coords, negative_anc_coords, \
        positive_anc_ind_sep, negative_anc_ind_sep, GT_bboxes_pos = get_req_anchors(anc_boxes_proj, gt_bboxes_proj, 
                                                                                gt_bboxes,self.pos_thresh, self.neg_thresh, return_all_negative_boxes=False)
        
        #combine positive and negative anchors into one set of data with their respectice classes by batch
        anchor_list, classes_list = get_labelled_data(positive_anc_ind, positive_anc_ind_sep, batch_size, negative_anc_ind, negative_anc_ind_sep, anc_boxes_all_flat)


        if isinstance(classes_list, list):
            classes_flat = np.array(list(itertools.chain(*classes_list)))
        else:
            classes_flat = np.array(classes_list).reshape(-1, 1)

        cls_loss =  self.classifier(feature_map, anchor_list, torch.Tensor(classes_flat))

        return cls_loss


    def inference(self, images, conf_thresh=0.5, nms_thresh=0.7):

        batch_size = images.size(dim=0)
        anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(self.out_h, self.out_w))

        anc_base = gen_anc_base(anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
        anc_boxes_all = anc_base.repeat(batch_size, 1, 1, 1, 1)
        anc_boxes_sep = anc_boxes_all.reshape(batch_size, anc_boxes_all.size(dim=1)*anc_boxes_all.size(dim=2)*anc_boxes_all.size(dim=3), 4)

        anchor_list = []

        for idx in range(batch_size):
            anchors_sep = anc_boxes_sep[idx] #
            anchor_list.append(anchors_sep)


        feature_map = self.feature_extractor(images)
        cls_scores = self.classifier(feature_map, anchor_list)
        
        # convert scores into probability
        cls_probs = F.softmax(cls_scores, dim=-1)
        # get classes with highest probability
        classes_all = torch.argmax(cls_probs, dim=-1)

        classes_final = []
        # slice classes to map to their corresponding image
        c = 0
        for i in range(batch_size):
            n_proposals = len(anchor_list[i]) # get the number of proposals for each image
            classes_final.append(classes_all[c: c+n_proposals])
            c += n_proposals
        
        return anchor_list, cls_probs, classes_final, cls_scores