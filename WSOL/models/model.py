from torchsummary import summary
from .resnet import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .feature_dictionary import FeatureDictionary

import yaml
import os

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetSeries_WithCrops(nn.Module):
    def __init__(self, pretrained, m=0.999, key_count=4000, temperature=0.07):
        super(ResNetSeries_WithCrops, self).__init__()

        # Get the current working directory
        cwd = os.getcwd()
        print(f'Current working directory: {cwd}')

        # Load the config paths
        # with open('../config/CCAM_CFG.yaml', 'r') as f:
        #     config = yaml.load(f, Loader=yaml.FullLoader)

        if pretrained == 'supervised':
            print(f'Loading supervised pretrained parameters!')
            model = resnet50(pretrained=True)
            
        elif pretrained == 'mocov2':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('moco_r50_v2-e3b0c442.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif pretrained == 'detco':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('detco_200ep.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # Set the momentum for the BatchNorm layers
        self.m = m
        self.key_count = key_count
        # Set the temperature for the softmax
        self.temperature = temperature

        self.feature_key_encoder = model
        self.feature_value_encoder = model

        for param_q, param_k in zip(self.feature_key_encoder.parameters(), self.feature_value_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the Queue
        self.register_buffer("queue", torch.randn(128, key_count))
        self.queue = F.normalize(self.queue, dim=1)

        # Create the Queue Pointer
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x2 = self.layer4(x1)

        # queries = self.feature_key_encoder(x2)
        # # Create random crops of the queries
        # # Quesries have the shape N x C x H x W
        # # Create 4 crops of the queries
        # crop_H = int(queries.size(2)/4)
        # crop_W = int(queries.size(3)/4)
        # # Crop the queries
        # queries_ = []
        # for i in range(4):
        #     queries_.append(queries[:, :, int(crop_H*i):int(crop_H*(i+1)), int(crop_W*i):int(crop_W*(i+1))])
        
        # # Concatenate the queries
        # queries = torch.cat(queries_, dim=0)

        queries = F.normalize(queries, dim=1)


        return torch.cat([x2, x1], dim=1)
    
    
class ResNetSeries(nn.Module):
    """
    ### ResNetSeries_WithCrops
    This model maintians a queue of size K, where K is the number of crops,
    """
    def __init__(self, pretrained):
        super(ResNetSeries, self).__init__()

        # Get the current working directory
        cwd = os.getcwd()
        print(f'Current working directory: {cwd}')

        # Load the config paths
        # with open('../config/CCAM_CFG.yaml', 'r') as f:
        #     config = yaml.load(f, Loader=yaml.FullLoader)

        if pretrained == 'supervised':
            print(f'Loading supervised pretrained parameters!')
            model = resnet50(pretrained=True)
            
        elif pretrained == 'mocov2':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('moco_r50_v2-e3b0c442.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif pretrained == 'detco':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('detco_200ep.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise NotImplementedError

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4


    # def __call__(self, x):
    #     self.key = self.

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x2 = self.layer4(x1)

        return torch.cat([x2, x1], dim=1)

class Disentangler(nn.Module):
    def __init__(self, cin):
        super(Disentangler, self).__init__()

        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)

    def forward(self, x):
        N, C, H, W = x.size()
        ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))

        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, C, H*W]
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam

class MultiCropDisentangler(Disentangler):
    def __init__(self, cin):
        super(MultiCropDisentangler, self).__init__(cin)

    def forward(self, x, fg_feats, bg_feats, ccam):
        """
        - x: [N, C, H, W]
        - fg_feats: [N, 1, C]
        - bg_feats: [N, 1, C]
        - ccam: [N, 1, H, W]
        Get the sizes of the input feature map
        """
        N, C, H, W = x.size()
        crop_H = int(H/4)
        crop_W = int(W/4)

        # Generate the forground activation map P (this means that the background is 1 - P)
        ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))

        # Get the foreground and background features from the input feature map
        fg_feats_noncrop = torch.matmul(ccam, x)
        bg_feats_noncrop = torch.matmul(1 - ccam, x)

        fg_feats_crops =[]
        bg_feats_crops = []

        # crop the ccam, foreground and background features to 4 crops
        for i in range(4):
            # Crop the foreground feature map
            fg_feats_ = fg_feats_noncrop[:, :, int(crop_H*i):int(crop_H*(i+1)), int(crop_W*i):int(crop_W*(i+1))]
            # Crop the background feature map
            bg_feats_ = bg_feats_noncrop[:, :, int(crop_H*i):int(crop_H*(i+1)), int(crop_W*i):int(crop_W*(i+1))]
            # Crop the class activation map
            ccam_ = ccam[:, :, int(crop_H*i):int(crop_H*(i+1)), int(crop_W*i):int(crop_W*(i+1))]

            # Append the cropped foreground and background features to the list
            fg_feats_crops.append(fg_feats_)
            bg_feats_crops.append(bg_feats_)

            # Flatten the CCAM to [N, 1, crop_H * crop_W]
            ccam_crop = ccam_.reshape(N, 1, crop_H * crop_W)                               # [N, 1, crop_H * crop_W]

            # Reshape the foreground and background features to [N, C, crop_H * crop_W]                          
            fg_feats_ = fg_feats_.reshape(N, C, crop_H * crop_W).permute(0, 2, 1).contiguous()
            bg_feats_ = bg_feats_.reshape(N, C, crop_H * crop_W).permute(0, 2, 1).contiguous()

            # Get the foreground and background features from the cropped feature maps
            fg_feats_ = torch.matmul(ccam_crop, fg_feats_)/ (crop_H * crop_W)             # [N, 1, C]
            bg_feats_ = torch.matmul(1 - ccam_crop, bg_feats_)/ (crop_H * crop_W)         # [N, 1, C]

            # # Reshape the foreground and background features to [N, 1, C]
            # fg_feats_ = torch.squeeze(fg_feats_, 2)                                  # [N, C]
            # bg_feats_ = torch.squeeze(bg_feats_, 2)                                  # [N, C]
            # fg_feats_ = torch.transpose(fg_feats_, 1, 2)                              # [N, C, 1]
            # bg_feats_ = torch.transpose(bg_feats_, 1, 2)                              # [N, C, 1]

            # Concatenate the foreground and background features
            if i == 0:
                fg_feats = fg_feats_
                bg_feats = bg_feats_
            else:
                # Concatenate the foreground and background features
                fg_feats = torch.cat((fg_feats, fg_feats_), dim=1)
                bg_feats = torch.cat((bg_feats, bg_feats_), dim=1)
        
        # Return the foreground and background features and the class activation map
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam, ((fg_feats_noncrop, fg_feats_crops), (bg_feats_noncrop, bg_feats_crops))


class Network(nn.Module):
    def __init__(self, pretrained='mocov2', cin=None):
        super(Network, self).__init__()

        # Set the backbone to ResNet50 
        self.backbone = ResNetSeries(pretrained=pretrained)

        # Set the activation head to the Disentangler
        self.ac_head_a = Disentangler(cin)

        # Set the activation head to the MultiCropDisentangler
        self.ac_head_b = MultiCropDisentangler(cin)

        # Add the feature dictionary
        self.feature_dict = FeatureDictionary()

        # crop loss computed by the feature dictionary
        self.crop_loss = 0.0


        # Set the layers to train from scratch  
        self.from_scratch_layers = [self.ac_head_b]

    def forward(self, x):

        feats = self.backbone(x)
        fg_feats, bg_feats, ccam = self.ac_head_a(feats)
        # Pass the features through the activation head
        # self.ac_head_b(feats, fg_feats, bg_feats, ccam)
        fg_feats, bg_feats, ccam,crop_tuple = self.ac_head_b(feats, fg_feats, bg_feats, ccam)

        self.crop_loss = self.feature_dict.forward(crop_tuple[0][0], crop_tuple[0][1], crop_tuple[1][0], crop_tuple[1][1])

        return fg_feats, bg_feats, ccam

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups


def get_model(pretrained, cin=2048+1024):
    return Network(pretrained=pretrained, cin=cin)

# Define the Intersection over Union (IoU) metric
def intersect_over_union(pred_box, gt_box):
    """
    Calculates the IoU between the predicted bounding box and the ground truth bounding box
    :param pred_box: predicted bounding box
    :param gt_box: ground truth bounding box
    :return: IoU between the predicted bounding box and the ground truth bounding box
    """
    # Get the coordinates of the predicted bounding box which is of type pytorch tensor
    # and convert it to numpy array 
    # eg tensor([x1_p, y1_p, x2_p, y2_p])
    pred_box = pred_box.cpu().numpy() if torch.is_tensor(pred_box) else pred_box
    # Get the coordinates of the ground truth bounding box which is of type pytorch tensor
    # and convert it to numpy array
    # eg tensor([x1_g, y1_g, x2_g, y2_g])
    gt_box = gt_box.cpu().numpy() if torch.is_tensor(gt_box) else gt_box

    # Get the coordinates of the predicted bounding box
    x1_p, y1_p, x2_p, y2_p = pred_box
    # Get the coordinates of the ground truth bounding box
    x1_g, y1_g, x2_g, y2_g = gt_box

    # Get the coordinates of the intersection rectangle
    x1_i = max(x1_p, x1_g)
    y1_i = max(y1_p, y1_g)
    x2_i = min(x2_p, x2_g)
    y2_i = min(y2_p, y2_g)

    # Calculate the area of the intersection rectangle
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

    # Calculate the area of both the predicted bounding box and the ground truth bounding box
    pred_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    gt_area = (x2_g - x1_g + 1) * (y2_g - y1_g + 1)

    # Calculate the intersection over union
    iou = inter_area / float(pred_area + gt_area - inter_area)

    # Return the intersection over union value
    return iou

# Define the Average Precision (AP) metric
def average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    """
    Calculates the average precision (AP) of the predicted bounding boxes
    :param pred_boxes: list of all predicted bounding boxes
    :param true_boxes: list of all ground truth bounding boxes
    :param iou_threshold: threshold value for deciding whether the predicted bounding box is correct or not
    :param box_format: format of the bounding boxes
    :param num_classes: number of classes
    :return: average precision of the predicted bounding boxes
    """
    # Initialize the average precision list
    average_precisions = []
    # Loop over the classes
    for c in range(num_classes):
        # Initialize the true positives, false positives and the false negatives
        true_positives = []
        false_positives = []
        false_negatives = []
        # Set the number of detections for the class c to 0
        num_detection = 0
        # Loop over the predictions
        for detection in pred_boxes:
            # If the detection is of class c
            if detection[1] == c:
                # Increment the number of detections for the class c
                num_detection += 1
                # Get the predicted bounding box
                pred_box = detection[2:]
                # Set the flag to false
                flag = False
                # Loop over the ground truth bounding boxes
                for true_box in true_boxes:
                    # Get the ground truth bounding box
                    gt_box = true_box[2:]
                    # Calculate the IoU between the predicted bounding box and the ground truth bounding box
                    iou = intersect_over_union(pred_box, gt_box)
                    # If the IoU is greater than the threshold
                    if iou > iou_threshold:
                        # Set the flag to true
                        flag = True
                        # Break out of the loop
                        break
                # If the flag is true
                if flag:
                    # Append 1.0 to the true positives list
                    true_positives.append(1.0)
                    # Append 0.0 to the false positives list
                    false_positives.append(0.0)
                    # Append 0.0 to the false negatives list
                    false_negatives.append(0.0)
                # If the flag is false
                else:
                    # Append 0.0 to the true positives list
                    true_positives.append(0.0)
                    # Append 1.0 to the false positives list
                    false_positives.append(1.0)
                    # Append 1.0 to the false negatives list
                    false_negatives.append(1.0)
        # Get the cummulative sum of the true positives
        cum_true_positives = np.cumsum(true_positives)
        # Get the cummulative sum of the false positives
        cum_false_positives = np.cumsum(false_positives)
        # Get the cummulative sum of the false negatives
        cum_false_negatives = np.cumsum(false_negatives)
        # Calculate the precision
        precision = cum_true_positives / (cum_true_positives + cum_false_positives + 1e-10)
        # Calculate the recall
        recall = cum_true_positives / (cum_true_positives + cum_false_negatives + 1e-10)
        # Pad the precision and recall
        precision = np.concatenate((np.zeros((1,)), precision, np.zeros((1,))))
        recall = np.concatenate((np.zeros((1,)), recall, np.ones((1,))))
        # Calculate the average precision
        average_precision = np.trapz(precision, recall)
        # Append the average precision to the average precision list
        average_precisions.append(average_precision)
    # Return the average precision list
    return sum(average_precisions) / len(average_precisions)
                

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model = Network(pretrained='mocov2', cin=2048+1024).to(device)
    # print(model)
    # print(model.get_parameter_groups())

    # Print model summary
    summary(model, [(3, 224, 224)])