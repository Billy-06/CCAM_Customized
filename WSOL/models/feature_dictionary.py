import torch 
import torch.nn as nn
import torch.nn.functional as F

# Assuming that .loss file contains SimMinLoss and SimMaxLoss
from .loss import SimMinLoss, SimMaxLoss 

class FeatureDictionary(nn.Module):
    def __init__(self, feature_dim=2048):
        super(FeatureDictionary, self).__init__()
        self.feature_dim = feature_dim
        # foreground: [ feature_dim, [crop_dim1, crop_dim2, ...] ]
        self.feature_dict = {
            "foreground": [None, []],
            "background": [None, []]
        }
        # add the dictionary to the device
        self.to(torch.device('cuda'))

    def forward(self, foreground, background, foreground_crops, background_crops):
        # Reduce the dimensions of foreground and background
        # foreground_crops = self._reduce_dimensions(foreground)
        # background_crops = self._reduce_dimensions(background)

        # Compute loss based on the feature dictionary
        loss = self._compute_loss(foreground, background, foreground_crops, background_crops)

        # Update feature dictionary
        self._update_feature_dict("foreground", foreground, foreground_crops)
        self._update_feature_dict("background", background, background_crops)

        return loss

    # def _reduce_dimensions(self, tensor):
    #     return F.interpolate(tensor, size=(tensor.size(2)//4, tensor.size(3)//4), mode='bilinear', align_corners=False).view(tensor.size(0), 1, -1)

    def _compute_loss(self, foreground, background, foreground_crops, background_crops):
        # Initialize loss
        loss = 0.0

        # Add to loss if foreground and background exist in feature_dict
        if self.feature_dict["foreground"][0] is not None and self.feature_dict["background"][0] is not None:
            # print("Foreground and background exist in feature dictionary")
            # print(f"Foreground Type: {type(self.feature_dict['foreground'][0])}")
            # print(f"Foreground crops: {len(self.feature_dict['foreground'][1])}")
            # Base foreground and background losses
            loss += SimMaxLoss()(foreground)
            loss += SimMinLoss()(foreground, self.feature_dict["background"][0])
            loss += SimMaxLoss()(background)
            loss += SimMinLoss()(background, self.feature_dict["foreground"][0])

            # Add crop-based losses
            for fg_crop in self.feature_dict["foreground"][1]:
                loss += SimMaxLoss()(fg_crop)
                # loss += SimMaxLoss()(foreground_crops, fg_crop)
                loss += SimMinLoss()(foreground_crops, self.feature_dict["background"][1][0])

            for bg_crop in self.feature_dict["background"][1]:
                loss += SimMaxLoss()(bg_crop)
                loss += SimMinLoss()(background_crops, self.feature_dict["foreground"][1][0])
        else:
            # Add the features to the dictionary    
            self.feature_dict["foreground"][0] = foreground
            self.feature_dict["background"][0] = background
            self.feature_dict["foreground"][1].append(foreground_crops)
            self.feature_dict["background"][1].append(background_crops)

        return loss

    def _update_feature_dict(self, key, feature, feature_crop):
        # If key is not present in the dictionary, initialize it
        if key not in self.feature_dict:
            self.feature_dict[key] = [feature, [feature_crop]]
        # If key is present, update it
        else:
            self.feature_dict[key][0] = feature
            self.feature_dict[key][1].append(feature_crop)

            # Optionally: Limit the number of crop features stored
            # if len(self.feature_dict[key][1]) > SOME_MAX_LENGTH:
            #     self.feature_dict[key][1].pop(0)

    def get(self, key):
        return self.feature_dict.get(key, None)
    
    def get_keys(self):
        return list(self.feature_dict.keys())
