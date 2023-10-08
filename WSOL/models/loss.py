import torch
import torch.nn as nn
import torch.nn.functional as F
# from infonce import InfoNCE


def cos_sim(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

# Minimize Similarity, e.g., push representation of foreground and background apart.
# Negative Pair Loss => L(neg) = -i/n^2 ()
# Lneg = −1/n^2 Sum n i=1 Sum n j=1 log(1 − s neg i,j ), 
class SimMinLoss(nn.Module):
    def __init__(self, margin=0.15, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.m = margin
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_sim(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
# Positive Pair Loss
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            # if embedded_bg_2 is not None:
            #     sim = cos_sim(embedded_bg, embedded_bg_2)
            # else:
            sim = cos_sim(embedded_bg, embedded_bg)

            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
        
# class ContractiveLoss(InfoNCE):
#     def __init__(self, temperature=0.07, reduction='mean'):
#         super(ContractiveLoss, self).__init__(temperature=temperature, reduction=reduction)

#         self.temperature = temperature
#         self.reduction = reduction

#         self.query = None
#         self.key = None
#         self.bg_features = None
#         self.fg_features = None

#     def forward(self, query, bg_features, fg_features):
#         """
#         :param query: [N, C]
#         :param bg_features: [N, C]
#         :param fg_features: [N, C]

#         - Take the query and foreground features as positive pairs
#         - Take the query and background features as negative pairs
#         - Take the foreground and background features as negative pairs

#         - Apply the InfoNCE loss on each pair
#         - Sum the losses

#         :return: the loss
#         """

#         self.query = query
#         self.key = torch.cat((fg_features, bg_features), dim=0)
#         self.bg_features = bg_features
#         self.fg_features = fg_features

#         # print(query.shape, bg_features.shape, fg_features.shape, self.key.shape)

#         return super(ContractiveLoss, self).forward(query, self.key)
        


if __name__ == '__main__':

    fg_embedding = torch.randn((4, 12))
    bg_embedding = torch.randn((4, 12))
    # print(fg_embedding, bg_embedding)

    examplar = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 4], [4, 2, 1, 3]])

    _, indices = examplar.sort(descending=True, dim=1)
    print(indices)
    _, rank = indices.sort(dim=1)
    print(rank)
    rank_weights = torch.exp(-rank.float() * 0.25)
    print(rank_weights)
