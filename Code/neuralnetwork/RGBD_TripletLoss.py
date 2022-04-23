import torch
import torch.nn as nn
import torch.nn.functional as func


class RGBD_TripletLoss(nn.Module):
    """
    RGB-D Triplet loss
    Takes the encodings of an anchor, a positive, a negative sample and computes
    the cosine similarities in pairs with respect to the anchor. Then the final
    loss is computed by combining the latent similarities all together.
    """

    def __init__(
        self,
        alpha=0.0
    ):
        """
        Constructor which takes just a hyperparameter.

        :param alpha:
            Loss mitigation hyperparameter
        """

        super(RGBD_TripletLoss, self).__init__()
        self.alpha = alpha

    def forward(
        self,
        latent_anchor,
        latent_pos,
        latent_neg
    ):
        """
        :param latent_anchor:
            Anchor sample latent representation

        :param latent_pos:
            Positive sample latent representation

        :param latent_neg:
            Negative sample latent representation

        :return:
            Triplet loss result
        """

        dist_pos = torch.cosine_similarity(latent_anchor, latent_pos)
        dist_neg = torch.cosine_similarity(latent_anchor, latent_neg)
        losses = func.relu(dist_pos - dist_neg + self.alpha)

        return losses.mean()
