import torch.nn as nn


class RGBD_TripletNetwork(nn.Module):
    """
    RGB-D Triplet Network
    Classical Triplet Network architecture which takes an encoder and trains it
    with a triplet loss in share weights.
    """

    def __init__(
        self,
        encoder
    ):
        """
        Constructor.

        :param encoder:
            Encoder architecture
        :type encoder: nn.Module
        """
        super(RGBD_TripletNetwork, self).__init__()
        self.encoder = encoder

    def forward(
        self,
        patch_anchor,
        patch_pos,
        patch_neg
    ):
        """
        :param patch_anchor:
            Anchor sample patch [4 @ 16 x 16]

        :param patch_pos:
            Positive sample patch [4 @ 16 x 16]

        :param patch_neg:
            Negative sample patch [4 @ 16 x 16]

        :return:
            Anchor, Positive and Negative latent representations
        """
        latent_anchor = self.encoder(patch_anchor)
        latent_pos = self.encoder(patch_pos)
        latent_neg = self.encoder(patch_neg)

        return latent_anchor, latent_pos, latent_neg

    def encode(
        self,
        x
    ):
        """
        Computes the embedding produced by the trained encoder.

        :param x:
            Input sample patch [4 @ 16 x 16]

        :return:
            Latent representation
        """
        return self.encoder(x)
