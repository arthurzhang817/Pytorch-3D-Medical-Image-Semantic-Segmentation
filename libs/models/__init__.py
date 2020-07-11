from __future__ import absolute_import
from .resunet_3d import resunet_3d


def ResUnet_3D(n_classes, base_filters, channel_in):
    return resunet_3d(n_classes, base_filters, channel_in)