from __future__ import absolute_import
from .deeper_resunet_3d import deeper_resunet_3d
from .resunet_3d import resunet_3d


def Deeper_ResUnet_3D(n_classes, base_filters, channel_in):
    return deeper_resunet_3d(n_classes, base_filters, channel_in)

def ResUnet_3D(n_classes, base_filters, channel_in):
    return resunet_3d(n_classes, base_filters, channel_in)