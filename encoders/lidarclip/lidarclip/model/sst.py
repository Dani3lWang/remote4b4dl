# from mmcv import Config
from mmengine.config import Config

# Compatibility shim must be imported BEFORE mmdet3d
from lidarclip.model._mmdet3d_compat import _patch_voxel_ops

import torch
from torch import nn
from torch.nn import functional as F

from lidarclip.model.attention_pool import AttentionPool2d
from lidarclip.model.sst_encoder_only_config import model as sst_model_conf

# Patch CUDA voxel ops with pure-Python implementations AFTER mmdet3d is loaded
_patch_voxel_ops()


class SSTEncoder(nn.Module):
    """Encoder-only SST model for feature extraction.

    Only builds the four components needed for extract_feat:
    voxel_layer, voxel_encoder, middle_encoder, backbone.

    Skips the detection head (bbox_head) to avoid OOM on 32GB GPUs.
    """

    def __init__(self, voxel_layer_cfg, voxel_encoder_cfg, middle_encoder_cfg, backbone_cfg):
        super().__init__()
        from mmdet3d.ops import Voxelization
        from mmdet3d.models import builder as b3d
        from mmdet.models.builder import build_backbone

        # Trigger registration of SST components before building
        import mmdet3d.models.voxel_encoders  # noqa: F401
        import mmdet3d.models.middle_encoders  # noqa: F401
        import mmdet3d.models.backbones  # noqa: F401

        self.voxel_layer = Voxelization(**voxel_layer_cfg)
        self.voxel_encoder = b3d.build_voxel_encoder(voxel_encoder_cfg)
        self.middle_encoder = b3d.build_middle_encoder(middle_encoder_cfg)
        self.backbone = build_backbone(backbone_cfg)

    def extract_feat(self, points, img_metas=None):
        voxels, coors = self.voxelize(points)
        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.backbone(x)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        coors = []
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch


def build_sst_encoder(config_path):
    cfg = Config.fromfile(config_path)
    model_cfg = cfg.model
    return SSTEncoder(
        voxel_layer_cfg=model_cfg["voxel_layer"],
        voxel_encoder_cfg=model_cfg["voxel_encoder"],
        middle_encoder_cfg=model_cfg["middle_encoder"],
        backbone_cfg=model_cfg["backbone"],
    )


def build_sst(config_path):
    """Build SST model directly, bypassing mmdet3d.apis.init_model to avoid
    cascading imports of training/eval/dataset modules."""
    from mmdet3d.models.builder import build_model

    cfg = Config.fromfile(config_path)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    model.eval()
    return model


class LidarEncoderSST(nn.Module):
    def __init__(self, sst_config_path, clip_embedding_dim=512):
        super().__init__()
        self._sst = build_sst_encoder(sst_config_path)
        self._pooler = AttentionPool2d(
            spacial_dim=sst_model_conf["backbone"]["output_shape"][0],
            embed_dim=clip_embedding_dim,
            num_heads=8,
            input_dim=sst_model_conf["backbone"]["conv_out_channel"],
        )

    def forward(self, point_cloud, no_pooling=False, return_attention=False):
        lidar_features = self._sst.extract_feat(point_cloud, None)[0]  # bs, d, h, w
        pooled_feature, attn_weights = self._pooler(lidar_features, no_pooling, return_attention)
        return pooled_feature, attn_weights


if __name__ == "__main__":
    model = LidarEncoderSST("sst_encoder_only.py")
    import torch

    model.to("cuda")
    points = [torch.rand(100, 4).cuda() for _ in range(16)]
    out = model(points)
    print(out.shape)
