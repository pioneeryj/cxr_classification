import torch
from huggingface_hub import hf_hub_download

from health_multimodal.image.model.model import MultiImageModel
from health_multimodal.image.model.types import ImageEncoderType

# 1) HF에서 BioViL-T "image encoder" 체크포인트 다운로드
ckpt_path = "/home/yoonji/mrg/cxr_classification/pretrained_weight/biovil_image_model.pt"

# 2) 논문 구조에 해당하는 multi-image encoder + projector까지 포함한 모델 로드
#    - img_encoder_type: resnet50_multi_image
#    - joint_feature_size: 파일명이 proj_size_128 이므로 128로 맞춤
biovil_t_img = MultiImageModel(
    img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE.value,
    joint_feature_size=128,
    pretrained_model_path=ckpt_path,
)

biovil_t_img.eval()
print("Loaded:", type(biovil_t_img), "ckpt:", ckpt_path)

import torch
import torch.nn as nn

class BioViLTForClassification(nn.Module):
    def __init__(self, biovil_t_img: nn.Module, num_classes: int, use_projected_128: bool = False, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = biovil_t_img  # MultiImageModel
        self.use_projected_128 = use_projected_128

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # feature dim 결정:
        # - projected_global_embedding: [B, 128]
        # - img_embedding: 보통 ResNet pooled dim (모델 설정에 따라 다를 수 있어 런타임에 한번 확인 권장)
        if use_projected_128:
            in_dim = 128
        else:
            # img_embedding dim은 체크포인트/구현에 따라 달라질 수 있어 더 안전하게 "더미 forward로 추정"하는 방식 권장
            in_dim = None

        self.classifier = nn.Linear(in_dim or 2048, num_classes)  # in_dim이 None이면 아래에서 자동 교체

        self._inferred = False

    @torch.no_grad()
    def _infer_dim_if_needed(self, device):
        if self._inferred or self.use_projected_128:
            return

        dummy = torch.zeros(2, 3, 224, 224, device=device)  # 입력 크기는 실제 transform과 맞추는 게 이상적
        out = self.encoder(current_image=dummy, previous_image=None)
        emb = out.img_embedding  # [B, D]
        d = emb.shape[-1]
        self.classifier = nn.Linear(d, self.classifier.out_features).to(device)
        self._inferred = True

    def forward(self, x, x_prev=None):
        """
        x:      [B, 3, H, W]
        x_prev: [B, 3, H, W] or None
        """
        if not self._inferred and (not self.use_projected_128):
            self._infer_dim_if_needed(x.device)

        out = self.encoder(current_image=x, previous_image=x_prev)

        feats = out.projected_global_embedding if self.use_projected_128 else out.img_embedding
        logits = self.classifier(feats)
        return logits