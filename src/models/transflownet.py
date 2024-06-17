import torch
from torch import nn
from src.models.base import *
from torchvision import models
from typing import Dict, Any
import torch.nn.functional as F

_BASE_CHANNELS = 64

class TransFlowNet(nn.Module):
    def __init__(self, args):
        super(TransFlowNet,self).__init__()
        self._args = args

        
        self.encoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.transformer_block = TransformerBlock(8*_BASE_CHANNELS, num_heads = 8, hidden_dim = 2048)

        # self.transformer_block = ViTBlock(in_channels=8*_BASE_CHANNELS)

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        skip_connections['skip0'] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
        inputs = self.encoder4(inputs)
        skip_connections['skip3'] = inputs.clone()

        # transition
        inputs = self.transformer_block(inputs)

        # decoder
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict['flow0'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict['flow1'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict['flow2'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()

        del skip_connections

        return flow, flow_dict
        

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, hidden_dim, dropout = 0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.mha = nn.MultiheadAttention(emb_size, num_heads, dropout = dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).permute(0, 2, 1) # (B, sequence_length, emb_size)

        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.ff(x)
        
        x = self.norm2(x + self.dropout(ff_output))
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x
'''
class ViTBlock(nn.Module):
    def __init__(self, in_channels, embed_dim=768, img_size = (30, 40), patch_size = 32):
        super(ViTBlock, self).__init__()

        self.vit = models.vit_b_32(pretrained = False)

        self.vit.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.vit.heads[0] = nn.Linear(self.vit.heads[0].in_features, embed_dim * 30 * 40)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.output_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

    def forward(self, x):

        b, c, h, w = x.shape

        x = F.interpolate(x, size = (224, 224), mode='bilinear', align_corners=False)

        x = self.vit(x)

        x = x.view(b, self.embed_dim, h, w)

        x = self.output_proj(x)

        return x
'''