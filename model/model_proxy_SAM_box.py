#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.segment_anything.modeling.common import LayerNorm2d
from model.segment_anything.modeling.image_encoder import Block
from model.segment_anything import sam_model_registry
from model.mobile_encoder.setup_mobile_sam import setup_model as build_sam_mobile

from model.loss_functions import dice_loss, multilabel_dice_loss
from model.utils import init_weights


class MSConv2d(nn.Module):
    def __init__(self, ch, groups=4):
        super(MSConv2d, self).__init__()
        assert ch % groups == 0
        group_ch = ch // groups
        self.convs = nn.ModuleList([
            nn.Conv2d(group_ch, group_ch, 1, 1)
        ])
        for i in range(1, groups):
            self.convs.append(
                nn.Conv2d(group_ch, group_ch, 3, 1, padding=i, dilation=i, groups=group_ch)
            )
        self.activate = nn.GELU()
        self.norm = nn.BatchNorm2d(ch)
        self.groups = groups

    def forward(self, x):
        features = x.chunk(self.groups, dim=1)
        outs = []
        for i in range(len(features)):
            outs.append(self.convs[i](features[i]))
        net = torch.cat(outs, dim=1)
        net = self.norm(net)
        net = self.activate(net)
        return net


class PromptGen(nn.Module):
    def __init__(self, blk, reduction=4, cls_token=False, reshape=False, seq_size=None) -> None:
        super(PromptGen, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        prompt_dim = dim // reduction
        self.prompt_learn = nn.Sequential(
            # nn.Linear(dim, 32),
            # nn.GELU(),
            # nn.Linear(32, dim),
            # nn.GELU()
            nn.Conv2d(dim, prompt_dim, 1, 1),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, groups=prompt_dim, bias=False),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, dim, 1, 1),
            LayerNorm2d(dim),
            nn.GELU()
        )
        self.cls_token = cls_token
        self.reshape = reshape
        self.seq_size = seq_size
        self.prompt_learn.apply(init_weights)
    
    def forward(self, x):
        if self.cls_token:
            tokens = x[:,1:]
            bs, seq_len, dim = tokens.size()
            if self.reshape:
                tokens = tokens.reshape(-1, self.seq_size, self.seq_size, dim).permute(0, 3, 1, 2)
            prompt = self.prompt_learn(tokens)
            promped = tokens + prompt
            promped = promped.reshape(bs, dim, seq_len).transpose(1, 2)
            promped = torch.cat([x[:, 0].unsqueeze(1), promped], dim=1)
        else:
            prompt = self.prompt_learn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # prompt = self.prompt_learn(x)
            promped = x + prompt
        net = self.block(promped)
        return net


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # qkv[:, :, :, : self.dim] += new_q
        # qkv[:, :, :, -self.dim:] += new_v
        qkv[..., : self.dim] += new_q
        qkv[..., -self.dim:] += new_v
        return qkv


class _LoRA_qkv_proj(nn.Module):
    def __init__(self, proj: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.proj = proj
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.proj(x) + self.w_b(self.w_a(x))
        return x


class SonarSAM(nn.Module):
    def __init__(self, model_name, checkpoint, num_classes=12, 
                 is_finetune_image_encoder=False,
                 use_adaptation=False,
                 adaptation_type='learnable_prompt_layer',
                 head_type='custom',
                 reduction=4, upsample_times=2, groups=4, rank=4) -> None:
        super(SonarSAM, self).__init__()
        
        #load same from the pretrained model
        if model_name == 'mobile':
            self.sam = build_sam_mobile(checkpoint=checkpoint, num_multimask_outputs=num_classes)
        else:
            self.sam = sam_model_registry[model_name](checkpoint=checkpoint, num_multimask_outputs=num_classes)
        self.is_finetune_image_encoder = is_finetune_image_encoder
        self.use_adaptation = use_adaptation
        self.adaptation_type = adaptation_type
        self.head_type = head_type
        self.num_classes = num_classes

        # freeze image encoder
        if not self.is_finetune_image_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False

        if self.use_adaptation:
            if self.adaptation_type == 'learnable_prompt_layer':
                if self.model_name != 'mobile':
                    blocks = []
                    for block in self.sam.image_encoder.blocks:
                        blocks.append(
                            PromptGen(block, reduction=reduction)
                        )
                    self.sam.image_encoder.blocks = nn.Sequential(
                        *blocks
                    )
                else:
                    raise ValueError('Not supported!')                  
            elif self.adaptation_type == 'LORA':
                if self.model_name != 'mobile':
                    for blk in self.sam.image_encoder.blocks:
                        w_qkv_linear = blk.attn.qkv
                        self.dim = w_qkv_linear.in_features
                        w_a_linear_q = nn.Linear(self.dim, rank, bias=False)
                        w_b_linear_q = nn.Linear(rank, self.dim, bias=False)
                        w_a_linear_v = nn.Linear(self.dim, rank, bias=False)
                        w_b_linear_v = nn.Linear(rank, self.dim, bias=False)

                        w_a_linear_q.apply(init_weights)
                        w_b_linear_q.apply(init_weights)
                        w_a_linear_v.apply(init_weights)
                        w_b_linear_v.apply(init_weights)

                        blk.attn.qkv = _LoRA_qkv(
                            w_qkv_linear,
                            w_a_linear_q,
                            w_b_linear_q,
                            w_a_linear_v,
                            w_b_linear_v,
                        )
                else:
                    for i_layer in range(1, len(self.sam.image_encoder.layers)):
                        for blk in self.sam.image_encoder.layers[i_layer].blocks:
                            w_qkv_linear = blk.attn.qkv
                            self.dim = w_qkv_linear.in_features
                            w_a_linear_q = nn.Linear(self.dim, rank, bias=False)
                            w_b_linear_q = nn.Linear(rank, self.dim, bias=False)
                            w_a_linear_v = nn.Linear(self.dim, rank, bias=False)
                            w_b_linear_v = nn.Linear(rank, self.dim, bias=False)

                            w_a_linear_q.apply(init_weights)
                            w_b_linear_q.apply(init_weights)
                            w_a_linear_v.apply(init_weights)
                            w_b_linear_v.apply(init_weights)

                            blk.attn.qkv = _LoRA_qkv(
                                w_qkv_linear,
                                w_a_linear_q,
                                w_b_linear_q,
                                w_a_linear_v,
                                w_b_linear_v,
                            ) 
            else:
                raise ValueError('unknown adaptation type: {}'.format(self.adaptation_type))
        
        out_dim = self.sam.image_encoder.neck[0].out_channels
        self.img_size = self.sam.image_encoder.img_size

        if self.head_type == 'custom':        
            del self.sam.prompt_encoder
            del self.sam.mask_decoder            
                                
            self.up_conv = nn.ModuleDict()
            self.up_times = upsample_times
            dim = out_dim
            for i in range(upsample_times):
                self.up_conv["up_{}".format(i+1)] = nn.Sequential(
                        nn.ConvTranspose2d(dim, dim//2, 2, 2),
                        LayerNorm2d(dim // 2),
                        nn.GELU()
                    )
                dim = dim // 2
            self.ms_conv = MSConv2d(dim, groups=groups)
            self.decoder = nn.Sequential(
                nn.Conv2d(dim, num_classes, 1, 1, 0),
            )
        elif self.head_type == 'semantic_mask_decoder':
            pass
        elif self.head_type == 'semantic_mask_decoder_LORA':
            for param in self.sam.mask_decoder.transformer.parameters():
                param.requires_grad = False
            decoder_transformer = self.sam.mask_decoder.transformer
            for layer_idx, blk in enumerate(decoder_transformer.layers):
                self_attn_q_proj = blk.self_attn.q_proj
                self_attn_v_proj = blk.self_attn.v_proj
                input_dim = blk.self_attn.embedding_dim
                output_dim = blk.self_attn.internal_dim
                w_a_linear_q_self_attn = nn.Linear(input_dim, rank, bias=False)
                w_b_linear_q_self_attn = nn.Linear(rank, output_dim, bias=False)
                w_a_linear_v_self_attn = nn.Linear(input_dim, rank, bias=False)
                w_b_linear_v_self_attn = nn.Linear(rank, output_dim, bias=False)
                w_a_linear_q_self_attn.apply(init_weights)
                w_b_linear_q_self_attn.apply(init_weights)
                w_a_linear_v_self_attn.apply(init_weights)
                w_b_linear_v_self_attn.apply(init_weights)
                blk.self_attn.q_proj = _LoRA_qkv_proj(self_attn_q_proj, w_a_linear_q_self_attn, w_b_linear_q_self_attn)
                blk.self_attn.v_proj = _LoRA_qkv_proj(self_attn_v_proj, w_a_linear_v_self_attn, w_b_linear_v_self_attn)

                cross_attn_ti_q_proj = blk.cross_attn_token_to_image.q_proj
                cross_attn_ti_v_proj = blk.cross_attn_token_to_image.v_proj
                ti_input_dim = blk.cross_attn_token_to_image.embedding_dim
                ti_output_dim = blk.cross_attn_token_to_image.internal_dim
                w_a_linear_q_cross_attn_ti = nn.Linear(ti_input_dim, rank, bias=False)
                w_b_linear_q_cross_attn_ti = nn.Linear(rank, ti_output_dim, bias=False)
                w_a_linear_v_cross_attn_ti = nn.Linear(ti_input_dim, rank, bias=False)
                w_b_linear_v_cross_attn_ti = nn.Linear(rank, ti_output_dim, bias=False)
                w_a_linear_q_cross_attn_ti.apply(init_weights)
                w_b_linear_q_cross_attn_ti.apply(init_weights)
                w_a_linear_v_cross_attn_ti.apply(init_weights)
                w_b_linear_v_cross_attn_ti.apply(init_weights)
                blk.cross_attn_token_to_image.q_proj = _LoRA_qkv_proj(cross_attn_ti_q_proj, w_a_linear_q_cross_attn_ti,
                                                                    w_b_linear_q_cross_attn_ti)
                blk.cross_attn_token_to_image.v_proj = _LoRA_qkv_proj(cross_attn_ti_v_proj, w_a_linear_v_cross_attn_ti,
                                                                    w_b_linear_v_cross_attn_ti)

                cross_attn_it_q_proj = blk.cross_attn_image_to_token.q_proj
                cross_attn_it_v_proj = blk.cross_attn_image_to_token.v_proj
                it_input_dim = blk.cross_attn_image_to_token.embedding_dim
                it_output_dim = blk.cross_attn_image_to_token.internal_dim
                w_a_linear_q_cross_attn_it = nn.Linear(it_input_dim, rank, bias=False)
                w_b_linear_q_cross_attn_it = nn.Linear(rank, it_output_dim, bias=False)
                w_a_linear_v_cross_attn_it = nn.Linear(it_input_dim, rank, bias=False)
                w_b_linear_v_cross_attn_it = nn.Linear(rank, it_output_dim, bias=False)
                w_a_linear_q_cross_attn_it.apply(init_weights)
                w_b_linear_q_cross_attn_it.apply(init_weights)
                w_a_linear_v_cross_attn_it.apply(init_weights)
                w_b_linear_v_cross_attn_it.apply(init_weights)
                blk.cross_attn_image_to_token.q_proj = _LoRA_qkv_proj(cross_attn_it_q_proj, w_a_linear_q_cross_attn_it,
                                                                    w_b_linear_q_cross_attn_it)
                blk.cross_attn_image_to_token.v_proj = _LoRA_qkv_proj(cross_attn_it_v_proj, w_a_linear_v_cross_attn_it,
                                                                    w_b_linear_v_cross_attn_it)

            # final attention token to image
            block = decoder_transformer.final_attn_token_to_image
            fa_ti_q_proj = block.q_proj
            fa_ti_v_proj = block.v_proj
            in_dim, out_dim = block.embedding_dim, block.internal_dim
            fa_ti_q_proj_A = nn.Linear(in_dim, rank, bias=False)
            fa_ti_q_proj_B = nn.Linear(rank, out_dim, bias=False)
            fa_ti_v_proj_A = nn.Linear(in_dim, rank, bias=False)
            fa_ti_v_proj_B = nn.Linear(rank, out_dim, bias=False)
            fa_ti_q_proj_A.apply(init_weights)
            fa_ti_q_proj_B.apply(init_weights)
            fa_ti_v_proj_A.apply(init_weights)
            fa_ti_v_proj_B.apply(init_weights)
            block.q_proj = _LoRA_qkv_proj(fa_ti_q_proj, fa_ti_q_proj_A, fa_ti_q_proj_B)
            block.v_proj = _LoRA_qkv_proj(fa_ti_v_proj, fa_ti_v_proj_A, fa_ti_v_proj_B)
        else:
            raise ValueError('unknow head type: {}'.format(self.head_type))

    def upscale(self, x, times=2):
        for i in range(times):
            x = self.up_conv["up_{}".format(i+1)](x)
        return x

    def forward(self, x, boxes=None):
        out = self.sam.image_encoder(x)
        seg_out = []
        if self.head_type in ['semantic_mask_decoder', 'semantic_mask_decoder_LORA']:            
            for idx, curr_embedding in enumerate(out):
                points = None
                if boxes is not None:
                    bboxes = boxes[idx]                    
                else:
                    bboxes = None
                masks = None
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=points,
                    boxes=bboxes,
                    masks=masks,
                )
                
                low_res_masks, iou_predictions = self.sam.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
                # print('low res masks', low_res_masks.shape)
                masks = F.interpolate(low_res_masks, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
                # print('masks', masks.shape)
                # low_res_masks = torch.sum(low_res_masks, dim=0, keepdim=True)
                output = []
                for idx in range(masks.shape[0]):                    
                    mask = masks[idx, ...]
                    # print('mask', mask.shape)                                        
                    output.append(mask.squeeze())
                seg_out.append(output)
        else:
            raise ValueError('unknow head type: {}'.format(self.head_type))

        return seg_out


class ModelWithLoss(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.bcewithlogit = nn.BCEWithLogitsLoss(reduction='mean')
        self.dice_loss = dice_loss

    def forward(self, images, masks, boxes):
        pred_masks = self.model(images, boxes)
        bce_loss, dice = 0, 0
        for idx, im_masks in enumerate(masks):
            assert len(pred_masks) == len(masks)
            p_im_masks = pred_masks[idx]
            for p_m, m in zip(p_im_masks, im_masks):                
                bce_loss += self.bcewithlogit(input=p_m, target=m)
                dice += self.dice_loss(label=m.unsqueeze(0), mask=p_m.unsqueeze(0))
        loss = bce_loss + dice
        return loss, pred_masks
    

if __name__ == "__main__":
    with torch.no_grad():                 
        model = SonarSAM("vit_b", "ckpts/sam_vit_b_01ec64.pth", 
                         num_classes=12, 
                         is_finetune_image_encoder=False, 
                         use_adaptation=False, 
                         adaptation_type='LORA', 
                         head_type='semantic_mask_decoder',
                         reduction=4, upsample_times=2, groups=4, rank=4).half().cuda()
        x = torch.randn(1, 3, 1024, 1024).half().cuda()

        out = model(x)
        print(out.shape)

