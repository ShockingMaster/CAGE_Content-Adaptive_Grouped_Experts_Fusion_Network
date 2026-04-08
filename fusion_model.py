import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_transformer_block import SwinStage, AGGatedSwinStage


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PatchMerging, self).__init__()
        self.reduction = nn.Linear(in_channels * 4, out_channels)
        self.norm = nn.LayerNorm(in_channels * 4)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, out_channels, H/2, W/2)
        """
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))
            H, W = x.shape[2], x.shape[3]
        # 提取4个patch: [0::2, 0::2], [1::2, 0::2], [0::2, 1::2], [1::2, 1::2]
        x0 = x[:, :, 0::2, 0::2]  # (B, C, H/2, W/2)
        x1 = x[:, :, 1::2, 0::2]  # (B, C, H/2, W/2)
        x2 = x[:, :, 0::2, 1::2]  # (B, C, H/2, W/2)
        x3 = x[:, :, 1::2, 1::2]  # (B, C, H/2, W/2)

        # 在通道维度拼接: (B, 4*C, H/2, W/2)
        x = torch.cat([x0, x1, x2, x3], dim=1)

        # (B, 4*C, H/2, W/2) -> (B, H/2, W/2, 4*C)
        x = x.permute(0, 2, 3, 1)

        # LayerNorm + Linear
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, out_channels)

        # (B, H/2, W/2, out_channels) -> (B, out_channels, H/2, W/2)
        x = x.permute(0, 3, 1, 2)

        return x


class MoEDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MoEDownsample, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = 4
        expert_out_channels = out_channels // 2

        # 4个专家：不同kernel size的空洞卷积
        # kernel_size, dilation, stride=2
        expert_configs = [
            (3, 1),  # 3x3, dilation=1
            (5, 2),  # 5x5, dilation=2
            (7, 3),  # 7x7, dilation=3
            (9, 4),  # 9x9, dilation=4
        ]

        self.experts = nn.ModuleList()
        for kernel_size, dilation in expert_configs:
            # 计算padding以保持特征图尺寸一致（stride=2时减半）
            padding = dilation * (kernel_size - 1) // 2
            expert = nn.Sequential(
                nn.Conv2d(in_channels, expert_out_channels,
                         kernel_size=kernel_size,
                         stride=2,
                         padding=padding,
                         dilation=dilation),
                nn.BatchNorm2d(expert_out_channels),
                nn.GELU()
            )
            self.experts.append(expert)

        # 门控网络（Gating Network）
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(in_channels, self.num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, out_channels, H/2, W/2)
        """
        B = x.shape[0]

        # 计算门控权重
        gate_weights = self.gate(x)  # (B, 4)

        # 获取Top-2专家的索引和权重
        top2_weights, top2_indices = torch.topk(gate_weights, k=2, dim=1)  # (B, 2)

        # 重新归一化Top-2权重
        top2_weights = top2_weights / (top2_weights.sum(dim=1, keepdim=True) + 1e-8)

        # 计算所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # 每个输出: (B, out_channels/2, H/2, W/2)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, 4, out_channels/2, H/2, W/2)

        # 选择Top-2专家并加权
        selected_outputs = []
        for i in range(2):  # Top-2
            # 为每个batch选择对应的专家
            batch_indices = torch.arange(B, device=x.device)
            expert_idx = top2_indices[:, i]  # (B,)
            weight = top2_weights[:, i]  # (B,)

            # 选择专家输出并加权
            selected = expert_outputs[batch_indices, expert_idx]  # (B, out_channels/2, H/2, W/2)
            weighted = selected * weight.view(B, 1, 1, 1)
            selected_outputs.append(weighted)

        output = torch.cat(selected_outputs, dim=1)  # (B, out_channels, H/2, W/2)
        return output

class GroupedMoEDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GroupedMoEDownsample, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = 4
        self.num_groups = 2
        self.experts_per_group = 2

        # 每个专家输出 out_channels/2
        expert_out_channels = out_channels // 2

        # 4个专家：不同kernel size的空洞卷积
        expert_configs = [
            (3, 1),  # Expert 0: 小感受野组
            (3, 1),  # Expert 1: 小感受野组
            (9, 4),  # Expert 2: 大感受野组
            (9, 4),  # Expert 3: 大感受野组
        ]

        self.experts = nn.ModuleList()
        for kernel_size, dilation in expert_configs:
            padding = dilation * (kernel_size - 1) // 2
            expert = nn.Sequential(
                nn.Conv2d(in_channels, expert_out_channels,
                         kernel_size=kernel_size,
                         stride=2,
                         padding=padding,
                         dilation=dilation),
                nn.BatchNorm2d(expert_out_channels),
                nn.GELU()
            )
            self.experts.append(expert)

        # 门控网络 - 为每组生成门控分数
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, self.num_experts),
        )

        # 专家分组: [0,1] 为第0组, [2,3] 为第1组
        self.expert_groups = [[0, 1], [2, 3]]

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, out_channels, H/2, W/2)
        """
        B = x.shape[0]

        # 计算门控logits
        gate_logits = self.gate(x)  # (B, 4)

        # 对所有4个专家做softmax
        gate_weights = F.softmax(gate_logits, dim=1)  # (B, 4)

        # 计算所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # 每个: (B, out_channels/2, H/2, W/2)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, 4, out_channels/2, H/2, W/2)

        # 为每组分别选择专家并加权
        selected_outputs = []

        for group_id, expert_ids in enumerate(self.expert_groups):
            # 提取该组的门控权重
            group_weights = gate_weights[:, expert_ids]  # (B, 2)

            # 选择组内权重最大的专家
            max_weight, max_idx_in_group = torch.max(group_weights, dim=1)  # (B,)

            # 转换为全局专家索引
            batch_indices = torch.arange(B, device=x.device)
            global_expert_idx = torch.tensor(expert_ids, device=x.device)[max_idx_in_group]  # (B,)

            # 选择专家输出并用权重加权
            selected = expert_outputs[batch_indices, global_expert_idx]  # (B, out_channels/2, H/2, W/2)
            weighted = selected * max_weight.view(B, 1, 1, 1)  # 用softmax权重加权
            selected_outputs.append(weighted)

        # Concat两组选出的加权专家输出
        output = torch.cat(selected_outputs, dim=1)  # (B, out_channels, H/2, W/2)
        return output


class StaticMultiKernelDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StaticMultiKernelDownsample, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 每个卷积核输出 out_channels/4
        expert_out_channels = out_channels // 4

        # 4个卷积核：3x3, 5x5, 7x7, 9x9 (与MoE中的expert一致)
        expert_configs = [
            (3, 1),  # 3x3, dilation=1
            (5, 2),  # 5x5, dilation=2
            (7, 3),  # 7x7, dilation=3
            (9, 4),  # 9x9, dilation=4
        ]

        self.convs = nn.ModuleList()
        for kernel_size, dilation in expert_configs:
            padding = dilation * (kernel_size - 1) // 2
            conv = nn.Sequential(
                nn.Conv2d(in_channels, expert_out_channels,
                         kernel_size=kernel_size,
                         stride=2,
                         padding=padding,
                         dilation=dilation),
                nn.BatchNorm2d(expert_out_channels),
                nn.GELU()
            )
            self.convs.append(conv)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, out_channels, H/2, W/2)
        """
        # 执行所有卷积
        outputs = [conv(x) for conv in self.convs]
        # 直接concat (4个expert各输出out_channels/4)
        return torch.cat(outputs, dim=1)


class DownsampleBlock(nn.Module):
    """下采样块（步长卷积）"""
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UpsampleBlock(nn.Module):
    """上采样块（双线性插值+卷积）"""
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.act(self.norm(self.conv(x)))


class EncoderBranch(nn.Module):
    """
    编码器分支
    """
    def __init__(self,
                 in_channels=1,
                 channels=[32, 64, 128],
                 image_size=(768, 1024),
                 depths=[2, 2, 2],
                 num_heads=[2, 4, 8],
                 window_size=7,
                 downsample_type='conv',
                 use_ag_gating=False,
                 ag_threshold=1.0):
        """
        Args:
            in_channels: 输入通道数
            channels: 各层通道数 [c1, c2, c3]
            image_size: 输入图像尺寸 (H, W)
            depths: 各层Swin Block数量
            num_heads: 各层注意力头数
            window_size: 窗口大小
            downsample_type: 下采样类型 ('conv', 'patch_merging', 'moe')
            use_ag_gating: 是否使用AG门控机制
            ag_threshold: AG阈值倍数
        """
        super(EncoderBranch, self).__init__()

        self.use_ag_gating = use_ag_gating
        h, w = image_size

        # 第一层: 输入 -> feat1
        self.conv1 = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels[0])

        if use_ag_gating:
            self.swin1 = AGGatedSwinStage(
                dim=channels[0],
                input_resolution=(h, w),
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=window_size,
                ag_threshold=ag_threshold
            )
        else:
            self.swin1 = SwinStage(
                dim=channels[0],
                input_resolution=(h, w),
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=window_size
            )

        # 下采样1
        self.down1 = self._create_downsample(downsample_type, channels[0], channels[1])

        # 第二层: feat1 -> feat2
        if use_ag_gating:
            self.swin2 = AGGatedSwinStage(
                dim=channels[1],
                input_resolution=(h // 2, w // 2),
                depth=depths[1],
                num_heads=num_heads[1],
                window_size=window_size,
                ag_threshold=ag_threshold
            )
        else:
            self.swin2 = SwinStage(
                dim=channels[1],
                input_resolution=(h // 2, w // 2),
                depth=depths[1],
                num_heads=num_heads[1],
                window_size=window_size
            )

        # 下采样2
        self.down2 = self._create_downsample(downsample_type, channels[1], channels[2])

        # 第三层: feat2 -> feat3
        if use_ag_gating:
            self.swin3 = AGGatedSwinStage(
                dim=channels[2],
                input_resolution=(h // 4, w // 4),
                depth=depths[2],
                num_heads=num_heads[2],
                window_size=window_size,
                ag_threshold=ag_threshold
            )
        else:
            self.swin3 = SwinStage(
                dim=channels[2],
                input_resolution=(h // 4, w // 4),
                depth=depths[2],
                num_heads=num_heads[2],
                window_size=window_size
            )

    def _create_downsample(self, downsample_type, in_channels, out_channels):
        """创建下采样模块"""
        if downsample_type == 'conv':
            return DownsampleBlock(in_channels, out_channels)
        elif downsample_type == 'patch_merging':
            return PatchMerging(in_channels, out_channels)
        elif downsample_type == 'moe':
            return MoEDownsample(in_channels, out_channels)
        elif downsample_type == 'grouped_moe':
            return GroupedMoEDownsample(in_channels, out_channels)
        elif downsample_type == 'static_concat':
            return StaticMultiKernelDownsample(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown downsample_type: {downsample_type}")

    def forward(self, x, x_other=None):
        # 第一层
        x_conv = F.gelu(self.norm1(self.conv1(x)))  # (B, 32, H, W)

        if self.use_ag_gating and x_other is not None:
            x_other_conv = F.gelu(self.norm1(self.conv1(x_other)))
            feat1 = self.swin1(x_conv, x_other_conv)  # AG-gated attention
            feat1_other = x_other_conv  # 保存对方模态在当前层的特征
        else:
            feat1 = self.swin1(x_conv)  # 普通Self-Attention
            feat1_other = None

        # 下采样1
        x = self.down1(feat1)  # (B, 64, H/2, W/2)

        # 第二层
        if self.use_ag_gating and feat1_other is not None:
            x_other_down1 = self.down1(feat1_other)  # 对方模态也下采样
            feat2 = self.swin2(x, x_other_down1)
            feat2_other = x_other_down1
        else:
            feat2 = self.swin2(x)
            feat2_other = None

        # 下采样2
        x = self.down2(feat2)  # (B, 128, H/4, W/4)

        # 第三层
        if self.use_ag_gating and feat2_other is not None:
            x_other_down2 = self.down2(feat2_other)  # 对方模态也下采样
            feat3 = self.swin3(x, x_other_down2)
        else:
            feat3 = self.swin3(x)

        return feat1, feat2, feat3


class FusionDecoder(nn.Module):
    def __init__(self,
                 channels=[32, 64, 128],
                 image_size=(768, 1024),
                 depths=[2, 2, 2],
                 num_heads=[2, 4, 8],
                 window_size=7):
        super(FusionDecoder, self).__init__()

        h, w = image_size

        # 融合层: concat(feat3_ir, feat3_vi) -> feat7
        self.fuse_conv3 = nn.Conv2d(channels[2] * 2, channels[2], kernel_size=1)
        self.fuse_norm3 = nn.BatchNorm2d(channels[2])
        self.swin_fuse3 = SwinStage(
            dim=channels[2],
            input_resolution=(h // 4, w // 4),
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size
        )

        # 上采样1
        self.up1 = UpsampleBlock(channels[2], channels[1])

        # 融合层2: concat(feat7_up, feat2_ir, feat2_vi) -> feat8
        self.fuse_conv2 = nn.Conv2d(channels[1] * 3, channels[1], kernel_size=1)
        self.fuse_norm2 = nn.BatchNorm2d(channels[1])
        self.swin_fuse2 = SwinStage(
            dim=channels[1],
            input_resolution=(h // 2, w // 2),
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size
        )

        # 上采样2
        self.up2 = UpsampleBlock(channels[1], channels[0])

        # 融合层3: concat(feat8_up, feat1_ir, feat1_vi) -> fused
        self.fuse_conv1 = nn.Conv2d(channels[0] * 3, channels[0], kernel_size=1)
        self.fuse_norm1 = nn.BatchNorm2d(channels[0])
        self.swin_fuse1 = SwinStage(
            dim=channels[0],
            input_resolution=(h, w),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size
        )

        # 输出层
        self.out_conv = nn.Conv2d(channels[0], 1, kernel_size=1)

    def forward(self, feat1_ir, feat2_ir, feat3_ir, feat1_vi, feat2_vi, feat3_vi):
        # 底层融合: concat(feat3_ir, feat3_vi) -> feat7
        x = torch.cat([feat3_ir, feat3_vi], dim=1)  # (B, 256, H/4, W/4)
        x = F.gelu(self.fuse_norm3(self.fuse_conv3(x)))  # (B, 128, H/4, W/4)
        feat7 = self.swin_fuse3(x)  # (B, 128, H/4, W/4)

        # 上采样1
        x = self.up1(feat7, target_size=feat2_ir.shape[2:])  # (B, 64, H/2, W/2)

        # 中层融合: concat(feat7_up, feat2_ir, feat2_vi) -> feat8
        x = torch.cat([x, feat2_ir, feat2_vi], dim=1)  # (B, 192, H/2, W/2)
        x = F.gelu(self.fuse_norm2(self.fuse_conv2(x)))  # (B, 64, H/2, W/2)
        feat8 = self.swin_fuse2(x)  # (B, 64, H/2, W/2)

        # 上采样2
        x = self.up2(feat8, target_size=feat1_ir.shape[2:])  # (B, 32, H, W)

        # 顶层融合: concat(feat8_up, feat1_ir, feat1_vi) -> fused
        x = torch.cat([x, feat1_ir, feat1_vi], dim=1)  # (B, 96, H, W)
        x = F.gelu(self.fuse_norm1(self.fuse_conv1(x)))  # (B, 32, H, W)
        x = self.swin_fuse1(x)  # (B, 32, H, W)

        # 输出层
        fused = self.out_conv(x)  # (B, 1, H, W)

        return fused, feat8, feat7


class ImageFusionModel(nn.Module):
    def __init__(self,
                 in_channels=1,
                 channels=[32, 64, 128],
                 image_size=(768, 1024),
                 depths=[2, 2, 2],
                 num_heads=[2, 4, 8],
                 window_size=7,
                 downsample_type='conv',
                 use_ag_gating=False,
                 ag_threshold=1.0):
        """
        Args:
            in_channels: 输入通道数
            channels: 各层通道数
            image_size: 输入图像尺寸
            depths: 各层深度
            num_heads: 各层注意力头数
            window_size: 窗口大小
            downsample_type: 下采样类型 ('conv', 'patch_merging', 'moe')
            use_ag_gating: 是否使用AG门控机制
            ag_threshold: AG阈值倍数
        """
        super(ImageFusionModel, self).__init__()

        self.use_ag_gating = use_ag_gating

        # 红外编码器
        self.encoder_ir = EncoderBranch(
            in_channels=in_channels,
            channels=channels,
            image_size=image_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            downsample_type=downsample_type,
            use_ag_gating=use_ag_gating,
            ag_threshold=ag_threshold
        )

        # 可见光编码器
        self.encoder_vi = EncoderBranch(
            in_channels=in_channels,
            channels=channels,
            image_size=image_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            downsample_type=downsample_type,
            use_ag_gating=use_ag_gating,
            ag_threshold=ag_threshold
        )

        # 融合解码器
        self.decoder = FusionDecoder(
            channels=channels,
            image_size=image_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size
        )

    def forward(self, img_ir, img_vi):
        # 编码
        if self.use_ag_gating:
            # AG-Gated模式：两个编码器需要交互
            feat1_ir, feat2_ir, feat3_ir = self.encoder_ir(img_ir, img_vi)
            feat1_vi, feat2_vi, feat3_vi = self.encoder_vi(img_vi, img_ir)
        else:
            # 普通模式：独立编码
            feat1_ir, feat2_ir, feat3_ir = self.encoder_ir(img_ir)
            feat1_vi, feat2_vi, feat3_vi = self.encoder_vi(img_vi)

        # 解码+融合
        fused, feat8, feat7 = self.decoder(
            feat1_ir, feat2_ir, feat3_ir,
            feat1_vi, feat2_vi, feat3_vi
        )

        return (fused, feat8, feat7,
                (feat1_ir, feat2_ir, feat3_ir),
                (feat1_vi, feat2_vi, feat3_vi))


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 ImageFusionModel:")
    print("=" * 60)

    # 使用较小尺寸进行测试
    model = ImageFusionModel(
        in_channels=1,
        channels=[32, 64, 128],
        image_size=(192, 256),
        depths=[1, 1, 1],  # 减少深度以加快测试
        num_heads=[2, 4, 8],
        window_size=7
    )

    img_ir = torch.randn(1, 1, 192, 256)  # 使用较小batch size和尺寸
    img_vi = torch.randn(1, 1, 192, 256)

    fused, feat8, feat7, (feat1_ir, feat2_ir, feat3_ir), (feat1_vi, feat2_vi, feat3_vi) = model(img_ir, img_vi)

    print(f"输入:")
    print(f"  img_ir: {img_ir.shape}")
    print(f"  img_vi: {img_vi.shape}")
    print(f"\n输出:")
    print(f"  fused: {fused.shape}")
    print(f"  feat8: {feat8.shape}")
    print(f"  feat7: {feat7.shape}")
    print(f"\n编码器特征:")
    print(f"  feat1_ir: {feat1_ir.shape}, feat1_vi: {feat1_vi.shape}")
    print(f"  feat2_ir: {feat2_ir.shape}, feat2_vi: {feat2_vi.shape}")
    print(f"  feat3_ir: {feat3_ir.shape}, feat3_vi: {feat3_vi.shape}")
    print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}")
