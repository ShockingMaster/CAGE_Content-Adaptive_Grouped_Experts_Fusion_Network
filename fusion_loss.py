import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):

    def __init__(self, window_size=11, channel=1, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

        window = self._create_window(window_size, channel)
        self.register_buffer('window', window)

    def _gaussian(self, window_size, sigma=1.5):
        gauss = torch.tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        if self.window.size(0) != channel or self.window.device != img1.device or self.window.dtype != img1.dtype:
            window = self._create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.register_buffer('window', window)

        window = self.window

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(dim=(1, 2, 3))


class SobelGradient(nn.Module):

    def __init__(self):
        super(SobelGradient, self).__init__()

        # Sobel X kernel
        self.sobel_x = nn.Parameter(torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0), requires_grad=False)

        # Sobel Y kernel
        self.sobel_y = nn.Parameter(torch.tensor([
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]]
        ], dtype=torch.float32).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        B, C, H, W = x.shape

        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(x)

        for c in range(C):
            grad_x[:, c:c + 1] = F.conv2d(
                x[:, c:c + 1],
                sobel_x,
                padding=1
            )
            grad_y[:, c:c + 1] = F.conv2d(
                x[:, c:c + 1],
                sobel_y,
                padding=1
            )

        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        return grad


class FusionLoss(nn.Module):

    def __init__(self, coeff_int=1.0, coeff_grad=10.0, coeff_ssim=0.0, use_window_weighting=False, window_size=8, window_stride=None, ssim_window_size=11, dataset_ag_ratio=None):
        super(FusionLoss, self).__init__()

        self.coeff_int = coeff_int
        self.coeff_grad = coeff_grad
        self.coeff_ssim = coeff_ssim
        self.use_window_weighting = use_window_weighting
        self.window_size = window_size
        self.window_stride = window_stride if window_stride is not None else window_size
        self.sobel = SobelGradient()

        self.dataset_ag_ratio = dataset_ag_ratio

        if self.coeff_ssim > 0:
            self.ssim = SSIM(window_size=ssim_window_size, channel=1)

    def _compute_window_weighted_grad_loss(self, grad_fused, grad_a, grad_b):
        B, C, H, W = grad_fused.shape
        ws = self.window_size
        stride = self.window_stride
        eps = 1e-6

        if self.dataset_ag_ratio is not None:
            global_ratio_a, global_ratio_b = self.dataset_ag_ratio
        else:
            global_mean_a = grad_a.mean()  
            global_mean_b = grad_b.mean()  
            global_ratio_a = global_mean_a / (global_mean_a + global_mean_b + eps)
            global_ratio_b = global_mean_b / (global_mean_a + global_mean_b + eps)

        grad_a_normalized = grad_a
        grad_b_normalized = grad_b

        if stride < ws:
            num_windows_h = max(1, ((H - ws) // stride) + 1)
            num_windows_w = max(1, ((W - ws) // stride) + 1)
            required_h = stride * (num_windows_h - 1) + ws
            required_w = stride * (num_windows_w - 1) + ws
            pad_h = max(0, required_h - H)
            pad_w = max(0, required_w - W)
        else:
            pad_h = (stride - H % stride) % stride
            pad_w = (stride - W % stride) % stride

        if pad_h > 0 or pad_w > 0:
            grad_fused = F.pad(grad_fused, (0, pad_w, 0, pad_h), mode='reflect')
            grad_a_normalized = F.pad(grad_a_normalized, (0, pad_w, 0, pad_h), mode='reflect')
            grad_b_normalized = F.pad(grad_b_normalized, (0, pad_w, 0, pad_h), mode='reflect')

        grad_fused_windows = F.unfold(grad_fused, kernel_size=ws, stride=stride)
        grad_a_windows = F.unfold(grad_a_normalized, kernel_size=ws, stride=stride)
        grad_b_windows = F.unfold(grad_b_normalized, kernel_size=ws, stride=stride)

        num_windows = grad_fused_windows.shape[2]
        grad_fused_windows = grad_fused_windows.view(B, C, ws * ws, num_windows)
        grad_a_windows = grad_a_windows.view(B, C, ws * ws, num_windows)
        grad_b_windows = grad_b_windows.view(B, C, ws * ws, num_windows)

        AG1 = grad_a_windows.mean(dim=2)  # (B, C, num_windows)
        AG2 = grad_b_windows.mean(dim=2)  # (B, C, num_windows)

        sum_AG = AG1 + AG2 + eps
        w1 = AG1 / sum_AG  # (B, C, num_windows)
        w2 = AG2 / sum_AG  # (B, C, num_windows)

        w1 = w1.unsqueeze(2).expand_as(grad_fused_windows)
        w2 = w2.unsqueeze(2).expand_as(grad_fused_windows)

        loss_a = torch.abs(grad_fused_windows - grad_a_windows)
        loss_b = torch.abs(grad_fused_windows - grad_b_windows)
        weighted_loss = w1 * loss_a / global_ratio_a + w2 * loss_b / global_ratio_b

        return weighted_loss.mean()

    def forward(self, fused, img_a, img_b):
        max_ab = torch.max(img_a, img_b)
        loss_int = F.l1_loss(fused, max_ab)

        grad_a = self.sobel(img_a)
        grad_b = self.sobel(img_b)
        grad_fused = self.sobel(fused)

        if self.use_window_weighting:
            loss_grad = self._compute_window_weighted_grad_loss(grad_fused, grad_a, grad_b)
        else:
            max_grad = torch.max(grad_a, grad_b)
            loss_grad = F.l1_loss(grad_fused, max_grad)

        if self.coeff_ssim > 0:
            loss_ssim_a = self.ssim(fused, img_a)
            loss_ssim_b = self.ssim(fused, img_b)
            loss_ssim = (loss_ssim_a + loss_ssim_b) / 2.0
        else:
            loss_ssim = torch.tensor(0.0).to(fused.device)

        # 总损失
        loss = self.coeff_int * loss_int + self.coeff_grad * loss_grad + self.coeff_ssim * loss_ssim

        return loss, loss_int, loss_grad, loss_ssim


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 SobelGradient:")
    print("=" * 60)

    sobel = SobelGradient()
    img = torch.randn(2, 1, 32, 32)
    grad = sobel(img)
    print(f"输入shape: {img.shape}")
    print(f"梯度shape: {grad.shape}")

    print("\n" + "=" * 60)
    print("测试 FusionLoss:")
    print("=" * 60)

    fusion_loss = FusionLoss(coeff_int=1.0, coeff_grad=10.0)
    fused = torch.randn(2, 1, 32, 32)
    img_a = torch.randn(2, 1, 32, 32)
    img_b = torch.randn(2, 1, 32, 32)

    loss, loss_int, loss_grad = fusion_loss(fused, img_a, img_b)
    print(f"总损失: {loss.item():.4f}")
    print(f"亮度损失: {loss_int.item():.4f}")
    print(f"梯度损失: {loss_grad.item():.4f}")

    print("\n" + "=" * 60)
    print("测试 MultiScaleFusionLoss (不使用相对损失):")
    print("=" * 60)