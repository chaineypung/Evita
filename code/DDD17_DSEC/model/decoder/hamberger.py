import torch
import torch.nn as nn
import torch.nn.functional as F
class _MatrixDecomposition2DBase(nn.Module):
    """
    Matrix Decomposition Base Class
    """
    def __init__(self, MD_S=1, MD_D=512, MD_R=64, train_steps=6, eval_steps=7, inv_t=100, rand_init=True):
        super().__init__()
        self.S = MD_S
        self.D = MD_D
        self.R = MD_R
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.inv_t = inv_t
        self.rand_init = rand_init
    def _build_bases(self, B, S, D, R, device):
        raise NotImplementedError
    def local_step(self, x, bases, coef):
        raise NotImplementedError
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)
        return bases, coef
    def compute_coef(self, x, bases, coef):
        raise NotImplementedError
    def forward(self, x):
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B * S, D, N)
        # Assume spatial=True
        D = C // self.S
        N = H * W
        x = x.view(B * self.S, D, N)
        if not self.rand_init and not hasattr(self, "bases"):
            bases = self._build_bases(1, self.S, D, self.R, device=x.device)
            self.register_buffer("bases", bases)
        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        else:
            bases = self.bases.repeat(B, 1, 1)
        bases, coef = self.local_inference(x, bases)
        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)
        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))
        # (B * S, D, N) -> (B, C, H, W)
        x = x.view(B, C, H, W)
        return x
class NMF2D(_MatrixDecomposition2DBase):
    """
    Non-negative Matrix Factorization
    """
    def __init__(self, ham_channels=512, MD_R=64, train_steps=6, eval_steps=7):
        super().__init__(MD_S=1, MD_D=ham_channels, MD_R=MD_R,
                         train_steps=train_steps, eval_steps=eval_steps, inv_t=1)
    def _build_bases(self, B, S, D, R, device):
        bases = torch.rand((B * S, D, R), device=device)
        bases = F.normalize(bases, dim=1)
        return bases
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef
    def compute_coef(self, x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef
class Hamburger(nn.Module):
    """
    Hamburger Module
    """
    def __init__(self, ham_channels=512, MD_R=64, train_steps=6, eval_steps=7):
        super().__init__()
        self.ham_in = nn.Conv2d(ham_channels, ham_channels, kernel_size=1)
        self.ham = NMF2D(ham_channels, MD_R, train_steps, eval_steps)
        self.ham_out = nn.Sequential(
            nn.Conv2d(ham_channels, ham_channels, kernel_size=1),
            nn.BatchNorm2d(ham_channels)
        )
    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)
        return ham
class LightHamHead(nn.Module):
    """
    LightHamHead for Semantic Segmentation (SegNeXt Style)
    Only uses stages 2, 3, 4 by default (indices [1, 2, 3]).
    """
    def __init__(self, in_channels=[64, 128, 320, 512], in_index=[1, 2, 3],
                 num_classes=40, ham_channels=512, dropout_ratio=0.1, align_corners=False):
        super().__init__()
        self.in_channels = in_channels
        self.in_index = in_index # [1, 2, 3] -> Stage 2, 3, 4
        self.num_classes = num_classes
        self.ham_channels = ham_channels
        self.align_corners = align_corners
        # Calculate input channels based on selected indices
        # Example: if in_index=[1, 2, 3], we take in_channels[1], [2], [3]
        selected_in_channels = [in_channels[i] for i in in_index]
        total_in_channels = sum(selected_in_channels)
        # 1. Squeeze: Fuse features from Stage 2, 3, 4 into ham_channels
        self.squeeze = nn.Sequential(
            nn.Conv2d(total_in_channels, ham_channels, kernel_size=1),
            nn.BatchNorm2d(ham_channels),
            nn.ReLU(inplace=True)
        )
        # 2. Hamburger Module
        self.hamburger = Hamburger(ham_channels=ham_channels, MD_R=16, train_steps=6, eval_steps=7)
        # 3. Align
        self.align = nn.Sequential(
            nn.Conv2d(ham_channels, ham_channels, kernel_size=1),
            nn.BatchNorm2d(ham_channels),
            nn.ReLU(inplace=True)
        )
        # 4. Classifier
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.cls_seg = nn.Conv2d(ham_channels, num_classes, kernel_size=1)
    def forward(self, inputs):
        # inputs is a list: [C1, C2, C3, C4]
        # 1. Select inputs based on in_index (default: C2, C3, C4)
        inputs = [inputs[i] for i in self.in_index]
        # 2. Resize all levels to the size of the first selected level
        # (Usually C2, which is 1/8 scale)
        target_size = inputs[0].shape[2:]
        resized_inputs = []
        for level in inputs:
            if level.shape[2:] != target_size:
                resized = F.interpolate(level, size=target_size, mode='bilinear', align_corners=self.align_corners)
            else:
                resized = level
            resized_inputs.append(resized)
        # 3. Concat
        x = torch.cat(resized_inputs, dim=1)
        # 4. Squeeze -> Hamburger -> Align
        x = self.squeeze(x)
        x = self.hamburger(x)
        x = self.align(x)
        # 5. Predict
        output = self.dropout(x)
        output = self.cls_seg(output)
        return output
# Test Code
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Simulate Backbone Output (B, C, H, W)
    # C1 (1/4), C2 (1/8), C3 (1/16), C4 (1/32)
    dummy_inputs = [
        torch.randn(2, 64, 128, 128).to(device), # C1 (will be ignored)
        torch.randn(2, 128, 64, 64).to(device), # C2
        torch.randn(2, 320, 32, 32).to(device), # C3
        torch.randn(2, 512, 16, 16).to(device) # C4
    ]
    # Instantiate Head: using stages 2, 3, 4
    head = LightHamHead(in_channels=[64, 128, 320, 512], in_index=[1, 2, 3],
                        num_classes=40, ham_channels=512).to(device)
    output = head(dummy_inputs)
    # Output shape should be same spatial size as C2 (64x64 in this example)
    print("Output Shape:", output.shape) # Expected: [2, 40, 64, 64]