# 修正されたmodel_utils.py（正確なパラメータカウント）

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

class TrainingConfig:
    def __init__(self, 
                 use_adversarial=True,
                 adversarial_attributes=['Sex', 'Race'],
                 lambda_adv=1.0,
                 dynamic_lambda=True,  # 保持
                 use_lora=True,
                 lora_r=8):
        self.use_adversarial = use_adversarial
        self.adversarial_attributes = adversarial_attributes if use_adversarial else []
        self.lambda_adv = lambda_adv
        self.dynamic_lambda = dynamic_lambda  # 保持
        self.use_lora = use_lora
        self.lora_r = lora_r

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

class AdversarialHead(nn.Module):
    """Adversarial head with GRL"""
    def __init__(self, input_dim=512, num_classes=2, dropout=0.3):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.grl(x)
        return self.head(x)
    
    def set_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)


class SimpleLoRA(nn.Module):
    """修正されたLoRA実装（最低限の修正）"""
    def __init__(self, original_layer, r=8, alpha=32, dropout=0.1):
        super().__init__()
        self.original = original_layer
        self.r = r
        self.scaling = alpha / r
        
        # 元のパラメータを明示的に固定
        for param in self.original.parameters():
            param.requires_grad = False
        
        # LoRAパラメータ（これらのみが訓練可能）
        self.lora_A = nn.Parameter(torch.randn(r, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, r))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 🚨 修正箇所：元の出力は勾配なしで計算、LoRA部分のみ勾配あり
        with torch.no_grad():
            original_output = self.original(x)
        
        # LoRA部分（勾配あり）
        lora_output = F.linear(x, (self.lora_B @ self.lora_A) * self.scaling)
        
        # 合成（LoRAの勾配が保持される）
        return original_output + self.dropout(lora_output)

class EchoEmbeddingModel(nn.Module):
    def __init__(self, base_encoder, config):
        super().__init__()
        self.config = config
        self.base_encoder = base_encoder
        
        # LoRAを適用（オプション）
        if config.use_lora:
            self._apply_lora()
        
        # Adversarial heads
        self.adversarial_heads = nn.ModuleDict()
        if config.use_adversarial:
            attr_classes = {'Sex': 2, 'Race': 4}
            for attr in config.adversarial_attributes:
                self.adversarial_heads[attr] = AdversarialHead(
                    input_dim=512, 
                    num_classes=attr_classes.get(attr, 2),
                    dropout=0.3
                )
        
        # パラメータ統計を表示
        self._print_trainable_params()
    
    def _apply_lora(self):
        """LoRAを主要なLinear層に適用"""
        target_modules = ['qkv', 'proj']
        
        # 最初に全てのbase_encoderパラメータを固定
        for param in self.base_encoder.parameters():
            param.requires_grad = False
        
        lora_applied_count = 0
        for name, module in self.base_encoder.named_modules():
            if isinstance(module, nn.Linear) and any(x in name for x in target_modules):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(self.base_encoder.named_modules())[parent_name]
                    lora_module = SimpleLoRA(
                        module, 
                        r=self.config.lora_r, 
                        alpha=32, 
                        dropout=0.1
                    )
                    setattr(parent, child_name, lora_module)
                    lora_applied_count += 1
        
        print(f"LoRA: adapted to {lora_applied_count} linear layes")
        
        # LoRA適用後、LoRAパラメータのみを訓練可能にする
        for name, module in self.base_encoder.named_modules():
            if isinstance(module, SimpleLoRA):
                # LoRAパラメータは訓練可能
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
                # 元のパラメータは固定を確認
                for param in module.original.parameters():
                    param.requires_grad = False
    
    def _print_trainable_params(self):
        # 全体のパラメータ統計
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Trainable propotion: {100 * trainable_params / total_params:.2f}%")
    
    def forward(self, x):
        features = self.base_encoder(x)
        return features