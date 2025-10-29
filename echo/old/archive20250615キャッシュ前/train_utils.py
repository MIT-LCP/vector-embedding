# 簡略化されたtrain_utils.py（Keras風trainer保持、dynamic lambda保持）

import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

def setup_logging(log_dir="./logs"):
    """ログ設定"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class SimpleTrainer:
    """Keras風trainer（保持）"""
    
    def __init__(self, model, log_dir="./logs"):
        self.model = model
        self.logger = setup_logging(log_dir)
        self.history = defaultdict(list)
        self.best_value = float('inf')
        
        # プロット保存パス
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.plot_path = os.path.join(log_dir, f"training_plot_{timestamp}.png")
    
    def train_epoch(self, train_loader, optimizer, criterion_triplet, criterion_adv, 
                   device, config, epoch):
        """1エポックの訓練（dynamic lambda保持）"""
        self.model.train()
        
        # Adversarial headsもトレーニングモードに
        if config.use_adversarial:
            for head in self.model.adversarial_heads.values():
                head.train()
        
        # 🔥 Dynamic lambda scheduling（保持）
        if config.use_adversarial and config.dynamic_lambda:
            p = float(epoch) / 100
            lambda_dynamic = 2. / (1. + np.exp(-10 * p)) - 1
            lambda_final = config.lambda_adv * lambda_dynamic
            
            for head in self.model.adversarial_heads.values():
                head.set_lambda(lambda_final)
        else:
            lambda_final = config.lambda_adv if config.use_adversarial else 0.0
        
        # メトリクス
        total_loss = 0
        total_triplet_loss = 0
        adversarial_losses = {attr: 0 for attr in config.adversarial_attributes}
        batch_count = 0
        
        # プログレスバー
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch in pbar:
            batch_count += 1
            
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            anchor_features = self.model(anchor)
            pos_features = self.model(positive)
            neg_features = self.model(negative)
            
            # Triplet loss
            triplet_loss = criterion_triplet(anchor_features, pos_features, neg_features)
            total_batch_loss = triplet_loss
            
            # Adversarial losses
            if config.use_adversarial:
                for attr in config.adversarial_attributes:
                    if attr in batch:
                        attr_labels = batch[attr].to(device)
                        attr_pred = self.model.adversarial_heads[attr](anchor_features)
                        attr_loss = criterion_adv(attr_pred, attr_labels)
                        total_batch_loss += attr_loss
                        adversarial_losses[attr] += attr_loss.item()
            
            # Backward pass
            total_batch_loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += total_batch_loss.item()
            total_triplet_loss += triplet_loss.item()
            
            # プログレスバー更新
            pbar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'triplet': f'{triplet_loss.item():.4f}',
                'lambda': f'{lambda_final:.3f}'
            })
        
        # エポック終了時のメトリクス
        if batch_count > 0:
            epoch_metrics = {
                'loss': total_loss / batch_count,
                'triplet_loss': total_triplet_loss / batch_count,
            }
            for attr in config.adversarial_attributes:
                if attr in adversarial_losses:
                    epoch_metrics[f'{attr}_loss'] = adversarial_losses[attr] / batch_count
        else:
            epoch_metrics = {'loss': float('nan'), 'triplet_loss': float('nan')}
            for attr in config.adversarial_attributes:
                epoch_metrics[f'{attr}_loss'] = float('nan')
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader, device, config):
        """1エポックの検証"""
        self.model.eval()
        
        if config.use_adversarial:
            for head in self.model.adversarial_heads.values():
                head.eval()
        
        adversarial_correct = {attr: 0 for attr in config.adversarial_attributes}
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                anchor = batch['anchor'].to(device)
                anchor_features = self.model(anchor)
                
                if config.use_adversarial:
                    for attr in config.adversarial_attributes:
                        if attr in batch:
                            attr_labels = batch[attr].to(device)
                            attr_pred = self.model.adversarial_heads[attr].head(anchor_features)
                            adversarial_correct[attr] += (attr_pred.argmax(1) == attr_labels).sum().item()
                
                total += anchor.size(0)
        
        # 検証メトリクス
        val_metrics = {}
        if config.use_adversarial and total > 0:
            for attr in config.adversarial_attributes:
                if attr in adversarial_correct:
                    val_metrics[f'{attr}_accuracy'] = adversarial_correct[attr] / total
            
            # Fairness score
            if val_metrics:
                fairness_score = sum(v for k, v in val_metrics.items() if 'accuracy' in k) / len([k for k in val_metrics.keys() if 'accuracy' in k])
                val_metrics['fairness_score'] = fairness_score
        
        return val_metrics
    
    def save_model(self, save_path, epoch, train_metrics, val_metrics, config):
        """モデル保存"""
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config.__dict__,
        }
        
        torch.save(save_dict, save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    def plot_training_history(self):
        """訓練履歴プロット"""
        if not self.history or 'epoch' not in self.history:
            return
        
        epochs = self.history['epoch']
        loss_metrics = [k for k in self.history.keys() if 'loss' in k and k != 'epoch']
        acc_metrics = [k for k in self.history.keys() if 'acc' in k or 'accuracy' in k]
        
        plot_count = sum([bool(loss_metrics), bool(acc_metrics)])
        if plot_count == 0:
            return
        
        fig, axes = plt.subplots(1, plot_count, figsize=(5 * plot_count, 4))
        if plot_count == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Loss プロット
        if loss_metrics:
            ax = axes[plot_idx]
            for metric in loss_metrics:
                ax.plot(epochs, self.history[metric], label=metric, marker='o', markersize=3)
            ax.set_title('Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Accuracy プロット
        if acc_metrics:
            ax = axes[plot_idx]
            for metric in acc_metrics:
                ax.plot(epochs, self.history[metric], label=metric, marker='o', markersize=3)
            ax.set_title('Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Training plot saved to {self.plot_path}")
    
    def fit(self, train_loader, val_loader, optimizer, criterion_triplet, criterion_adv,
            device, config, epochs, scheduler=None, save_best_path='best_model.pth'):
        """Keras風fit関数（保持）"""
        
        self.logger.info("Training Started")
        print("=" * 60)
        
        # トレーニングループ
        for epoch in range(epochs):
            start_time = time.time()
            
            # トレーニング
            train_metrics = self.train_epoch(
                train_loader, optimizer, criterion_triplet, criterion_adv,
                device, config, epoch
            )
            
            # 検証
            val_metrics = self.validate_epoch(val_loader, device, config)
            
            # スケジューラ更新
            if scheduler:
                scheduler.step()
            
            # 履歴更新
            self.history['epoch'].append(epoch + 1)
            for k, v in train_metrics.items():
                self.history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                self.history[f'val_{k}'].append(v)
            
            # ベストモデル保存
            current_loss = train_metrics.get('loss', float('inf'))
            if current_loss < self.best_value:
                self.best_value = current_loss
                self.save_model(save_best_path, epoch, train_metrics, val_metrics, config)
            
            # エポック結果ログ
            epoch_time = time.time() - start_time
            log_msg = f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
            
            # メトリクスをログ
            all_metrics = []
            for k, v in train_metrics.items():
                all_metrics.append(f"train_{k}: {v:.4f}")
            for k, v in val_metrics.items():
                all_metrics.append(f"val_{k}: {v:.4f}")
            
            log_msg += " - ".join(all_metrics)
            self.logger.info(log_msg)
        
        # 終了処理
        self.logger.info("Training Completed!")
        print("=" * 60)
        self.plot_training_history()

def create_trainer(model, log_dir="./logs"):
    """トレーナー作成ヘルパー"""
    return SimpleTrainer(model, log_dir)