import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, config):
        """
        Initialize the trainer
        
        Args:
            model: Model to train
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            test_loader: DataLoader for testing
            device: Device to use for training (cpu or cuda)
            config: Dictionary containing hyperparameters
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Setup loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config['learning_rate'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay']
            )
        else:
            raise ValueError(f"Optimizer {config['optimizer']} not supported")
            
        # Setup learning rate scheduler
        if config['lr_scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['lr_step_size'],
                gamma=config['lr_gamma']
            )
        elif config['lr_scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=config['lr_min']
            )
        else:
            self.scheduler = None
            
        # Create directory for saving model and logs
        self.save_dir = config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize best validation accuracy and early stopping parameters
        self.best_val_acc = 0
        self.patience = config.get('patience', 5)
        self.early_stop_counter = 0
        
    def train_epoch(self, epoch):
        """
        Train the model for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            train_loss: Average training loss
            train_acc: Training accuracy
        """
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Apply gradient clipping if specified
            if self.config['grad_clip']:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_value'])
                
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
            
        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc
        
    def validate(self):
        """
        Validate the model
        
        Returns:
            val_loss: Average validation loss
            val_acc: Validation accuracy
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        # 创建验证进度条
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                current_loss = val_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
                
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def test(self):
        """
        Test the model
        
        Returns:
            test_loss: Average test loss
            test_acc: Test accuracy
            test_f1: F1 score
            conf_matrix: Confusion matrix
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        # 创建测试进度条
        pbar = tqdm(self.test_loader, desc="Testing", leave=False)
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # 更新进度条
                current_loss = test_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
                
        test_loss = test_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        
        # Calculate F1 score and confusion matrix
        test_f1 = f1_score(all_targets, all_preds, average='macro')
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        return test_loss, test_acc, test_f1, conf_matrix
    
    def train(self):
        """
        Train the model for the specified number of epochs
        
        Returns:
            train_losses: List of training losses
            train_accs: List of training accuracies
            val_losses: List of validation losses
            val_accs: List of validation accuracies
        """
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Check early stopping condition
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0  # Reset counter
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.early_stop_counter += 1
                print(f"EarlyStopping counter: {self.early_stop_counter} out of {self.patience}")
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered!")
                    break
                
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                  
            # Save checkpoint every save_interval epochs
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(epoch)
                
        train_time = time.time() - start_time
        print(f"Training completed in {train_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Test the model
        self.load_checkpoint()  # Load best model
        test_loss, test_acc, test_f1, conf_matrix = self.test()
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, F1 Score: {test_f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
        return train_losses, train_accs, val_losses, val_accs
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint (only model weights, no optimizer state)
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        # 只保存超参数配置，不保存包含数据的对象
        config_to_save = {k: v for k, v in self.config.items() 
                         if k not in ['train_loader', 'val_loader', 'test_loader']}
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': config_to_save
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
        
        torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
    def load_checkpoint(self, path=None):
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint file (if None, load best model)
        """
        if path is None:
            path = os.path.join(self.save_dir, 'best_model.pth')
            
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 不再加载优化器和调度器状态，因为我们不再保存它们
        # 如果需要继续训练，优化器会从当前学习率重新开始
            
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1} with val_acc: {self.best_val_acc:.2f}%")