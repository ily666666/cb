import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class FedAvgClient:
	"""标准FedAvg边侧（无正则化）"""
	
	def __init__(self, client_id, model, train_loader, val_loader, device, config, test_loader=None):
		"""
		初始化边侧
		
		Args:
			client_id: 边侧ID
			model: 模型
			train_loader: 训练数据加载器
			val_loader: 验证数据加载器
			device: 设备
			config: 配置参数
			test_loader: 边侧测试数据加载器（可选）
		"""
		self.client_id = client_id
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.device = device
		self.config = config
		
		# 设置损失函数
		self.criterion = nn.CrossEntropyLoss()
		
		# 设置优化器
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
		
		# 设置学习率调度器
		if config['lr_scheduler'] == 'step':
			self.scheduler = optim.lr_scheduler.StepLR(
				self.optimizer,
				step_size=config['lr_step_size'],
				gamma=config['lr_gamma']
			)
		elif config['lr_scheduler'] == 'cosine':
			self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
				self.optimizer,
				T_max=config['local_epochs'],
				eta_min=config['lr_min']
			)
		else:
			self.scheduler = None
	
	def train_local(self, global_model_state):
		"""
		标准FedAvg本地训练（无正则化）
		
		Args:
			global_model_state: 全局模型状态
			
		Returns:
			local_model_state: 本地训练后的模型状态
			train_loss: 训练损失
			train_acc: 训练准确率
		"""
		# 加载全局模型参数
		self.model.load_state_dict(global_model_state)
		
		# 将模型移到指定设备
		self.model = self.model.to(self.device)
		
		# 重新创建优化器和调度器（确保状态正确）
		if self.config['optimizer'] == 'adam':
			self.optimizer = optim.Adam(
				self.model.parameters(), 
				lr=self.config['learning_rate'],
				weight_decay=self.config['weight_decay']
			)
		elif self.config['optimizer'] == 'sgd':
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr=self.config['learning_rate'],
				momentum=self.config['momentum'],
				weight_decay=self.config['weight_decay']
			)
		
		# 重新创建学习率调度器
		if self.config['lr_scheduler'] == 'step':
			self.scheduler = optim.lr_scheduler.StepLR(
				self.optimizer,
				step_size=self.config['lr_step_size'],
				gamma=self.config['lr_gamma']
			)
		elif self.config['lr_scheduler'] == 'cosine':
			self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
				self.optimizer,
				T_max=self.config['local_epochs'],
				eta_min=self.config['lr_min']
			)
		else:
			self.scheduler = None
		
		# 本地训练
		self.model.train()
		train_loss = 0
		correct = 0
		total = 0

		for epoch in range(self.config['local_epochs']):
			epoch_loss = 0
			epoch_correct = 0
			epoch_total = 0
			
			# 创建进度条
			pbar = tqdm(
				self.train_loader, 
				desc=f"Client {self.client_id+1} - Epoch {epoch+1}/{self.config['local_epochs']}",
				leave=True
			)
			
			for batch_idx, (data, targets) in enumerate(pbar):
				data, targets = data.to(self.device), targets.to(self.device)
				
				self.optimizer.zero_grad()
				outputs = self.model(data)
				loss = self.criterion(outputs, targets)  # 标准交叉熵损失，无正则化
				
				loss.backward()
				
				# 检查梯度是否有效
				grad_norm = None
				try:
					grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
					# 检查梯度范数是否合理
					if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 100:
						print(f"警告：Client {self.client_id+1} 检测到异常梯度范数 {grad_norm:.4f}，跳过更新")
						# 清零梯度并跳过该batch
						self.optimizer.zero_grad()
						continue
				except Exception as e:
					print(f"警告：Client {self.client_id+1} 梯度计算错误: {e}，跳过该batch")
					self.optimizer.zero_grad()
					continue
				
				self.optimizer.step()
				
				# 检查模型输出是否有效
				if torch.isnan(outputs).any() or torch.isinf(outputs).any():
					print(f"警告：Client {self.client_id+1} 检测到无效模型输出，清零梯度")
					self.optimizer.zero_grad()
					# 重置模型参数
					self.model.load_state_dict(global_model_state)
					continue
				
				self.optimizer.step()
				
				epoch_loss += loss.item()
				_, predicted = outputs.max(1)
				epoch_total += targets.size(0)
				epoch_correct += predicted.eq(targets).sum().item()
				
				# 更新进度条
				current_loss = epoch_loss / (batch_idx + 1)
				current_acc = 100. * epoch_correct / epoch_total
				pbar.set_postfix({
					'loss': f'{current_loss:.4f}',
					'acc': f'{current_acc:.2f}%'
				})
			
			# 更新学习率
			if self.scheduler is not None:
				self.scheduler.step()
			
			train_loss = epoch_loss / len(self.train_loader)
			train_acc = 100. * epoch_correct / epoch_total
		
		# 返回本地模型状态
		local_model_state = copy.deepcopy(self.model.state_dict())
		
		# 清理GPU内存
		if self.device.type == 'cuda':
			del self.optimizer
			if self.scheduler is not None:
				del self.scheduler
			self.model = self.model.cpu()
			torch.cuda.empty_cache()
		
		return local_model_state, train_loss, train_acc
	
	def validate_local(self, global_model_state):
		"""
		本地验证
		
		Args:
			global_model_state: 全局模型状态
			
		Returns:
			val_loss: 验证损失
			val_acc: 验证准确率
		"""
		# 加载全局模型参数
		self.model.load_state_dict(global_model_state)
		self.model = self.model.to(self.device)
		
		self.model.eval()
		val_loss = 0
		correct = 0
		total = 0
		
		# 创建验证进度条
		pbar = tqdm(
			self.val_loader, 
			desc=f"Client {self.client_id+1} - Validation",
			leave=False
		)
		
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
		
		# 清理GPU内存
		if self.device.type == 'cuda':
			self.model = self.model.cpu()
			torch.cuda.empty_cache()
		
		return val_loss, val_acc

	def test_local(self, global_model_state):
		"""本地测试（在边侧自己的测试集上）"""
		if self.test_loader is None:
			return None
		
		self.model.load_state_dict(global_model_state)
		self.model = self.model.to(self.device)
		self.model.eval()
		
		criterion = nn.CrossEntropyLoss()
		test_loss = 0
		correct = 0
		total = 0
		all_preds = []
		all_targets = []
		
		pbar = tqdm(self.test_loader, desc=f"Client {self.client_id+1} - Testing", leave=False)
		with torch.no_grad():
			for batch_idx, (data, targets) in enumerate(pbar):
				data, targets = data.to(self.device), targets.to(self.device)
				outputs = self.model(data)
				loss = criterion(outputs, targets)
				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()
				all_preds.extend(predicted.cpu().numpy())
				all_targets.extend(targets.cpu().numpy())
				current_loss = test_loss / (batch_idx + 1)
				current_acc = 100. * correct / total
				pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
		
		test_loss = test_loss / len(self.test_loader)
		test_acc = 100. * correct / total
		test_f1 = f1_score(all_targets, all_preds, average='macro')
		conf_matrix = confusion_matrix(all_targets, all_preds)
		
		if self.device.type == 'cuda':
			self.model = self.model.cpu()
			torch.cuda.empty_cache()
		
		return test_loss, test_acc, test_f1, conf_matrix


class FedProxClient:
	"""FedProx边侧（带正则化）"""
	
	def __init__(self, client_id, model, train_loader, val_loader, device, config, test_loader=None):
		"""
		初始化边侧
		
		Args:
			client_id: 边侧ID
			model: 模型
			train_loader: 训练数据加载器
			val_loader: 验证数据加载器
			device: 设备
			config: 配置参数
			test_loader: 边侧测试数据加载器（可选）
		"""
		self.client_id = client_id
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.test_loader = test_loader
		self.device = device
		self.config = config
		
		# 设置损失函数
		self.criterion = nn.CrossEntropyLoss()
		# 近端与分类头正则配置
		self.prox_mu = config.get('prox_mu', 0.0) or 0.0
		self.head_reg_lambda = config.get('head_reg_lambda', 0.0) or 0.0
		
		# 设置优化器
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
		
		# 设置学习率调度器
		if config['lr_scheduler'] == 'step':
			self.scheduler = optim.lr_scheduler.StepLR(
				self.optimizer,
				step_size=config['lr_step_size'],
				gamma=config['lr_gamma']
			)
		elif config['lr_scheduler'] == 'cosine':
			self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
				self.optimizer,
				T_max=config['local_epochs'],
				eta_min=config['lr_min']
			)
		else:
			self.scheduler = None
	
	def train_local(self, global_model_state):
		"""
		本地训练
		
		Args:
			global_model_state: 全局模型状态
			
		Returns:
			local_model_state: 本地训练后的模型状态
			train_loss: 训练损失
			train_acc: 训练准确率
		"""
		# 加载全局模型参数
		self.model.load_state_dict(global_model_state)
		
		# 将模型移到指定设备
		self.model = self.model.to(self.device)
		
		# 重新创建优化器和调度器（确保状态正确）
		if self.config['optimizer'] == 'adam':
			self.optimizer = optim.Adam(
				self.model.parameters(), 
				lr=self.config['learning_rate'],
				weight_decay=self.config['weight_decay']
			)
		elif self.config['optimizer'] == 'sgd':
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr=self.config['learning_rate'],
				momentum=self.config['momentum'],
				weight_decay=self.config['weight_decay']
			)
		
		# 重新创建学习率调度器
		if self.config['lr_scheduler'] == 'step':
			self.scheduler = optim.lr_scheduler.StepLR(
				self.optimizer,
				step_size=self.config['lr_step_size'],
				gamma=self.config['lr_gamma']
			)
		elif self.config['lr_scheduler'] == 'cosine':
			self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
				self.optimizer,
				T_max=self.config['local_epochs'],
				eta_min=self.config['lr_min']
			)
		else:
			self.scheduler = None
		
		# 本地训练
		self.model.train()
		train_loss = 0
		correct = 0
		total = 0

		# 记录全局参数用于FedProx（仅可训练参数）
		# 将全局state加载到一个临时模型，以便与named_parameters对齐
		temp_model = copy.deepcopy(self.model)
		temp_model.load_state_dict(global_model_state)
		global_params = {name: p.detach().clone() for name, p in temp_model.named_parameters() if p.requires_grad}
		del temp_model
 
		# 推断边侧包含的标签集合（从train_loader抽取一次）
		client_present_classes = None
		if self.head_reg_lambda > 0.0 and hasattr(self.model, 'fc') and hasattr(self.model.fc, 'weight'):
			# 尝试从一个batch统计出现的类，作为近似
			try:
				for data_tmp, targets_tmp in self.train_loader:
					client_present_classes = set(targets_tmp.view(-1).tolist())
					break
			except Exception:
				client_present_classes = None
 
		for epoch in range(self.config['local_epochs']):
			epoch_loss = 0
			epoch_correct = 0
			epoch_total = 0
			epoch_prox_sum = 0.0
			epoch_head_sum = 0.0
			batch_count = 0
			
			# 创建进度条
			pbar = tqdm(
				self.train_loader, 
				desc=f"Client {self.client_id+1} - Epoch {epoch+1}/{self.config['local_epochs']}",
				leave=True
			)
			
			for batch_idx, (data, targets) in enumerate(pbar):
				data, targets = data.to(self.device), targets.to(self.device)
				
				self.optimizer.zero_grad()
				outputs = self.model(data)
				loss = self.criterion(outputs, targets)
				
				# 检查损失是否为有效数值
				if torch.isnan(loss) or torch.isinf(loss):
					print(f"警告：Client {self.client_id+1} 检测到无效损失值，跳过该batch")
					continue
				
				# FedProx近端正则：约束参数不偏离全局（仅可训练参数；按参数规模做归一化）
				prox_val = 0.0
				if self.prox_mu > 0.0:
					prox_sum = None
					total_numel = 0
					for name, p in self.model.named_parameters():
						if not p.requires_grad:
							continue
						if name not in global_params:
							continue
						# 确保global_params[name]在同一设备上
						param_global = global_params[name].to(p.device)
						diff = p - param_global
						term = (diff * diff).sum()
						prox_sum = term if prox_sum is None else (prox_sum + term)
						total_numel += p.numel()
					if prox_sum is not None and total_numel > 0:
						prox_mean = prox_sum / float(total_numel)
						prox_val = prox_mean.detach().item()
						loss = loss + (self.prox_mu / 2.0) * prox_mean
				
				# 分类头未见类的L2保持（使用均值项，稳定规模）
				head_val = 0.0
				if self.head_reg_lambda > 0.0 and client_present_classes is not None and hasattr(self.model, 'fc'):
					fc_weight = self.model.fc.weight
					fc_bias = getattr(self.model, 'fc').bias if hasattr(self.model, 'fc') else None
					all_classes = set(range(fc_weight.size(0)))
					absent = list(all_classes - set(client_present_classes))
					if len(absent) > 0:
						gw = global_params['fc.weight'] if 'fc.weight' in global_params else None
						gb = global_params['fc.bias'] if 'fc.bias' in global_params else None
						head_terms = []
						if gw is not None:
							term_w = (fc_weight[absent] - gw[absent].to(fc_weight.device))
							head_terms.append((term_w * term_w).mean())
						if gb is not None and fc_bias is not None:
							term_b = (fc_bias[absent] - gb[absent].to(fc_bias.device))
							head_terms.append((term_b * term_b).mean())
						if len(head_terms) > 0:
							head_term = torch.stack(head_terms).mean()
							head_val = head_term.detach().item()
							loss = loss + self.head_reg_lambda * head_term
				
				loss.backward()
				
				# 梯度裁剪和数值稳定性检查
				grad_norm = None
				try:
					# 更严格的梯度裁剪和数值稳定性检查
					grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
					# 检查梯度范数是否合理
					if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 50:
						print(f"警告：Client {self.client_id+1} 检测到异常梯度范数 {grad_norm:.4f}，跳过更新")
						self.optimizer.zero_grad()
						continue
				except Exception as e:
					print(f"警告：Client {self.client_id+1} 梯度计算错误: {e}，跳过该batch")
					self.optimizer.zero_grad()
					continue
				
				# 检查模型输出是否有效
				if torch.isnan(outputs).any() or torch.isinf(outputs).any():
					print(f"警告：Client {self.client_id+1} 检测到无效模型输出，清零梯度")
					self.optimizer.zero_grad()
					continue
				
				self.optimizer.step()
				
				epoch_loss += loss.item()
				_, predicted = outputs.max(1)
				epoch_total += targets.size(0)
				epoch_correct += predicted.eq(targets).sum().item()
				epoch_prox_sum += float(prox_val)
				epoch_head_sum += float(head_val)
				batch_count += 1
				
				# 更新进度条
				current_loss = epoch_loss / (batch_idx + 1)
				current_acc = 100. * epoch_correct / epoch_total
				pbar.set_postfix({
					'loss': f'{current_loss:.4f}',
					'acc': f'{current_acc:.2f}%',
					'prox': f'{(epoch_prox_sum/max(1,batch_count)):.2e}',
					'head': f'{(epoch_head_sum/max(1,batch_count)):.2e}'
				})
			
			# 更新学习率
			if self.scheduler is not None:
				self.scheduler.step()
			
			train_loss = epoch_loss / len(self.train_loader)
			train_acc = 100. * epoch_correct / epoch_total
			avg_prox = epoch_prox_sum / max(1, batch_count)
			avg_head = epoch_head_sum / max(1, batch_count)
			print(f"Client {self.client_id+1} - Epoch {epoch+1} avg prox={avg_prox:.2e}, head={avg_head:.2e}")
		
		# 返回本地模型状态
		local_model_state = copy.deepcopy(self.model.state_dict())
		
		# 清理GPU内存
		if self.device.type == 'cuda':
			del self.optimizer
			if self.scheduler is not None:
				del self.scheduler
			self.model = self.model.cpu()
			torch.cuda.empty_cache()
		
		return local_model_state, train_loss, train_acc
	
	def validate_local(self, global_model_state):
		"""
		本地验证
		
		Args:
			global_model_state: 全局模型状态
			
		Returns:
			val_loss: 验证损失
			val_acc: 验证准确率
		"""
		# 加载全局模型参数
		self.model.load_state_dict(global_model_state)
		self.model = self.model.to(self.device)
		
		self.model.eval()
		val_loss = 0
		correct = 0
		total = 0
		
		# 创建验证进度条
		pbar = tqdm(
			self.val_loader, 
			desc=f"Client {self.client_id+1} - Validation",
			leave=False
		)
		
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
		
		# 清理GPU内存
		if self.device.type == 'cuda':
			self.model = self.model.cpu()
			torch.cuda.empty_cache()
		
		return val_loss, val_acc

	def test_local(self, global_model_state):
		"""本地测试（在边侧自己的测试集上）"""
		if self.test_loader is None:
			return None
		
		self.model.load_state_dict(global_model_state)
		self.model = self.model.to(self.device)
		self.model.eval()
		
		criterion = nn.CrossEntropyLoss()
		test_loss = 0
		correct = 0
		total = 0
		all_preds = []
		all_targets = []
		
		pbar = tqdm(self.test_loader, desc=f"Client {self.client_id+1} - Testing", leave=False)
		with torch.no_grad():
			for batch_idx, (data, targets) in enumerate(pbar):
				data, targets = data.to(self.device), targets.to(self.device)
				outputs = self.model(data)
				loss = criterion(outputs, targets)
				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()
				all_preds.extend(predicted.cpu().numpy())
				all_targets.extend(targets.cpu().numpy())
				current_loss = test_loss / (batch_idx + 1)
				current_acc = 100. * correct / total
				pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
		
		test_loss = test_loss / len(self.test_loader)
		test_acc = 100. * correct / total
		test_f1 = f1_score(all_targets, all_preds, average='macro')
		conf_matrix = confusion_matrix(all_targets, all_preds)
		
		if self.device.type == 'cuda':
			self.model = self.model.cpu()
			torch.cuda.empty_cache()
		
		return test_loss, test_acc, test_f1, conf_matrix


class FederatedServer:
	"""联邦学习云侧"""
	
	def __init__(self, global_model, test_loader, device, config, save_dir):
		"""
		初始化云侧
		
		Args:
			global_model: 全局模型
			test_loader: 测试数据加载器
			device: 设备
			config: 配置参数
			save_dir: 保存目录
		"""
		self.global_model = global_model
		self.test_loader = test_loader
		self.device = device
		self.config = config
		self.save_dir = save_dir
		
		# 设置损失函数
		self.criterion = nn.CrossEntropyLoss()
		
		# 初始化全局模型状态
		self.global_model_state = copy.deepcopy(global_model.state_dict())
		
		# 训练历史
		self.train_history = {
			'round': [],
			'global_test_acc': [],
			'global_test_loss': [],
			'global_test_f1': []
		}
	
	def aggregate_models(self, client_models, client_weights=None):
		"""
		聚合边侧模型
		
		Args:
			client_models: 边侧模型状态列表
			client_weights: 边侧权重列表（可选）
		"""
		if client_weights is None:
			# 平均聚合
			client_weights = [1.0 / len(client_models)] * len(client_models)
		
		# 归一化权重
		total_weight = sum(client_weights)
		client_weights = [w / total_weight for w in client_weights]
		
		# 初始化聚合后的模型状态
		aggregated_state = {}
		
		# 获取模型参数键
		param_keys = client_models[0].keys()
		
		# 加权聚合每个参数
		for key in param_keys:
			# 对于 BatchNorm 的计数缓冲（整型，如 num_batches_tracked），不做加权平均，直接取第一个边侧的值
			if key.endswith('num_batches_tracked'):
				aggregated_state[key] = client_models[0][key].clone()
				continue
			
			# 其他张量：浮点参数与缓冲做加权平均，整数类型（若有）保持第一个边侧的值
			tensor_template = client_models[0][key]
			if tensor_template.dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]:
				aggregated_state[key] = tensor_template.clone()
				continue
			
			# 浮点参数与缓冲
			aggregated = torch.zeros_like(tensor_template)
			for i, state in enumerate(client_models):
				aggregated += state[key] * client_weights[i]
			aggregated_state[key] = aggregated
		
		# 更新全局模型
		self.global_model_state = aggregated_state
		self.global_model.load_state_dict(self.global_model_state)
	
	def evaluate_global_model(self):
		"""
		评估全局模型
		
		Returns:
			test_loss: 测试损失
			test_acc: 测试准确率
			test_f1: F1分数
			conf_matrix: 混淆矩阵
		"""
		self.global_model = self.global_model.to(self.device)
		self.global_model.eval()
		
		test_loss = 0
		correct = 0
		total = 0
		all_preds = []
		all_targets = []
		
		# 创建全局评估进度条
		pbar = tqdm(
			self.test_loader, 
			desc="Global Model Evaluation",
			leave=True
		)
		
		with torch.no_grad():
			for batch_idx, (data, targets) in enumerate(pbar):
				data, targets = data.to(self.device), targets.to(self.device)
				outputs = self.global_model(data)
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
		test_f1 = f1_score(all_targets, all_preds, average='macro')
		conf_matrix = confusion_matrix(all_targets, all_preds)
		
		# 清理GPU内存
		if self.device.type == 'cuda':
			self.global_model = self.global_model.cpu()
			torch.cuda.empty_cache()
		
		return test_loss, test_acc, test_f1, conf_matrix
	
	def save_global_model(self, round_num):
		"""
		保存全局模型
		
		Args:
			round_num: 当前轮次
		"""
		if round_num % self.config['global_save_interval'] == 0:
			save_path = os.path.join(self.save_dir, f'global_model_round_{round_num}.pth')
			# 只保存超参数配置，不保存 DataLoader 等对象
			config_to_save = {k: v for k, v in self.config.items() 
			                 if k not in ['train_loader', 'val_loader', 'test_loader']}
			torch.save({
				'round': round_num,
				'model_state_dict': self.global_model_state,
				'config': config_to_save
			}, save_path)
			print(f"Global model saved at round {round_num}: {save_path}")
	
	def save_training_history(self):
		"""保存训练历史"""
		history_path = os.path.join(self.save_dir, 'federated_training_history.json')
		import json
		with open(history_path, 'w') as f:
			json.dump(self.train_history, f, indent=4)
	
	def update_history(self, round_num, test_loss, test_acc, test_f1):
		"""更新训练历史"""
		self.train_history['round'].append(round_num)
		self.train_history['global_test_loss'].append(test_loss)
		self.train_history['global_test_acc'].append(test_acc)
		self.train_history['global_test_f1'].append(test_f1)


class FederatedTrainer:
	"""联邦学习训练器"""
	
	def __init__(self, global_model, client_configs, server_config, save_dir, fed_algorithm='fedprox'):
		"""
		初始化联邦学习训练器
		
		Args:
			global_model: 全局模型
			client_configs: 边侧配置列表
			server_config: 云侧配置
			save_dir: 保存目录
			fed_algorithm: 联邦学习算法 ('fedavg' 或 'fedprox')
		"""
		self.global_model = global_model
		self.client_configs = client_configs
		self.server_config = server_config
		self.save_dir = save_dir
		self.fed_algorithm = fed_algorithm
		
		# 设置设备
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		# 创建云侧
		self.server = FederatedServer(
			global_model=global_model,
			test_loader=server_config['test_loader'],
			device=self.device,
			config=server_config,
			save_dir=save_dir
		)
		
		# 创建边侧
		self.clients = []
		for i, config in enumerate(client_configs):
			# 创建边侧配置（不包含数据加载器）
			client_config = {k: v for k, v in config.items() if k not in ['train_loader', 'val_loader', 'test_loader']}
			
			# 根据算法选择边侧类型
			if fed_algorithm == 'fedavg':
				client = FedAvgClient(
					client_id=i,
					model=copy.deepcopy(global_model),
				 train_loader=config['train_loader'],
					val_loader=config['val_loader'],
					device=self.device,
					config=client_config,
					test_loader=config.get('test_loader')
				)
			elif fed_algorithm == 'fedprox':
				client = FedProxClient(
					client_id=i,
					model=copy.deepcopy(global_model),
				 train_loader=config['train_loader'],
					val_loader=config['val_loader'],
					device=self.device,
					config=client_config,
					test_loader=config.get('test_loader')
				)
			elif fed_algorithm == 'fedaware':
				# 导入FedAWARE相关的类和函数
				from project import FedAWAREClient
				client = FedAWAREClient(
					client_id=i,
					client_model=copy.deepcopy(global_model),
					test_loader=config.get('test_loader'),
					val_loader=config['val_loader'],
					device=self.device,
					config=client_config,
					local_test_loader=None,
				 train_loader=config['train_loader']
				)
			else:
				raise ValueError(f"Unsupported federated algorithm: {fed_algorithm}")
			
			self.clients.append(client)
	
	def train_federated(self):
		"""
		执行联邦学习训练
		
		Returns:
			training_history: 训练历史
		"""
		print(f"Starting federated learning with {len(self.clients)} clients")
		print(f"Algorithm: {self.fed_algorithm.upper()}")
		print(f"Total rounds: {self.server_config['num_rounds']}")
		print(f"Local epochs per round: {self.server_config['local_epochs']}")
		
		# 创建边侧模型保存目录
		client_save_dir = os.path.join(self.save_dir, 'client_models')
		os.makedirs(client_save_dir, exist_ok=True)
		print(f"边侧模型保存目录: {client_save_dir}")
		
		start_time = time.time()
		
		for round_num in range(self.server_config['num_rounds']):
			print(f"\n=== Federated Round {round_num + 1}/{self.server_config['num_rounds']} ===")
			# 打印本轮正则超参数（仅FedProx算法）
			if self.fed_algorithm == 'fedprox' and len(self.clients) > 0:
				print(f"Regularizers - prox_mu: {self.clients[0].prox_mu}, head_reg_lambda: {self.clients[0].head_reg_lambda}")
			
			# 边侧本地训练
			client_models = []
			client_train_losses = []
			client_train_accs = []
			
			for i, client in enumerate(self.clients):
				print(f"Training client {i + 1}/{len(self.clients)}...")
				
				# 本地训练
				local_model_state, train_loss, train_acc = client.train_local(
					self.server.global_model_state
				)
				
				# 立刻在该边侧自己的测试集上评估本地模型
				if client.test_loader is not None:
					cl_loss, cl_acc, cl_f1, _ = client.test_local(local_model_state)
					print(f"Client {i + 1} - Test (own subset, local model): Loss={cl_loss:.4f}, Acc={cl_acc:.2f}%, F1={cl_f1:.4f}")
				
				client_models.append(local_model_state)
				client_train_losses.append(train_loss)
				client_train_accs.append(train_acc)
				
				print(f"Client {i + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
				
				# 保存边侧模型（根据配置的间隔）
				if (round_num + 1) % self.server_config.get('client_save_interval', 5) == 0:
					client_save_path = os.path.join(client_save_dir, f'client_{i+1:03d}_round_{round_num+1:03d}.pth')
					torch.save({
						'round': round_num + 1,
						'client_id': i + 1,
						'model_state_dict': local_model_state,
						'train_loss': train_loss,
						'train_acc': train_acc
					}, client_save_path)
					print(f"  → 边侧 {i + 1} 模型已保存: client_{i+1:03d}_round_{round_num+1:03d}.pth")
			
			# 云侧聚合
			print("Aggregating models...")
			# 基于各边侧训练样本数进行加权（FedAvg）
			client_sample_counts = [len(c.train_loader.dataset) for c in self.clients]
			total_samples = float(sum(client_sample_counts)) if len(client_sample_counts) > 0 else 1.0
			client_weights = [n / total_samples for n in client_sample_counts]
			self.server.aggregate_models(client_models, client_weights=client_weights)
			
			# 全局评估
			print("Evaluating global model...")
			test_loss, test_acc, test_f1, conf_matrix = self.server.evaluate_global_model()
			
			print(f"Global Test - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%, F1: {test_f1:.4f}")
			
			# 更新历史
			self.server.update_history(round_num + 1, test_loss, test_acc, test_f1)
			
			# 保存全局模型
			self.server.save_global_model(round_num + 1)
			
			# 打印平均边侧性能
			avg_train_loss = np.mean(client_train_losses)
			avg_train_acc = np.mean(client_train_accs)
			print(f"Average Client - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%")
		
		# 保存最终模型
		# 只保存超参数配置，不保存 DataLoader 等对象
		config_to_save = {k: v for k, v in self.server_config.items() 
		                 if k not in ['train_loader', 'val_loader', 'test_loader']}
		final_save_path = os.path.join(self.save_dir, 'final_global_model.pth')
		torch.save({
			'model_state_dict': self.server.global_model_state,
			'config': config_to_save,
			'final_round': self.server_config['num_rounds']
		}, final_save_path)
		
		# 保存训练历史
		history_path = os.path.join(self.save_dir, f'{self.fed_algorithm}_training_history.json')
		import json
		with open(history_path, 'w') as f:
			json.dump(self.server.train_history, f, indent=4)
		
		training_time = time.time() - start_time
		print(f"\nFederated learning completed in {training_time/60:.2f} minutes")
		
		return self.server.train_history