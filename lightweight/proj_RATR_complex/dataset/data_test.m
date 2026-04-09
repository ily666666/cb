clear; clc; close all;

load Class1_data.mat

% 1. 加载数据
load('Class1_data.mat'); % 替换为你的mat文件路径

% 2. 查看数据基本信息（确认维度和数值范围）
disp(['数据维度：', num2str(size(tarHRRP_inScene_db))]);
disp(['数值范围：最小值=', num2str(min(tarHRRP_inScene_db(:))), ' dB，最大值=', num2str(max(tarHRRP_inScene_db(:))), ' dB']);

% 3. 可视化1：单样本HRRP（比如第1列，第一个样本）
figure('Name','单个HRRP样本（距离-功率）');
plot(tarHRRP_inScene_db(:,1)); % 绘制第1个样本的距离-功率曲线
xlabel('距离单元（Range Bin）');
ylabel('回波功率（dB）');
title('第1个HRRP样本的距离像');
grid on;

% 4. 可视化2：所有样本的热力图（距离×样本）
figure('Name','所有HRRP样本热力图');
imagesc(tarHRRP_inScene_db); % 热力图展示100个样本的整体分布
colorbar; % 显示颜色刻度（对应dB值）
xlabel('样本序号（Sample Index）');
ylabel('距离单元（Range Bin）');
title('HRRP样本矩阵热力图（颜色越深=功率越低）');
colormap(jet); % 用jet色标（红=高功率，蓝=低功率）

% 5. 可视化3：所有样本的平均HRRP（更清晰看目标特征）
figure('Name','100个样本的平均HRRP');
mean_HRRP = mean(tarHRRP_inScene_db, 2); % 按行求平均（所有样本的平均距离像）
plot(mean_HRRP);
xlabel('距离单元（Range Bin）');
ylabel('平均回波功率（dB）');
title('100个HRRP样本的平均距离像');
grid on;