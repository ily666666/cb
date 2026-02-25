function RadarSignalGenerator_Batch2()
    % 雷达信号数据集生成器 - 批量版本
    % 根据7架飞机的航线生成雷达信号数据集
    
    fprintf('开始生成雷达信号数据集...\n');
    
    % 基本参数设置
    params.Pt = 50;                    % 发射功率 (W)
    params.Gt = 12;                    % 发射天线增益 (dBi)
    params.Gr = 8;                     % 接收天线增益 (dBi)
    params.T_sys = 290;                % 系统温度 (K)
    params.F = 3;                      % 噪声系数 (dB)
    params.num_samples_per_point = 6000; % 每个点样本数量
    params.pulse_width = 5e-6;         % 脉冲宽度 (s)
    params.fs = 100e6;                 % 采样率 (Hz)
    params.repeat_factor = 1;          % 信号因子
    % 接收机位置 (台湾海峡附近)
    receiver_pos.lat = 25.45;          % 纬度 (°N)
    receiver_pos.lon = 122.07;         % 经度 (°E)
    receiver_pos.alt = 0;              % 高度 (m) - 海平面
    
    % 定义飞机参数
    aircraftParams = defineAircraftParameters();
    
    % 定义飞机航线
    flightPaths = defineFlightPaths();
    
    % 为每条航线生成航点
    fprintf('生成航线航点...\n');
    waypoints = generateWaypoints(flightPaths, 22); % 22段，23个点
    
    % 初始化数据集
    all_X = [];
    all_Y = [];
    all_info = [];
    
    % 生成信号
    fprintf('开始生成雷达信号...\n');
    total_samples = 0;
    
    % 遍历每架飞机
    aircraft_names = fieldnames(flightPaths);
    for ac_idx = 1:length(aircraft_names)
        ac_name = aircraft_names{ac_idx};
        ac_type = flightPaths.(ac_name).type;
        
        fprintf('生成%s信号...\n', ac_name);
        
        % 获取该飞机的航点
        ac_waypoints = waypoints.(ac_name);
        
        % 遍历每个航点
        for wp_idx = 1:size(ac_waypoints, 1)
            wp = ac_waypoints(wp_idx, :);
            
            % 计算距离
            distance_km = calculateDistance(receiver_pos.lat, receiver_pos.lon, ...
                                          wp(1), wp(2), receiver_pos.alt, wp(3));
            
            % 计算信噪比
            ac_param = aircraftParams.(ac_type);
            [~, ~, ~, SNR_dB] = radarEquation(distance_km*1000, params.Pt, params.Gt, ...
                                            params.Gr, ac_param.fc, ac_param.B, ...
                                            params.T_sys, params.F);
            
        % 修改输出，加入经纬度
        fprintf('  %s - 航点%d: 距离=%.2fkm, 经纬度=(%.2f°N, %.2f°E), SNR=%.2fdB\n', ...
               ac_name, wp_idx, distance_km, wp(1), wp(2), SNR_dB);
            
            % 生成信号样本
            individual_idx = flightPaths.(ac_name).individual;
            [X_batch, Y_batch, info_batch] = generateAircraftSignals(...
                ac_param, individual_idx, SNR_dB, params.num_samples_per_point, ...
                params.pulse_width, params.fs, distance_km, params.repeat_factor);
            
            % 合并数据
            if isempty(all_X)
                all_X = X_batch;
                all_Y = Y_batch;
                all_info = info_batch;
            else
                all_X = [all_X; X_batch];
                all_Y = [all_Y; Y_batch];
                all_info = [all_info; info_batch];
            end
            
            total_samples = total_samples + size(X_batch, 1);
        end
    end
    
    % 保存数据集
    fprintf('保存数据集...\n');
    saveDataset(all_X, all_Y, all_info, params.fs, params.pulse_width, total_samples, params.repeat_factor);
    
    fprintf('数据集生成完成! 总样本数: %d\n', total_samples);
end

function aircraftParams = defineAircraftParameters()
    % 定义飞机雷达参数
    aircraftParams = struct();
    aircraftParams.P8A = struct('fc', 9.6e9, 'B', 42e6, 'name', 'P-8A');
    aircraftParams.P3C = struct('fc', 9.3e9, 'B', 30e6, 'name', 'P-3C');
    aircraftParams.E2D = struct('fc', 1e9, 'B', 20e6, 'name', 'E-2D');
end

function flightPaths = defineFlightPaths()
    % 定义7架飞机的航线
    flightPaths = struct();
    
    % P-8A 飞机
    flightPaths.P8A_1 = struct('type', 'P8A', 'individual', 1, ...
        'start_lat', 26.38, 'start_lon', 127.68, 'end_lat', 25.11, 'end_lon', 122.44, 'altitude', 6000);
    flightPaths.P8A_2 = struct('type', 'P8A', 'individual', 2, ...
        'start_lat', 26.57, 'start_lon', 127.67, 'end_lat', 25.09, 'end_lon', 122.42, 'altitude', 6200);
    flightPaths.P8A_3 = struct('type', 'P8A', 'individual', 3, ...
        'start_lat', 26.32, 'start_lon', 127.71, 'end_lat', 25.08, 'end_lon', 122.43, 'altitude', 5800);
    
    % P-3C 飞机
    flightPaths.P3C_1 = struct('type', 'P3C', 'individual', 1, ...
        'start_lat', 26.48, 'start_lon', 127.63, 'end_lat', 25.04, 'end_lon', 122.44, 'altitude', 3000);
    flightPaths.P3C_2 = struct('type', 'P3C', 'individual', 2, ...
        'start_lat', 26.27, 'start_lon', 127.75, 'end_lat', 25.02, 'end_lon', 122.42, 'altitude', 2500);
    
    % E-2D 飞机
    flightPaths.E2D_1 = struct('type', 'E2D', 'individual', 1, ...
        'start_lat', 26.16, 'start_lon', 127.66, 'end_lat', 24.99, 'end_lon', 122.41, 'altitude', 8000);
    flightPaths.E2D_2 = struct('type', 'E2D', 'individual', 2, ...
        'start_lat', 26.09, 'start_lon', 127.67, 'end_lat', 25.01, 'end_lon', 122.43, 'altitude', 9000);
end

function waypoints = generateWaypoints(flightPaths, num_segments)
    % 为每条航线生成航点
    % num_segments: 分段数量，航点数量为 num_segments + 1
    
    waypoints = struct();
    aircraft_names = fieldnames(flightPaths);
    
    for i = 1:length(aircraft_names)
        ac_name = aircraft_names{i};
        path = flightPaths.(ac_name);
        
        % 计算纬度、经度、高度的增量
        lat_step = (path.end_lat - path.start_lat) / num_segments;
        lon_step = (path.end_lon - path.start_lon) / num_segments;
        alt_step = 0; % 高度保持不变
        
        % 生成航点
        ac_waypoints = zeros(num_segments + 1, 3);
        for j = 0:num_segments
            ac_waypoints(j+1, 1) = path.start_lat + j * lat_step;  % 纬度
            ac_waypoints(j+1, 2) = path.start_lon + j * lon_step;  % 经度
            ac_waypoints(j+1, 3) = path.altitude + j * alt_step;   % 高度
        end
        
        waypoints.(ac_name) = ac_waypoints;
    end
end

function distance = calculateDistance(lat1, lon1, lat2, lon2, alt1, alt2)
    % 使用哈弗辛公式计算两点间距离（考虑地球曲率）
    % 输入：纬度、经度（度），高度（米）
    % 输出：距离（千米）
    
    R = 6371; % 地球半径（千米）
    
    % 转换为弧度
    lat1_rad = deg2rad(lat1);
    lon1_rad = deg2rad(lon1);
    lat2_rad = deg2rad(lat2);
    lon2_rad = deg2rad(lon2);
    
    % 哈弗辛公式
    dlat = lat2_rad - lat1_rad;
    dlon = lon2_rad - lon1_rad;
    
    a = sin(dlat/2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)^2;
    c = 2 * atan2(sqrt(a), sqrt(1-a));
    
    % 地面距离
    ground_distance = R * c;
    
    % 高度差（转换为千米）
    alt_diff = abs(alt2 - alt1) / 1000;
    
    % 直线距离（勾股定理）
    distance = sqrt(ground_distance^2 + alt_diff^2);
end

function [Pr, Lfs_dB, Pn, SNR_dB] = radarEquation(R, Pt, Gt_dB, Gr_dB, fc, B, T, F)
    % 雷达方程计算函数
    % 输入：距离R(m), 功率Pt(W), 增益Gt/Gr(dB), 频率fc(Hz), 带宽B(Hz), 温度T(K), 噪声系数F(dB)
    
    % 常数定义
    c = 3e8; % 光速 (m/s)
    k = 1.38e-23; % 玻尔兹曼常数
    
    % 转换为线性值
    Gt_lin = 10^(Gt_dB/10);
    Gr_lin = 10^(Gr_dB/10);
    
    % 波长 (m)
    lambda = c / fc;
    
    % 自由空间损耗 (dB)
    d_km = R / 1000; % 距离转换为km
    f_MHz = fc / 1e6; % 频率转换为MHz
    Lfs_dB = 32.44 + 20*log10(d_km) + 20*log10(f_MHz);
    
    % 接收功率计算 (dB方法)
    Pt_dB = 10*log10(Pt);
    Pr_dB = Pt_dB + Gt_dB + Gr_dB - Lfs_dB;
    Pr = 10^(Pr_dB/10);
    
    % 噪声功率
    Pn = k * T * B * 10^(F/10);
    
    % 信噪比
    if Pr > 0 && Pn > 0
        SNR = Pr / Pn;
        SNR_dB = 10*log10(SNR);
    else
        SNR_dB = -inf;
    end
end

function [X, Y, info] = generateAircraftSignals(ac, individual_idx, SNR_dB, num_samples, T, fs, distance_km, repeat_factor)
    % 生成单个飞机类型信号的函数
    
    % 信号参数
    pulse_samples = round(fs * T);
    total_samples_after_repeat = pulse_samples * repeat_factor;
    t_pulse = linspace(-T/2, T/2, pulse_samples);
    % 根据飞机类型和个体索引获取对应的相位噪声参数
    [f_m, M_m] = getDetailedPhaseNoiseParams(ac.name, individual_idx, ac.B);
    
    % 初始化输出
    X = zeros(num_samples, total_samples_after_repeat, 2);
    Y = zeros(num_samples, 1);
    info = cell(num_samples, 1);
    
    % 标签编码: P-8A:1-3, P-3C:4-5, E-2D:6-7
    if strcmp(ac.name, 'P-8A')
        label_base = 0;
    elseif strcmp(ac.name, 'P-3C')
        label_base = 3;
    else % E-2D
        label_base = 5;
    end
    
    label = label_base + individual_idx;
    
    % 生成理想LFM复信号 (基带)
    chirp_slope = ac.B / T;
    ideal_signal = exp(1j * pi * chirp_slope * t_pulse.^2);
    
    for sample_idx = 1:num_samples
        % 生成带相位噪声的信号
        noisy_signal = ideal_signal;
        
        % 添加相位噪声 - 使用详细的相位噪声模型
        for m = 1:length(f_m)
            % 为每个样本添加随机相位偏移，增加多样性
%             random_phase = 2 * pi * rand();
              random_phase = 0;
            phase_noise = M_m(m) * sin(2*pi*f_m(m)*1e6 * t_pulse + random_phase);
            noisy_signal = noisy_signal .* exp(1j * phase_noise);
        end
        
        % 添加高斯白噪声
        signal_power = mean(abs(noisy_signal).^2);
        noise_power = signal_power / (10^(SNR_dB/10));
        
        % 生成复高斯噪声
        real_noise = sqrt(noise_power/2) * randn(1, pulse_samples);
        imag_noise = sqrt(noise_power/2) * randn(1, pulse_samples);
        noisy_signal = noisy_signal + (real_noise + 1j*imag_noise);
        
        current_power = mean(abs(noisy_signal).^2);
        if current_power > 0
            normalization_factor = 1 / sqrt(current_power);%对信号进行功率归一化
            noisy_signal = noisy_signal * normalization_factor;
        end
        % 分离I/Q双路
        I = real(noisy_signal);
        Q = imag(noisy_signal);
        
        I_repeated = repmat(I, 1, repeat_factor);
        Q_repeated = repmat(Q, 1, repeat_factor);
        
        % 存储数据 - 使用重复后的信号
        X(sample_idx, :, 1) = I_repeated;
        X(sample_idx, :, 2) = Q_repeated;
        Y(sample_idx) = label;
        
        % 存储信息
        info{sample_idx} = struct(...
            'aircraft_type', ac.name, ...
            'individual', individual_idx, ...
            'fc', ac.fc, ...
            'B', ac.B, ...
            'SNR_dB', SNR_dB, ...
            'distance_km', distance_km, ...
            'pulse_width', T, ...
            'fs', fs,...
            'repeated_factor', repeat_factor);
    end
end

function [f_m, M_m] = getDetailedPhaseNoiseParams(aircraft_type, individual_idx, B)
    % 基于飞机类型和个体索引返回详细的相位噪声参数
    
    switch aircraft_type
        case 'P-8A'
            % P-8A战机 - 3架不同的相位噪声特征
            switch individual_idx
                case 1
                    % 第1架P-8A
                    f_m_base = [1e-6, 1e-5, 1e-4, 1e-3, 5e-2, 5e-1, 5, 50, 250, 3e3, 9e4];
                    M_m_base = [0.21, 0.011, 0.081, 0.03, 0.0312, 0.09, 0.092, 0.021, 0.0098, 0.022, 0.0097];
                    factor = 1.2;
                case 2
                    % 第2架P-8A（轻微差异）
                    f_m_base = [1e-6, 1e-5, 1e-4, 1e-3, 4e-2, 6e-1, 4, 55, 260, 2.8e3, 8.5e4];
                    M_m_base = [0.25, 0.015, 0.075, 0.035, 0.028, 0.095, 0.088, 0.025, 0.011, 0.020, 0.0085];
                    factor = 1.1;
                case 3
                    % 第3架P-8A（轻微差异）
                    f_m_base = [1e-6, 1e-5, 1e-4, 1e-3, 6e-2, 4e-1, 6, 45, 240, 3.2e3, 9.5e4];
                    M_m_base = [0.18, 0.008, 0.085, 0.028, 0.035, 0.085, 0.095, 0.018, 0.0085, 0.024, 0.0105];
                    factor = 1.3;
            end
            
        case 'P-3C'
            % P-3C战机 - 2架不同的相位噪声特征
            switch individual_idx
                case 1
                    % 第1架P-3C
                    f_m_base = [1e-6, 8e-5, 1.2e-4, 1.5e-3, 2e-2, 2e-1, 3, 40, 200, 2.5e3, 8e4];
                    M_m_base = [0.15, 0.02, 0.06, 0.04, 0.025, 0.07, 0.08, 0.03, 0.015, 0.018, 0.007];
                    factor = 0.9;
                case 2
                    % 第2架P-3C（硬件差异）
                    f_m_base = [1e-6, 1.2e-5, 0.8e-4, 1.8e-3, 3e-2, 1.5e-1, 3.5, 35, 220, 2.8e3, 7.5e4];
                    M_m_base = [0.18, 0.025, 0.055, 0.045, 0.03, 0.065, 0.085, 0.028, 0.017, 0.022, 0.008];
                    factor = 1.1;
            end
            
        case 'E-2D'
            % E-2D预警机 - 2架不同的相位噪声特征
            switch individual_idx
                case 1
                    % 第1架E-2D
                    f_m_base = [1e-6, 5e-5, 0.5e-4, 0.8e-3, 1e-2, 0.5e-1, 2, 30, 150, 1.5e3, 5e4];
                    M_m_base = [0.12, 0.015, 0.04, 0.025, 0.015, 0.05, 0.06, 0.02, 0.01, 0.012, 0.005];
                    factor = 0.8;
                case 2
                    % 第2架E-2D（硬件差异）
                    f_m_base = [1e-6, 7e-5, 0.6e-4, 1.2e-3, 1.5e-2, 0.8e-1, 2.5, 25, 170, 1.8e3, 6e4];
                    M_m_base = [0.14, 0.018, 0.045, 0.03, 0.018, 0.055, 0.065, 0.025, 0.012, 0.015, 0.006];
                    factor = 1.2;
            end
    end
    
    % 根据带宽调整频率分量
    f_m = f_m_base * (B/100e6);
    
    % 应用调整因子
    M_m = M_m_base * factor;
end

function saveDataset(X, Y, info, fs, pulse_width, total_samples, repeat_factor)
    % 保存数据集到文件
    
    % 指定保存文件夹
    saveFolder = 'E:\BaiduNet_Download\dataGen\';
    
    % 确保文件夹存在
    if ~exist(saveFolder, 'dir')
        try
            mkdir(saveFolder);
            fprintf('创建文件夹: %s\n', saveFolder);
        catch
            fprintf('无法创建文件夹: %s，使用当前目录\n', saveFolder);
            saveFolder = pwd;
        end
    end
    
    % 生成文件名
    filename = sprintf('RadarDataset_%s_%dsamples_Repeat%d.mat', ...
                      datestr(now, 'yyyymmdd_HHMMSS'), total_samples, repeat_factor);
    
    fullPath = fullfile(saveFolder, filename);
    
    % 保存数据
    save(fullPath, 'X', 'Y', 'info', 'fs', 'pulse_width','repeat_factor', '-v7.3');
    
    fprintf('数据集已保存到: %s\n', fullPath);
end
