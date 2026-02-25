"""
雷达信号数据集生成器 - 训练数据版本
等效于 MATLAB 版本 RadarSignalGenerator_Batch2.m

根据 7 架飞机的航线生成 LFM 雷达信号数据集，整合保存为单个 PKL 文件，
输出到 run/data/radar.pkl，供 prepare_data_splits.py 和 readdata_radar.py 使用。

用法:
    python training_data_generation/radar_signal_generator.py

输出格式:
    PKL 文件包含 {'X': ndarray(num_samples, 2, 500), 'Y': ndarray(num_samples,)}
    其中 Y 标签为 1~7（readdata_radar.py 会自动转换为 0~6）
"""

import numpy as np
import pickle
import os
import time


# ========================== 物理常数 ==========================
C_LIGHT = 3e8
K_BOLTZMANN = 1.38e-23
R_EARTH = 6371.0  # km


# ========================== 飞机雷达参数 ==========================
AIRCRAFT_PARAMS = {
    'P8A': {'fc': 9.6e9, 'B': 42e6, 'name': 'P-8A'},
    'P3C': {'fc': 9.3e9, 'B': 30e6, 'name': 'P-3C'},
    'E2D': {'fc': 1e9,   'B': 20e6, 'name': 'E-2D'},
}


# ========================== 航线定义 ==========================
FLIGHT_PATHS = {
    'P8A_1': {'type': 'P8A', 'individual': 1,
              'start_lat': 26.38, 'start_lon': 127.68, 'end_lat': 25.11, 'end_lon': 122.44, 'altitude': 6000},
    'P8A_2': {'type': 'P8A', 'individual': 2,
              'start_lat': 26.57, 'start_lon': 127.67, 'end_lat': 25.09, 'end_lon': 122.42, 'altitude': 6200},
    'P8A_3': {'type': 'P8A', 'individual': 3,
              'start_lat': 26.32, 'start_lon': 127.71, 'end_lat': 25.08, 'end_lon': 122.43, 'altitude': 5800},
    'P3C_1': {'type': 'P3C', 'individual': 1,
              'start_lat': 26.48, 'start_lon': 127.63, 'end_lat': 25.04, 'end_lon': 122.44, 'altitude': 3000},
    'P3C_2': {'type': 'P3C', 'individual': 2,
              'start_lat': 26.27, 'start_lon': 127.75, 'end_lat': 25.02, 'end_lon': 122.42, 'altitude': 2500},
    'E2D_1': {'type': 'E2D', 'individual': 1,
              'start_lat': 26.16, 'start_lon': 127.66, 'end_lat': 24.99, 'end_lon': 122.41, 'altitude': 8000},
    'E2D_2': {'type': 'E2D', 'individual': 2,
              'start_lat': 26.09, 'start_lon': 127.67, 'end_lat': 25.01, 'end_lon': 122.43, 'altitude': 9000},
}

# 标签编码 (1-indexed，与原 MATLAB 版一致): P-8A:1-3, P-3C:4-5, E-2D:6-7
# readdata_radar.py 在加载后会自动做 Y_adjusted = Y - 1 转为 0-indexed
LABEL_BASE = {'P-8A': 0, 'P-3C': 3, 'E-2D': 5}


# ========================== 相位噪声参数 ==========================
def get_phase_noise_params(aircraft_type, individual_idx, B):
    """根据飞机类型和个体索引返回相位噪声参数 (f_m, M_m)。"""
    params_table = {
        ('P-8A', 1): {
            'f_m_base': [1e-6, 1e-5, 1e-4, 1e-3, 5e-2, 5e-1, 5, 50, 250, 3e3, 9e4],
            'M_m_base': [0.21, 0.011, 0.081, 0.03, 0.0312, 0.09, 0.092, 0.021, 0.0098, 0.022, 0.0097],
            'factor': 1.2,
        },
        ('P-8A', 2): {
            'f_m_base': [1e-6, 1e-5, 1e-4, 1e-3, 4e-2, 6e-1, 4, 55, 260, 2.8e3, 8.5e4],
            'M_m_base': [0.25, 0.015, 0.075, 0.035, 0.028, 0.095, 0.088, 0.025, 0.011, 0.020, 0.0085],
            'factor': 1.1,
        },
        ('P-8A', 3): {
            'f_m_base': [1e-6, 1e-5, 1e-4, 1e-3, 6e-2, 4e-1, 6, 45, 240, 3.2e3, 9.5e4],
            'M_m_base': [0.18, 0.008, 0.085, 0.028, 0.035, 0.085, 0.095, 0.018, 0.0085, 0.024, 0.0105],
            'factor': 1.3,
        },
        ('P-3C', 1): {
            'f_m_base': [1e-6, 8e-5, 1.2e-4, 1.5e-3, 2e-2, 2e-1, 3, 40, 200, 2.5e3, 8e4],
            'M_m_base': [0.15, 0.02, 0.06, 0.04, 0.025, 0.07, 0.08, 0.03, 0.015, 0.018, 0.007],
            'factor': 0.9,
        },
        ('P-3C', 2): {
            'f_m_base': [1e-6, 1.2e-5, 0.8e-4, 1.8e-3, 3e-2, 1.5e-1, 3.5, 35, 220, 2.8e3, 7.5e4],
            'M_m_base': [0.18, 0.025, 0.055, 0.045, 0.03, 0.065, 0.085, 0.028, 0.017, 0.022, 0.008],
            'factor': 1.1,
        },
        ('E-2D', 1): {
            'f_m_base': [1e-6, 5e-5, 0.5e-4, 0.8e-3, 1e-2, 0.5e-1, 2, 30, 150, 1.5e3, 5e4],
            'M_m_base': [0.12, 0.015, 0.04, 0.025, 0.015, 0.05, 0.06, 0.02, 0.01, 0.012, 0.005],
            'factor': 0.8,
        },
        ('E-2D', 2): {
            'f_m_base': [1e-6, 7e-5, 0.6e-4, 1.2e-3, 1.5e-2, 0.8e-1, 2.5, 25, 170, 1.8e3, 6e4],
            'M_m_base': [0.14, 0.018, 0.045, 0.03, 0.018, 0.055, 0.065, 0.025, 0.012, 0.015, 0.006],
            'factor': 1.2,
        },
    }

    p = params_table[(aircraft_type, individual_idx)]
    f_m = np.array(p['f_m_base']) * (B / 100e6)
    M_m = np.array(p['M_m_base']) * p['factor']
    return f_m, M_m


# ========================== 核心计算函数 ==========================
def haversine_distance(lat1, lon1, lat2, lon2, alt1, alt2):
    """哈弗辛公式计算两点间直线距离 (km)。"""
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    ground_distance = R_EARTH * c
    alt_diff = abs(alt2 - alt1) / 1000.0
    return np.sqrt(ground_distance ** 2 + alt_diff ** 2)


def radar_equation(R_m, Pt, Gt_dB, Gr_dB, fc, B, T_sys, F_dB):
    """雷达方程计算 SNR (dB)。"""
    d_km = R_m / 1000.0
    f_MHz = fc / 1e6
    Lfs_dB = 32.44 + 20 * np.log10(d_km) + 20 * np.log10(f_MHz)

    Pt_dB = 10 * np.log10(Pt)
    Pr_dB = Pt_dB + Gt_dB + Gr_dB - Lfs_dB
    Pr = 10 ** (Pr_dB / 10)

    Pn = K_BOLTZMANN * T_sys * B * 10 ** (F_dB / 10)

    if Pr > 0 and Pn > 0:
        return 10 * np.log10(Pr / Pn)
    return -np.inf


def generate_waypoints(flight_path, num_segments=22):
    """为一条航线均匀生成 num_segments+1 个航点。"""
    n = num_segments + 1
    lats = np.linspace(flight_path['start_lat'], flight_path['end_lat'], n)
    lons = np.linspace(flight_path['start_lon'], flight_path['end_lon'], n)
    alts = np.full(n, flight_path['altitude'], dtype=np.float64)
    return np.column_stack([lats, lons, alts])


# ========================== 信号生成 ==========================
def generate_aircraft_signals(ac_param, individual_idx, SNR_dB, num_samples,
                              pulse_width, fs, repeat_factor=1):
    """
    生成 LFM 雷达信号样本。
    返回:
        X: ndarray (num_samples, 2, signal_length)  — I/Q 双通道
        label: int (1-indexed, 与 MATLAB 原版一致)
    """
    pulse_samples = round(fs * pulse_width)
    total_len = pulse_samples * repeat_factor
    t_pulse = np.linspace(-pulse_width / 2, pulse_width / 2, pulse_samples)

    ac_name = ac_param['name']
    f_m, M_m = get_phase_noise_params(ac_name, individual_idx, ac_param['B'])

    # 1-indexed 标签（与 MATLAB 原版保持一致）
    label = LABEL_BASE[ac_name] + individual_idx

    chirp_slope = ac_param['B'] / pulse_width
    ideal_signal = np.exp(1j * np.pi * chirp_slope * t_pulse ** 2)

    omega_t = 2 * np.pi * (f_m * 1e6)[:, None] * t_pulse[None, :]

    X = np.zeros((num_samples, 2, total_len), dtype=np.float32)

    for s in range(num_samples):
        noisy_signal = ideal_signal.copy()

        for m in range(len(f_m)):
            phase_noise = M_m[m] * np.sin(omega_t[m])
            noisy_signal = noisy_signal * np.exp(1j * phase_noise)

        signal_power = np.mean(np.abs(noisy_signal) ** 2)
        noise_power = signal_power / (10 ** (SNR_dB / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(pulse_samples) + 1j * np.random.randn(pulse_samples)
        )
        noisy_signal = noisy_signal + noise

        current_power = np.mean(np.abs(noisy_signal) ** 2)
        if current_power > 0:
            noisy_signal = noisy_signal / np.sqrt(current_power)

        I = np.real(noisy_signal).astype(np.float32)
        Q = np.imag(noisy_signal).astype(np.float32)

        if repeat_factor > 1:
            I = np.tile(I, repeat_factor)
            Q = np.tile(Q, repeat_factor)

        X[s, 0, :] = I
        X[s, 1, :] = Q

    return X, label


# ========================== 主函数 ==========================
def main():
    # 基本参数（与 MATLAB RadarSignalGenerator_Batch2 一致）
    Pt = 50
    Gt = 12
    Gr = 8
    T_sys = 290
    F = 3
    num_samples_per_point = 6000
    pulse_width = 5e-6
    fs = 100e6
    repeat_factor = 1
    num_segments = 22

    receiver_lat = 25.45
    receiver_lon = 122.07
    receiver_alt = 0.0

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'run', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'radar.pkl')

    # 生成航点
    aircraft_names = list(FLIGHT_PATHS.keys())
    waypoints = {name: generate_waypoints(FLIGHT_PATHS[name], num_segments) for name in aircraft_names}

    total_waypoints = sum(wp.shape[0] for wp in waypoints.values())
    expected_total = total_waypoints * num_samples_per_point
    pulse_samples = round(fs * pulse_width)
    signal_length = pulse_samples * repeat_factor

    print('开始生成雷达信号数据集（训练数据）...')
    print('=' * 60)
    print(f'  飞机数量: {len(aircraft_names)}')
    print(f'  总航点数: {total_waypoints}')
    print(f'  每航点样本数: {num_samples_per_point}')
    print(f'  预计总样本数: {expected_total}')
    print(f'  信号长度: {signal_length}')
    print(f'  输出路径: {output_path}')
    print('=' * 60)

    start_time = time.time()

    all_X = []
    all_Y = []
    total_samples = 0

    for ac_idx, ac_name in enumerate(aircraft_names):
        path_info = FLIGHT_PATHS[ac_name]
        ac_type = path_info['type']
        ac_param = AIRCRAFT_PARAMS[ac_type]

        print(f'\n[{ac_idx + 1}/{len(aircraft_names)}] 正在处理: {ac_name} ({ac_param["name"]})')

        ac_wp = waypoints[ac_name]
        for wp_idx in range(ac_wp.shape[0]):
            wp_lat, wp_lon, wp_alt = ac_wp[wp_idx]

            distance_km = haversine_distance(
                receiver_lat, receiver_lon, wp_lat, wp_lon, receiver_alt, wp_alt
            )
            SNR_dB = radar_equation(
                distance_km * 1000, Pt, Gt, Gr, ac_param['fc'], ac_param['B'], T_sys, F
            )

            print(f'  {ac_name} - 航点{wp_idx + 1}: 距离={distance_km:.2f}km, '
                  f'经纬度=({wp_lat:.2f}°N, {wp_lon:.2f}°E), SNR={SNR_dB:.2f}dB')

            X_batch, label = generate_aircraft_signals(
                ac_param, path_info['individual'], SNR_dB,
                num_samples_per_point, pulse_width, fs, repeat_factor
            )

            all_X.append(X_batch)
            all_Y.extend([label] * X_batch.shape[0])
            total_samples += X_batch.shape[0]

    # 合并所有数据
    print(f'\n合并数据 ({total_samples} 个样本)...')
    X = np.concatenate(all_X, axis=0)   # (total_samples, 2, signal_length)
    Y = np.array(all_Y, dtype=np.int64) # (total_samples,) labels 1-7

    # 保存为 PKL（readdata_radar.py 会优先加载 .pkl）
    print(f'保存数据集到: {output_path}')
    save_data = {'X': X, 'Y': Y}
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f, protocol=4)

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)

    print(f'\n{"=" * 60}')
    print('数据集生成完成!')
    print(f'{"=" * 60}')
    print(f'  输出路径: {output_path}')
    print(f'  文件大小: {file_size_mb:.1f} MB')
    print(f'  总样本数: {total_samples}')
    print(f'  数据shape: X={X.shape}, Y={Y.shape}')
    print(f'  标签范围: {Y.min()}~{Y.max()} (7类, readdata_radar.py 会自动转为 0~6)')
    print(f'  类别分布: {dict(zip(*np.unique(Y, return_counts=True)))}')
    print(f'  生成耗时: {elapsed / 60:.2f} 分钟')
    print(f'  平均速度: {total_samples / elapsed:.0f} 样本/秒')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
