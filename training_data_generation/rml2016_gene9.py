import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from scipy.signal import hilbert, lfilter

# 物理常数（用于SNR计算）
c = 3e8  # 光速 (m/s)
k = 1.38e-23  # 玻尔兹曼常数
T = 290  # 绝对温度 (K)
R_earth = 6371e3  # 地球半径 (m)

# -------------------------- 关键修改1：定义7个个体参数（航线+高度+平台映射） --------------------------
aircraft_parameters = {
    # E-2D预警机
    'E-2D_1': {
        'type': 'E-2D',
        'frequency': 225e6,
        'transmit_power': 100,
        'tx_gain': 12,
        'rx_gain': 2,
        'height': 8000,          # 固定高度8km（用户指定）
        'start_lat': 26.16,      # 起点纬度（度）
        'start_lon': 127.66,     # 起点经度（度）
        'end_lat': 24.09,        # 终点纬度（度）
        'end_lon': 122.41,       # 终点经度（度）
        'noise_figure': 4,
        'bandwidth': 16e3,
        'symbol_rate': 1200
    },
    'E-2D_2': {
        'type': 'E-2D',
        'frequency': 225e6,
        'transmit_power': 100,
        'tx_gain': 12,
        'rx_gain': 1,
        'height': 9000,          # 固定高度9km（用户指定）
        'start_lat': 26.09,
        'start_lon': 127.67,
        'end_lat': 25.01,
        'end_lon': 122.43,
        'noise_figure': 4,
        'bandwidth': 16e3,
        'symbol_rate': 1200
    },
    'P-3C_1': {
        'type': 'P-3C',
        'frequency': 270e6,
        'transmit_power': 30,
        'tx_gain': 8,
        'rx_gain': 2,
        'height': 3000,          # 固定高度3km（用户指定）
        'start_lat': 26.48,
        'start_lon': 127.63,
        'end_lat': 25.14,
        'end_lon': 122.44,
        'noise_figure': 4,
        'bandwidth': 16e3,
        'symbol_rate': 1200
    },
    'P-3C_2': {
        'type': 'P-3C',
        'frequency': 270e6,
        'transmit_power': 30,
        'tx_gain': 8,
        'rx_gain': 2,
        'height': 2500,          # 固定高度2.5km（用户指定）
        'start_lat': 26.27,
        'start_lon': 127.75,
        'end_lat': 25.02,
        'end_lon': 122.42,
        'noise_figure': 4,
        'bandwidth': 16e3,
        'symbol_rate': 1200
    },
    'P-8A_1': {
        'type': 'P-8A',
        'frequency': 300e6,
        'transmit_power': 50,
        'tx_gain': 10,
        'rx_gain': 1,
        'height': 6000,          # 固定高度6km（用户指定）
        'start_lat': 26.38,
        'start_lon': 127.68,
        'end_lat': 25.11,
        'end_lon': 122.44,
        'noise_figure': 5,
        'bandwidth': 16e3,
        'symbol_rate': 1200
    },
    'P-8A_2': {
        'type': 'P-8A',
        'frequency': 300e6,
        'transmit_power': 50,
        'tx_gain': 10,
        'rx_gain': 2,
        'height': 6200,          # 固定高度6.2km（用户指定）
        'start_lat': 26.57,
        'start_lon': 127.67,
        'end_lat': 25.09,
        'end_lon': 122.42,
        'noise_figure': 5,
        'bandwidth': 16e3,
        'symbol_rate': 1200
    },
    'P-8A_3': {
        'type': 'P-8A',
        'frequency': 300e6,
        'transmit_power': 50,
        'tx_gain': 10,
        'rx_gain': 2,
        'height': 5800,          # 固定高度5.8km（用户指定）
        'start_lat': 26.32,
        'start_lon': 127.71,
        'end_lat': 25.08,
        'end_lon': 122.43,
        'noise_figure': 5,
        'bandwidth': 16e3,
        'symbol_rate': 1200
    }
}
# -------------------------- 关键修改2：定义双接收机参数 --------------------------
receivers = {
    'receiver_1': {'lat': 25.45, 'lon': 122.07, 'height': 2},  # 2m高度
    'receiver_2': {'lat': 24.69, 'lon': 122.43, 'height': 2}
}


# 调制专属参数（保持不变）
mod_exclusive_params = {
    '8PSK': {'symbol_rate': 2500, 'phase_offset': np.pi/4, 'filter_sigma': 0.3, 'noise_type': 'phase', 'phase_noise_ratio': 0.5},
    'BPSK': {'symbol_rate': 800, 'phase_offset': 0, 'filter_sigma': 1.0, 'noise_type': 'phase', 'phase_noise_ratio': 0.8},
    'GMSK': {'bt': 0.3, 'samples_per_bit': 8, 'symbol_rate': 500, 'filter_sigma': 0.2, 'noise_type': 'frequency'},
    '16QAM': {'constellation_scale': 5.0, 'filter_sigma': 0.5, 'noise_type': 'amplitude', 'amp_noise_ratio': 0.3},
    '64QAM': {'constellation_scale': 0.4, 'filter_sigma': 0.4, 'noise_type': 'amplitude', 'amp_noise_ratio': 0.2},
    'QPSK': {'symbol_rate': 1500, 'phase_offset': 0, 'filter_sigma': 0.7, 'noise_type': 'phase', 'phase_noise_ratio': 0.6},
    'CPFSK': {'h': 0.8, 'samples_per_bit': 8, 'symbol_rate': 600, 'filter_sigma': 0.3, 'noise_type': 'frequency'},
    'PAM4': {'symbol_rate': 1200, 'filter_sigma': 0.6, 'noise_type': 'amplitude', 'amp_noise_ratio': 0.4},
    'AM-DSB': {'carrier_freq': 0.1, 'mod_index': 0.8, 'filter_sigma': 0.7, 'noise_type': 'amplitude', 'amp_noise_ratio': 0.5},
    'AM-SSB': {'carrier_freq': 0.15, 'mod_index': 0.7, 'filter_sigma': 0.7, 'noise_type': 'amplitude', 'amp_noise_ratio': 0.5},
    'WBFM': {'carrier_freq': 0.08, 'mod_index': 8, 'filter_sigma': 0.4, 'noise_type': 'frequency'}
}

# -------------------------- 关键修改3：新增工具函数 --------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine公式计算两点水平距离（米），输入经纬度为度"""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R_earth * c  # 地球半径×圆心角=距离

def generate_waypoints(start_lat, start_lon, end_lat, end_lon, num_points=600):
    """生成起点到终点的均匀航点（经纬度）"""
    latitudes = np.linspace(start_lat, end_lat, num_points)
    longitudes = np.linspace(start_lon, end_lon, num_points)
    return latitudes, longitudes

# -------------------------- 关键修改4：适配个体与接收机的SignalSimulator --------------------------
class SignalSimulator:
    def __init__(self, individual_id, receiver_height=2):
        # 从个体参数中获取配置
        self.individual_id = individual_id
        indiv_params = aircraft_parameters[individual_id]
        #self.platform_type = indiv_params['platform_type']
        self.Ht = indiv_params['height']  # 个体固定高度（米）
        self.Hr = receiver_height  # 接收机高度（外部传入）
        
        # 加载对应平台的参数（用于SNR计算）
        self.f = indiv_params['frequency']
        self.type = indiv_params['type']
        self.lambda_ = c / self.f  # 波长
        self.Pt = indiv_params['transmit_power']
        self.Gt = indiv_params['tx_gain']
        self.Gr = indiv_params['rx_gain']
        self.noise_figure = indiv_params['noise_figure']
        self.B = indiv_params['bandwidth']
        
        # 计算噪声功率 (dBW)
        noise_power_watts = k * T * self.B
        self.noise_power_dBW = 10 * np.log10(noise_power_watts) + self.noise_figure

    # 计算自由空间损耗 (dB)
    def free_space_loss(self, distance):
        return 20 * np.log10(4 * np.pi * distance / self.lambda_)

    # 计算视线距离 (m)
    def line_of_sight_distance(self):
        return np.sqrt(2 * R_earth * self.Ht) + np.sqrt(2 * R_earth * self.Hr)

    # 计算额外损耗 (dB)
    def additional_loss1(self, d_horizontal):
        d_los = self.line_of_sight_distance()
        if d_horizontal <= d_los:
            return 0
        excess_distance = d_horizontal - d_los
        # 按平台类型区分额外损耗
        #return 0.01 * (excess_distance / 1000) if self.platform_type == 'awacs' else 0.05 * (excess_distance / 1000)
        return 0.1 * (excess_distance / 1000) if self.platform_type == 'awacs' else 0.08 * (excess_distance / 1000)
    
    def additional_loss(self, d_horizontal):
        d_los = self.line_of_sight_distance()
        if d_horizontal <= d_los:
            return 0  # 视线传播
        
        excess_distance = d_horizontal - d_los
        # 不同机型的额外损耗模型
        if self.type == 'E-2D':
            return 0.1 * (excess_distance / 1000)#0.015 * (excess_distance / 1000)
        elif self.type == 'P-8A':
            return 0.08 * (excess_distance / 1000)#0.03 * (excess_distance / 1000)
        elif self.type == 'P-3C':
            return 0.07 * (excess_distance / 1000)#0.025 * (excess_distance / 1000)
        else:
            return 0.05 * (excess_distance / 1000)#0.02 * (excess_distance / 1000)

    # 计算总损耗 (dB)
    def total_loss(self, d_horizontal):
        # 直线距离（考虑高度差）
        d_line = np.sqrt(d_horizontal**2 + (self.Ht - self.Hr)**2)
        fs_loss = self.free_space_loss(d_line)
        extra_loss = self.additional_loss(d_horizontal)
        return fs_loss + extra_loss

    # 计算接收功率 (dBW)
    def received_power(self, d_horizontal):
        loss = self.total_loss(d_horizontal)
        Pt_dBW = 10 * np.log10(self.Pt)
        return Pt_dBW + self.Gt + self.Gr - loss

    # 关键修改：由水平距离直接计算实际SNR（无需反向查找）
    def calculate_snr_from_distance(self, d_horizontal):
        pr = self.received_power(d_horizontal)
        return pr - self.noise_power_dBW

    # 信号生成逻辑（保持不变）
    def generate_signal(self, modulation, snr_db, num_samples=600):
        mod_params = mod_exclusive_params[modulation]
        signal = None

        if modulation == '8PSK':
            symbol_rate = mod_params['symbol_rate']
            phase_offset = mod_params['phase_offset']
            samples_per_symbol = int(num_samples / (num_samples // (symbol_rate // 10)))
            num_symbols = num_samples // samples_per_symbol
            symbols = np.random.randint(0, 8, num_symbols)
            phases = (2 * np.pi * symbols / 8) + phase_offset
            phase_jitter = np.random.normal(0, 0.05, num_symbols)
            phases += phase_jitter
            signal = np.exp(1j * phases)
            signal = np.repeat(signal, samples_per_symbol)[:num_samples]

        elif modulation == 'BPSK':
            symbol_rate = mod_params['symbol_rate']
            samples_per_symbol = int(num_samples / (num_samples // (symbol_rate // 10)))
            num_symbols = num_samples // samples_per_symbol
            symbols = np.random.randint(0, 2, num_symbols)
            phases = symbols * np.pi
            signal = np.exp(1j * phases)
            signal = np.repeat(signal, samples_per_symbol)[:num_samples]
        
        elif modulation == 'GMSK':
            bt = mod_params['bt']
            samples_per_bit = mod_params['samples_per_bit']
            num_bits = num_samples // samples_per_bit + 2
            bits = np.random.randint(0, 2, num_bits)
            data = 2 * bits - 1
            t = np.linspace(-3, 3, 17)
            gaussian = np.exp(-(np.pi**2 * bt**2 * t**2) / np.log(2))
            gaussian /= np.sum(gaussian)
            shaped_data = np.zeros(num_bits * samples_per_bit)
            shaped_data[::samples_per_bit] = data
            filtered_data = np.convolve(shaped_data, gaussian, mode='same')
            phase = np.cumsum(filtered_data) * (np.pi / 2)
            signal = np.exp(1j * phase)[:num_samples]
        
        elif modulation == '16QAM':
            scale = mod_params['constellation_scale']
            samples_per_symbol = 4
            num_symbols = num_samples // samples_per_symbol
            symbols = np.random.randint(0, 16, num_symbols)
            i = 2 * ((symbols // 4) % 4) - 3
            q = 2 * (symbols % 4) - 3
            signal = (i + 1j * q) * scale
            signal = np.repeat(signal, samples_per_symbol)[:num_samples]

        elif modulation == '64QAM':
            scale = mod_params['constellation_scale']
            samples_per_symbol = 4
            num_symbols = num_samples // samples_per_symbol
            symbols = np.random.randint(0, 64, num_symbols)
            i = 2 * ((symbols // 8) % 8) - 7
            q = 2 * (symbols % 8) - 7
            signal = (i + 1j * q) * scale
            signal = np.repeat(signal, samples_per_symbol)[:num_samples]

        elif modulation == 'QPSK':
            symbol_rate = mod_params['symbol_rate']
            samples_per_symbol = int(num_samples / (num_samples // (symbol_rate // 10)))
            num_symbols = num_samples // samples_per_symbol
            symbols = np.random.randint(0, 4, num_symbols)
            phases = (2 * np.pi * symbols / 4) + mod_params['phase_offset']
            signal = np.exp(1j * phases)
            signal = np.repeat(signal, samples_per_symbol)[:num_samples]

        elif modulation == 'CPFSK':
            h = mod_params['h']
            symbol_rate = mod_params['symbol_rate']
            samples_per_bit = mod_params['samples_per_bit']
            num_bits = num_samples // samples_per_bit
            bits = np.random.randint(0, 2, num_bits)
            freq = np.repeat(2 * bits - 1, samples_per_bit)
            phase = np.cumsum(freq) * (np.pi * h / samples_per_bit)
            signal = np.exp(1j * phase)[:num_samples]

        elif modulation == 'PAM4':
            symbols = np.random.randint(0, 4, num_samples)
            signal = (2 * symbols - 3) * mod_params.get('constellation_scale', 1.0)

        elif modulation == 'AM-DSB':
            carrier_freq = mod_params['carrier_freq']
            mod_index = mod_params['mod_index']
            t = np.arange(num_samples)
            carrier = np.cos(2 * np.pi * carrier_freq * t)
            message = mod_index * np.random.randn(num_samples)
            signal = (1 + message) * carrier

        elif modulation == 'AM-SSB':
            carrier_freq = mod_params['carrier_freq']
            mod_index = mod_params['mod_index']
            t = np.arange(num_samples)
            carrier_cos = np.cos(2 * np.pi * carrier_freq * t)
            carrier_sin = np.sin(2 * np.pi * carrier_freq * t)
            message = mod_index * np.random.randn(num_samples)
            message_hilbert = np.imag(hilbert(message))
            signal = message * carrier_cos - message_hilbert * carrier_sin

        elif modulation == 'WBFM':
            carrier_freq = mod_params['carrier_freq']
            mod_index = mod_params['mod_index']
            t = np.arange(num_samples)
            message = np.random.randn(num_samples)
            phase = 2 * np.pi * mod_index * np.cumsum(message) / num_samples
            signal = np.cos(2 * np.pi * carrier_freq * t + phase)

        # 统一转为复数格式
        if not np.iscomplexobj(signal):
            signal = signal + 1j * 0

        # 差异化滤波
        filter_sigma = mod_params['filter_sigma']
        if filter_sigma > 0:
            b = np.exp(-(np.arange(-16, 17) ** 2) / (2 * filter_sigma ** 2))
            b /= np.sum(b)
            signal_real = lfilter(b, 1, np.real(signal))
            signal_imag = lfilter(b, 1, np.imag(signal))
            signal = signal_real + 1j * signal_imag

        # 噪声添加
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

        # 确保信号长度一致
        if len(signal) < num_samples:
            pad_length = num_samples - len(signal)
            signal = np.pad(signal, (0, pad_length), mode='edge')            
        elif len(signal) > num_samples:
            signal = signal[:num_samples]            
        if len(noise) < num_samples:
            pad_length = num_samples - len(noise)
            noise = np.pad(noise, (0, pad_length), mode='edge')
        elif len(noise) > num_samples:
            noise = noise[:num_samples]

        return signal + noise

# -------------------------- 关键修改5：主程序（个体+航点+双接收机逻辑） --------------------------
def main():
    # 1. 核心参数配置（不变）
    # modulations = [
    #     '8PSK', 'BPSK', 'GMSK', '16QAM', '64QAM', 
    #     'QPSK', 'AM-DSB', 'AM-SSB', 'CPFSK', 'PAM4', 'WBFM'
    # ]
    modulations = [
        '8PSK', 'BPSK', 'GMSK', '16QAM', '64QAM', 
        'QPSK'
    ]
    num_waypoints = 25000
    samples_per_signal = 600
    snr_round_decimals = 1
    individual_ids = list(aircraft_parameters.keys())

    # 2. 初始化数据集字典（关键修改：键改为 (调制类型, SNR)）
    data_dict = {}

    # 3. 预验证信号长度（不变）
    temp_sim = SignalSimulator(individual_ids[0], receivers['receiver_1']['height'])
    temp_signal = temp_sim.generate_signal(modulations[0], snr_db=0, num_samples=samples_per_signal)
    signal_length = len(temp_signal)
    print(f"单个信号样本长度：{signal_length} 采样点")

    # 4. 遍历逻辑（不变），但样本存储按 (调制, SNR) 分组（关键修改）
    for individual_id in individual_ids:
        print(f"\n===== 正在处理个体：{individual_id} =====")
        indiv_params = aircraft_parameters[individual_id]
        waypoint_lats, waypoint_lons = generate_waypoints(
            start_lat=indiv_params['start_lat'],
            start_lon=indiv_params['start_lon'],
            end_lat=indiv_params['end_lat'],
            end_lon=indiv_params['end_lon'],
            num_points=num_waypoints
        )

        for receiver_name, receiver in receivers.items():
            print(f"  处理接收机：{receiver_name}")
            simulator = SignalSimulator(individual_id, receiver_height=receiver['height'])

            for wp_idx in range(num_waypoints):
                wp_lat = waypoint_lats[wp_idx]
                wp_lon = waypoint_lons[wp_idx]

                # 计算距离和SNR（不变）
                horizontal_dist = haversine_distance(
                    receiver['lat'], receiver['lon'], wp_lat, wp_lon
                )
                snr_db = simulator.calculate_snr_from_distance(horizontal_dist)
                snr_rounded = np.round(snr_db, snr_round_decimals)

                # 遍历调制类型（不变）
                for modulation in modulations:
                    complex_signal = simulator.generate_signal(
                        modulation=modulation, snr_db=snr_db, num_samples=samples_per_signal
                    )
                    real_part = np.real(complex_signal).astype(np.float32)
                    imag_part = np.imag(complex_signal).astype(np.float32)

                    # 关键修改：键改为 (调制类型, SNR)
                    key = (modulation, snr_rounded)
                    print(key)
                    if key not in data_dict:
                       data_dict[key] = []
                    data_dict[key].append([real_part, imag_part])
                    # if len(data_dict[key]) == 0:
                    #     data_dict[key] = platform_samples
                    # else:
                    #     data_dict[key] = np.concatenate([data_dict[key], platform_samples], axis=0)

                if (wp_idx + 1) % 100 == 0:
                    print(f"    已处理 {wp_idx + 1}/{num_waypoints} 个航点")

    #转换为numpy数组（不变）
    print(f"  转换 {individual_id} 的样本为numpy数组...")
    for key in list(data_dict.keys()):
        if key[0] in modulations:  # 按调制类型过滤
            samples_array = np.array(data_dict[key], dtype=np.float32)
            data_dict[key] = samples_array
            print(f"    键 {key}：{samples_array.shape[0]} 个样本")
    
    # 5. 数据格式验证
    assert isinstance(data_dict, dict), f"数据集类型错误，应为dict，实际为{type(data_dict)}"
    first_key = next(iter(data_dict.keys())) if data_dict else None
    if first_key:
        assert len(first_key) == 2, f"字典键应为(个体ID, 信噪比)元组，实际为{first_key}"
        assert data_dict[first_key].shape[1:] == (2, signal_length), \
            f"样本数组形状错误，应为(样本数, 2, {signal_length})，实际为{data_dict[first_key].shape}"

    # 保存数据集（不变）
    # 输出到项目 run/data/ 目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'run', 'data')
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, 'rml2016.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f, protocol=4)

    # 7. 结果统计
    total_samples = sum([arr.shape[0] for arr in data_dict.values()])
    total_keys = len(data_dict.keys())
    print(f"\n===== 数据生成完成 =====")
    print(f"存储文件：{filename}")
    print(f"数据集键数量（个体×信噪比组合）：{total_keys}")
    print(f"总样本数：{total_samples}（7个体 × 2接收机 × 600航点 × 6调制 = {7*2*600*6}）")
    print(f"样本格式：(样本数, 2, {signal_length}) → 2=实部/虚部，{signal_length}=信号采样点")
    print(f"SNR精度：保留{snr_round_decimals}位小数")

    # 验证前3个键的格式
    print(f"\n前3个键的样本信息：")
    count = 0
    for (indiv_id, snr), samples in data_dict.items():
        if count >= 3:
            break
        print(f"  键：({indiv_id}, {snr}dB) → 样本形状：{samples.shape}")
        count += 1

if __name__ == "__main__":
    main()