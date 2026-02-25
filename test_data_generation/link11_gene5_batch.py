import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import os
from scipy.signal import hilbert, lfilter

# 物理常数
c = 3e8  # 光速 (m/s)
k = 1.38e-23  # 玻尔兹曼常数
T = 290  # 绝对温度 (K)
R_earth = 6371e3  # 地球半径 (m)

# -------------------------- 关键修改1：修正个体相位偏移（确保唯一） --------------------------
individual_phase_offset = {
    'E-2D_1': 0.2 * np.pi,    # 0.628 rad
    'E-2D_2': 0.5 * np.pi,    # 1.571 rad
    'P-3C_1': 0.8 * np.pi,    # 2.513 rad
    'P-3C_2': 1.1 * np.pi,    # 3.456 rad
    'P-8A_1': 1.4 * np.pi,    # 4.398 rad
    'P-8A_2': 1.7 * np.pi,    # 5.341 rad
    'P-8A_3': 2.3 * np.pi     # 7.226 rad（修正为唯一值）
}

# -------------------------- 关键修改2：更新个体参数（添加航线+固定高度） --------------------------
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
        'symbol_rate': 1200,
        'modulation': '2FSK'
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
        'symbol_rate': 1200,
        'modulation': '2FSK'
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
        'symbol_rate': 1200,
        'modulation': '2FSK'
    },
    'P-3C_2': {
        'type': 'P-3C',
        'frequency': 270e6,
        'transmit_power': 30,
        'tx_gain': 8,
        'rx_gain': 1,
        'height': 2500,          # 固定高度2.5km（用户指定）
        'start_lat': 26.27,
        'start_lon': 127.75,
        'end_lat': 25.02,
        'end_lon': 122.42,
        'noise_figure': 4,
        'bandwidth': 16e3,
        'symbol_rate': 1200,
        'modulation': '2FSK'
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
        'symbol_rate': 1200,
        'modulation': '2FSK'
    },
    'P-8A_2': {
        'type': 'P-8A',
        'frequency': 300e6,
        'transmit_power': 50,
        'tx_gain': 10,
        'rx_gain': 1,
        'height': 6200,          # 固定高度6.2km（用户指定）
        'start_lat': 26.57,
        'start_lon': 127.67,
        'end_lat': 25.09,
        'end_lon': 122.42,
        'noise_figure': 5,
        'bandwidth': 16e3,
        'symbol_rate': 1200,
        'modulation': '2FSK'
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
        'symbol_rate': 1200,
        'modulation': '2FSK'
    }
}

# -------------------------- 关键修改3：新增工具函数 --------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    用Haversine公式计算两点间水平距离（大圆距离），单位：米
    输入：两点经纬度（度）
    输出：水平距离（米）
    """
    # 转换为弧度
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine公式
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R_earth * c  # 地球半径×圆心角=距离

def generate_waypoints(start_lat, start_lon, end_lat, end_lon, num_points=6000):
    """
    生成起点到终点的均匀分布航点（经纬度）
    输入：起点/终点经纬度（度）、航点数量
    输出：(latitudes, longitudes) 纬度数组、经度数组（度）
    """
    latitudes = np.linspace(start_lat, end_lat, num_points)
    longitudes = np.linspace(start_lon, end_lon, num_points)
    return latitudes, longitudes

# -------------------------- 模拟器类（少量修改，适配动态距离和接收机高度） --------------------------
class Link11Simulator:
    def __init__(self, aircraft_id, receiver_height=2):
        # 初始化飞机参数
        self.aircraft_id = aircraft_id
        params = aircraft_parameters[aircraft_id]
        self.type = params['type']
        self.f = params['frequency']
        self.lambda_ = c / self.f  # 波长
        self.Pt = params['transmit_power']
        self.Gt = params['tx_gain']
        self.Gr = params['rx_gain']
        self.Ht = params['height']  # 飞机固定高度（从参数中读取）
        self.Hr = receiver_height  # 接收机高度（外部传入，适配双接收机）
        self.noise_figure = params['noise_figure']
        self.B = params['bandwidth']
        self.symbol_rate = params['symbol_rate']
        self.modulation = params['modulation']
        
        # 计算噪声功率 (dBW)
        noise_power_watts = k * T * self.B
        self.noise_power_dBW = 10 * np.log10(noise_power_watts) + self.noise_figure
        
        # 生成该个体的唯一识别码
        self.identifier = self._generate_unique_identifier()
        
        # 加载个体专属相位偏移
        self.phase_offset = individual_phase_offset[aircraft_id]

    def _generate_unique_identifier(self):
        # Link11使用16位地址码，为每个个体生成唯一地址
        id_hash = hash(self.aircraft_id) % (2**16)
        return np.unpackbits(np.array([id_hash], dtype=np.uint16).view(np.uint8))

    def _generate_link11_frame(self, frame_length=128):
        # Link11帧结构: 前同步码 + 地址码 + 数据 + 校验码
        preamble = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        address = self.identifier
        data = np.random.randint(0, 2, frame_length - len(preamble) - len(address) - 8)
        checksum = np.sum(np.concatenate([address, data])) % 256
        checksum_bits = np.unpackbits(np.array([checksum], dtype=np.uint8))
        return np.concatenate([preamble, address, data, checksum_bits])

    def free_space_loss(self, distance):
        return 20 * np.log10(4 * np.pi * distance / self.lambda_)

    def line_of_sight_distance(self):
        return np.sqrt(2 * R_earth * self.Ht) + np.sqrt(2 * R_earth * self.Hr)

    def additional_loss(self, d_horizontal):
        d_los = self.line_of_sight_distance()
        if d_horizontal <= d_los:
            return 0  # 视线传播
        
        excess_distance = d_horizontal - d_los
        # 不同机型的额外损耗模型
        if self.type == 'E-2D':
            return 0.3 * (excess_distance / 1000)
        elif self.type == 'P-8A':
            return 0.25 * (excess_distance / 1000)
        elif self.type == 'P-3C':
            return 0.07 * (excess_distance / 1000)
        else:
            return 0.22 * (excess_distance / 1000)

    def total_loss(self, d_horizontal):
        # 直线距离（考虑高度差）
        d_line = np.sqrt(d_horizontal**2 + (self.Ht - self.Hr)**2)
        # 自由空间损耗 + 额外损耗
        fs_loss = self.free_space_loss(d_line)
        extra_loss = self.additional_loss(d_horizontal)
        return fs_loss + extra_loss

    def received_power(self, d_horizontal):
        loss = self.total_loss(d_horizontal)
        Pt_dBW = 10 * np.log10(self.Pt)
        return Pt_dBW + self.Gt + self.Gr - loss

    def snr(self, d_horizontal):
        # 根据水平距离计算实际SNR（核心修改：从距离→SNR，而非反向）
        pr = self.received_power(d_horizontal)
        return pr - self.noise_power_dBW

    def generate_link11_signal(self, snr_db, samples_per_symbol=8):
        # 生成Link11帧数据
        frame_data = self._generate_link11_frame()
        num_symbols = len(frame_data)
        num_samples = num_symbols * samples_per_symbol
        
        # 2FSK调制参数
        f0 = 1000  # 空号频率
        f1 = 2000  # 传号频率
        fs = samples_per_symbol * self.symbol_rate  # 采样率
        t = np.arange(num_samples) / fs
        
        # 生成2FSK实信号
        real_signal = np.zeros(num_samples)
        for i, bit in enumerate(frame_data):
            start_idx = i * samples_per_symbol
            end_idx = start_idx + samples_per_symbol
            if bit == 1:
                real_signal[start_idx:end_idx] = np.cos(2 * np.pi * f1 * t[start_idx:end_idx])
            else:
                real_signal[start_idx:end_idx] = np.cos(2 * np.pi * f0 * t[start_idx:end_idx])
        
        # 高斯滤波
        b = np.exp(-(np.arange(-16, 17) ** 2) / (2 * 3 ** 2))
        b /= np.sum(b)
        real_signal = lfilter(b, 1, real_signal)
        
        # 融入个体专属相位偏移（实信号转复信号）
        theta = self.phase_offset
        real_part = real_signal * np.cos(theta)
        imag_part = real_signal * np.sin(theta)
        complex_signal = real_part + 1j * imag_part
        
        # 添加复噪声以匹配实际SNR
        signal_power = np.mean(np.abs(complex_signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        return complex_signal + noise, frame_data

# -------------------------- 主程序（核心修改：批次保存） --------------------------
def main():
    # 1. 核心参数设置
    samples_per_symbol = 8        # 每个符号的采样数
    num_waypoints = 100000        # 每条航线的航点数
    aircraft_ids = list(aircraft_parameters.keys())  # 7个个体
    # 双接收机参数（经纬度+高度）
    receivers = {
        'receiver_1': {'lat': 25.45, 'lon': 122.07, 'height': 2},
        'receiver_2': {'lat': 24.69, 'lon': 122.43, 'height': 2}
    }
    snr_round_decimals = 1        # SNR保留1位小数，统一字典键格式
    
    # 批次保存参数
    batch_size = 10000            # 每个batch包含10000个样本
    # 输出到项目 dataset/link11/ 目录（推理时 device_load 从此目录读取）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'dataset', 'link11')

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建输出目录：{output_dir}/")
    
    # 2. 初始化数据集字典和计数器
    data_dict = {}
    total_sample_count = 0        # 全局样本计数
    batch_count = 0               # 批次计数
    
    # 3. 预获取信号长度（所有样本长度一致）
    temp_sim = Link11Simulator(aircraft_ids[0], receivers['receiver_1']['height'])
    temp_signal, _ = temp_sim.generate_link11_signal(snr_db=0, samples_per_symbol=samples_per_symbol)
    signal_length = len(temp_signal)
    print(f"单个信号样本长度：{signal_length} 采样点")
    print(f"批次大小：{batch_size} 个样本")
    print(f"预计生成批次数：{(7 * 2 * num_waypoints) // batch_size} 个\n")
    
    # 4. 遍历每个个体生成数据
    for aircraft_id in aircraft_ids:
        print(f"\n===== 正在处理 {aircraft_id} =====")
        params = aircraft_parameters[aircraft_id]
        
        # 生成100000个航点
        waypoint_lats, waypoint_lons = generate_waypoints(
            start_lat=params['start_lat'],
            start_lon=params['start_lon'],
            end_lat=params['end_lat'],
            end_lon=params['end_lon'],
            num_points=num_waypoints
        )
        print(f"  航线：({params['start_lat']:.2f}°N, {params['start_lon']:.2f}°E) → "
              f"({params['end_lat']:.2f}°N, {params['end_lon']:.2f}°E)，共{num_waypoints}个航点")
        
        # 遍历两个接收机
        for receiver_name, receiver in receivers.items():
            print(f"  处理接收机：{receiver_name}（{receiver['lat']:.2f}°N, {receiver['lon']:.2f}°E）")
            simulator = Link11Simulator(aircraft_id, receiver_height=receiver['height'])
            
            # 遍历每个航点
            for wp_idx in range(num_waypoints):
                # 航点经纬度
                wp_lat = waypoint_lats[wp_idx]
                wp_lon = waypoint_lons[wp_idx]
                
                # 计算接收机到航点的水平距离（米）
                horizontal_dist = haversine_distance(
                    receiver['lat'], receiver['lon'],
                    wp_lat, wp_lon
                )
                
                # 计算该距离对应的实际SNR
                snr_db = simulator.snr(horizontal_dist)
                snr_rounded = np.round(snr_db, snr_round_decimals)  # 统一键格式
                
                # 生成Link11复数信号
                complex_signal, _ = simulator.generate_link11_signal(
                    snr_db=snr_db,
                    samples_per_symbol=samples_per_symbol
                )
                
                # 分离实虚部（适配原数据集格式）
                real_part = np.real(complex_signal).astype(np.float32)
                imag_part = np.imag(complex_signal).astype(np.float32)

                # 存入数据字典
                key = (aircraft_id, snr_rounded)
                if key not in data_dict:
                    data_dict[key] = []
                data_dict[key].append([real_part, imag_part])
                
                # 样本计数
                total_sample_count += 1
                
                # 检查是否达到batch_size，如果是则保存
                if total_sample_count % batch_size == 0:
                    # 将列表转为numpy数组
                    batch_dict = {}
                    for k, v in data_dict.items():
                        batch_dict[k] = np.array(v, dtype=np.float32)
                    
                    # 保存batch
                    batch_filename = os.path.join(output_dir, f'batch_{batch_count:04d}.pkl')
                    with open(batch_filename, 'wb') as f:
                        pickle.dump(batch_dict, f, protocol=4)
                    
                    print(f"    ✓ 保存 batch_{batch_count:04d}.pkl ({len(batch_dict)} 个键, {total_sample_count} 个样本)")
                    
                    # 清空内存，准备下一个batch
                    data_dict = {}
                    batch_count += 1
                
                # 进度提示
                if (wp_idx + 1) % 10000 == 0:
                    print(f"    已处理 {wp_idx + 1}/{num_waypoints} 个航点，当前SNR：{snr_rounded:.1f}dB")
    
    # 5. 保存最后一个不满batch_size的batch（如果有剩余数据）
    if len(data_dict) > 0:
        batch_dict = {}
        for k, v in data_dict.items():
            batch_dict[k] = np.array(v, dtype=np.float32)
        
        batch_filename = os.path.join(output_dir, f'batch_{batch_count:04d}.pkl')
        with open(batch_filename, 'wb') as f:
            pickle.dump(batch_dict, f, protocol=4)
        
        print(f"\n✓ 保存最后一个batch: batch_{batch_count:04d}.pkl ({len(batch_dict)} 个键)")
        batch_count += 1
    
    # 6. 结果统计
    print(f"\n{'='*60}")
    print(f"===== 数据生成完成 =====")
    print(f"{'='*60}")
    print(f"输出目录：{output_dir}/")
    print(f"批次数量：{batch_count} 个")
    print(f"总样本数：{total_sample_count}（7个体 × 2接收机 × {num_waypoints}航点）")
    print(f"样本格式：(样本数, 2, {signal_length})，2对应实部/虚部")
    print(f"SNR精度：保留{snr_round_decimals}位小数")
    print(f"批次大小：{batch_size} 个样本/batch")
    print(f"平均文件大小：约{(total_sample_count * 2 * signal_length * 4 / batch_count) / (1024**2):.1f} MB/batch")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
