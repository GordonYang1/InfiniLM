#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门用于提取 blk.61.ffn_gate_up.weight 张量的独立脚本
基于原始jiexitiqu.py代码，针对单个张量进行优化
修改：scale使用FP16格式，每块结构为2字节scale + 32字节int8权重 = 34字节
"""

import struct
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from enum import IntEnum
from dataclasses import dataclass
import os


class GGMLType(IntEnum):
    """GGML张量数据类型枚举"""
    F32 = 0     # 32位浮点数
    F16 = 1     # 16位浮点数
    Q4_0 = 2    # 4位量化，块大小32
    Q4_1 = 3    # 4位量化，块大小32，带偏置
    Q5_0 = 6    # 5位量化，块大小32
    Q5_1 = 7    # 5位量化，块大小32，带偏置
    Q8_0 = 8    # 8位量化，块大小32
    Q8_1 = 9    # 8位量化，块大小32，带偏置
    Q2_K = 10   # 2位量化，K变体
    Q3_K = 11   # 3位量化，K变体
    Q4_K = 12   # 4位量化，K变体
    Q5_K = 13   # 5位量化，K变体
    Q6_K = 14   # 6位量化，K变体
    Q8_K = 15   # 8位量化，K变体
    I8 = 16     # 8位整数
    I16 = 17    # 16位整数
    I32 = 18    # 32位整数


class GGUFValueType(IntEnum):
    """GGUF元数据值类型"""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


@dataclass
class Q8_0Block:
    """Q8_0量化块结构（修改为使用FP16 scale）"""
    scale: float          # FP16缩放因子（转换为float32存储）
    weights: np.ndarray   # 32个int8权重值


@dataclass
class TensorInfo:
    """张量信息结构"""
    name: str              # 张量名称
    dimensions: List[int]  # 维度数组
    dtype: int            # 数据类型ID
    dtype_name: str       # 数据类型名称
    offset: int           # 数据偏移量（字节）
    element_count: int    # 元素总数
    size_bytes: int       # 占用字节数


class SingleTensorExtractor:
    """单个张量提取器"""
    
    def __init__(self, gguf_filepath: str):
        self.gguf_filepath = gguf_filepath
        self.file = None
        self.target_tensor = None
        
    def __enter__(self):
        self.file = open(self.gguf_filepath, 'rb')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
    
    def _read_string(self) -> str:
        """读取GGUF字符串格式：8字节长度 + UTF-8内容"""
        length = struct.unpack('<Q', self.file.read(8))[0]
        if length == 0:
            return ""
        return self.file.read(length).decode('utf-8')
    
    def _read_value_by_type(self, value_type: int) -> Any:
        """根据类型标识读取对应格式的值"""
        type_readers = {
            GGUFValueType.UINT8: lambda: struct.unpack('<B', self.file.read(1))[0],
            GGUFValueType.INT8: lambda: struct.unpack('<b', self.file.read(1))[0],
            GGUFValueType.UINT16: lambda: struct.unpack('<H', self.file.read(2))[0],
            GGUFValueType.INT16: lambda: struct.unpack('<h', self.file.read(2))[0],
            GGUFValueType.UINT32: lambda: struct.unpack('<I', self.file.read(4))[0],
            GGUFValueType.INT32: lambda: struct.unpack('<i', self.file.read(4))[0],
            GGUFValueType.UINT64: lambda: struct.unpack('<Q', self.file.read(8))[0],
            GGUFValueType.INT64: lambda: struct.unpack('<q', self.file.read(8))[0],
            GGUFValueType.FLOAT32: lambda: struct.unpack('<f', self.file.read(4))[0],
            GGUFValueType.FLOAT64: lambda: struct.unpack('<d', self.file.read(8))[0],
            GGUFValueType.BOOL: lambda: bool(struct.unpack('<B', self.file.read(1))[0]),
            GGUFValueType.STRING: self._read_string,
        }
        
        if value_type == GGUFValueType.ARRAY:
            # 读取数组：类型 + 长度 + 元素
            array_type = struct.unpack('<I', self.file.read(4))[0]
            array_length = struct.unpack('<Q', self.file.read(8))[0]
            return [self._read_value_by_type(array_type) for _ in range(array_length)]
        
        return type_readers.get(value_type, lambda: None)()
    
    def _load_metadata_and_tensors(self) -> bool:
        """加载GGUF文件头和张量信息"""
        # 验证文件头
        magic = self.file.read(4)
        if magic != b'GGUF':
            print("错误: 不是有效的GGUF文件")
            return False
        
        # 读取版本
        version = struct.unpack('<I', self.file.read(4))[0]
        print(f"GGUF版本: {version}")
        
        # 读取张量数量和元数据数量
        tensor_count = struct.unpack('<Q', self.file.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', self.file.read(8))[0]
        
        print(f"张量数量: {tensor_count}")
        print(f"元数据键值对数量: {metadata_kv_count}")
        
        # 跳过元数据
        for _ in range(metadata_kv_count):
            key = self._read_string()
            value_type = struct.unpack('<I', self.file.read(4))[0]
            value = self._read_value_by_type(value_type)
        
        # 读取张量信息
        for _ in range(tensor_count):
            name = self._read_string()
            n_dimensions = struct.unpack('<I', self.file.read(4))[0]
            dimensions = [struct.unpack('<Q', self.file.read(8))[0] for _ in range(n_dimensions)]
            dtype = struct.unpack('<I', self.file.read(4))[0]
            offset = struct.unpack('<Q', self.file.read(8))[0]
            
            # 计算元素总数
            element_count = 1
            for dim in dimensions:
                element_count *= dim
            
            # 获取数据类型名称
            dtype_name = "UNKNOWN"
            for ggml_type in GGMLType:
                if ggml_type.value == dtype:
                    dtype_name = ggml_type.name
                    break
            
            # 计算大小（字节）
            if dtype == GGMLType.Q8_0:
                # Q8_0: 每32个元素一个块，每块34字节（2字节FP16 scale + 32字节int8）
                num_blocks = (element_count + 31) // 32
                size_bytes = num_blocks * 34
            elif dtype == GGMLType.F32:
                size_bytes = element_count * 4
            elif dtype == GGMLType.F16:
                size_bytes = element_count * 2
            else:
                size_bytes = 0  # 未知类型
            
            # 如果找到目标张量，保存信息
            if name == "blk.61.ffn_gate_up.weight":
                self.target_tensor = TensorInfo(
                    name=name,
                    dimensions=dimensions,
                    dtype=dtype,
                    dtype_name=dtype_name,
                    offset=offset,
                    element_count=element_count,
                    size_bytes=size_bytes
                )
                print(f"\n找到目标张量: {name}")
                print(f"  数据类型: {dtype_name}")
                print(f"  维度: {dimensions}")
                print(f"  元素总数: {element_count:,}")
                print(f"  数据偏移: {offset:,}")
                print(f"  预估大小: {size_bytes:,} 字节")
                return True
        
        print("错误: 未找到目标张量 blk.61.ffn_gate_up.weight")
        return False
    
    def _print_q8_0_statistics(self, blocks: List[Q8_0Block], tensor_info: TensorInfo):
        """打印Q8_0数据的详细统计信息"""
        print(f"\n{'='*80}")
        print(f"Q8_0张量统计信息 - {tensor_info.name}")
        print(f"{'='*80}")
        
        # 基本信息
        print(f"张量形状: {tensor_info.dimensions}")
        print(f"元素总数: {tensor_info.element_count:,}")
        print(f"量化块数: {len(blocks)}")
        print(f"块结构: 2字节FP16 scale + 32字节int8权重 = 34字节")
        
        # 缩放因子统计
        scales = np.array([block.scale for block in blocks])
        print(f"\n缩放因子统计:")
        print(f"  数量: {len(scales)}")
        print(f"  范围: [{scales.min():.8f}, {scales.max():.8f}]")
        print(f"  均值: {scales.mean():.8f}")
        print(f"  标准差: {scales.std():.8f}")
        print(f"  中位数: {np.median(scales):.8f}")
        
        # 权重统计
        all_weights = np.concatenate([block.weights for block in blocks])
        print(f"\n权重统计:")
        print(f"  总权重数: {len(all_weights):,}")
        print(f"  范围: [{all_weights.min()}, {all_weights.max()}]")
        print(f"  均值: {all_weights.mean():.6f}")
        print(f"  标准差: {all_weights.std():.6f}")
        
        # 权重分布
        unique, counts = np.unique(all_weights, return_counts=True)
        print(f"  唯一值数量: {len(unique)}")
        print(f"  权重分布（前10个最常见的值）:")
        for i in range(min(10, len(unique))):
            idx = np.argmax(counts)
            print(f"    {unique[idx]:3d}: {counts[idx]:8,} 次 ({counts[idx]/len(all_weights)*100:.2f}%)")
            counts[idx] = 0  # 移除已显示的
    
    def _print_regular_statistics(self, data: np.ndarray, tensor_info: TensorInfo):
        """打印常规数据的统计信息"""
        print(f"\n{'='*80}")
        print(f"{tensor_info.dtype_name}张量统计信息 - {tensor_info.name}")
        print(f"{'='*80}")
        
        print(f"张量形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"元素总数: {data.size:,}")
        print(f"内存占用: {data.nbytes:,} 字节")
        
        print(f"\n数值统计:")
        print(f"  范围: [{data.min():.8f}, {data.max():.8f}]")
        print(f"  均值: {data.mean():.8f}")
        print(f"  标准差: {data.std():.8f}")
        print(f"  中位数: {np.median(data):.8f}")
        
        # 检查特殊值
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        zero_count = (data == 0).sum()
        
        print(f"\n特殊值统计:")
        print(f"  NaN值: {nan_count:,} ({nan_count/data.size*100:.2f}%)")
        print(f"  无穷值: {inf_count:,} ({inf_count/data.size*100:.2f}%)")
        print(f"  零值: {zero_count:,} ({zero_count/data.size*100:.2f}%)")
    
    def extract_q8_0_blocks(self, tensor: TensorInfo) -> List[Q8_0Block]:
        """提取Q8_0量化权重数据（使用FP16 scale）"""
        if tensor.dtype != GGMLType.Q8_0:
            raise ValueError(f"张量 {tensor.name} 不是Q8_0类型，而是 {tensor.dtype_name}")
        
        print(f"\n开始提取Q8_0数据...")
        print(f"定位到偏移量: {tensor.offset}")
        
        # 获取文件大小进行验证
        current_pos = self.file.tell()
        self.file.seek(0, 2)  # 移到文件末尾
        file_size = self.file.tell()
        self.file.seek(current_pos)  # 恢复原位置
        print(f"文件总大小: {file_size:,} 字节")
        
        # 定位到张量数据位置
        self.file.seek(tensor.offset)
        
        # 计算可用的剩余字节数
        remaining_bytes = file_size - tensor.offset
        print(f"从偏移量到文件末尾的可用字节: {remaining_bytes:,}")
        
        # 计算理论块数量和实际可能的块数量
        theoretical_blocks = (tensor.element_count + 31) // 32
        max_possible_blocks = remaining_bytes // 34  # 每块34字节 (2+32)
        
        print(f"理论块数量: {theoretical_blocks:,}")
        print(f"文件剩余空间最多可容纳块数: {max_possible_blocks:,}")
        print(f"每块结构: 2字节FP16 scale + 32字节int8权重 = 34字节")
        
        # 使用较小的值作为实际处理的块数
        actual_blocks = min(theoretical_blocks, max_possible_blocks)
        if actual_blocks < theoretical_blocks:
            print(f"警告: 文件空间不足，将只提取 {actual_blocks:,} 个块（预期 {theoretical_blocks:,}）")
        
        blocks = []
        last_progress = 0
        
        try:
            for i in range(actual_blocks):
                # 检查是否还有足够的字节
                current_pos = self.file.tell()
                if current_pos + 34 > file_size:
                    print(f"警告: 在块 {i} 处剩余字节不足，停止读取")
                    print(f"当前位置: {current_pos}, 需要: 34字节, 剩余: {file_size - current_pos}字节")
                    break
                
                # 读取FP16缩放因子（2字节）
                scale_bytes = self.file.read(2)
                if len(scale_bytes) != 2:
                    print(f"警告: 在块{i}读取缩放因子时只读到{len(scale_bytes)}字节")
                    break
                
                # 将FP16转换为FP32
                scale_fp16 = struct.unpack('<H', scale_bytes)[0]  # 读取为uint16
                scale = np.frombuffer(np.array([scale_fp16], dtype=np.uint16).tobytes(), dtype=np.float16)[0]
                scale = float(scale)  # 转换为float32
                
                # 读取32个int8权重
                weight_bytes = self.file.read(32)
                if len(weight_bytes) != 32:
                    print(f"警告: 在块{i}读取权重数据时只读到{len(weight_bytes)}字节")
                    break
                
                weights = np.frombuffer(weight_bytes, dtype=np.int8)
                
                block = Q8_0Block(scale=scale, weights=weights)
                blocks.append(block)
                
                # 打印前几个块的详细信息
                if i < 3:
                    print(f"\n块 {i} 详情:")
                    print(f"  缩放因子 (FP16->FP32): {scale}")
                    print(f"  权重范围: [{weights.min()}, {weights.max()}]")
                    print(f"  权重前10个值: {weights[:10].tolist()}")
                
                # 显示进度（每1000块或每1%）
                progress_interval = max(1000, actual_blocks // 100)
                if (i + 1) % progress_interval == 0:
                    progress = (i + 1) / actual_blocks * 100
                    print(f"已处理块: {i + 1:,}/{actual_blocks:,} ({progress:.1f}%)")
        
        except Exception as e:
            print(f"在处理第 {i} 个块时出错: {e}")
            print(f"当前文件位置: {self.file.tell()}")
            print(f"已成功提取的块数: {len(blocks)}")
            # 不再抛出异常，而是返回已提取的数据
            print(f"将返回已成功提取的 {len(blocks)} 个块")
        
        print(f"\n提取完成:")
        print(f"  成功提取块数: {len(blocks):,}")
        print(f"  预期块数: {theoretical_blocks:,}")
        print(f"  完成度: {len(blocks)/theoretical_blocks*100:.1f}%")
        
        if len(blocks) == 0:
            raise ValueError("未能提取到任何有效的块数据")
        
        return blocks
    
    def extract_fp32_weights(self, tensor: TensorInfo) -> np.ndarray:
        """提取FP32权重数据"""
        if tensor.dtype != GGMLType.F32:
            raise ValueError(f"张量 {tensor.name} 不是F32类型")
        
        # 定位到张量数据位置
        self.file.seek(tensor.offset)
        
        # 读取数据
        data_bytes = self.file.read(tensor.element_count * 4)
        if len(data_bytes) != tensor.element_count * 4:
            raise ValueError(f"读取FP32数据时到达文件末尾")
        
        # 转换为numpy数组并重塑
        weights = np.frombuffer(data_bytes, dtype='<f4')  # 小端序float32
        return weights.reshape(tensor.dimensions)
    
    def extract_fp16_weights(self, tensor: TensorInfo) -> np.ndarray:
        """提取FP16权重数据"""
        if tensor.dtype != GGMLType.F16:
            raise ValueError(f"张量 {tensor.name} 不是F16类型")
        
        # 定位到张量数据位置
        self.file.seek(tensor.offset)
        
        # 读取数据
        data_bytes = self.file.read(tensor.element_count * 2)
        if len(data_bytes) != tensor.element_count * 2:
            raise ValueError(f"读取FP16数据时到达文件末尾")
        
        # 转换为numpy数组并重塑
        weights = np.frombuffer(data_bytes, dtype='<f2')  # 小端序float16
        return weights.reshape(tensor.dimensions)
    
    def save_q8_0_data(self, blocks: List[Q8_0Block], output_path: str):
        """保存Q8_0数据到文件"""
        print(f"\n保存Q8_0数据到: {output_path}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为npz格式，包含所有必要信息
        scales = np.array([block.scale for block in blocks], dtype=np.float32)
        weights_matrix = np.array([block.weights for block in blocks], dtype=np.int8)
        
        np.savez_compressed(
            output_path,
            scales=scales,
            weights=weights_matrix,
            num_blocks=len(blocks),
            block_size=32,
            dtype='Q8_0',
            scale_format='FP16'  # 标记使用FP16格式的scale
        )
        
        print(f"保存完成:")
        print(f"  缩放因子数组形状: {scales.shape}")
        print(f"  权重矩阵形状: {weights_matrix.shape}")
        print(f"  文件大小: {os.path.getsize(output_path):,} 字节")
        print(f"  Scale格式: FP16 (已转换为FP32存储)")
    
    def save_regular_data(self, data: np.ndarray, output_path: str, dtype_name: str):
        """保存常规权重数据到文件"""
        print(f"\n保存{dtype_name}数据到: {output_path}")
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为npz格式
        np.savez_compressed(
            output_path,
            weights=data,
            shape=data.shape,
            dtype=dtype_name
        )
        
        print(f"保存完成:")
        print(f"  权重形状: {data.shape}")
        print(f"  数据类型: {dtype_name}")
        print(f"  文件大小: {os.path.getsize(output_path):,} 字节")
    
    def extract_target_tensor(self, target_name: str = "blk.61.ffn_gate_up.weight") -> bool:
        """提取指定的张量"""
        print(f"开始提取张量: {target_name}")
        print(f"GGUF文件: {self.gguf_filepath}")
        
        # 加载文件头和张量信息
        if not self._load_metadata_and_tensors():
            return False
        
        if not self.target_tensor:
            print(f"错误: 未找到目标张量 {target_name}")
            return False
        
        tensor_info = self.target_tensor
        
        try:
            # 根据数据类型提取权重
            if tensor_info.dtype == GGMLType.Q8_0:
                print(f"\n提取Q8_0量化数据 (FP16 scale)...")
                blocks = self.extract_q8_0_blocks(tensor_info)
                
                if len(blocks) > 0:
                    # 保存数据
                    output_path = f"extracted_weights/{target_name.replace('.', '_')}_q8_0_fp16scale.npz"
                    self.save_q8_0_data(blocks, output_path)
                    
                    # 详细统计信息
                    self._print_q8_0_statistics(blocks, tensor_info)
                    
                    # 计算预期块数
                    expected_blocks = (tensor_info.element_count + 31) // 32
                    if len(blocks) < expected_blocks:
                        print(f"\n警告: 只提取了部分数据")
                        print(f"提取块数: {len(blocks):,}")
                        print(f"预期块数: {expected_blocks:,}")
                        print(f"完成度: {len(blocks)/expected_blocks*100:.1f}%")
                        print(f"数据已保存到: {output_path}")
                        return True  # 部分提取也算成功
                    else:
                        return True  # 完全提取成功
                else:
                    print(f"错误: 未能提取到任何有效数据")
                    return False
                
            elif tensor_info.dtype == GGMLType.F32:
                print(f"\n提取FP32数据...")
                data = self.extract_fp32_weights(tensor_info)
                
                # 保存数据
                output_path = f"extracted_weights/{target_name.replace('.', '_')}_fp32.npz"
                self.save_regular_data(data, output_path, "F32")
                
                # 详细统计信息
                self._print_regular_statistics(data, tensor_info)
                
                return True
                
            elif tensor_info.dtype == GGMLType.F16:
                print(f"\n提取FP16数据...")
                data = self.extract_fp16_weights(tensor_info)
                
                # 保存数据
                output_path = f"extracted_weights/{target_name.replace('.', '_')}_fp16.npz"
                self.save_regular_data(data, output_path, "F16")
                
                # 详细统计信息
                self._print_regular_statistics(data, tensor_info)
                
                return True
                
            else:
                print(f"错误: 暂不支持 {tensor_info.dtype_name} 类型的权重提取")
                return False
                
        except Exception as e:
            print(f"错误: 提取张量时失败: {e}")
            return False


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python extract_blk61.py <gguf文件路径>")
        print("示例: python extract_blk61.py /path/to/model.gguf")
        sys.exit(1)
    
    gguf_file = sys.argv[1]
    
    if not os.path.exists(gguf_file):
        print(f"错误: 文件不存在: {gguf_file}")
        sys.exit(1)
    
    print("="*80)
    print("blk.61.ffn_gate_up.weight 张量提取工具")
    print("修改版本：使用FP16格式的scale，每块34字节")
    print("="*80)
    
    try:
        with SingleTensorExtractor(gguf_file) as extractor:
            success = extractor.extract_target_tensor("blk.61.ffn_gate_up.weight")
            
            if success:
                print("\n" + "="*80)
                print("✅ 提取完成！数据已保存到 extracted_weights/ 目录")
                print("="*80)
            else:
                print("\n" + "="*80)
                print("❌ 提取失败")
                print("="*80)
                sys.exit(1)
    
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
