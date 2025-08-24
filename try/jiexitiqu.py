#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUF文件格式深度解析器 + 权重数据提取器
专门用于理解llama.cpp生成的GGUF模型文件的二进制结构
支持完整的文件头解析、元数据提取、架构参数分析、量化信息解读和权重数据提取
"""

import struct
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
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
class GGUFHeader:
    """GGUF文件头结构"""
    magic: str              # 4字节魔数 "GGUF"
    version: int           # 4字节版本号
    tensor_count: int      # 8字节张量数量
    metadata_count: int    # 8字节元数据键值对数量


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


@dataclass
class Q8_0Block:
    """Q8_0量化块结构"""
    scale: float          # FP32缩放因子
    weights: np.ndarray   # 32个int8权重值


@dataclass
class ExtractedWeight:
    """提取的权重数据"""
    original_name: str      # GGUF中的原始名称
    infinicore_name: str   # InfiniCore对应名称
    shape: Tuple[int, ...]  # 张量形状
    dtype: str             # 数据类型
    data: Union[np.ndarray, List[Q8_0Block]]  # 权重数据
    layer_id: Optional[int] = None            # 层ID（如果是层权重）


@dataclass
class QuantizationInfo:
    """量化参数详细信息"""
    type_name: str        # 量化类型名称
    bits_per_weight: float # 每权重平均位数
    block_size: int       # 量化块大小
    bytes_per_block: int  # 每块字节数
    description: str      # 量化方法描述
    compression_ratio: float # 压缩比率


class InfiniCoreNameMapper:
    """GGUF到InfiniCore的权重名称映射器"""
    
    @staticmethod
    def map_tensor_name(gguf_name: str) -> str:
        """将GGUF张量名称映射到InfiniCore命名规范"""
        # 根据InfiniCore的JiugeWeights结构进行映射
        
        # 输入嵌入层
        if gguf_name == "token_embd.weight":
            return "input_embd"
        
        # 输出归一化和嵌入层
        if gguf_name == "output_norm.weight":
            return "output_norm"
        if gguf_name == "output.weight":
            return "output_embd"
        
        # 层权重映射
        if gguf_name.startswith("blk."):
            parts = gguf_name.split(".")
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                weight_type = parts[2]
                
                # 注意力层权重
                if weight_type == "attn_norm":
                    return f"attn_norm.{layer_idx}"
                elif weight_type == "attn_qkv":
                    return f"attn_qkv.{layer_idx}"
                elif weight_type == "attn_output":
                    return f"attn_o.{layer_idx}"
                
                # FFN层权重
                elif weight_type == "ffn_norm":
                    return f"ffn_norm.{layer_idx}"
                elif weight_type == "ffn_gate_up":
                    return f"ffn_gate_up.{layer_idx}"
                elif weight_type == "ffn_down":
                    return f"ffn_down.{layer_idx}"
        
        # 如果没有匹配到，返回原名称
        return gguf_name
    
    @staticmethod
    def extract_layer_info(gguf_name: str) -> Tuple[Optional[int], str]:
        """从GGUF名称中提取层ID和权重类型"""
        if gguf_name.startswith("blk."):
            parts = gguf_name.split(".")
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                weight_type = parts[2]
                return layer_idx, weight_type
        return None, gguf_name


class GGUFWeightExtractor:
    """GGUF权重数据提取器"""
    
    # 量化类型详细参数
    QUANTIZATION_SPECS = {
        GGMLType.F32: QuantizationInfo("F32", 32.0, 1, 4, "32位IEEE754浮点数", 1.0),
        GGMLType.F16: QuantizationInfo("F16", 16.0, 1, 2, "16位IEEE754浮点数", 2.0),
        GGMLType.Q8_0: QuantizationInfo("Q8_0", 8.5, 32, 34, "8位量化，32元素块，1个FP16缩放因子", 3.76),
        GGMLType.Q8_1: QuantizationInfo("Q8_1", 9.0, 32, 36, "8位量化，32元素块，FP32缩放+偏置", 3.56),
        GGMLType.Q4_0: QuantizationInfo("Q4_0", 4.5, 32, 18, "4位量化，32元素块，1个FP32缩放因子", 7.11),
        GGMLType.Q4_1: QuantizationInfo("Q4_1", 5.0, 32, 20, "4位量化，32元素块，FP32缩放+偏置", 6.40),
        GGMLType.Q5_0: QuantizationInfo("Q5_0", 5.5, 32, 22, "5位量化，32元素块，1个FP32缩放因子", 5.82),
        GGMLType.Q5_1: QuantizationInfo("Q5_1", 6.0, 32, 24, "5位量化，32元素块，FP32缩放+偏置", 5.33),
    }
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.header: Optional[GGUFHeader] = None
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[TensorInfo] = []
        self.tensor_data_offset: int = 0  # 张量数据区开始偏移
        
    def __enter__(self):
        self.file = open(self.filepath, 'rb')
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
            GGUFValueType.STRING: lambda: self._read_string(),
            GGUFValueType.ARRAY: lambda: self._read_array(),
        }
        
        if value_type not in type_readers:
            raise ValueError(f"不支持的GGUF值类型: {value_type}")
        
        return type_readers[value_type]()
    
    def _read_array(self) -> List[Any]:
        """读取GGUF数组格式：4字节类型 + 8字节长度 + 数据"""
        array_type = struct.unpack('<I', self.file.read(4))[0]
        array_length = struct.unpack('<Q', self.file.read(8))[0]
        
        return [self._read_value_by_type(array_type) for _ in range(array_length)]
    
    def parse_header(self) -> GGUFHeader:
        """解析GGUF文件头部结构"""
        # 读取并验证魔数
        magic_bytes = self.file.read(4)
        if magic_bytes != b'GGUF':
            raise ValueError(f"无效的GGUF魔数: {magic_bytes.hex()}, 期望: 47475546")
        
        # 读取版本号（小端序）
        version = struct.unpack('<I', self.file.read(4))[0]
        if version not in [1, 2, 3]:
            print(f"警告: 未知的GGUF版本 {version}, 可能存在兼容性问题")
        
        # 读取张量和元数据计数
        tensor_count = struct.unpack('<Q', self.file.read(8))[0]
        metadata_count = struct.unpack('<Q', self.file.read(8))[0]
        
        self.header = GGUFHeader(
            magic=magic_bytes.decode('ascii'),
            version=version,
            tensor_count=tensor_count,
            metadata_count=metadata_count
        )
        
        return self.header
    
    def parse_metadata(self) -> Dict[str, Any]:
        """解析GGUF元数据部分"""
        metadata = {}
        
        for i in range(self.header.metadata_count):
            try:
                # 读取键名
                key = self._read_string()
                
                # 读取值类型标识
                value_type = struct.unpack('<I', self.file.read(4))[0]
                
                # 读取实际值
                value = self._read_value_by_type(value_type)
                
                metadata[key] = {
                    'type_id': value_type,
                    'type_name': GGUFValueType(value_type).name,
                    'value': value
                }
                
            except Exception as e:
                print(f"警告: 解析第{i+1}个元数据项时出错: {e}")
                continue
        
        self.metadata = metadata
        return metadata
    
    def parse_tensors(self) -> List[TensorInfo]:
        """解析张量信息部分"""
        tensors = []
        
        for i in range(self.header.tensor_count):
            try:
                # 读取张量名称
                name = self._read_string()
                
                # 读取维度数量
                n_dims = struct.unpack('<I', self.file.read(4))[0]
                
                # 读取各维度大小
                dimensions = []
                for _ in range(n_dims):
                    dim = struct.unpack('<Q', self.file.read(8))[0]
                    dimensions.append(dim)
                
                # 读取数据类型和偏移量
                dtype = struct.unpack('<I', self.file.read(4))[0]
                offset = struct.unpack('<Q', self.file.read(8))[0]
                
                # 计算元素总数和字节大小
                element_count = 1
                for dim in dimensions:
                    element_count *= dim
                
                # 根据量化类型计算字节大小
                size_bytes = self._calculate_tensor_size(dtype, element_count)
                
                tensor_info = TensorInfo(
                    name=name,
                    dimensions=dimensions,
                    dtype=dtype,
                    dtype_name=GGMLType(dtype).name if dtype in [t.value for t in GGMLType] else f"UNKNOWN_{dtype}",
                    offset=offset,
                    element_count=element_count,
                    size_bytes=size_bytes
                )
                
                tensors.append(tensor_info)
                
            except Exception as e:
                print(f"警告: 解析第{i+1}个张量时出错: {e}")
                continue
        
        self.tensors = tensors

        # GGUF v3 要求数据区对齐到32字节边界
        current_pos = self.file.tell()
        alignment = 32
        self.tensor_data_offset = (current_pos + alignment - 1) // alignment * alignment
        self.file.seek(self.tensor_data_offset)

        return tensors
    
    def _calculate_tensor_size(self, dtype: int, element_count: int) -> int:
        """计算张量实际占用的字节数"""
        if dtype == GGMLType.F32:
            return element_count * 4
        elif dtype == GGMLType.F16:
            return element_count * 2
        elif dtype == GGMLType.Q8_0:
            # Q8_0: 每32个元素一块，每块34字节
            blocks = (element_count + 31) // 32
            return blocks * 34
        elif dtype == GGMLType.Q8_1:
            blocks = (element_count + 31) // 32
            return blocks * 36
        elif dtype == GGMLType.Q4_0:
            blocks = (element_count + 31) // 32
            return blocks * 18
        elif dtype == GGMLType.Q4_1:
            blocks = (element_count + 31) // 32
            return blocks * 20
        else:
            # 对于未知类型，使用保守估计
            return element_count * 4
    
    def extract_q8_0_weights(self, tensor: TensorInfo) -> List[Q8_0Block]:
        """提取Q8_0量化权重数据"""
        if tensor.dtype != GGMLType.Q8_0:
            raise ValueError(f"张量 {tensor.name} 不是Q8_0类型")
        
        # 使用正确的偏移量（数据区起始 + 相对偏移）
        actual_offset = self.tensor_data_offset + tensor.offset
        self.file.seek(actual_offset)
        
        # 计算块数量
        num_blocks = (tensor.element_count + 31) // 32
        blocks = []
        
        for i in range(num_blocks):
            # 读取FP16缩放因子
            scale_bytes = self.file.read(2)
            if len(scale_bytes) != 2:
                raise ValueError(f"读取缩放因子时到达文件末尾")
            # 读取FP16并显式转换为标准float以便后续计算
            scale_fp16 = np.frombuffer(scale_bytes, dtype='<f2')[0]
            scale = float(scale_fp16)  # 显式转换为Python标准float

            # 检查并处理NaN或无穷大
            if np.isnan(scale) or np.isinf(scale):
                print(f"警告: 块 {i} 的缩放因子异常: {scale}")
                # 可以选择跳过或使用默认值
                scale = 0.0  # 或者使用其他合理的默认值
            
            # 读取32个int8权重
            weight_bytes = self.file.read(32)
            if len(weight_bytes) != 32:
                raise ValueError(f"读取权重数据时到达文件末尾")
            
            weights = np.frombuffer(weight_bytes, dtype=np.int8)
            
            blocks.append(Q8_0Block(scale=scale, weights=weights))
        
        return blocks
    
    def extract_fp32_weights(self, tensor: TensorInfo) -> np.ndarray:
        """提取FP32权重数据"""
        if tensor.dtype != GGMLType.F32:
            raise ValueError(f"张量 {tensor.name} 不是F32类型")
        
        # 在这两个方法中也要使用正确的偏移量
        actual_offset = self.tensor_data_offset + tensor.offset
        self.file.seek(actual_offset)
        
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
        
        # 在这两个方法中也要使用正确的偏移量
        actual_offset = self.tensor_data_offset + tensor.offset
        self.file.seek(actual_offset)
        
        # 读取数据
        data_bytes = self.file.read(tensor.element_count * 2)
        if len(data_bytes) != tensor.element_count * 2:
            raise ValueError(f"读取FP16数据时到达文件末尾")
        
        # 转换为numpy数组并重塑
        weights = np.frombuffer(data_bytes, dtype='<f2')  # 小端序float16
        return weights.reshape(tensor.dimensions)
    
    def extract_all_weights(self, target_tensors: Optional[List[str]] = None) -> List[ExtractedWeight]:
        """提取所有权重数据并映射到InfiniCore命名"""
        extracted_weights = []
        
        for tensor in self.tensors:
            # 如果指定了目标张量列表，检查是否在列表中
            if target_tensors and tensor.name not in target_tensors:
                continue
            
            try:
                # 映射到InfiniCore命名
                infinicore_name = InfiniCoreNameMapper.map_tensor_name(tensor.name)
                layer_id, weight_type = InfiniCoreNameMapper.extract_layer_info(tensor.name)
                
                # 根据数据类型提取权重
                if tensor.dtype == GGMLType.Q8_0:
                    data = self.extract_q8_0_weights(tensor)
                elif tensor.dtype == GGMLType.F32:
                    data = self.extract_fp32_weights(tensor)
                elif tensor.dtype == GGMLType.F16:
                    data = self.extract_fp16_weights(tensor)
                else:
                    print(f"警告: 暂不支持提取 {tensor.dtype_name} 类型的权重: {tensor.name}")
                    continue
                
                # 特殊处理：打印blk.60.ffn_gate_up.weight和blk.61.ffn_gate_up.weight的详细数据
                if tensor.name in ["blk.60.ffn_gate_up.weight", "blk.61.ffn_gate_up.weight"]:
                    print(f"\n{'='*80}")
                    print(f"详细数据输出 - {tensor.name}")
                    print(f"{'='*80}")
                    print(f"张量信息:")
                    print(f"  原始名称: {tensor.name}")
                    print(f"  InfiniCore名称: {infinicore_name}")
                    print(f"  数据类型: {tensor.dtype_name}")
                    print(f"  维度: {tensor.dimensions}")
                    print(f"  形状: {tensor.dimensions}")
                    print(f"  元素总数: {tensor.element_count:,}")
                    print(f"  数据大小: {tensor.size_bytes:,} 字节")
                    print(f"  数据偏移: {tensor.offset:,}")
                    
                    if tensor.dtype == GGMLType.Q8_0:
                        # Q8_0量化数据的详细输出
                        print(f"\nQ8_0量化数据详情:")
                        print(f"  量化块数量: {len(data)}")
                        print(f"  每块元素数: 32")
                        
                        # 打印前几个块的详细信息
                        blocks_to_show = min(5, len(data))
                        print(f"\n前{blocks_to_show}个量化块的详细数据:")
                        for i, block in enumerate(data[:blocks_to_show]):
                            print(f"\n  块 {i}:")
                            print(f"    缩放因子 (scale): {block.scale}")
                            print(f"    权重数据 (32个int8值): {block.weights.tolist()}")
                            print(f"    权重范围: [{block.weights.min()}, {block.weights.max()}]")
                            print(f"    权重均值: {block.weights.mean():.6f}")
                            print(f"    权重标准差: {block.weights.std():.6f}")
                        
                        if len(data) > blocks_to_show:
                            print(f"\n  ... 还有 {len(data) - blocks_to_show} 个块未显示")
                        
                        # 统计所有块的缩放因子
                        all_scales = [block.scale for block in data]
                        scales_array = np.array(all_scales)
                        print(f"\n所有块的缩放因子统计:")
                        print(f"  最小值: {scales_array.min()}")
                        print(f"  最大值: {scales_array.max()}")
                        print(f"  均值: {scales_array.mean():.6f}")
                        print(f"  标准差: {scales_array.std():.6f}")
                        print(f"  中位数: {np.median(scales_array):.6f}")
                        
                        # 统计所有权重值
                        all_weights = np.concatenate([block.weights for block in data])
                        print(f"\n所有量化权重值统计:")
                        print(f"  最小值: {all_weights.min()}")
                        print(f"  最大值: {all_weights.max()}")
                        print(f"  均值: {all_weights.mean():.6f}")
                        print(f"  标准差: {all_weights.std():.6f}")
                        print(f"  中位数: {np.median(all_weights):.6f}")
                        
                    elif tensor.dtype in [GGMLType.F32, GGMLType.F16]:
                        # 浮点数数据的详细输出
                        print(f"\n{tensor.dtype_name}浮点数据详情:")
                        print(f"  数据形状: {data.shape}")
                        print(f"  数据类型: {data.dtype}")
                        
                        # 统计信息
                        print(f"\n数据统计:")
                        print(f"  最小值: {data.min()}")
                        print(f"  最大值: {data.max()}")
                        print(f"  均值: {data.mean():.6f}")
                        print(f"  标准差: {data.std():.6f}")
                        print(f"  中位数: {np.median(data):.6f}")
                        print(f"  零值数量: {np.sum(data == 0):,}")
                        print(f"  非零值数量: {np.sum(data != 0):,}")
                        
                        # 打印部分实际数据
                        print(f"\n部分实际数据 (展平后前50个值):")
                        flat_data = data.flatten()
                        values_to_show = min(50, len(flat_data))
                        for i in range(0, values_to_show, 10):
                            chunk = flat_data[i:i+10]
                            values_str = ", ".join([f"{val:.6f}" for val in chunk])
                            print(f"    [{i:3d}-{i+len(chunk)-1:3d}]: {values_str}")
                        
                        if len(flat_data) > values_to_show:
                            print(f"    ... 还有 {len(flat_data) - values_to_show:,} 个值未显示")
                        
                        # 如果是二维张量，显示矩阵的一些行
                        if len(data.shape) == 2:
                            rows, cols = data.shape
                            rows_to_show = min(3, rows)
                            cols_to_show = min(10, cols)
                            print(f"\n矩阵数据示例 (前{rows_to_show}行，前{cols_to_show}列):")
                            for i in range(rows_to_show):
                                row_data = data[i, :cols_to_show]
                                values_str = ", ".join([f"{val:.6f}" for val in row_data])
                                print(f"    行{i}: [{values_str}]")
                                if cols > cols_to_show:
                                    print(f"           ... (还有{cols - cols_to_show}列)")
                            
                            if rows > rows_to_show:
                                print(f"    ... (还有{rows - rows_to_show}行)")
                    
                    print(f"{'='*80}")
                
                extracted_weight = ExtractedWeight(
                    original_name=tensor.name,
                    infinicore_name=infinicore_name,
                    shape=tuple(tensor.dimensions),
                    dtype=tensor.dtype_name,
                    data=data,
                    layer_id=layer_id
                )
                
                extracted_weights.append(extracted_weight)
                print(f"✓ 提取权重: {tensor.name} -> {infinicore_name} ({tensor.dtype_name})")
                
            except Exception as e:
                print(f"错误: 提取张量 {tensor.name} 时失败: {e}")
                continue
        
        return extracted_weights
    
    def extract_architecture_info(self) -> Dict[str, Any]:
        """提取模型架构信息"""
        arch_info = {}
        
        # 核心架构参数
        arch_mappings = {
            'architecture': 'general.architecture',
            'model_name': 'general.name',
            'model_size': 'general.size_label',
            'vocab_size': 'llama.vocab_size',
            'context_length': 'llama.context_length',
            'embedding_length': 'llama.embedding_length',
            'block_count': 'llama.block_count',
            'attention_head_count': 'llama.attention.head_count',
            'attention_head_count_kv': 'llama.attention.head_count_kv',
            'feed_forward_length': 'llama.feed_forward_length',
            'rope_dimension_count': 'llama.rope.dimension_count',
            'rope_freq_base': 'llama.rope.freq_base',
            'layer_norm_rms_epsilon': 'llama.attention.layer_norm_rms_epsilon',
        }
        
        for param_name, metadata_key in arch_mappings.items():
            if metadata_key in self.metadata:
                arch_info[param_name] = self.metadata[metadata_key]['value']
        
        # 计算派生参数
        if 'embedding_length' in arch_info and 'attention_head_count' in arch_info:
            arch_info['head_dimension'] = arch_info['embedding_length'] // arch_info['attention_head_count']
        
        if 'attention_head_count' in arch_info and 'attention_head_count_kv' in arch_info:
            arch_info['gqa_ratio'] = arch_info['attention_head_count'] // arch_info['attention_head_count_kv']
        
        return arch_info
    
    def analyze_quantization(self) -> Dict[str, Any]:
        """深度分析量化策略"""
        quant_analysis = {
            'type_distribution': {},
            'total_parameters': 0,
            'total_size_bytes': 0,
            'compression_analysis': {},
            'layer_patterns': {}
        }
        
        # 统计各量化类型
        for tensor in self.tensors:
            dtype_name = tensor.dtype_name
            
            if dtype_name not in quant_analysis['type_distribution']:
                quant_analysis['type_distribution'][dtype_name] = {
                    'tensor_count': 0,
                    'total_elements': 0,
                    'total_bytes': 0,
                    'tensors': []
                }
            
            dist = quant_analysis['type_distribution'][dtype_name]
            dist['tensor_count'] += 1
            dist['total_elements'] += tensor.element_count
            dist['total_bytes'] += tensor.size_bytes
            dist['tensors'].append(tensor.name)
            
            quant_analysis['total_parameters'] += tensor.element_count
            quant_analysis['total_size_bytes'] += tensor.size_bytes
        
        # 分析压缩效果
        if quant_analysis['total_parameters'] > 0:
            uncompressed_size = quant_analysis['total_parameters'] * 4  # FP32基线
            compression_ratio = uncompressed_size / quant_analysis['total_size_bytes']
            
            quant_analysis['compression_analysis'] = {
                'uncompressed_size_gb': uncompressed_size / (1024**3),
                'compressed_size_gb': quant_analysis['total_size_bytes'] / (1024**3),
                'overall_compression_ratio': compression_ratio,
                'space_saved_percent': (1 - 1/compression_ratio) * 100
            }
        
        # 分析层级量化模式
        layer_types = {}
        for tensor in self.tensors:
            if 'blk.' in tensor.name:
                # 提取层类型（如 attn_qkv, ffn_down等）
                parts = tensor.name.split('.')
                if len(parts) >= 3:
                    layer_type = parts[2]
                    if layer_type not in layer_types:
                        layer_types[layer_type] = {}
                    
                    dtype = tensor.dtype_name
                    if dtype not in layer_types[layer_type]:
                        layer_types[layer_type][dtype] = 0
                    layer_types[layer_type][dtype] += 1
        
        quant_analysis['layer_patterns'] = layer_types
        
        return quant_analysis
    
    def save_weights_to_format(self, extracted_weights: List[ExtractedWeight], 
                              output_dir: str, format: str = 'numpy') -> None:
        """保存提取的权重到指定格式"""
        os.makedirs(output_dir, exist_ok=True)
        
        if format == 'numpy':
            for weight in extracted_weights:
                filename = f"{weight.infinicore_name.replace('.', '_')}.npz"
                filepath = os.path.join(output_dir, filename)
                
                if weight.dtype == 'Q8_0':
                    # 保存Q8_0量化数据
                    scales = np.array([block.scale for block in weight.data])
                    weights_data = np.stack([block.weights for block in weight.data])
                    np.savez_compressed(filepath, 
                                      scales=scales, 
                                      weights=weights_data,
                                      shape=weight.shape,
                                      dtype=weight.dtype,
                                      original_name=weight.original_name)
                else:
                    # 保存常规numpy数组
                    np.savez_compressed(filepath, 
                                      data=weight.data,
                                      shape=weight.shape,
                                      dtype=weight.dtype,
                                      original_name=weight.original_name)
                
                print(f"✓ 保存权重: {filepath}")
        
        # 保存权重映射信息
        mapping_info = {
            'weight_mappings': [
                {
                    'original_name': w.original_name,
                    'infinicore_name': w.infinicore_name,
                    'shape': w.shape,
                    'dtype': w.dtype,
                    'layer_id': w.layer_id
                }
                for w in extracted_weights
            ]
        }
        
        mapping_file = os.path.join(output_dir, 'weight_mapping.json')
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_info, f, indent=2, ensure_ascii=False)
        print(f"✓ 保存映射信息: {mapping_file}")
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成完整的文件结构分析报告"""
        print("正在解析GGUF文件结构...")
        
        # 解析各部分
        header = self.parse_header()
        print(f"✓ 文件头解析完成 - GGUF v{header.version}")
        
        metadata = self.parse_metadata()
        print(f"✓ 元数据解析完成 - {len(metadata)} 项")
        
        tensors = self.parse_tensors()
        print(f"✓ 张量信息解析完成 - {len(tensors)} 个张量")
        
        architecture = self.extract_architecture_info()
        print("✓ 架构信息提取完成")
        
        quantization = self.analyze_quantization()
        print("✓ 量化分析完成")
        
        return {
            'file_info': {
                'path': self.filepath,
                'header': header.__dict__,
            },
            'metadata': metadata,
            'architecture': architecture,
            'quantization': quantization,
            'tensors': [t.__dict__ for t in tensors]
        }
    
    def print_structure_summary(self, report: Dict[str, Any]):
        """打印结构化的分析摘要"""
        print("\n" + "="*80)
        print("GGUF文件结构深度分析报告")
        print("="*80)
        
        # 文件基本信息
        header = report['file_info']['header']
        print(f"\n文件格式信息:")
        print(f"  魔数: {header['magic']} (0x{header['magic'].encode().hex().upper()})")
        print(f"  版本: {header['version']}")
        print(f"  张量数量: {header['tensor_count']:,}")
        print(f"  元数据条目: {header['metadata_count']:,}")
        
        # 模型架构
        arch = report['architecture']
        if arch:
            print(f"\n模型架构参数:")
            print(f"  基础架构: {arch.get('architecture', 'Unknown')}")
            print(f"  模型名称: {arch.get('model_name', 'Unknown')}")
            print(f"  参数规模: {arch.get('model_size', 'Unknown')}")
            print(f"  词汇表大小: {arch.get('vocab_size', 0):,}")
            print(f"  上下文长度: {arch.get('context_length', 0):,}")
            print(f"  嵌入维度: {arch.get('embedding_length', 0):,}")
            print(f"  Transformer层数: {arch.get('block_count', 0)}")
            print(f"  注意力头数: {arch.get('attention_head_count', 0)}")
            print(f"  KV头数: {arch.get('attention_head_count_kv', 0)}")
            if 'gqa_ratio' in arch:
                print(f"  GQA分组比例: 1:{arch['gqa_ratio']}")
            print(f"  头维度: {arch.get('head_dimension', 0)}")
            print(f"  FFN维度: {arch.get('feed_forward_length', 0):,}")
            print(f"  RoPE频率基数: {arch.get('rope_freq_base', 0):,.0f}")
        
        # 量化详细分析
        quant = report['quantization']
        print(f"\n量化策略分析:")
        print(f"  总参数量: {quant['total_parameters']:,}")
        print(f"  文件大小: {quant['total_size_bytes'] / (1024**3):.2f} GB")
        
        if 'compression_analysis' in quant:
            comp = quant['compression_analysis']
            print(f"  未压缩大小: {comp['uncompressed_size_gb']:.2f} GB (FP32)")
            print(f"  压缩比率: {comp['overall_compression_ratio']:.2f}x")
            print(f"  空间节省: {comp['space_saved_percent']:.1f}%")
        
        print(f"\n量化类型分布:")
        for dtype, info in quant['type_distribution'].items():
            pct = info['total_elements'] / quant['total_parameters'] * 100
            print(f"  {dtype}: {info['tensor_count']} 张量, {info['total_elements']:,} 参数 ({pct:.1f}%)")
            
            # 显示量化规格
            if dtype in ['Q8_0', 'Q8_1', 'Q4_0', 'Q4_1', 'F32', 'F16']:
                dtype_enum = getattr(GGMLType, dtype, None)
                if dtype_enum and dtype_enum in self.QUANTIZATION_SPECS:
                    spec = self.QUANTIZATION_SPECS[dtype_enum]
                    print(f"    规格: {spec.description}")
                    print(f"    块大小: {spec.block_size}, 每块字节: {spec.bytes_per_block}")
        
        # Q8_0详细分析（如果存在）
        if 'Q8_0' in quant['type_distribution']:
            print(f"\nQ8_0量化详细信息:")
            q8_info = quant['type_distribution']['Q8_0']
            spec = self.QUANTIZATION_SPECS[GGMLType.Q8_0]
            
            total_blocks = sum(
                (t['element_count'] + 31) // 32 
                for t in report['tensors'] 
                if t['dtype_name'] == 'Q8_0'
            )
            
            print(f"  量化方法: 8位块量化，每32个权重为一组")
            print(f"  存储格式: 每块包含32个int8值 + 1个float16缩放因子")
            print(f"  总量化块数: {total_blocks:,}")
            print(f"  相比FP32压缩比: {spec.compression_ratio:.2f}x")
        
        # 张量层级模式
        if quant['layer_patterns']:
            print(f"\n层级量化模式:")
            for layer_type, dtypes in quant['layer_patterns'].items():
                print(f"  {layer_type}:")
                for dtype, count in dtypes.items():
                    print(f"    {dtype}: {count} 层")
    
    def print_weight_extraction_summary(self, extracted_weights: List[ExtractedWeight]):
        """打印权重提取摘要"""
        print("\n" + "="*80)
        print("权重数据提取报告")
        print("="*80)
        
        # 按类型分组统计
        type_stats = {}
        for weight in extracted_weights:
            dtype = weight.dtype
            if dtype not in type_stats:
                type_stats[dtype] = {'count': 0, 'total_elements': 0}
            
            type_stats[dtype]['count'] += 1
            total_elements = 1
            for dim in weight.shape:
                total_elements *= dim
            type_stats[dtype]['total_elements'] += total_elements
        
        print(f"\n提取统计:")
        print(f"  总提取权重数: {len(extracted_weights)}")
        
        for dtype, stats in type_stats.items():
            print(f"  {dtype}: {stats['count']} 个权重, {stats['total_elements']:,} 参数")
        
        # 按层级分组显示
        layer_weights = {}
        global_weights = []
        
        for weight in extracted_weights:
            if weight.layer_id is not None:
                if weight.layer_id not in layer_weights:
                    layer_weights[weight.layer_id] = []
                layer_weights[weight.layer_id].append(weight)
            else:
                global_weights.append(weight)
        
        if global_weights:
            print(f"\n全局权重:")
            for weight in global_weights:
                shape_str = "×".join(map(str, weight.shape))
                print(f"  {weight.infinicore_name}: [{shape_str}] ({weight.dtype})")
        
        if layer_weights:
            print(f"\n层级权重 (共{len(layer_weights)}层):")
            # 只显示前3层作为示例
            for layer_id in sorted(layer_weights.keys())[:3]:
                print(f"  第{layer_id}层:")
                for weight in layer_weights[layer_id]:
                    shape_str = "×".join(map(str, weight.shape))
                    print(f"    {weight.infinicore_name}: [{shape_str}] ({weight.dtype})")
            
            if len(layer_weights) > 3:
                print(f"  ... 还有 {len(layer_weights) - 3} 层权重")
        
        # Q8_0权重的详细分析
        q8_weights = [w for w in extracted_weights if w.dtype == 'Q8_0']
        if q8_weights:
            print(f"\nQ8_0权重详细分析:")
            print(f"  Q8_0权重数量: {len(q8_weights)}")
            
            # 分析缩放因子分布
            all_scales = []
            for weight in q8_weights[:5]:  # 只分析前5个权重避免过多输出
                if isinstance(weight.data, list):
                    scales = [block.scale for block in weight.data]
                    all_scales.extend(scales)
            
            if all_scales:
                scales_array = np.array(all_scales)
                print(f"  缩放因子统计 (前5个权重):")
                print(f"    范围: [{scales_array.min():.6f}, {scales_array.max():.6f}]")
                print(f"    均值: {scales_array.mean():.6f}")
                print(f"    标准差: {scales_array.std():.6f}")


def main():
    """主函数"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='GGUF文件结构分析和权重提取工具')
    parser.add_argument('filepath', help='GGUF文件路径')
    parser.add_argument('--extract-weights', action='store_true', help='提取权重数据')
    parser.add_argument('--output-dir', default='./extracted_weights', help='权重输出目录')
    parser.add_argument('--target-tensors', nargs='*', help='指定要提取的张量名称')
    parser.add_argument('--save-report', action='store_true', help='保存详细分析报告')
    parser.add_argument('--weights-format', choices=['numpy'], default='numpy', help='权重保存格式')
    
    args = parser.parse_args()
    
    try:
        with GGUFWeightExtractor(args.filepath) as extractor:
            # 基础结构分析
            report = extractor.generate_comprehensive_report()
            extractor.print_structure_summary(report)
            
            # 权重提取
            if args.extract_weights:
                print(f"\n开始提取权重数据...")
                extracted_weights = extractor.extract_all_weights(args.target_tensors)
                
                if extracted_weights:
                    extractor.print_weight_extraction_summary(extracted_weights)
                    
                    # 保存权重
                    extractor.save_weights_to_format(
                        extracted_weights, 
                        args.output_dir, 
                        args.weights_format
                    )
                    print(f"\n权重提取完成，已保存到: {args.output_dir}")
                else:
                    print("未提取到任何权重数据")
            
            # 保存报告
            if args.save_report:
                output_file = args.filepath.replace('.gguf', '_structure_analysis.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                print(f"\n完整分析报告已保存至: {output_file}")
                
    except FileNotFoundError:
        print(f"错误: 找不到文件 {args.filepath}")
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUF文件格式深度解析器
专门用于理解llama.cpp生成的GGUF模型文件的二进制结构
支持完整的文件头解析、元数据提取、架构参数分析和量化信息解读
"""

import struct
import json
from typing import Dict, Any, List, Tuple, Optional
from enum import IntEnum
from dataclasses import dataclass


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
class GGUFHeader:
    """GGUF文件头结构"""
    magic: str              # 4字节魔数 "GGUF"
    version: int           # 4字节版本号
    tensor_count: int      # 8字节张量数量
    metadata_count: int    # 8字节元数据键值对数量


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


@dataclass
class QuantizationInfo:
    """量化参数详细信息"""
    type_name: str        # 量化类型名称
    bits_per_weight: float # 每权重平均位数
    block_size: int       # 量化块大小
    bytes_per_block: int  # 每块字节数
    description: str      # 量化方法描述
    compression_ratio: float # 压缩比率


class GGUFStructureAnalyzer:
    """GGUF文件结构深度分析器"""
    
    # 量化类型详细参数
    QUANTIZATION_SPECS = {
        GGMLType.F32: QuantizationInfo("F32", 32.0, 1, 4, "32位IEEE754浮点数", 1.0),
        GGMLType.F16: QuantizationInfo("F16", 16.0, 1, 2, "16位IEEE754浮点数", 2.0),
        GGMLType.Q8_0: QuantizationInfo("Q8_0", 8.5, 32, 34, "8位量化，32元素块，1个FP16缩放因子", 3.76),
        GGMLType.Q8_1: QuantizationInfo("Q8_1", 9.0, 32, 36, "8位量化，32元素块，FP32缩放+偏置", 3.56),
        GGMLType.Q4_0: QuantizationInfo("Q4_0", 4.5, 32, 18, "4位量化，32元素块，1个FP32缩放因子", 7.11),
        GGMLType.Q4_1: QuantizationInfo("Q4_1", 5.0, 32, 20, "4位量化，32元素块，FP32缩放+偏置", 6.40),
        GGMLType.Q5_0: QuantizationInfo("Q5_0", 5.5, 32, 22, "5位量化，32元素块，1个FP32缩放因子", 5.82),
        GGMLType.Q5_1: QuantizationInfo("Q5_1", 6.0, 32, 24, "5位量化，32元素块，FP32缩放+偏置", 5.33),
    }
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.header: Optional[GGUFHeader] = None
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[TensorInfo] = []
        
    def __enter__(self):
        self.file = open(self.filepath, 'rb')
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
            GGUFValueType.STRING: lambda: self._read_string(),
            GGUFValueType.ARRAY: lambda: self._read_array(),
        }
        
        if value_type not in type_readers:
            raise ValueError(f"不支持的GGUF值类型: {value_type}")
        
        return type_readers[value_type]()
    
    def _read_array(self) -> List[Any]:
        """读取GGUF数组格式：4字节类型 + 8字节长度 + 数据"""
        array_type = struct.unpack('<I', self.file.read(4))[0]
        array_length = struct.unpack('<Q', self.file.read(8))[0]
        
        return [self._read_value_by_type(array_type) for _ in range(array_length)]
    
    def parse_header(self) -> GGUFHeader:
        """解析GGUF文件头部结构"""
        # 读取并验证魔数
        magic_bytes = self.file.read(4)
        if magic_bytes != b'GGUF':
            raise ValueError(f"无效的GGUF魔数: {magic_bytes.hex()}, 期望: 47475546")
        
        # 读取版本号（小端序）
        version = struct.unpack('<I', self.file.read(4))[0]
        if version not in [1, 2, 3]:
            print(f"警告: 未知的GGUF版本 {version}, 可能存在兼容性问题")
        
        # 读取张量和元数据计数
        tensor_count = struct.unpack('<Q', self.file.read(8))[0]
        metadata_count = struct.unpack('<Q', self.file.read(8))[0]
        
        self.header = GGUFHeader(
            magic=magic_bytes.decode('ascii'),
            version=version,
            tensor_count=tensor_count,
            metadata_count=metadata_count
        )
        
        return self.header
    
    def parse_metadata(self) -> Dict[str, Any]:
        """解析GGUF元数据部分"""
        metadata = {}
        
        for i in range(self.header.metadata_count):
            try:
                # 读取键名
                key = self._read_string()
                
                # 读取值类型标识
                value_type = struct.unpack('<I', self.file.read(4))[0]
                
                # 读取实际值
                value = self._read_value_by_type(value_type)
                
                metadata[key] = {
                    'type_id': value_type,
                    'type_name': GGUFValueType(value_type).name,
                    'value': value
                }
                
            except Exception as e:
                print(f"警告: 解析第{i+1}个元数据项时出错: {e}")
                continue
        
        self.metadata = metadata
        return metadata
    
    def parse_tensors(self) -> List[TensorInfo]:
        """解析张量信息部分"""
        tensors = []
        
        for i in range(self.header.tensor_count):
            try:
                # 读取张量名称
                name = self._read_string()
                
                # 读取维度数量
                n_dims = struct.unpack('<I', self.file.read(4))[0]
                
                # 读取各维度大小
                dimensions = []
                for _ in range(n_dims):
                    dim = struct.unpack('<Q', self.file.read(8))[0]
                    dimensions.append(dim)
                
                # 读取数据类型和偏移量
                dtype = struct.unpack('<I', self.file.read(4))[0]
                offset = struct.unpack('<Q', self.file.read(8))[0]
                
                # 计算元素总数和字节大小
                element_count = 1
                for dim in dimensions:
                    element_count *= dim
                
                # 根据量化类型计算字节大小
                size_bytes = self._calculate_tensor_size(dtype, element_count)
                
                tensor_info = TensorInfo(
                    name=name,
                    dimensions=dimensions,
                    dtype=dtype,
                    dtype_name=GGMLType(dtype).name if dtype in [t.value for t in GGMLType] else f"UNKNOWN_{dtype}",
                    offset=offset,
                    element_count=element_count,
                    size_bytes=size_bytes
                )
                
                tensors.append(tensor_info)
                
            except Exception as e:
                print(f"警告: 解析第{i+1}个张量时出错: {e}")
                continue
        
        self.tensors = tensors
        return tensors
    
    def _calculate_tensor_size(self, dtype: int, element_count: int) -> int:
        """计算张量实际占用的字节数"""
        if dtype == GGMLType.F32:
            return element_count * 4
        elif dtype == GGMLType.F16:
            return element_count * 2
        elif dtype == GGMLType.Q8_0:
            # Q8_0: 每32个元素一块，每块34字节（2字节FP16 scale + 32字节权重）
            blocks = (element_count + 31) // 32
            return blocks * 34
        elif dtype == GGMLType.Q8_1:
            blocks = (element_count + 31) // 32
            return blocks * 36
        elif dtype == GGMLType.Q4_0:
            blocks = (element_count + 31) // 32
            return blocks * 18
        elif dtype == GGMLType.Q4_1:
            blocks = (element_count + 31) // 32
            return blocks * 20
        else:
            # 对于未知类型，使用保守估计
            return element_count * 4
    
    def extract_architecture_info(self) -> Dict[str, Any]:
        """提取模型架构信息"""
        arch_info = {}
        
        # 核心架构参数
        arch_mappings = {
            'architecture': 'general.architecture',
            'model_name': 'general.name',
            'model_size': 'general.size_label',
            'vocab_size': 'llama.vocab_size',
            'context_length': 'llama.context_length',
            'embedding_length': 'llama.embedding_length',
            'block_count': 'llama.block_count',
            'attention_head_count': 'llama.attention.head_count',
            'attention_head_count_kv': 'llama.attention.head_count_kv',
            'feed_forward_length': 'llama.feed_forward_length',
            'rope_dimension_count': 'llama.rope.dimension_count',
            'rope_freq_base': 'llama.rope.freq_base',
            'layer_norm_rms_epsilon': 'llama.attention.layer_norm_rms_epsilon',
        }
        
        for param_name, metadata_key in arch_mappings.items():
            if metadata_key in self.metadata:
                arch_info[param_name] = self.metadata[metadata_key]['value']
        
        # 计算派生参数
        if 'embedding_length' in arch_info and 'attention_head_count' in arch_info:
            arch_info['head_dimension'] = arch_info['embedding_length'] // arch_info['attention_head_count']
        
        if 'attention_head_count' in arch_info and 'attention_head_count_kv' in arch_info:
            arch_info['gqa_ratio'] = arch_info['attention_head_count'] // arch_info['attention_head_count_kv']
        
        return arch_info
    
    def analyze_quantization(self) -> Dict[str, Any]:
        """深度分析量化策略"""
        quant_analysis = {
            'type_distribution': {},
            'total_parameters': 0,
            'total_size_bytes': 0,
            'compression_analysis': {},
            'layer_patterns': {}
        }
        
        # 统计各量化类型
        for tensor in self.tensors:
            dtype_name = tensor.dtype_name
            
            if dtype_name not in quant_analysis['type_distribution']:
                quant_analysis['type_distribution'][dtype_name] = {
                    'tensor_count': 0,
                    'total_elements': 0,
                    'total_bytes': 0,
                    'tensors': []
                }
            
            dist = quant_analysis['type_distribution'][dtype_name]
            dist['tensor_count'] += 1
            dist['total_elements'] += tensor.element_count
            dist['total_bytes'] += tensor.size_bytes
            dist['tensors'].append(tensor.name)
            
            quant_analysis['total_parameters'] += tensor.element_count
            quant_analysis['total_size_bytes'] += tensor.size_bytes
        
        # 分析压缩效果
        if quant_analysis['total_parameters'] > 0:
            uncompressed_size = quant_analysis['total_parameters'] * 4  # FP32基线
            compression_ratio = uncompressed_size / quant_analysis['total_size_bytes']
            
            quant_analysis['compression_analysis'] = {
                'uncompressed_size_gb': uncompressed_size / (1024**3),
                'compressed_size_gb': quant_analysis['total_size_bytes'] / (1024**3),
                'overall_compression_ratio': compression_ratio,
                'space_saved_percent': (1 - 1/compression_ratio) * 100
            }
        
        # 分析层级量化模式
        layer_types = {}
        for tensor in self.tensors:
            if 'blk.' in tensor.name:
                # 提取层类型（如 attn_qkv, ffn_down等）
                parts = tensor.name.split('.')
                if len(parts) >= 3:
                    layer_type = parts[2]
                    if layer_type not in layer_types:
                        layer_types[layer_type] = {}
                    
                    dtype = tensor.dtype_name
                    if dtype not in layer_types[layer_type]:
                        layer_types[layer_type][dtype] = 0
                    layer_types[layer_type][dtype] += 1
        
        quant_analysis['layer_patterns'] = layer_types
        
        return quant_analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成完整的文件结构分析报告"""
        print("正在解析GGUF文件结构...")
        
        # 解析各部分
        header = self.parse_header()
        print(f"✓ 文件头解析完成 - GGUF v{header.version}")
        
        metadata = self.parse_metadata()
        print(f"✓ 元数据解析完成 - {len(metadata)} 项")
        
        tensors = self.parse_tensors()
        print(f"✓ 张量信息解析完成 - {len(tensors)} 个张量")
        
        architecture = self.extract_architecture_info()
        print("✓ 架构信息提取完成")
        
        quantization = self.analyze_quantization()
        print("✓ 量化分析完成")
        
        return {
            'file_info': {
                'path': self.filepath,
                'header': header.__dict__,
            },
            'metadata': metadata,
            'architecture': architecture,
            'quantization': quantization,
            'tensors': [t.__dict__ for t in tensors]
        }
    
    def print_structure_summary(self, report: Dict[str, Any]):
        """打印结构化的分析摘要"""
        print("\n" + "="*80)
        print("GGUF文件结构深度分析报告")
        print("="*80)
        
        # 文件基本信息
        header = report['file_info']['header']
        print(f"\n文件格式信息:")
        print(f"  魔数: {header['magic']} (0x{header['magic'].encode().hex().upper()})")
        print(f"  版本: {header['version']}")
        print(f"  张量数量: {header['tensor_count']:,}")
        print(f"  元数据条目: {header['metadata_count']:,}")
        
        # 模型架构
        arch = report['architecture']
        if arch:
            print(f"\n模型架构参数:")
            print(f"  基础架构: {arch.get('architecture', 'Unknown')}")
            print(f"  模型名称: {arch.get('model_name', 'Unknown')}")
            print(f"  参数规模: {arch.get('model_size', 'Unknown')}")
            print(f"  词汇表大小: {arch.get('vocab_size', 0):,}")
            print(f"  上下文长度: {arch.get('context_length', 0):,}")
            print(f"  嵌入维度: {arch.get('embedding_length', 0):,}")
            print(f"  Transformer层数: {arch.get('block_count', 0)}")
            print(f"  注意力头数: {arch.get('attention_head_count', 0)}")
            print(f"  KV头数: {arch.get('attention_head_count_kv', 0)}")
            if 'gqa_ratio' in arch:
                print(f"  GQA分组比例: 1:{arch['gqa_ratio']}")
            print(f"  头维度: {arch.get('head_dimension', 0)}")
            print(f"  FFN维度: {arch.get('feed_forward_length', 0):,}")
            print(f"  RoPE频率基数: {arch.get('rope_freq_base', 0):,.0f}")
        
        # 量化详细分析
        quant = report['quantization']
        print(f"\n量化策略分析:")
        print(f"  总参数量: {quant['total_parameters']:,}")
        print(f"  文件大小: {quant['total_size_bytes'] / (1024**3):.2f} GB")
        
        if 'compression_analysis' in quant:
            comp = quant['compression_analysis']
            print(f"  未压缩大小: {comp['uncompressed_size_gb']:.2f} GB (FP32)")
            print(f"  压缩比率: {comp['overall_compression_ratio']:.2f}x")
            print(f"  空间节省: {comp['space_saved_percent']:.1f}%")
        
        print(f"\n量化类型分布:")
        for dtype, info in quant['type_distribution'].items():
            pct = info['total_elements'] / quant['total_parameters'] * 100
            print(f"  {dtype}: {info['tensor_count']} 张量, {info['total_elements']:,} 参数 ({pct:.1f}%)")
            
            # 显示量化规格
            if dtype in ['Q8_0', 'Q8_1', 'Q4_0', 'Q4_1', 'F32', 'F16']:
                dtype_enum = getattr(GGMLType, dtype, None)
                if dtype_enum and dtype_enum in self.QUANTIZATION_SPECS:
                    spec = self.QUANTIZATION_SPECS[dtype_enum]
                    print(f"    规格: {spec.description}")
                    print(f"    块大小: {spec.block_size}, 每块字节: {spec.bytes_per_block}")
        
        # Q8_0详细分析（如果存在）
        if 'Q8_0' in quant['type_distribution']:
            print(f"\nQ8_0量化详细信息:")
            q8_info = quant['type_distribution']['Q8_0']
            spec = self.QUANTIZATION_SPECS[GGMLType.Q8_0]
            
            total_blocks = sum(
                (t['element_count'] + 31) // 32 
                for t in report['tensors'] 
                if t['dtype_name'] == 'Q8_0'
            )
            
            print(f"  量化方法: 8位块量化，每32个权重为一组")
            print(f"  存储格式: 每块包含32个int8值 + 1个float16缩放因子")
            print(f"  总量化块数: {total_blocks:,}")
            print(f"  相比FP32压缩比: {spec.compression_ratio:.2f}x")
        
        # 张量层级模式
        if quant['layer_patterns']:
            print(f"\n层级量化模式:")
            for layer_type, dtypes in quant['layer_patterns'].items():
                print(f"  {layer_type}:")
                for dtype, count in dtypes.items():
                    print(f"    {dtype}: {count} 层")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python gguf_analyzer.py <gguf_file_path> [--save-report]")
        print("示例: python gguf_analyzer.py fm9g-4B-sft-v1.0-Q8_0.gguf --save-report")
        return
    
    filepath = sys.argv[1]
    save_report = '--save-report' in sys.argv
    
    try:
        with GGUFStructureAnalyzer(filepath) as analyzer:
            report = analyzer.generate_comprehensive_report()
            analyzer.print_structure_summary(report)
            
            if save_report:
                output_file = filepath.replace('.gguf', '_structure_analysis.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                print(f"\n完整分析报告已保存至: {output_file}")
                
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filepath}")
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()