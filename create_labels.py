import numpy as np
import os

# 配置路径
DATA_DIR = "processed_frequency/"  # Removed leading slash for relative path
SPATIAL_DIR = "processed_spatial/"
OUTPUT_DIR = "labels/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_file_based_labels():
    """为每个.npy文件生成独立标签"""
    # 获取实际存在的文件列表（而不是硬编码）
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_bands.npy')]

    # 分离文件名前缀（不带扩展名）
    file_prefixes = [f.replace('_bands.npy', '') for f in all_files]

    # 按文件名分类
    healthy_files = sorted([f for f in file_prefixes if 'healthy' in f])
    depressed_files = sorted([f for f in file_prefixes if 'depressed' in f])

    # 训练集和测试集划分 (调整以匹配你的实际文件数量)
    # 健康组: 29个文件 (训练1-20, 测试20-29)
    # 抑郁组: 24个文件 (训练1-17, 测试17-24)
    train_files = (
        healthy_files[:20] +  # healthy_01 到 healthy_10
        depressed_files[:17]  # depressed_01 到 depressed_12
    )

    test_files = (
        healthy_files[20:] +   # healthy_11 到 healthy_13
        depressed_files[17:]   # depressed_13 到 depressed_15
    )

    # 生成标签映射
    def create_label_mapping(files):
        labels = []
        file_paths = []
        for prefix in files:
            # 确定标签 (0=健康, 1=抑郁)
            label = 0 if 'healthy' in prefix else 1
            # 构建完整文件名
            filename = f"{prefix}_bands.npy"
            # 检查文件是否存在
            if os.path.exists(os.path.join(DATA_DIR, filename)):
                labels.append(label)
                file_paths.append(filename)
        return np.array(labels), file_paths

    # 创建标签
    train_labels, train_file_list = create_label_mapping(train_files)
    test_labels, test_file_list = create_label_mapping(test_files)

    # 保存结果
    np.save(os.path.join(OUTPUT_DIR, "train_labels.npy"), train_labels)
    np.save(os.path.join(OUTPUT_DIR, "test_labels.npy"), test_labels)

    # 保存文件列表（用于调试）
    with open(os.path.join(OUTPUT_DIR, "train_files.txt"), "w") as f:
        f.write("\n".join(train_file_list))
    with open(os.path.join(OUTPUT_DIR, "test_files.txt"), "w") as f:
        f.write("\n".join(test_file_list))

    # 打印统计信息
    print(f"训练集: {len(train_labels)}个文件 (健康={sum(train_labels == 0)}, 抑郁={sum(train_labels == 1)})")
    print(f"测试集: {len(test_labels)}个文件 (健康={sum(test_labels == 0)}, 抑郁={sum(test_labels == 1)})")
    print(f"标签已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_file_based_labels()