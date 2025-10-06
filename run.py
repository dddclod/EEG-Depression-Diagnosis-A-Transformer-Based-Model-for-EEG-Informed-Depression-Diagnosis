import argparse
import configparser
import os
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
import model_transformer as sst_model  # 使用Transformer混合模型
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.ndimage import rotate, shift
from scipy.ndimage import map_coordinates, gaussian_filter

# Global variables
train_specInput_root_path = None
train_tempInput_root_path = None
train_label_root_path = None
test_specInput_root_path = None
test_tempInput_root_path = None
test_label_root_path = None
result_path = None
model_save_path = None

input_width = None
specInput_length = None
temInput_length = None
depth_spec = None
depth_tem = None
gr_spec = None
gr_tem = None
nb_dense_block = None
nb_class = None

nbEpoch = None
batch_size = None
lr = None


def read_config(config_path):
    conf = configparser.ConfigParser()
    conf.read(config_path)

    global train_specInput_root_path, train_tempInput_root_path, train_label_root_path, test_specInput_root_path, test_tempInput_root_path, test_label_root_path
    train_specInput_root_path = conf['path']['train_specInput_root_path']
    train_tempInput_root_path = conf['path']['train_tempInput_root_path']
    train_label_root_path = conf['path']['train_label_root_path']
    test_specInput_root_path = conf['path']['test_specInput_root_path']
    test_tempInput_root_path = conf['path']['test_tempInput_root_path']
    test_label_root_path = conf['path']['test_label_root_path']

    global result_path, model_save_path
    result_path = conf['path']['result_path']
    model_save_path = conf['path']['model_save_path']

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    global input_width, specInput_length, temInput_length, depth_spec, depth_tem, gr_spec, gr_tem, nb_dense_block, nb_class
    input_width = int(conf['model']['input_width'])
    specInput_length = int(conf['model']['specInput_length'])
    temInput_length = int(conf['model']['temInput_length'])
    depth_spec = int(conf['model']['depth_spec'])
    depth_tem = int(conf['model']['depth_tem'])
    gr_spec = int(conf['model']['gr_spec'])
    gr_tem = int(conf['model']['gr_tem'])
    nb_dense_block = int(conf['model']['nb_dense_block'])
    nb_class = int(conf['model']['nb_class'])

    global nbEpoch, batch_size, lr
    nbEpoch = int(conf['training']['nbEpoch'])
    batch_size = int(conf['training']['batch_size'])
    lr = float(conf['training']['lr'])


def augment_data(spec_data, temp_data):
    """对单个样本进行数据增强"""
    augmented_spec = []
    augmented_temp = []

    for i in range(spec_data.shape[0]):  # 遍历每个样本
        # 原始样本
        spec = spec_data[i]
        temp = temp_data[i]

        # 1. 随机旋转 (0-30度)
        angle = np.random.uniform(-30, 30)
        rotated_spec = rotate(spec, angle, axes=(0, 1), reshape=False, mode='reflect')
        rotated_temp = rotate(temp, angle, axes=(0, 1), reshape=False, mode='reflect')

        # 2. 随机平移 (最多2个像素)
        shift_val = np.random.uniform(-2, 2, size=2)
        shifted_spec = shift(rotated_spec, shift=(shift_val[0], shift_val[1], 0, 0), mode='reflect')
        shifted_temp = shift(rotated_temp, shift=(shift_val[0], shift_val[1], 0, 0), mode='reflect')

        # 3. 随机添加高斯噪声
        noise_spec = shifted_spec + np.random.normal(0, 0.01, size=shifted_spec.shape)
        noise_temp = shifted_temp + np.random.normal(0, 0.01, size=shifted_temp.shape)

        # 4. 随机水平/垂直翻转
        if np.random.rand() > 0.5:
            noise_spec = np.flip(noise_spec, axis=0)
            noise_temp = np.flip(noise_temp, axis=0)
        if np.random.rand() > 0.5:
            noise_spec = np.flip(noise_spec, axis=1)
            noise_temp = np.flip(noise_temp, axis=1)

        augmented_spec.append(noise_spec)
        augmented_temp.append(noise_temp)

    return np.array(augmented_spec), np.array(augmented_temp)

def normalize_data(data):
    mean = np.mean(data, axis=(1, 2, 3), keepdims=True)
    std = np.std(data, axis=(1, 2, 3), keepdims=True)
    return (data - mean) / (std + 1e-8)


def load_data(spec_path, temp_path, label_path, train=True):
    """加载数据（按新数据分布）
    参数:
        train=True: 加载训练集 (20健康 + 17患病)
        train=False: 加载测试集 (9健康 + 7患病)
    """
    all_spec_samples = []
    all_temp_samples = []
    all_labels = []

    if train:
        # 训练集文件范围
        healthy_range = range(1, 21)  # healthy_01到healthy_20
        depressed_range = range(1, 18)  # depressed_01到depressed_17
    else:
        # 测试集文件范围
        healthy_range = range(21, 30)  # healthy_21到healthy_29
        depressed_range = range(18, 25)  # depressed_18到depressed_24

    # 加载健康样本
    for i in healthy_range:
        prefix = f"healthy_{i:02d}"
        try:
            spec_data = np.load(os.path.join(spec_path, f'{prefix}_bands.npy'))
            spec_data = np.expand_dims(spec_data, axis=-1)
            temp_data = np.load(os.path.join(temp_path, f'{prefix}_spatial.npy'))
            temp_data = np.expand_dims(temp_data, axis=-1)

            all_spec_samples.append(spec_data)
            all_temp_samples.append(temp_data)
            all_labels.append(0)  # 健康标签为0
        except FileNotFoundError:
            print(f"警告: 文件 {prefix} 不存在，已跳过")
            continue

    # 加载抑郁样本
    for i in depressed_range:
        prefix = f"depressed_{i:02d}"
        try:
            spec_data = np.load(os.path.join(spec_path, f'{prefix}_bands.npy'))
            spec_data = np.expand_dims(spec_data, axis=-1)
            temp_data = np.load(os.path.join(temp_path, f'{prefix}_spatial.npy'))
            temp_data = np.expand_dims(temp_data, axis=-1)

            all_spec_samples.append(spec_data)
            all_temp_samples.append(temp_data)
            all_labels.append(1)  # 抑郁标签为1
        except FileNotFoundError:
            print(f"警告: 文件 {prefix} 不存在，已跳过")
            continue

    # 转换为numpy数组
    all_spec = np.array(all_spec_samples)  # shape: (n_files, 20, 20, 5, 1)
    all_temp = np.array(all_temp_samples)  # shape: (n_files, 20, 20, 512, 1)
    all_labels = np.array(all_labels)  # shape: (n_files,)

    # 数据增强 - 仅对训练集
    if train:
        augmented_spec = [all_spec]
        augmented_temp = [all_temp]
        augmented_labels = [all_labels]

        for _ in range(3):  # 增强3次 (最终数据量=原始4倍)
            spec_aug, temp_aug = augment_data(all_spec, all_temp)
            augmented_spec.append(spec_aug)
            augmented_temp.append(temp_aug)
            augmented_labels.append(all_labels)

        all_spec = np.concatenate(augmented_spec, axis=0)
        all_temp = np.concatenate(augmented_temp, axis=0)
        all_labels = np.concatenate(augmented_labels, axis=0)

    # 归一化
    all_spec = normalize_data(all_spec)
    all_temp = normalize_data(all_temp)

    # 打乱训练集顺序
    if train:
        indices = np.arange(len(all_spec))
        np.random.shuffle(indices)
        all_spec = all_spec[indices]
        all_temp = all_temp[indices]
        all_labels = all_labels[indices]

    return all_spec, all_temp, all_labels


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from tabulate import tabulate


# Add this function (anywhere before the run() function)
def plot_training_history(history, result_path):
    """Plot training and validation accuracy/loss curves with y-axis from 0 to 1"""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)  # 设置y轴范围为0~1
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)  # 设置y轴范围为0~1
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'training_history.png'))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, result_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(result_path, 'confusion_matrix.png'))
    plt.close()


def save_classification_report(y_true, y_pred, result_path):
    """Save classification metrics as table and text file"""
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Create metrics table
    metrics = [
        ['Accuracy', report['accuracy']],
        ['Precision', report['weighted avg']['precision']],
        ['Recall', report['weighted avg']['recall']],
        ['F1 Score', report['weighted avg']['f1-score']]
    ]

    # Save as text file
    with open(os.path.join(result_path, 'classification_report.txt'), 'w') as f:
        f.write(tabulate(metrics, headers=['Metric', 'Value'], tablefmt='grid'))
        f.write('\n\nDetailed Report:\n')
        f.write(classification_report(y_true, y_pred))

    # Save as image
    plt.figure(figsize=(8, 3))
    plt.axis('off')
    plt.table(cellText=metrics,
              colLabels=['Metric', 'Value'],
              loc='center',
              cellLoc='center')
    plt.savefig(os.path.join(result_path, 'metrics_table.png'),
                bbox_inches='tight', pad_inches=0.5)
    plt.close()


def run():
    # 配置GPU内存增长
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    K.set_image_data_format('channels_last')

    all_result_file = open(os.path.join(result_path, 'all_result.txt'), "w")
    all_result_file.close()

    # Load all data
    train_specInput, train_tempInput, train_label = load_data(
        train_specInput_root_path, train_tempInput_root_path, train_label_root_path, train=True)
    test_specInput, test_tempInput, test_label = load_data(
        test_specInput_root_path, test_tempInput_root_path, test_label_root_path, train=False)

    # Print shapes for debugging
    print("Train spec input shape:", train_specInput.shape)
    print("Train temp input shape:", train_tempInput.shape)
    print("Test spec input shape:", test_specInput.shape)
    print("Test temp input shape:", test_tempInput.shape)

    # Convert labels to categorical
    train_label = to_categorical(train_label, num_classes=nb_class)
    test_label = to_categorical(test_label, num_classes=nb_class)

    # Create and train model
    model = sst_model.sst_emotionnet(
        input_width=input_width,
        specInput_length=specInput_length,
        temInput_length=temInput_length,
        depth_spec=depth_spec,
        depth_tem=depth_tem,
        gr_spec=gr_spec,
        gr_tem=gr_tem,
        nb_dense_block=nb_dense_block,
        nb_class=nb_class)

    adam = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # 计算类别权重以处理类别不平衡
    from sklearn.utils.class_weight import compute_class_weight
    y_integers = np.argmax(train_label, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"类别权重: {class_weight_dict}")

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
    save_model = ModelCheckpoint(
        filepath=os.path.join(model_save_path, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True)

    history = model.fit(
        [train_specInput, train_tempInput],
        train_label,
        epochs=nbEpoch,
        batch_size=batch_size,
        validation_data=([test_specInput, test_tempInput], test_label),
        callbacks=[early_stopping, save_model],
        class_weight=class_weight_dict,
        verbose=1)

    # Evaluate best model (already restored by early stopping)
    loss, accuracy = model.evaluate([test_specInput, test_tempInput], test_label)

    print('\nTest loss:', loss)
    print('Test accuracy:', accuracy)

    # Save results
    with open(os.path.join(result_path, 'training_history.txt'), "w") as f:
        print(history.history, file=f)

    with open(os.path.join(result_path, 'all_result.txt'), "a") as all_result_file:
        print('Final Test Accuracy:', accuracy, file=all_result_file)

    keras.backend.clear_session

    y_pred = model.predict([test_specInput, test_tempInput])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_label, axis=1)

    # Plot training history
    plot_training_history(history, result_path)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred_classes, ['Healthy', 'Depressed'], result_path)

    # Save classification report
    save_classification_report(y_true, y_pred_classes, result_path)

    print("\nTraining visualization plots saved to:", result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument of running SST-EmotionNet.')
    parser.add_argument('-c', type=str, help='Config file path.', required=True)
    args = parser.parse_args()
    read_config(args.c)
    run()