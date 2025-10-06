# 导入必要的库和模块
from __future__ import absolute_import, division, print_function
import keras
import keras.backend as K
from keras.layers import Input, Conv3D, Activation, Dense, Dropout, Lambda, concatenate, BatchNormalization
from keras.layers import AveragePooling3D, GlobalAveragePooling3D, MaxPooling3D
from keras.models import Model
from keras.regularizers import l2

"""
参数说明：
- input_width和input_height：输入数据的空间维度（宽度和高度）。
- specInput_length和temInput_length：频谱和时间维度的长度。
- depth_spec和depth_tem：频谱流和时间流的深度（层数）。
- gr_spec和gr_tem：频谱流和时间流的增长率（每层增加的通道数）。
- nb_dense_block：密集块的数量。
- attention：是否启用注意力机制。
- spatial_attention和temporal_attention：是否启用空间和时间注意力。
- nb_class：分类的类别数。
"""


def DenseNet(input_shape=None, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
             bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
             include_top=True, weights=None, input_tensor=None, classes=10, activation='softmax'):
    """
    构建3D DenseNet模型。

    参数：
    - input_shape: 输入张量的形状。
    - depth: 网络的总深度。
    - nb_dense_block: 密集块的数量。
    - growth_rate: 每层的增长率（通道数增加量）。
    - nb_filter: 初始卷积核数量，若为-1则自动计算。
    - nb_layers_per_block: 每个密集块的层数，若为-1则自动分配。
    - bottleneck: 是否使用瓶颈层。
    - reduction: 压缩因子（用于过渡块）。
    - dropout_rate: Dropout率。
    - weight_decay: 权重衰减系数。
    - subsample_initial_block: 是否对初始块进行降采样。
    - include_top: 是否包含顶部分类层。
    - classes: 分类类别数。
    - activation: 输出激活函数（'softmax'或'sigmoid'）。

    返回：
    - Model: 构建好的Keras模型。
    """
    # 检查激活函数是否合法
    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')
    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    # 定义输入层
    img_input = Input(tensor=input_tensor, shape=input_shape)

    # 创建DenseNet网络
    x = __create_dense_net(classes, img_input, include_top, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck, reduction,
                           dropout_rate, weight_decay, subsample_initial_block, activation)

    # 构建并返回模型
    model = Model(img_input, x, name='densenet')
    return model


def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    """
    3D卷积块，包含批归一化、激活函数和卷积操作。

    参数：
    - ip: 输入张量。
    - nb_filter: 卷积核数量。
    - bottleneck: 是否使用瓶颈层。
    - dropout_rate: Dropout率。
    - weight_decay: 权重衰减系数。

    返回：
    - 输出张量。
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # 批归一化和ReLU激活
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)

    # 使用瓶颈层（1x1x1卷积降维）
    if bottleneck:
        inter_channel = nb_filter * 4
        x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    # 空间卷积（3x3x1）和时间卷积（1x1x3）
    x = Conv3D(nb_filter, (3, 3, 1), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)
    x = Conv3D(nb_filter, (1, 1, 3), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)

    # 可选Dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  grow_nb_filters=True, return_concat_list=False):
    """
    密集块（Dense Block），包含多个卷积块，每层的输入是前面所有层的拼接。

    参数：
    - x: 输入张量。
    - nb_layers: 当前块的层数。
    - nb_filter: 当前通道数。
    - growth_rate: 每层增加的通道数。
    - bottleneck: 是否使用瓶颈层。
    - dropout_rate: Dropout率。
    - weight_decay: 权重衰减系数。
    - grow_nb_filters: 是否动态增加通道数。
    - return_concat_list: 是否返回所有中间层的特征图。

    返回：
    - 输出张量和更新后的通道数。
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x_list = [x]  # 存储所有层的输出

    for i in range(nb_layers):
        # 构建卷积块
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)

        # 将当前层的输出与前面所有层的输出拼接
        x = concatenate([x, cb], axis=concat_axis)

        # 动态增加通道数
        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    """
    过渡块（Transition Block），用于压缩通道数和降采样。

    参数：
    - ip: 输入张量。
    - nb_filter: 当前通道数。
    - compression: 压缩因子（0-1）。
    - weight_decay: 权重衰减系数。

    返回：
    - 输出张量。
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # 批归一化和ReLU激活
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)

    # 1x1x1卷积压缩通道数
    x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)

    # 平均池化降采样
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    return x


def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, activation='softmax', attention=True, spatial_attention=True,
                       temporal_attention=True):
    """
    创建3D DenseNet网络。

    参数：
    - nb_classes: 分类类别数。
    - img_input: 输入张量。
    - include_top: 是否包含顶部分类层。
    - depth: 网络总深度。
    - nb_dense_block: 密集块数量。
    - growth_rate: 每层增长率。
    - nb_filter: 初始卷积核数量。
    - nb_layers_per_block: 每个密集块的层数。
    - bottleneck: 是否使用瓶颈层。
    - reduction: 压缩因子。
    - dropout_rate: Dropout率。
    - weight_decay: 权重衰减系数。
    - subsample_initial_block: 是否对初始块降采样。
    - activation: 输出激活函数。
    - attention: 是否启用注意力机制。
    - spatial_attention: 是否启用空间注意力。
    - temporal_attention: 是否启用时间注意力。

    返回：
    - 输出张量。
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # 检查压缩因子是否合法
    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # 计算每个密集块的层数
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
        assert len(nb_layers) == (
            nb_dense_block), 'If list, nb_layer is used as provided. Note that list size must be (nb_dense_block)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
            count = int((depth - 4) / 3)
            if bottleneck:
                count = count // 2
            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    # 计算初始卷积核数量
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # 计算压缩因子
    compression = 1.0 - reduction

    # 初始卷积
    if subsample_initial_block:
        initial_kernel = (5, 5, 3)
        initial_strides = (2, 2, 1)
    else:
        initial_kernel = (3, 3, 1)
        initial_strides = (1, 1, 1)

    x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    # 初始块降采样
    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    # 添加密集块和过渡块
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # 添加过渡块
        x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
        # 添加注意力块
        if attention:
            x = Attention_block(x, spatial_attention=spatial_attention, temporal_attention=temporal_attention)

    # 最后一个密集块（不接过渡块）
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

    # 批归一化和ReLU激活
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)

    # 顶部分类层
    if include_top:
        x = Dense(nb_classes, activation=activation)(x)

    return x


def channel_wise_mean(x):
    """计算通道维度的均值。"""
    import tensorflow as tf
    mid = tf.reduce_mean(x, axis=-1)
    return mid

def channel_wise_mean_output_shape(input_shape):
    """计算channel_wise_mean的输出形状"""
    return input_shape[:-1]


def Attention_block(input_tensor, spatial_attention=True, temporal_attention=True):
    """
    注意力块（Attention Block），包含空间注意力和时间注意力。

    参数：
    - input_tensor: 输入张量。
    - spatial_attention: 是否启用空间注意力。
    - temporal_attention: 是否启用时间注意力。

    返回：
    - 输出张量。
    """
    tem = input_tensor
    # 获取输入形状
    input_shape = input_tensor.shape
    h, w, d, c = input_shape[1], input_shape[2], input_shape[3], input_shape[4]
    
    # 计算通道均值并重塑
    x = Lambda(channel_wise_mean, output_shape=channel_wise_mean_output_shape)(input_tensor)
    x = keras.layers.Reshape([h, w, d, 1])(x)

    # 空间注意力
    if spatial_attention:
        spatial = AveragePooling3D(pool_size=[1, 1, d])(x)
        spatial = keras.layers.Flatten()(spatial)
        spatial = Dense(h * w)(spatial)
        spatial = Activation('sigmoid')(spatial)
        spatial = keras.layers.Reshape([h, w, 1, 1])(spatial)
        tem = keras.layers.multiply([input_tensor, spatial])

    # 时间注意力
    if temporal_attention:
        temporal = AveragePooling3D(pool_size=[h, w, 1])(x)
        temporal = keras.layers.Flatten()(temporal)
        temporal = Dense(d)(temporal)
        temporal = Activation('sigmoid')(temporal)
        temporal = keras.layers.Reshape([1, 1, d, 1])(temporal)
        tem = keras.layers.multiply([temporal, tem])

    return tem


def sst_emotionnet(input_width, specInput_length, temInput_length, depth_spec, depth_tem, gr_spec, gr_tem,
                   nb_dense_block, attention=True, spatial_attention=True, temporal_attention=True, nb_class=3):
    """
    构建双流3D DenseNet模型（空间-频谱流和空间-时间流）。

    参数：
    - input_width: 输入数据的空间宽度。
    - specInput_length: 频谱维度的长度。
    - temInput_length: 时间维度的长度。
    - depth_spec: 频谱流的深度。
    - depth_tem: 时间流的深度。
    - gr_spec: 频谱流的增长率。
    - gr_tem: 时间流的增长率。
    - nb_dense_block: 密集块的数量。
    - attention: 是否启用注意力机制。
    - spatial_attention: 是否启用空间注意力。
    - temporal_attention: 是否启用时间注意力。
    - nb_class: 分类类别数。

    返回：
    - Model: 构建好的双流Keras模型。
    """
    # 空间-频谱流
    specInput = Input([input_width, input_width, specInput_length, 1])
    x_s = __create_dense_net(img_input=specInput, depth=depth_spec, nb_dense_block=nb_dense_block,
                             growth_rate=gr_spec, nb_classes=nb_class, reduction=0.5, bottleneck=True,
                             include_top=False, attention=attention, spatial_attention=spatial_attention,
                             temporal_attention=temporal_attention)

    # 空间-时间流
    temInput = Input([input_width, input_width, temInput_length, 1])
    x_t = __create_dense_net(img_input=temInput, depth=depth_tem, nb_dense_block=nb_dense_block,
                             growth_rate=gr_tem, nb_classes=nb_class, bottleneck=True, include_top=False,
                             subsample_initial_block=True, attention=attention)

    # 双流特征融合
    y = keras.layers.concatenate([x_s, x_t], axis=-1)
    y = keras.layers.Dense(128, activation='relu')(y)
    y = keras.layers.Dropout(0.5)(y)
    y = keras.layers.Dense(64, activation='relu')(y)
    y = keras.layers.Dropout(0.5)(y)

    # 分类输出
    if nb_class == 2:
        y = keras.layers.Dense(nb_class, activation='sigmoid')(y)
    else:
        y = keras.layers.Dense(nb_class, activation='softmax')(y)

    # 构建并返回模型
    model = Model([specInput, temInput], y)
    return model
