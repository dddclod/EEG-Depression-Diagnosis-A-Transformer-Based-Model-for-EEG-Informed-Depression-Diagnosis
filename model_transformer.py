# 导入必要的库和模块
from __future__ import absolute_import, division, print_function
import keras
import keras.backend as K
from keras.layers import Input, Conv3D, Activation, Dense, Dropout, Lambda, concatenate, BatchNormalization
from keras.layers import AveragePooling3D, GlobalAveragePooling3D, MaxPooling3D
from keras.layers import MultiHeadAttention, LayerNormalization, Add, Reshape, Flatten
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf


def channel_wise_mean(x):
    """计算通道维度的均值。"""
    mid = tf.reduce_mean(x, axis=-1)
    return mid


def channel_wise_mean_output_shape(input_shape):
    """计算channel_wise_mean的输出形状"""
    return input_shape[:-1]


# ==================== Transformer 模块 ====================

def transformer_encoder_block(x, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
    """
    Transformer编码器块
    
    参数:
    - x: 输入张量 (batch, seq_len, embed_dim)
    - embed_dim: 嵌入维度
    - num_heads: 多头注意力的头数
    - ff_dim: 前馈网络的隐藏层维度
    - dropout_rate: Dropout率
    """
    # 多头自注意力
    attn_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=embed_dim // num_heads,
        dropout=dropout_rate
    )(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([x, attn_output]))
    
    # 前馈网络
    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ffn]))
    
    return out2


def spatial_embedding_layer(x, embed_dim):
    """
    将空间特征(20x20x5)转换为序列嵌入
    
    参数:
    - x: 输入张量 (batch, 20, 20, 5, 1)
    - embed_dim: 嵌入维度
    
    返回:
    - 序列张量 (batch, seq_len, embed_dim)
    """
    # 移除最后一个维度
    x = Reshape((20, 20, 5))(x)  # (batch, 20, 20, 5)
    
    # 重塑为序列: 将空间维度展平
    x = Reshape((400, 5))(x)  # (batch, 400, 5)
    
    # 线性投影到嵌入维度
    x = Dense(embed_dim, name='spectral_embedding')(x)  # (batch, 400, embed_dim)
    
    return x


def positional_encoding(seq_len, embed_dim):
    """
    生成位置编码
    
    参数:
    - seq_len: 序列长度
    - embed_dim: 嵌入维度
    """
    position = tf.range(start=0, limit=seq_len, delta=1, dtype=tf.float32)
    position = tf.expand_dims(position, 1)
    
    div_term = tf.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) * 
                     -(tf.math.log(10000.0) / embed_dim))
    
    pos_encoding = tf.zeros((seq_len, embed_dim))
    pos_encoding = tf.tensor_scatter_nd_update(
        pos_encoding,
        tf.stack([tf.range(seq_len), tf.zeros(seq_len, dtype=tf.int32)], axis=1),
        tf.sin(position * div_term)[:, 0]
    )
    
    # 简化版本：使用可学习的位置编码
    return None  # 将在模型中使用可学习的位置嵌入


class PositionalEmbedding(keras.layers.Layer):
    """可学习的位置嵌入层"""
    def __init__(self, seq_len, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pos_emb = self.add_weight(
            name='pos_emb',
            shape=(1, seq_len, embed_dim),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, x):
        return x + self.pos_emb


def transformer_spectral_stream(input_tensor, embed_dim=64, num_heads=4, 
                               num_layers=2, ff_dim=128, dropout_rate=0.1):
    """
    Transformer频谱流
    
    参数:
    - input_tensor: 输入张量 (batch, 20, 20, 5, 1)
    - embed_dim: 嵌入维度
    - num_heads: 注意力头数
    - num_layers: Transformer层数
    - ff_dim: 前馈网络维度
    - dropout_rate: Dropout率
    
    返回:
    - 特征向量
    """
    # 空间嵌入
    x = spatial_embedding_layer(input_tensor, embed_dim)  # (batch, 400, embed_dim)
    
    # 添加位置编码
    seq_len = 400  # 20 * 20
    x = PositionalEmbedding(seq_len, embed_dim)(x)
    x = Dropout(dropout_rate)(x)
    
    # 堆叠Transformer编码器块
    for i in range(num_layers):
        x = transformer_encoder_block(x, embed_dim, num_heads, ff_dim, dropout_rate)
    
    # 全局平均池化 - 使用Keras层而不是Lambda
    x = keras.layers.GlobalAveragePooling1D()(x)  # (batch, embed_dim)
    
    return x


# ==================== DenseNet 时空流（保持不变）====================

def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    """3D卷积块"""
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    
    if bottleneck:
        inter_channel = nb_filter * 4
        x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_normal', 
                   padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
    
    x = Conv3D(nb_filter, (3, 3, 1), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)
    x = Conv3D(nb_filter, (1, 1, 3), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, 
                  dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    """密集块"""
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    for i in range(nb_layers):
        cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x = concatenate([x, cb], axis=concat_axis)
        if grow_nb_filters:
            nb_filter += growth_rate
    
    return x, nb_filter


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    """过渡块"""
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer='he_normal',
               padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    
    return x


class ChannelMeanLayer(keras.layers.Layer):
    """自定义层：计算通道维度的均值"""
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=False)
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class CrossAttentionFusion(keras.layers.Layer):
    """交叉注意力融合层：让两个流相互关注"""
    def __init__(self, units=64, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        
        # 投影层
        self.query_proj = Dense(units)
        self.key_proj = Dense(units)
        self.value_proj = Dense(units)
        
        # 注意力层
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=units//num_heads)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        
        # 前馈网络
        self.ffn = keras.Sequential([
            Dense(units * 2, activation='relu'),
            Dropout(0.1),
            Dense(units)
        ])
    
    def call(self, query_features, key_value_features, training=False):
        """
        query_features: 查询特征 (batch, dim1)
        key_value_features: 键值特征 (batch, dim2)
        """
        # 扩展维度以适配MultiHeadAttention
        query = tf.expand_dims(query_features, 1)  # (batch, 1, dim1)
        key_value = tf.expand_dims(key_value_features, 1)  # (batch, 1, dim2)
        
        # 投影到相同维度
        query = self.query_proj(query)
        key = self.key_proj(key_value)
        value = self.value_proj(key_value)
        
        # 交叉注意力
        attn_output = self.attention(query, key, value, training=training)
        attn_output = self.norm1(query + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(attn_output, training=training)
        output = self.norm2(attn_output + ffn_output)
        
        # 移除序列维度
        output = tf.squeeze(output, axis=1)  # (batch, units)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads
        })
        return config


def Attention_block(input_tensor, spatial_attention=True, temporal_attention=True):
    """注意力块"""
    tem = input_tensor
    input_shape = input_tensor.shape
    h, w, d, c = input_shape[1], input_shape[2], input_shape[3], input_shape[4]
    
    # 使用自定义层代替Lambda
    x = ChannelMeanLayer()(input_tensor)
    x = Reshape([h, w, d, 1])(x)
    
    if spatial_attention:
        spatial = AveragePooling3D(pool_size=[1, 1, d])(x)
        spatial = Flatten()(spatial)
        spatial = Dense(h * w)(spatial)
        spatial = Activation('sigmoid')(spatial)
        spatial = Reshape([h, w, 1, 1])(spatial)
        tem = keras.layers.multiply([input_tensor, spatial])
    
    if temporal_attention:
        temporal = AveragePooling3D(pool_size=[h, w, 1])(x)
        temporal = Flatten()(temporal)
        temporal = Dense(d)(temporal)
        temporal = Activation('sigmoid')(temporal)
        temporal = Reshape([1, 1, d, 1])(temporal)
        tem = keras.layers.multiply([temporal, tem])
    
    return tem


def densenet_temporal_stream(input_tensor, depth=40, nb_dense_block=2, growth_rate=12,
                             nb_filter=24, reduction=0.5, dropout_rate=0.0, 
                             weight_decay=1e-4, subsample_initial_block=True):
    """
    DenseNet时空流
    
    参数:
    - input_tensor: 输入张量 (batch, 20, 20, 512, 1)
    - depth: 网络深度
    - nb_dense_block: 密集块数量
    - growth_rate: 增长率
    - nb_filter: 初始滤波器数量
    - reduction: 压缩因子
    - dropout_rate: Dropout率
    - weight_decay: 权重衰减
    - subsample_initial_block: 是否降采样
    
    返回:
    - 特征向量
    """
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    # 计算每个密集块的层数
    if depth == 40:
        nb_layers = [6, 6]
    else:
        count = int((depth - 4) / 3)
        nb_layers = [count] * nb_dense_block
    
    compression = 1.0 - reduction
    
    # 初始卷积
    if subsample_initial_block:
        initial_kernel = (5, 5, 3)
        initial_strides = (2, 2, 1)
    else:
        initial_kernel = (3, 3, 1)
        initial_strides = (1, 1, 1)
    
    x = Conv3D(nb_filter, initial_kernel, kernel_initializer='he_normal',
               padding='same', strides=initial_strides, use_bias=False,
               kernel_regularizer=l2(weight_decay))(input_tensor)
    
    if subsample_initial_block:
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    
    # 密集块和过渡块
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate,
                                     bottleneck=True, dropout_rate=dropout_rate,
                                     weight_decay=weight_decay)
        x = __transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)
        x = Attention_block(x, spatial_attention=True, temporal_attention=True)
    
    # 最后一个密集块
    x, nb_filter = __dense_block(x, nb_layers[-1], nb_filter, growth_rate,
                                 bottleneck=True, dropout_rate=dropout_rate,
                                 weight_decay=weight_decay)
    
    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)
    
    return x


# ==================== 混合模型：Transformer + DenseNet ====================

def transformer_densenet_hybrid(input_width=20, specInput_length=5, temInput_length=512,
                               # Transformer参数
                               transformer_embed_dim=64, transformer_heads=4,
                               transformer_layers=2, transformer_ff_dim=128,
                               # DenseNet参数
                               depth_tem=40, gr_tem=12, nb_dense_block=2,
                               # 通用参数
                               nb_class=2, dropout_rate=0.1):
    """
    混合模型：Transformer频谱流 + DenseNet时空流
    
    参数:
    - input_width: 输入空间维度
    - specInput_length: 频谱维度长度
    - temInput_length: 时间维度长度
    - transformer_embed_dim: Transformer嵌入维度
    - transformer_heads: Transformer注意力头数
    - transformer_layers: Transformer层数
    - transformer_ff_dim: Transformer前馈网络维度
    - depth_tem: DenseNet深度
    - gr_tem: DenseNet增长率
    - nb_dense_block: 密集块数量
    - nb_class: 分类类别数
    - dropout_rate: Dropout率
    
    返回:
    - Keras模型
    """
    # 频谱输入 - Transformer流
    specInput = Input([input_width, input_width, specInput_length, 1], name='spectral_input')
    x_spec = transformer_spectral_stream(
        specInput,
        embed_dim=transformer_embed_dim,
        num_heads=transformer_heads,
        num_layers=transformer_layers,
        ff_dim=transformer_ff_dim,
        dropout_rate=dropout_rate
    )
    
    # 时空输入 - DenseNet流
    temInput = Input([input_width, input_width, temInput_length, 1], name='temporal_input')
    x_temp = densenet_temporal_stream(
        temInput,
        depth=depth_tem,
        nb_dense_block=nb_dense_block,
        growth_rate=gr_tem,
        reduction=0.5,
        dropout_rate=dropout_rate,
        subsample_initial_block=True
    )
    
    # 特征融合：简单拼接（无交叉注意力）
    y = concatenate([x_spec, x_temp], axis=-1, name='feature_fusion')
    
    # 分类头
    y = Dense(128, activation='relu', name='fc1')(y)
    y = Dropout(0.5)(y)
    y = Dense(64, activation='relu', name='fc2')(y)
    y = Dropout(0.5)(y)
    
    # 输出层
    if nb_class == 2:
        y = Dense(nb_class, activation='softmax', name='output')(y)
    else:
        y = Dense(nb_class, activation='softmax', name='output')(y)
    
    # 构建模型
    model = Model([specInput, temInput], y, name='Transformer_DenseNet_Hybrid')
    
    return model


# 为了兼容性，保留原函数名
def sst_emotionnet(input_width, specInput_length, temInput_length, depth_spec, depth_tem,
                   gr_spec, gr_tem, nb_dense_block, attention=True, 
                   spatial_attention=True, temporal_attention=True, nb_class=3):
    """
    兼容性包装函数，调用新的混合模型
    """
    return transformer_densenet_hybrid(
        input_width=input_width,
        specInput_length=specInput_length,
        temInput_length=temInput_length,
        transformer_embed_dim=64,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff_dim=128,
        depth_tem=depth_tem,
        gr_tem=gr_tem,
        nb_dense_block=nb_dense_block,
        nb_class=nb_class,
        dropout_rate=0.1
    )
