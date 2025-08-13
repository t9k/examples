import tensorflow as tf

# 定义 CNN 模型
class SimpleCNN(tf.keras.Model):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10)  # 假设分类 10 类

    def call(self, inputs, training=False):
        x = self.conv1(inputs)   # [B, 28, 28, 16]
        x = self.pool1(x)        # [B, 14, 14, 16]
        x = self.conv2(x)        # [B, 14, 14, 32]
        x = self.pool2(x)        # [B, 7, 7, 32]
        x = self.flatten(x)      # [B, 7*7*32]
        x = self.fc1(x)
        return self.fc2(x)

if __name__ == "__main__":
    # 创建模型
    model = SimpleCNN()

    # 先构建模型（给定输入尺寸），以便初始化权重
    dummy_input = tf.random.normal([1, 28, 28, 1])
    _ = model(dummy_input, training=False)

    # 保存为 SavedModel 格式
    tf.saved_model.save(model, "simple_cnn_tf_savedmodel")
    print("模型已保存到 simple_cnn_tf_savedmodel 文件夹")
