import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pickle import dump

# VAE 클래스 정의
class VariantionalAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, learning_rate=1e-4, batch_size=64, n_z=16):
        super(VariantionalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def build_encoder(self):
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        x = Dense(512, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        z_mean = Dense(self.n_z)(x)
        z_log_var = Dense(self.n_z)(x)
        return tf.keras.Model(inputs, [z_mean, z_log_var])

    def build_decoder(self):
        latent_inputs = tf.keras.layers.Input(shape=(self.n_z,))
        x = Dense(32, activation='relu')(latent_inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(self.input_dim, activation='sigmoid')(x)
        return tf.keras.Model(latent_inputs, outputs)

    def sample(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_loss(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sample(z_mean, z_log_var)
        x_hat = self.decoder(z)
        recon_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_hat))
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        return recon_loss + kl_loss

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

if __name__ == "__main__":
    # 데이터 로드 및 전처리
    data = pd.read_pickle('gamelog_agg.pkl')
    x = data.iloc[:, 13:-1].values
    y = data.iloc[:, -1].values

    # 라벨 인코딩
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("클래스 매핑:", class_mapping)

    # 데이터 분할 (학습, 테스트)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    dump(scaler, open('scaler.pkl', 'wb'))

    # 최대 수 클래스 찾기
    majority_class = np.argmax(np.bincount(y_train))

    # 모든 소수 클래스 데이터 오버샘플링
    x_train_augmented = []
    y_train_augmented = []

    for cls in np.unique(y_train):
        if cls == majority_class:
            x_train_augmented.append(x_train[y_train == cls])
            y_train_augmented.append(y_train[y_train == cls])
            continue

        x_train_minority = x_train[y_train == cls]
        input_dim = x_train.shape[1]
        vae = VariantionalAutoencoder(input_dim=input_dim)
        for epoch in range(100):
            np.random.shuffle(x_train_minority)
            for i in range(0, len(x_train_minority), vae.batch_size):
                batch = x_train_minority[i:i + vae.batch_size]
                vae.train_step(batch)

        n_samples_to_generate = len(x_train[y_train == majority_class]) - len(x_train_minority)
        z_sample = np.random.normal(size=(n_samples_to_generate, vae.n_z))
        generated_samples = vae.decoder(z_sample)

        x_train_augmented.append(np.vstack([x_train_minority, generated_samples]))
        y_train_augmented.append(np.full(len(x_train_minority) + n_samples_to_generate, cls))

    x_train_augmented = np.vstack(x_train_augmented)
    y_train_augmented = np.hstack(y_train_augmented)

    # 데이터 분할 (학습, 검증)
    x_train_final, x_val, y_train_final, y_val = train_test_split(x_train_augmented, y_train_augmented, test_size=0.25, random_state=42)

    # 원-핫 인코딩
    y_train_final = tf.keras.utils.to_categorical(y_train_final, num_classes=len(class_mapping))
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(class_mapping))
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(class_mapping))

    # 모델 체크포인트 설정 및 조기 종료
    filepath = 'models/mlb_model.{epoch}.keras'
    modelckpt = ModelCheckpoint(filepath=filepath, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)

    # Sequential 모델 생성
    model = Sequential()
    model.add(Dense(units=512, kernel_regularizer=L2(1e-4), input_shape=(np.shape(x_train_final)[1],)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=512, kernel_regularizer=L2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=len(class_mapping), activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    # 모델 학습
    history = model.fit(x_train_final, y_train_final, epochs=100, batch_size=32, validation_data=(x_val, y_val), callbacks=[modelckpt, earlystopping])

    # 학습 곡선 시각화
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_learning_curve.png', dpi=300)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_learning_curve.png', dpi=300)

    # 모델 평가
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

    # 예측 및 성능 평가
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes))
    print(classification_report(np.argmax(y_test, axis=1), y_pred_classes))
