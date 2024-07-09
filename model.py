import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pickle import dump
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    data = pd.read_pickle('gamelog_agg.pkl')

    x = data.iloc[:, 13:-1].values
    y = data.iloc[:, -1].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("클래스 매핑:", class_mapping)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # StandardScaler 객체 생성 및 데이터 스케일링
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # scaler 저장하기
    dump(scaler, open('scaler.pkl', 'wb'))

    # ADASYN 오버샘플링
    adasyn = ADASYN(random_state=42)
    x_resampled, y_resampled = adasyn.fit_resample(x_train, y_train)

    # 검증 데이터셋 분할
    x_resampled, x_val, y_resampled, y_val = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

    # CNN을 위한 입력 데이터 형태 변환
    x_resampled = x_resampled.reshape(x_resampled.shape[0], x_resampled.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # CNN 모델 정의
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_resampled.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=len(np.unique(y_resampled)), activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 체크포인트 설정
    filepath = 'models/cnn_model.{epoch:02d}.hdf5'
    modelckpt = ModelCheckpoint(filepath=filepath, save_best_only=True)

    # 모델 학습
    history = model.fit(x_resampled, y_resampled, epochs=100, batch_size=32, validation_data=(x_val, y_val), callbacks=[modelckpt])

    loss = history.history['loss']
    val_loss = history.history['val_loss']

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

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

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

    print(confusion_matrix(y_test, y_pred_classes))
    print(classification_report(y_test, y_pred_classes))
