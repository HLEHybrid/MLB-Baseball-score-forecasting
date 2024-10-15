import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from pickle import dump

if __name__ == "__main__":
    # 데이터 로드 및 전처리
    data = pd.read_pickle('gamelog_agg.pkl')
    X = data.iloc[:, 13:-1].values
    Y = data.iloc[:, -1].values

    # 라벨 인코딩
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    class_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("클래스 매핑:", class_mapping)

    # 데이터 분할 (학습, 검증, 테스트)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    dump(scaler, open('scaler.pkl', 'wb'))

    # 원-핫 인코딩
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=len(class_mapping))
    Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=len(class_mapping))
    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=len(class_mapping))

    # 모델 체크포인트 설정 및 조기 종료
    filepath = 'models/mlb_model_no_aug.{epoch}.keras'
    modelckpt = ModelCheckpoint(filepath=filepath, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    # Sequential 모델 생성
    model = Sequential()
    model.add(Dense(units=512, input_shape=(np.shape(X_train)[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=len(class_mapping), activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                  loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=0.5), metrics=['accuracy'])
    
    # 모델 학습
    history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val), callbacks=[modelckpt, earlystopping])

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
    plt.savefig('loss_learning_curve_no_aug.png', dpi=300)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_learning_curve_no_aug.png', dpi=300)

    # 모델 평가
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

    # 예측 및 성능 평가
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    print(confusion_matrix(np.argmax(Y_test, axis=1), Y_pred_classes))
    print(classification_report(np.argmax(Y_test, axis=1), Y_pred_classes))
