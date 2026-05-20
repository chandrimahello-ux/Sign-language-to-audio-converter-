import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('asl_data.csv')

X = df.drop('label', axis=1).values
y = df['label'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

SEQ_LEN = 20
X_seq, y_seq = [], []

for i in range(len(X) - SEQ_LEN):
    X_seq.append(X[i:i+SEQ_LEN])
    y_seq.append(y_cat[i+SEQ_LEN])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)


model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 42)),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(y_seq.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save('asl_lstm_model.keras')
print("[INFO] LSTM model saved")
