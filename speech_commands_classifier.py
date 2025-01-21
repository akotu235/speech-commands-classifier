import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Input

# 1. Przygotowanie zbioru danych
# Pobieranie Speech Commands Dataset
dataset_name = "speech_commands"
data_dir = ".\\dataset"
data, info = tfds.load(dataset_name, data_dir=data_dir, with_info=True, as_supervised=True)

# Informacje o danych
print(info)

# Rozdzielenie zbiorów na trening, walidację i test
train_data = data['train']
validation_data = data['validation']
test_data = data['test']

# 2. Normalizacja danych i przygotowanie do modelu
def preprocess(audio, label):
    audio = tf.squeeze(audio)
    audio = tf.cast(audio, tf.float32) / 32768.0
    if tf.size(audio) < 16000:
        audio = tf.pad(audio, [[0, 16000 - tf.size(audio)]])
    else:
        audio = audio[:16000]
    audio = tf.reshape(audio, [16000, 1])
    return audio, label

train_data = train_data.map(preprocess).batch(16)
validation_data = validation_data.map(preprocess).batch(16)
test_data = test_data.map(preprocess).batch(16)

# 3. Definicja modelu
model = Sequential([
    Input(shape=(16000, 1)),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.4),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.4),
    Conv1D(256, 3, activation='relu'),
    MaxPooling1D(2),
    GlobalAveragePooling1D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(12, activation='softmax')
])

# 4. Kompilacja modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Trening modelu
history = model.fit(train_data, validation_data=validation_data, epochs=20)

# 6. Ewaluacja modelu
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")