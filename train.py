import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import train_model_GRU, train_model_LSTM
from mltu.preprocessors import WavReader
from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric


metadata_df = pd.read_csv('vivos/vivos/train/prompts.csv')
dataset = [[f"vivos/vivos/train/waves/{file}.wav", label] for file, label in metadata_df.values.tolist()]

vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ "
max_text_length, max_spectrogram_length = 0, 0
for file_path, label in tqdm(dataset):
    spectrogram = WavReader.get_spectrogram(file_path, frame_length=256, frame_step=160, fft_length=384)
    valid_label = [c for c in label.upper() if c in vocab]
    max_text_length = max(max_text_length, len(valid_label))
    max_spectrogram_length = max(max_spectrogram_length, spectrogram.shape[0])
    input_shape = [max_spectrogram_length, spectrogram.shape[1]]
max_spectrogram_length = max_spectrogram_length
max_text_length = max_text_length

data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=32,
    data_preprocessors=[
        WavReader(frame_length=256, frame_step=160, fft_length=384),
        ],
    transformers=[
        SpectrogramPadding(max_spectrogram_length=max_spectrogram_length, padding_value=0),
        LabelIndexer(vocab),
        LabelPadding(max_word_length=max_text_length, padding_value=len(vocab)),
        ],
)

train_data_provider, val_data_provider = data_provider.split(split = 0.8)

model = train_model_GRU(
    input_dim = input_shape,
    output_dim = len(vocab),
    dropout=0.5
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
    loss=CTCloss(), 
    metrics=[
        CERMetric(vocabulary=vocab),
        WERMetric(vocabulary=vocab)
        ],
    run_eagerly=False
)

model.summary(line_length=110)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model/best_model_GRU.h5', monitor='val_loss', 
							 verbose=1, save_best_only=True, 
							 mode='auto')
history = model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=40,
    callbacks=checkpoint
)

fig, ax = plt.subplots()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()
plt.savefig('eval/loss_GRU.png')

fig, ax = plt.subplots()
plt.plot(history.history['WER'])
plt.plot(history.history['val_WER'])
plt.title('WER')
plt.ylabel('WER')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.show()
plt.savefig('eval/WER_GRU.png')

model.save('model/model_GRU.h5')

