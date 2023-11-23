import tensorflow as tf
from keras.models import load_model
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import train_model_GRU, train_model_LSTM
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.preprocessors import WavReader
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric


vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ "
max_spectrogram_length = 2361
input_shape = [2361, 193]
model = train_model_LSTM(
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
model.load_weights('model/best_model_LSTM.h5')
# model = load_model('Model/model_GRU.h5',compile=False)
df = pd.read_csv("vivos/vivos/test/prompts.csv")
dataset = [[f"vivos/vivos/test/waves/{file}.wav", label] for file, label in df.values.tolist()]
accum_cer, accum_wer = [], []
for wav_path, label in tqdm(dataset):
        
    spectrogram = WavReader.get_spectrogram(wav_path, frame_length=256, frame_step=160, fft_length=384)

    padded_spectrogram = np.pad(spectrogram, ((0, max_spectrogram_length - spectrogram.shape[0]),(0,0)), mode="constant", constant_values=0)
    reshaped_input_data = np.expand_dims(padded_spectrogram, axis=0)

    text = model.predict(reshaped_input_data)
    pred = ctc_decoder(text, vocab)[0]
    print("Predict: ", pred)
    print("Label: ", label)

    cer = get_cer(pred, label)
    wer = get_wer(pred, label)
    print("CER: ", cer)
    print("WER: ", wer)

print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")