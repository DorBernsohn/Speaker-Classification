import random
import librosa
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def batch_generator(data, batch_size, label_binarizer, max_length):
    while 1:
        random.shuffle(data)
        X, y = [], []
        for i in range(batch_size):
            wav = data[i]
            wave, sr = librosa.load(wav, mono=True)
            label = wav.split('/')[-1].split('_')[0]
            y.append(one_hot_encode([label], label_binarizer)[0])
            mfcc = librosa.feature.mfcc(wave, sr)
            mfcc = np.pad(mfcc, ((0,0), (0, max_length - len(mfcc[0]))), mode='constant', constant_values=0)
            X.append(np.array(mfcc))
        yield np.array(X), np.array(y)

def split_wav_files(files):
    X_train, X_val = train_test_split(files, test_size=0.2, random_state=42)
    print('# Training examples: {}'.format(len(X_train)))
    print('# Validation examples: {}'.format(len(X_val)))

    return X_train, X_val

def one_hot_encode(x, label_binarizer):
    return label_binarizer.transform(x)