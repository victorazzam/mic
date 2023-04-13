# System
import os, sys, time, json, random, pickle, warnings, threading, traceback

# Audio
import wave, librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence

# ML helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ML classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# Live time counter https://stackoverflow.com/a/44381654
class Timer(threading.Thread):
    def __init__(self, message):
        super(Timer, self).__init__()
        self._stop_event = threading.Event()
        self._message = message
    def stop(self):
        self._stop_event.set()
    def stopped(self):
        return self._stop_event.is_set()
    def run(self):
        thread_start = time.time()
        while not self.stopped():
            print(f'\r{self._message} [{time.time() - thread_start:.1f}s]', end='')
            time.sleep(0.1)
        print()
    def __enter__(self):
        global threads
        threads.append(self)
        self.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.join()

def trim_sound(input_file, output_file, start_minute, end_minute):
    with wave.open(input_file, 'rb') as wav_in:
        params = wav_in.getparams()
        start_frame = int(start_minute * 60 * wav_in.getframerate())
        end_frame = int(end_minute * 60 * wav_in.getframerate())
        frames_to_read = min(end_frame - start_frame, wav_in.getnframes() - start_frame)
        wav_in.setpos(start_frame)
        audio_data = wav_in.readframes(frames_to_read)
        with wave.open(output_file, 'wb') as wav_out:
            wav_out.setparams(params)
            wav_out.writeframes(audio_data)

# Split by silence
def break_silence(input_file, output_folder, min_duration=1, silence_thresh=-70, keep_silence=100):
    audio = AudioSegment.from_wav(input_file)
    segments = split_on_silence(
        audio,
        min_silence_len = int(min_duration * 1000),
        silence_thresh = silence_thresh,
        keep_silence = keep_silence
    )
    for i, segment in enumerate(segments):
        output_file = os.path.join(output_folder, f'part-{i:03d}.wav')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        segment.export(output_file, format='wav')

# Prepare dataset
def load_data(data_folder):
    X, y = [], []
    for mic_folder in os.listdir(data_folder):
        path = os.path.join(data_folder, mic_folder)
        if os.path.isdir(path):
            label = int(path.split('-')[-1])
            for mic_part in os.listdir(path):
                X.append(os.path.join(path, mic_part))
                y.append(label)
    return X, np.array(y)

# Feature extraction: MFCC, ZCR, RMS
def extract_features(file):
    audio, sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc = np.mean(mfcc.T, axis=0)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr = np.mean(zcr.T, axis=0)
    rms = librosa.feature.rms(y=audio)
    rms = np.mean(rms.T, axis=0)
    return np.concatenate((mfcc, zcr, rms), axis=0)

# Show the confusion matrix in a nice-looking graph
def draw_cm(cm, f1, saveas, model_name, actual):

    # Normalize confusion matrix
    cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create plot
    fig, ax = plt.subplots()
    img = ax.imshow(cm_normalised, interpolation='nearest', cmap=plt.cm.OrRd)
    ax.figure.colorbar(img, ax=ax)

    # Set labels and ticks
    num_classes = len(np.unique(actual))
    ax.set(xticks = np.arange(num_classes),
           yticks = np.arange(num_classes),
           xticklabels = [mics[i] for i in range(num_classes)],
           yticklabels = [mics[i] for i in range(num_classes)],
           title = f'Confusion Matrix: {model_name}\nF1-score: {f1}',
           ylabel = 'True label',
           xlabel = 'Predicted label')

    # Rotate tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=36, ha='right', rotation_mode='anchor')

    # Loop over data dimensions and create text annotations
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center', fontsize=9,
                        color=('black', 'white')[cm_normalised[i, j] > 0.5])

    plt.tight_layout()
    plt.savefig(saveas, bbox_inches='tight')
    #plt.show()

# Consolidate the confusion matrices by summing them
def draw_all(matrices, actual):

    # Normalize the consolidated confusion matrix
    consolidated = np.sum(matrices, axis=0)
    consolidated = consolidated.astype('float') / consolidated.sum(axis=1)[:, np.newaxis]

    # Create the heatmap plot
    fig, ax = plt.subplots()
    img = ax.imshow(consolidated, interpolation='nearest', cmap=plt.cm.OrRd)
    ax.figure.colorbar(img, ax=ax)

    # Set labels and ticks
    num_classes = len(np.unique(actual))
    ax.set(xticks = np.arange(num_classes),
           yticks = np.arange(num_classes),
           xticklabels = [mics[i] for i in range(num_classes)],
           yticklabels = [mics[i] for i in range(num_classes)],
           title = 'Consolidated Confusion Matrix',
           ylabel = 'True label',
           xlabel = 'Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=36, ha='right', rotation_mode='anchor')

    # Loop over data dimensions and create text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(consolidated[i, j], '.2f'),
                    ha='center', va='center', fontsize=9,
                    color='white' if consolidated[i, j] > np.max(consolidated) else 'black')

    plt.tight_layout()
    plt.savefig('confusion-matrix.svg')
    #plt.show()

def mean_all(reports):
    combined = reports.groupby(reports.index).mean()
    print('\nCombined classification report:')
    print(combined.to_string(float_format=f'{{:.2f}}'.format))

def main():
    '''
    Fingerprinting microphones via acoustic features such as:

     - Mel-Frequency Cepstral Coefficient
     - Zero Cross Rate
     - Root-Mean-Square power
    
    Sounds are assumed to be stored as "P-N.wav"
     - P: any prefix (e.g. mic)
     - N: the labels as integers (e.g. 1, then 2, ...)
    '''

    # Load configuration parameters
    conf = {}
    with Timer('Loading configuration'):
        try:
            with open('parameters.json') as f:
                settings = json.load(f)
                for v in settings.values():
                    if type(v) == dict:
                        for k, w in v.items():
                            value = w if k in settings['evaluate'] else repr(w)
                            #print(f'{k} = {value}')
                            exec(f'{k} = {value}', globals(), globals())
        except:
            sys.exit('Bad configuration. Traceback:\n' + traceback.format_exc())
    for k, v in conf.items():
        globals()[k] = v

    with Timer('Trimming audio and splitting on silence'):
        if not os.path.isdir(destination):
            os.makedirs(destination)
        if not os.listdir(destination):
            for source in sources:
                source = root_path + source
                for mic in os.listdir(source):
                    if mic.endswith('.wav'):
                        mic_in = os.path.join(source, mic)
                        mic_out = os.path.join(destination, mic)
                        trim_sound(mic_in, mic_out, start_minute=time_start, end_minute=time_end)
                        mic_parts = mic_out.rsplit('.')[0]
                        break_silence(mic_out, mic_parts, min_duration=min_silence, silence_thresh=threshold, keep_silence=keep_silence)

    with Timer('Loading labels'):
        X, y = load_data(destination)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    if all(os.path.isfile(x) for x in (train_x_ft, train_y_lb, test_x_ft, test_y_lb)):
        with Timer('Loading sounds and labels'):
            with open(train_x_ft, 'rb') as f,   \
                 open(train_y_lb, 'rb') as g,   \
                 open(test_x_ft, 'rb') as h,    \
                 open(test_y_lb, 'rb') as i:
                X_train, X_test, y_train, y_test = pickle.load(f), pickle.load(g), pickle.load(h), pickle.load(i)
    else:
        with Timer('Extracting and normalising features'):
            with open(train_x_ft, 'wb') as f,   \
                 open(train_y_lb, 'wb') as g,   \
                 open(test_x_ft, 'wb') as h,    \
                 open(test_y_lb, 'wb') as i:

                # Normalise features
                scaler = StandardScaler()
                X_train = np.array(list(map(extract_features, X_train)))
                X_test = np.array(list(map(extract_features, X_test)))
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)

                # Apply PCA
                pca = PCA(n_components=8, random_state=random_state)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

                # Save features
                pickle.dump(X_train, f)
                pickle.dump(X_test, g)
                pickle.dump(y_train, h)
                pickle.dump(y_test, i)

    models = {
        'svc.model':  SVC(random_state=random_state),
        'mlp.model':  MLPClassifier(random_state=random_state),
        'knn.model':  KNeighborsClassifier(n_neighbors=len(set(test_y_lb))),
        'rf.model':   RandomForestClassifier(n_estimators=100, random_state=random_state),
        'gbc.model':  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state),
        'lr.model':   LogisticRegression(solver='lbfgs', random_state=random_state),
        'nb.model':   GaussianNB(),
        'dt.model':   DecisionTreeClassifier(random_state=random_state)
    }
    reports = None
    matrices = []

    for model_name, model in models.items():
        if os.path.isfile(model_name):
            with Timer('Loading saved model'), open(model_name, 'rb') as f:
                model = pickle.load(f)

        with Timer('Training and saving model'):
            model.fit(X_train, y_train)
            with open(model_name, 'wb') as f:
                pickle.dump(model, f)

        # Model evaluation
        predictions = model.predict(X_test)
        actual = y_test
        model_ = model_name.split('.')[0]

        # Classification report
        print(model)
        report = classification_report(actual, predictions, target_names=mics, output_dict=True, zero_division=0)
        #print(f'Report:\n{report}')
        df_report = pd.DataFrame(report).transpose()
        reports = df_report if reports is None else reports.append(df_report)

        # Confusion matrix
        cm = confusion_matrix(actual, predictions)
        cm_f1 = round(report['weighted avg']['f1-score'], 2)
        cm_name = 'confusion-matrix-' + model_ + '.svg'
        draw_cm(cm, cm_f1, cm_name, model_.upper(), actual)
        matrices.append(cm)

    # Final results
    mean_all(reports)
    draw_all(matrices, actual)

if __name__ == '__main__':
    try:
        print()
        threads = []
        main()
    except KeyboardInterrupt:
        print('Exiting.')
    except:
        traceback.print_exc()

    [thread.stop() for thread in threads]
    [thread.join() for thread in threads]
    print('Successfully stopped threads.')
