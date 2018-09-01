import librosa
import numpy as np

def rhythmic_extraction(y,sr):
    '''this funtion return beats and tempo. Beats is a array, tempo is number (beat per minute)'''
    tempo, beats = librosa.beat.beat_track(y, sr=sr, start_bpm=40, units='time', hop_length=512)
    times = librosa.frames_to_time(beats, sr=sr)
    y_beat_times = librosa.clicks(times=times, sr=sr)
    return y_beat_times, tempo

def timbral_extraction(y,sr):
    '''return timbral matrix including spectral centroid, spectral_rolloff, spectral flux, rms,
    zero crossing rate and mfcc matrix. You get spectral centroid by instruction:
       spectral_centroid = timbral_feature[:,0]
       spectral_flux=timbral_feature[:,1]. From index 5-17 is mfcc_matrix
    '''
    'Spectral Centroid'
    spectral_centroid = librosa.feature.spectral_centroid(y, sr=sr, S=None,n_fft=2048, hop_length=512, freq=None)
    'Spectral roll off'
    spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=sr, S=None, n_fft=2048, hop_length=512, freq=None, roll_percent = 0.85)
    'Spectral Flux'
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    'Low Energy'
    rms = librosa.feature.rmse(y=y)
    'Zero Crossing Rate'
    zr_rate = librosa.feature.zero_crossing_rate(y, frame_length=512)
    'MFCC-Mel Frequency Content Coefficient'
    mfcc_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, dct_type=2)
    timbral_matrix = np.vstack((spectral_centroid, spectral_flux, spectral_rolloff, rms, zr_rate, mfcc_matrix))
    return timbral_matrix.T

if __name__ == '__main__':
    filename = 'C:/Users/MINH QUAN/PycharmProjects/MusicAnalysis/test5.mp3'
    y, sr = librosa.core.load(filename)
    timbral_feature= timbral_extraction(y,sr)
    beats,tempo = rhythmic_extraction(y,sr)



