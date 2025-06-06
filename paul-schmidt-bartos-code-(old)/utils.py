import ctypes
import platform
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import librosa
from scipy.stats import wasserstein_distance
import numpy as np
import torch
from paule.util import VTL


# This should be done on all cp_deltas
# np.max(np.stack((np.abs(np.min(delta, axis=0)), np.max(delta, axis=0))), axis=0)
# np.max(np.stack((np.abs(np.min(cp_param, axis=0)), np.max(cp_param, axis=0))), axis=0)

# absolute value from max / min

# Vocal tract parameters: "HX HY JX JA LP LD VS VO TCX TCY TTX TTY TBX TBY TRX TRY TS1 TS2 TS3"
# Glottis parameters: "f0 pressure x_bottom x_top chink_area lag rel_amp double_pulsing pulse_skewness flutter aspiration_strength "

cp_theoretical_means = np.array([5.00000e-01, -4.75000e+00, -2.50000e-01, -3.50000e+00,
                                 0.00000e+00, 1.00000e+00, 5.00000e-01, 4.50000e-01,
                                 5.00000e-01, -1.00000e+00, 3.50000e+00, -2.50000e-01,
                                 5.00000e-01, 1.00000e+00, -1.00000e+00, -3.00000e+00,
                                 5.00000e-01, 5.00000e-01, 0.00000e+00, 3.20000e+02,
                                 1.00000e+04, 1.25000e-01, 1.25000e-01, 0.00000e+00,
                                 1.57075e+00, 0.00000e+00, 5.00000e-01, 0.00000e+00,
                                 5.00000e+01, -2.00000e+01])

cp_theoretical_stds = np.array([5.00000e-01, 1.25000e+00, 2.50000e-01, 3.50000e+00, 1.00000e+00,
                                3.00000e+00, 5.00000e-01, 5.50000e-01, 3.50000e+00, 2.00000e+00,
                                2.00000e+00, 2.75000e+00, 3.50000e+00, 4.00000e+00, 3.00000e+00,
                                3.00000e+00, 5.00000e-01, 5.00000e-01, 1.00000e+00, 2.80000e+02,
                                1.00000e+04, 1.75000e-01, 1.75000e-01, 2.50000e-01, 1.57075e+00,
                                1.00000e+00, 5.00000e-01, 5.00000e-01, 5.00000e+01, 2.00000e+01])


def librosa_melspec(wav, sample_rate):
    melspec = librosa.feature.melspectrogram(wav, n_fft=1024, hop_length=220, n_mels=60, sr=sample_rate, power=1.0,
                                             fmin=10, fmax=12000)
    melspec_db = librosa.amplitude_to_db(melspec, ref=0.15)
    return np.array(melspec_db.T, order='C')


def normalize_cp(cp):
    return (cp - cp_theoretical_means) / cp_theoretical_stds


def inv_normalize_cp(norm_cp):
    return cp_theoretical_stds * norm_cp + cp_theoretical_means


# -83.52182518111363
mel_mean_librosa = librosa_melspec(np.zeros(5000), 44100)[0, 0]
mel_std_librosa = abs(mel_mean_librosa)


def normalize_mel_librosa(mel):
    return (mel - mel_mean_librosa) / mel_std_librosa


def inv_normalize_mel_librosa(norm_mel):
    return mel_std_librosa * norm_mel + mel_mean_librosa


def read_cp(filename):
    with open(filename, 'rt') as cp_file:
        # skip first 6 lines
        for _ in range(6):
            cp_file.readline()
        glottis_model = cp_file.readline().strip()
        if glottis_model != 'Geometric glottis':
            print(glottis_model)
            raise ValueError(f'glottis model is not "Geometric glottis" in file {filename}')
        n_states = int(cp_file.readline().strip())
        cp_param = np.zeros((n_states, 19 + 11))
        for ii, line in enumerate(cp_file):
            kk = ii // 2
            if kk >= n_states:
                raise ValueError(f'more states saved in file {filename} than claimed in the beginning')
            # even numbers are glottis params
            elif ii % 2 == 0:
                glottis_param = line.strip()
                cp_param[kk, 19:30] = np.mat(glottis_param)
            # odd numbers are tract params
            elif ii % 2 == 1:
                tract_param = line.strip()
                cp_param[kk, 0:19] = np.mat(tract_param)
    return cp_param


def speak(cp_param):
    """
    Calls the vocal tract lab to synthesize an audio signal from the cp_param.

    Parameters
    ==========
    cp_param : np.array
        array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms)

    Returns
    =======
    (signal, sampling rate) : np.array, int
        returns the signal which is number of time steps in the cp_param array
        minus one times the time step length, i. e. ``(cp_param.shape[0] - 1) *
        110 / 44100``

    """
    # initialize vtl
    speaker_file_name = ctypes.c_char_p('JD2.speaker'.encode())

    failure = VTL.vtlInitialize(speaker_file_name)
    if failure != 0:
        raise ValueError('Error in vtlInitialize! Errorcode: %i' % failure)

    # get some constants
    audio_sampling_rate = ctypes.c_int(0)
    number_tube_sections = ctypes.c_int(0)
    number_vocal_tract_parameters = ctypes.c_int(0)
    number_glottis_parameters = ctypes.c_int(0)

    VTL.vtlGetConstants(ctypes.byref(audio_sampling_rate),
                        ctypes.byref(number_tube_sections),
                        ctypes.byref(number_vocal_tract_parameters),
                        ctypes.byref(number_glottis_parameters))

    assert audio_sampling_rate.value == 44100
    assert number_vocal_tract_parameters.value == 19
    assert number_glottis_parameters.value == 11

    number_frames = cp_param.shape[0]
    frame_steps = 110  # 2.5 ms
    # within first parenthesis type definition, second initialisation
    # 2000 samples more in the audio signal for safety
    audio = (ctypes.c_double * int((number_frames - 1) * frame_steps + 2000))()

    # init the arrays
    tract_params = (ctypes.c_double * (number_frames * number_vocal_tract_parameters.value))()
    glottis_params = (ctypes.c_double * (number_frames * number_glottis_parameters.value))()

    # fill in data
    tmp = np.ascontiguousarray(cp_param[:, 0:19])  # am einen stück speicher
    tmp.shape = (number_frames * 19,)
    tract_params[:] = tmp
    del tmp

    tmp = np.ascontiguousarray(cp_param[:, 19:30])
    tmp.shape = (number_frames * 11,)
    glottis_params[:] = tmp
    del tmp

    # Reset time-domain synthesis
    failure = VTL.vtlSynthesisReset()
    if failure != 0:
        raise ValueError(f'Error in vtlSynthesisReset! Errorcode: {failure}')

    # Set initial state of time-domain synthesis
    failure = VTL.vtlSynthesisAddTract(
        0,
        ctypes.byref(audio),  # output
        ctypes.byref(tract_params),  # input
        ctypes.byref(glottis_params))  # input
    if failure != 0:
        raise ValueError(f'Error in vtlSynthesisAddTract in setting initial state! Errorcode: {failure}')

    # Call the synthesis function. It may calculate a few seconds.
    failure = VTL.vtlSynthBlock(
        ctypes.byref(tract_params),  # input
        ctypes.byref(glottis_params),  # input
        number_frames,
        frame_steps,
        ctypes.byref(audio),  # output
        0)
    if failure != 0:
        raise ValueError('Error in vtlSynthBlock! Errorcode: %i' % failure)

    VTL.vtlClose()

    return (np.array(audio[:-2000]), 44100)


def audio_padding(sig, samplerate, winlen=0.010):
    """
    Pads the signal by half a window length on each side with zeros.

    Parameters
    ==========
    sig : np.array
        the audio signal
    samplerate : int
        sampling rate
    winlen : float
        the window size in seconds

    """
    pad = int(np.ceil(samplerate * winlen) / 2)
    z = np.zeros(pad)
    pad_signal = np.concatenate((z, sig, z))
    return pad_signal

def mel_to_sig(mel, mel_min=0.0):
    """
    creates audio from a normlised log mel spectrogram.

    Parameters
    ==========
    mel : np.array
        normalised log mel spectrogram (n_mel, seq_length)
    mel_min : float
        original min value (default: 0.0)

    Returns
    =======
    (sig, sampling_rate) : (np.array, int)

    """
    mel = mel + mel_min
    mel = inv_normalize_mel_librosa(mel)
    mel = np.array(mel.T, order='C')
    mel = librosa.db_to_amplitude(mel, ref=0.15)
    sig = librosa.feature.inverse.mel_to_audio(mel, sr=44100, n_fft=1024,
                                             hop_length=220, win_length=1024,
                                             power=1.0, fmin=10, fmax=12000)
    # there are always 110 data points missing compared to the speak function using VTL
    # add 55 zeros to the beginning and the end
    sig = np.concatenate((np.zeros(55), sig, np.zeros(55)))
    return (sig, 44100)

ARTICULATOR = {0: 'vocal folds',
               1: 'tongue',
               2: 'lower incisors',
               3: 'lower lip',
               4: 'other articulator',
               5: 'num articulators',
               }


def speak_and_extract_tube_information(cp_param):
    """
    Calls the vocal tract lab to synthesize an audio signal from the cp_param.
    Parameters
    ==========
    cp_param : np.array
        array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms)
    Returns
    =======
    (signal, sampling rate, tube_info) : np.array, int, dict
        returns the signal which is number of time steps in the cp_param array
        minus one times the time step length, i. e. ``(cp_param.shape[0] - 1) *
        110 / 44100``
    """
    # get some constants
    audio_sampling_rate = ctypes.c_int(0)
    number_tube_sections = ctypes.c_int(0)
    number_vocal_tract_parameters = ctypes.c_int(0)
    number_glottis_parameters = ctypes.c_int(0)
    number_audio_samples_per_tract_state = ctypes.c_int(0)
    internal_sampling_rate = ctypes.c_double(0)

    VTL.vtlGetConstants(ctypes.byref(audio_sampling_rate),
                        ctypes.byref(number_tube_sections),
                        ctypes.byref(number_vocal_tract_parameters),
                        ctypes.byref(number_glottis_parameters),
                        ctypes.byref(number_audio_samples_per_tract_state),
                        ctypes.byref(internal_sampling_rate))

    assert audio_sampling_rate.value == 44100
    assert number_vocal_tract_parameters.value == 19
    assert number_glottis_parameters.value == 11

    number_frames = cp_param.shape[0]
    frame_steps = 110  # 2.5 ms
    # within first parenthesis type definition, second initialisation
    audio = [(ctypes.c_double * int(frame_steps))() for _ in range(number_frames - 1)]

    # init the arrays
    tract_params = [(ctypes.c_double * (number_vocal_tract_parameters.value))() for _ in range(number_frames)]
    glottis_params = [(ctypes.c_double * (number_glottis_parameters.value))() for _ in range(number_frames)]

    # fill in data
    tmp = np.ascontiguousarray(cp_param[:, 0:19])
    for i in range(number_frames):
        tract_params[i][:] = tmp[i]
    del tmp

    tmp = np.ascontiguousarray(cp_param[:, 19:30])
    for i in range(number_frames):
        glottis_params[i][:] = tmp[i]
    del tmp

    # tube sections
    tube_length_cm = [(ctypes.c_double * 40)() for _ in range(number_frames)]
    tube_area_cm2 = [(ctypes.c_double * 40)() for _ in range(number_frames)]
    tube_articulator = [(ctypes.c_int * 40)() for _ in range(number_frames)]
    incisor_pos_cm = [ctypes.c_double(0) for _ in range(number_frames)]
    tongue_tip_side_elevation = [ctypes.c_double(0) for _ in range(number_frames)]
    velum_opening_cm2 = [ctypes.c_double(0) for _ in range(number_frames)]

    # Reset time-domain synthesis
    failure = VTL.vtlSynthesisReset()
    if failure != 0:
        raise ValueError(f'Error in vtlSynthesisReset! Errorcode: {failure}')

    for i in range(number_frames):
        if i == 0:
            failure = VTL.vtlSynthesisAddTract(0, ctypes.byref(audio[0]),
                                               ctypes.byref(tract_params[i]),
                                               ctypes.byref(glottis_params[i]))
        else:
            failure = VTL.vtlSynthesisAddTract(frame_steps, ctypes.byref(audio[i-1]),
                                               ctypes.byref(tract_params[i]),

                                               ctypes.byref(glottis_params[i]))
        if failure != 0:
            raise ValueError('Error in vtlSynthesisAddTract! Errorcode: %i' % failure)

        # export
        failure = VTL.vtlTractToTube(ctypes.byref(tract_params[i]),
                                     ctypes.byref(tube_length_cm[i]),
                                     ctypes.byref(tube_area_cm2[i]),
                                     ctypes.byref(tube_articulator[i]),
                                     ctypes.byref(incisor_pos_cm[i]),
                                     ctypes.byref(tongue_tip_side_elevation[i]),
                                     ctypes.byref(velum_opening_cm2[i]))

        if failure != 0:
            raise ValueError('Error in vtlTractToTube! Errorcode: %i' % failure)

    audio = np.ascontiguousarray(audio)
    audio.shape = ((number_frames - 1) * frame_steps,)

    arti = [[ARTICULATOR[sec] for sec in list(tube_articulator_i)] for tube_articulator_i in list(tube_articulator)]
    incisor_pos_cm = [x.value for x in incisor_pos_cm]
    tongue_tip_side_elevation = [x.value for x in tongue_tip_side_elevation]
    velum_opening_cm2 = [x.value for x in velum_opening_cm2]

    tube_info = {"tube_length_cm": np.array(tube_length_cm),
                 "tube_area_cm2": np.array(tube_area_cm2),
                 "tube_articulator": np.array(arti),
                 "incisor_pos_cm": np.array(incisor_pos_cm),
                 "tongue_tip_side_elevation": np.array(tongue_tip_side_elevation),
                 "velum_opening_cm2": np.array(velum_opening_cm2)}

    return (audio, 44100, tube_info)

def stereo_to_mono(wave, which="both"):
    """
    Extract a channel from a stereo wave

    Parameters
    ==========
    wave: np.array
        Input wave data.
    which: {"left", "right", "both"} default = "both"
        if `mono`, `which` indicates whether the *left* or the *right* channel
        should be extracted, or whether *both* channels should be averaged.

    Returns
    =======
    data: numpy.array

    """
    if which == "left":
        return wave[:, 0]
    if which == "right":
        return wave[:, 1]
    return (wave[:, 0] + wave[:, 1]) / 2


def pad_same_to_even_seq_length(array):
    """
    Pad array of cp-trajectories to have an even length (cps sampled every 2.5ms while log-mel spectrogram every 5ms sample --> half of cps)
    - pad at the end
    - pad with last value

    :param array: np.array
        cp-trajectories (seq_length, control parameters)
    :return: array: np.array
        padded cp-trajectory (seq_length, control parameters)
    """
    if not array.shape[0] % 2 == 0:
        return np.concatenate((array, array[-1:, :]), axis=0)
    else:
        return array

def half_seq_by_average_pooling(seq):
    if len(seq) % 2:
        seq = pad_same_to_even_seq_length(seq)
    half_seq = (seq[::2,:] + seq[1::2,:])/2
    return half_seq


def mel_wasserstein_distance(mel1, mel2):
    """
    perform the 1d Wasserstein distance function over mel bands, time points and energy

    :param mel1: np.array
        log-mel spectrogram (seq_length, mel channels)
    :param: mel1: np.array
        log-mel spectrogram (seq_length, mel channels)

    :return mean_time_dist, mean_mel_dist, energy_dist: np.float, np.float, np.float
        average 1d distance over time, mel_channel and energy
    """
    assert mel1.shape == mel2.shape

    time_dist = []
    mel_dist = []

    for time_point in range(mel1.shape[0]):
        time_dist.append(wasserstein_distance(mel1[time_point],mel2[time_point]))
    for mel_channel in range(mel1.shape[1]):
        mel_dist.append(wasserstein_distance(mel1[:,mel_channel], mel2[:,mel_channel]))

    energy1 = np.mean(mel1, axis = 1)
    energy2 = np.mean(mel2, axis=1)

    energy_dist = wasserstein_distance(energy1, energy2)

    return np.mean(time_dist), np.mean(mel_dist), energy_dist

class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6, reduction = "mean"):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction = reduction)
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

rmse_loss = RMSELoss(eps=0)

def get_vel_acc_jerk(trajectory, *, lag=1):
    """returns (velocity, acceleration, jerk) tuple"""
    velocity = (trajectory[:, lag:, :] - trajectory[:, :-lag, :]) / lag
    acc = (velocity[:, 1:, :] - velocity[:, :-1, :]) / 1.0
    jerk = (acc[:, 1:, :] - acc[:, :-1, :]) / 1.0
    return velocity, acc, jerk

def cp_trajacetory_loss(Y_hat, tgts):
    """
    Calculate additive loss using the RMSE of position velocity , acc and jerk

    :param Y_hat: 3D torch.Tensor
        model prediction
    :param tgts: 3D torch.Tensor
        target tensor
    :return loss, pos_loss, vel_loss, acc_loss, jerk_loss: torch.Tensors
        summed total loss with all individual losses
    """

    velocity, acc, jerk = get_vel_acc_jerk(tgts)
    velocity2, acc2, jerk2 = get_vel_acc_jerk(tgts, lag=2)
    velocity4, acc4, jerk4 = get_vel_acc_jerk(tgts, lag=4)

    Y_hat_velocity, Y_hat_acceleration, Y_hat_jerk = get_vel_acc_jerk(Y_hat)
    Y_hat_velocity2, Y_hat_acceleration2, Y_hat_jerk2 = get_vel_acc_jerk(Y_hat, lag=2)
    Y_hat_velocity4, Y_hat_acceleration4, Y_hat_jerk4 = get_vel_acc_jerk(Y_hat, lag=4)

    pos_loss = rmse_loss(Y_hat, tgts)
    vel_loss = rmse_loss(Y_hat_velocity, velocity) + rmse_loss(Y_hat_velocity2, velocity2) + rmse_loss(Y_hat_velocity4, velocity4)
    jerk_loss = rmse_loss(Y_hat_jerk, jerk) + rmse_loss(Y_hat_jerk2, jerk2) + rmse_loss(Y_hat_jerk4, jerk4)
    acc_loss = rmse_loss(Y_hat_acceleration, acc) + rmse_loss(Y_hat_acceleration2, acc2) + rmse_loss(Y_hat_acceleration4, acc4)

    loss = pos_loss + vel_loss + acc_loss + jerk_loss
    return loss, pos_loss, vel_loss, acc_loss, jerk_loss


def add_and_pad(xx, max_len, with_onset_dim=False):
    """
    Pad a sequence with last value to maximal length

    Parameters
    ==========
    xx : 2D np.array
        seuence to be padded (seq_length, feeatures)
    max_len : int
        maximal length to be padded to
    with_onset_dim : bool
        add one features with 1 for the first time step and rest 0 to indicate
        sound onset

    Returns
    =======
    pad_seq : torch.Tensor
        2D padded sequence

    """
    seq_length = xx.shape[0]
    if with_onset_dim:
        onset = np.zeros((seq_length, 1))
        onset[0, 0] = 1
        xx = np.concatenate((xx, onset), axis=1)  # shape len X (features +1)
    padding_size = max_len - seq_length
    padding_size = tuple([padding_size] + [1 for i in range(len(xx.shape) - 1)])
    xx = np.concatenate((xx, np.tile(xx[-1:], padding_size)), axis=0)
    return torch.from_numpy(xx)


def pad_batch_online(lens, data_to_pad, device="cpu", with_onset_dim=False):
    """
    pads and batches data into one single padded batch.

    Parameters
    ==========
    lens : 1D torch.Tensor
        Tensor containing the length of each sample in data_to_pad of one batch
    data_to_pad : series
        series containing the data to pad

    Returns
    =======
    padded_data : torch.Tensors
        Tensors containing the padded and stacked to one batch

    """
    max_len = int(max(lens))
    padded_data = torch.stack(list(data_to_pad.apply(
        lambda x: add_and_pad(x, max_len, with_onset_dim=with_onset_dim)))).to(device)

    return padded_data


def cps_to_ema_and_mesh(cps, file_prefix, *, path=""):
    """
    Calls the vocal tract lab to generate synthesized EMA trajectories.

    Parameters
    ==========
    cps : np.array
        2D array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms); first dimension is
        sequence and second is vocal tract lab parameters, i. e. (n_sequence,
        30)
    file_prefix : str
        the prefix of the files written
    path : str
        path where to put the output files

    Returns
    =======
    None : None
        all output is writen to files.

    """
    # get some constants
    audio_sampling_rate = ctypes.c_int(0)
    number_tube_sections = ctypes.c_int(0)
    number_vocal_tract_parameters = ctypes.c_int(0)
    number_glottis_parameters = ctypes.c_int(0)
    number_audio_samples_per_tract_state = ctypes.c_int(0)
    internal_sampling_rate = ctypes.c_double(0)

    VTL.vtlGetConstants(ctypes.byref(audio_sampling_rate),
                        ctypes.byref(number_tube_sections),
                        ctypes.byref(number_vocal_tract_parameters),
                        ctypes.byref(number_glottis_parameters),
                        ctypes.byref(number_audio_samples_per_tract_state),
                        ctypes.byref(internal_sampling_rate))

    assert audio_sampling_rate.value == 44100
    assert number_vocal_tract_parameters.value == 19
    assert number_glottis_parameters.value == 11

    number_frames = cps.shape[0]

    # init the arrays
    tract_params = (ctypes.c_double * (number_frames * number_vocal_tract_parameters.value))()
    glottis_params = (ctypes.c_double * (number_frames * number_glottis_parameters.value))()

    # fill in data
    tmp = np.ascontiguousarray(cps[:, 0:19])
    tmp.shape = (number_frames * 19,)
    tract_params[:] = tmp
    del tmp

    tmp = np.ascontiguousarray(cps[:, 19:30])
    tmp.shape = (number_frames * 11,)
    glottis_params[:] = tmp
    del tmp

    number_ema_points = 3
    surf = (ctypes.c_int * number_ema_points)()
    surf[:] = np.array([16, 16, 16])  # 16 = TONGUE

    vert = (ctypes.c_int * number_ema_points)()
    vert[:] = np.array([115, 225, 335])  # Tongue Back (TB) = 115; Tongue Middle (TM) = 225; Tongue Tip (TT) = 335

    if not os.path.exists(path):
        os.mkdir(path)

    failure = VTL.vtlTractSequenceToEmaAndMesh(
            ctypes.byref(tract_params), ctypes.byref(glottis_params),
            number_vocal_tract_parameters, number_glottis_parameters,
            number_frames, number_ema_points,
            ctypes.byref(surf), ctypes.byref(vert),
            path.encode(), file_prefix.encode())
    if failure != 0:
        raise ValueError('Error in vtlTractSequenceToEmaAndMesh! Errorcode: %i' % failure)


def cps_to_ema(cps):
    """
    Calls the vocal tract lab to generate synthesized EMA trajectories.

    Parameters
    ==========
    cps : np.array
        2D array containing the vocal and glottis parameters for each time step
        which is 110 / 44100 seoconds (roughly 2.5 ms); first dimension is
        sequence and second is vocal tract lab parameters, i. e. (n_sequence,
        30)

    Returns
    =======
    emas : pd.DataFrame
        returns the 3D ema points for different virtual EMA sensors in a
        pandas.DataFrame

    """
    with tempfile.TemporaryDirectory(prefix='python_paule_') as path:
        file_name = 'pyndl_util_ema_export'
        cps_to_ema_and_mesh(cps, file_prefix=file_name, path=path)
        emas = pd.read_table(os.path.join(path, f"{file_name}-ema.txt"), sep=' ')
    return emas


def seg_to_cps(seg_file):
    """
    Calls the vocal tract lab to read a segment file (seg_file) and returns the
    unnormalised cps.

    Parameters
    ==========
    seg_file : str
        path to the segment file

    Returns
    =======
    cps : np.array
        two dimensional numpy array of the unnormalised control parameter
        trajectories

    """
    segment_file_name = seg_file.encode()

    with tempfile.TemporaryDirectory() as tmpdirname:
        gesture_file_name = os.path.join(tmpdirname, 'vtl_ges_file.txt').encode()
        failure = VTL.vtlSegmentSequenceToGesturalScore(segment_file_name, gesture_file_name)
        if failure != 0:
            raise ValueError('Error in vtlSegmentSequenceToGesturalScore! Errorcode: %i' % failure)
        cps = ges_to_cps(gesture_file_name.decode())
    return cps


def ges_to_cps(ges_file):
    """
    Calls the vocal tract lab to read a gesture file (ges_file) and returns the
    unnormalised cps.

    Parameters
    ==========
    ges_file : str
        path to the gesture file

    Returns
    =======
    cps : np.array
        two dimensional numpy array of the unnormalised control parameter
        trajectories

    """
    gesture_file_name = ges_file.encode()

    with tempfile.TemporaryDirectory() as tmpdirname:
        tract_sequence_file_name = os.path.join(tmpdirname, 'vtl_tract_seq.txt').encode()
        failure = VTL.vtlGesturalScoreToTractSequence(gesture_file_name, tract_sequence_file_name)
        if failure != 0:
            raise ValueError('Error in vtlGesturalScoreToTractSequence! Errorcode: %i' % failure)

        cps = read_cp(tract_sequence_file_name.decode())
    return cps



def export_svgs(cps, path='svgs/', hop_length=5):
    """
    hop_length == 5 : roughly 80 frames per second
    hop_length == 16 : roughly 25 frames per second

    """
    n_tract_parameter = 19
    for ii in range(cps.shape[0] // hop_length):
        jj = ii * hop_length

        tract_params = (ctypes.c_double * 19)()
        tract_params[:] = cps[jj, :n_tract_parameter]

        file_name = os.path.join(path, f'tract{ii:05d}.svg')
        file_name = ctypes.c_char_p(file_name.encode())

        if not os.path.exists(path):
            os.mkdir(path)

        VTL.vtlExportTractSvg(tract_params, file_name)
