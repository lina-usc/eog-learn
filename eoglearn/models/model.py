# Author: Scott Huberty <seh33@uw.edu>
#         Christian O"Reilly <christian.oreilly@sc.edu>
#
# License: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from mne.utils import logger
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad

from eoglearn.viz import plot_values_topomap


class EOGDenoiser:
    """Use simultaneous EEG and Eyetracking to Denoise EOG from the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        An instance of ``mne.io.Raw``, with EEG, eyegaze, and pupil channels.
    downsample : int
        The factor by which to downsample the EEG and eyetracking data. EEG channels
        will be low-pass filtered before downsampling using ``mne.io.filter.resample``.
        Eyetracking channels will be decimated without any filtering. resampling and
        decimating will be done on copies of the data, so the original input data will
        be preserved.
    filter : tuple
        The bandpass filter to apply to the EEG data. The filter will only be applied
        to a copy of the raw data, and the original data will be preserved.
    n_units : int
        The number of units to pass into the initial LSTM layer. Defaults to 50.
    n_times : int
        The number of timepoints to pass into the LSTM model at once. Defaults to 100.

    Attributes
    ----------
    raw : mne.io.Raw
        The original input ``mne.io.Raw`` instance.
    downsample : int
        The factor by which the data was downsampled.
    filter : tuple
        The bandpass filter that was applied to the EEG data.
    n_units : int
        The number of units in the initial LSTM layer.
    n_times : int
        The number of timepoints passed into the LSTM model at once.
    model : eoglearn.model.Model
        The Keras LSTM model instance.
    X : np.ndarray
        The raw eyetracking data.
    Y : np.ndarray
        The raw EEG data.
    X_train : np.ndarray
        The eyetracking data (``X``)reshaped to fit into the LSTM model.
    Y_train : np.ndarray
        The EEG data (``Y``) reshaped to fit into the LSTM model.
    downsampled_sfreq : float
        The sampling frequency after downsampling.
    scaler_X : sklearn.preprocessing.StandardScaler
        The StandardScaler instance used to scale the eyetracking data.
    scaler_Y : sklearn.preprocessing.StandardScaler
        The StandardScaler instance used to scale the EEG data.

    Notes
    -----
    See the MNE-Python tutorial on aligning EEG and eyetracking data for information
    on how to create a raw object with both EEG and eyetracking channels.
    """

    def __init__(
        self,
        raw,
        downsample=10,
        n_units=50,
        n_times=100,
    ):
        self.__x = None
        self.__y = None
        #############################################
        # MNE Raw object and preprocessing parameters
        #############################################
        self.raw = raw

        self.downsample = downsample
        #############################
        # Set up the Keras LSTM Model
        #############################
        self.n_units = n_units
        self.n_times = n_times
        self.model = self.setup_model()
        self.train_test_split()  # i.e. self.X_train, self.Y_train

    @property
    def downsampled_sfreq(self):
        """Return the sampling frequency after downsampling."""
        return (
            self.raw.info["sfreq"] // self.downsample
            if self.downsample
            else self.raw.info["sfreq"]
        )

    @property
    def X(self):
        """Return an array of the raw eye-tracking data."""
        if self.__x is None:
            eye_data = self.raw.get_data(picks=["eyetrack"]).T
            if self.downsample is not None:
                eye_data = eye_data[:: self.downsample, :]  # i.e. eye_data[::10, :]
            self.scaler_X = StandardScaler().fit(np.nan_to_num(eye_data))
            self.__x = self.scaler_X.transform(np.nan_to_num(eye_data))
        return self.__x

    @property
    def Y(self):
        """Return an array of the raw EEG data."""
        if self.__y is None:
            eeg_data = self.raw.copy()
            if self.downsample:
                eeg_data.resample(self.downsampled_sfreq)
            eeg_data = eeg_data.get_data(picks="eeg").T
            self.scaler_Y = StandardScaler().fit(np.nan_to_num(eeg_data))
            self.__y = self.scaler_Y.transform(np.nan_to_num(eeg_data))
        return self.__y

    def setup_model(self):
        """Return a model instance given a raw instance.

        Returns
        -------
        model : eoglearn.model.Model
            a model instance.
        """
        model = Sequential()

        # LSTM layer accepts 3D array of shape of (n_sample, n_timesteps, n_features)
        model.add(
            LSTM(
                self.n_units,
                input_shape=(self.n_times, self.X.shape[1]),
                dropout=0.5,
                return_sequences=True,
            )
        )
        model.add(LSTM(self.Y.shape[1], dropout=0.5, return_sequences=True))

        adagrad = Adagrad(learning_rate=1)
        model.compile(loss="mean_squared_error", optimizer=adagrad)
        return model

    def train_test_split(self):
        """Split Eyetrack and EEG data into training and testing sets.

        Notes
        -----
        The ``X_train`` and ``Y_train`` attributes will be reshaped into 3D arrays of
        shape ``(n_samples, n_timesteps, n_features)``. The ``n_samples`` dimension
        will be the number of samples in the raw data divided by the ``n_times``
        attribute. The ``n_timesteps`` dimension will be the ``n_times`` attribute.
        The ``n_features`` dimension will be the number of channels in the eyetracking
        or EEG data. For example, if the raw data has 1492 samples, and the ``n_times``
        parameter is 100, then the ``X_train`` and ``Y_train`` arrays will have shape
        ``(14, 100, 3)`` and ``(14, 100, 129)``, respectively. ``X_train`` contains
        only the eyetracking data, and ``Y_train`` contains only the EEG data.
        """
        n = self.X.shape[0] // self.n_times  # i.e 1492 / 100
        self.X_train = self.X[: self.n_times * n, :].reshape(
            (-1, self.n_times, self.X.shape[-1]), order="C"
        )
        self.Y_train = self.Y[: self.n_times * n, :].reshape(
            (-1, self.n_times, self.Y.shape[-1]), order="C"
        )

    def fit_model(
        self,
        epochs=10,
        validation_split=0.2,
        batch_size=1,
        verbose=2,
        fitting_kwargs=None,
    ):
        """Fit the EOGDenoiser model using the input Raw object.

        Parameters
        ----------
        fitting_kwargs : dict
            A dictionary of keyword arguments to pass into the ``fit`` method of the
            Keras ``Sequential`` model. Defaults to ``None``, which will use
            ``dict(epochs=50, validation_split=0.2, batch_size=1, verbose=2)``.
        """
        if fitting_kwargs is None:
            fitting_kwargs = dict()
        self.model.fit(
            self.X_train,
            self.Y_train,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
            verbose=verbose,
            **fitting_kwargs,
        )

    def predict(self, data, predict_kwargs=None):
        """Return Model predictions.

        Parameters
        ----------
        data : np.ndarray
            The data to predict. Must be the same shape as ``self.X_train``.
        predict_kwargs : dict
            A dictionary of keyword arguments to pass into the ``predict`` method of
            the Keras ``Sequential`` model. Defaults to ``None``.

        Returns
        -------
        predictions : np.ndarray
            The predicted data.

        Notes
        -----
        This method is a wrapper for the ``predict`` method of the Keras
        :class:`~tensorflow.keras.models.Sequential` model.
        """
        if not predict_kwargs:
            predict_kwargs = dict()
        return self.model.predict(data, **predict_kwargs)

    def predict_eog(self):
        """Return the predicted EOG data.

        Returns
        -------
        predicted_eog : np.ndarray
            The predicted EOG data, scaled back to the original units and shape, i.e.
            ``(n_samples, n_meeg_channels)`` of the input :class:`~mne.io.Raw` object.
            The predictions are saved to the ``predicted_eog_`` attribute.
        """
        if not len(self.X_train):
            logger.info("setting up train/test split")
            _ = self.train_test_split()
        predictions = self.predict(self.X_train)
        # reshape to back to 2D array matching the original raw data shape
        predicted_eog = predictions.reshape(
            (np.prod(predictions.shape[:-1]), predictions.shape[-1]), order="C"
        )
        # inverse transform to get back to original raw data units
        self.predicted_eog_ = self.scaler_Y.inverse_transform(predicted_eog)
        return self.predicted_eog_

    def get_denoised_neural(self):
        """Return the denoised M/EEG neural data.

        Returns
        -------
        denoised_neural : np.ndarray
            The denoised neural data, scaled back to the original units and shape,
            i.e. ``(n_samples, n_meeg_channels)`` of the input :class:`~mne.io.Raw`
            object. The denoised neural data are saved to the ``denoised_neural_``
            attribute.
        """
        if not len(self.Y_train):
            logger.info("setting up train/test split")
            _ = self.train_test_split()
        if not hasattr(self, "predicted_eog_"):
            logger.info("Predicting EOG data, saving to ``predicted_eog_`` attribute.")
            _ = self.predict_eog()
        predicted_eog = self.predicted_eog_
        # reshape to back to 2D array matching the original raw data shape
        Y_train = self.Y_train.reshape(
            (np.prod(self.Y_train.shape[:-1]), self.Y_train.shape[-1]), order="C"
        )
        # scale back to original units of the Raw data
        Y_train = self.scaler_Y.inverse_transform(Y_train)
        assert Y_train.shape == predicted_eog.shape
        self.denoised_neural_ = Y_train - predicted_eog
        return self.denoised_neural_

    def plot_eog_topo(self, montage, show=True):
        """Plot the topography of the eyetracking data.

        Parameters
        ----------
        montage : mne.channels.DigMontage | str
            Montage for digitized electrode and headshape position data.
            See mne.channels.make_standard_montage(), and
            mne.channels.get_builtin_montages() for more information
            on making montage objects in MNE.
        show : bool
            Whether to show the plot or not. Defaults to True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The resulting figure object for the topomap plot.
        """
        if not hasattr(self, "denoised_neural_"):
            logger.info(
                "Denoising neural data, saving to ``denoised_neural_`` attribute."
            )
            _ = self.get_denoised_neural()
        squared_errors = np.square(self.denoised_neural_)
        mean_squared_erros = np.mean(squared_errors, axis=0)
        noise = np.sqrt(mean_squared_erros)
        # reshape to back to 2D array matching the original raw data shape
        Y_train = self.Y_train.reshape(
            (np.prod(self.Y_train.shape[:-1]), self.Y_train.shape[-1]), order="C"
        )
        # scale back to original units of the Raw data
        Y_train = self.scaler_Y.inverse_transform(Y_train)
        signal = np.sqrt((Y_train**2).mean(axis=0))
        assert noise.shape == signal.shape
        snr = (noise / signal)[:-1]
        percent_noise = 1 - snr
        percent_noise *= 100
        eeg_names = self.raw.copy().pick("eeg").ch_names[:-1]
        data_dict = dict(list(zip(eeg_names, percent_noise)))
        montage = montage
        fig, ax = plt.subplots(constrained_layout=True)

        plot_values_topomap(
            data_dict,
            montage,
            axes=ax,
            vmin=percent_noise.min(),
            vmax=percent_noise.max(),
            names=None,
            image_interp="linear",
            sensors=True,
            show=show,
        )
        ax.set_title(
            "Percentage of EEG signal that is accounted for by Ocular Artifact"
        )
        return fig
