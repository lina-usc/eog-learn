import matplotlib.pyplot as plt


def plot_training(model, axes=None):
    """Plot the training history of a model.

    Parameters
    ----------
    model : keras.Model
        A compiled Keras model.

    Returns
    -------
    None
    """
    if axes:
        ax = axes
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(model.history.history["loss"], label="Training Loss")
    ax.plot(model.history.history["val_loss"], label="Validation Loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    return fig.show()
