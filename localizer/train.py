import warnings

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, History, ReduceLROnPlateau
from keras.utils import multi_gpu_model

from localizer import data, model, util


def get_data_generators(Xtr, Ytr, Xte, Yte, batchsize=128):
    train_slice = slice(0, Xtr.shape[0])
    val_slice = slice(0, Xte.shape[0])

    seq, hooks_label = data.get_trivial_augmenter()

    def data_gen_train(bs, sl): return data.data_generator(Xtr, Ytr, sl, bs)

    def data_gen_test(bs, sl): return data.data_generator(Xte, Yte, sl, bs)

    train_gen = data.imgaug_generator(lambda: data_gen_train(batchsize, train_slice),
                                      seq, hooks_label, zoom_factor=1)
    val_gen = data.no_imgaug_generator(lambda: data_gen_test(batchsize, val_slice), zoom_factor=1)

    steps_per_epoch = Xtr.shape[0] // batchsize
    validation_steps = Xte.shape[0] // batchsize

    return train_gen, val_gen, steps_per_epoch, validation_steps


def plot_samples_grid(gen, num_plot=4):
    xt, yt = next(gen)

    _, axes = plt.subplots(num_plot, num_plot, figsize=(3 * num_plot, 3 * num_plot))

    for idx in range(num_plot * num_plot):
        r, c = divmod(idx, num_plot)

        axes[r, c].imshow(xt[idx, :, :, 0], cmap=plt.cm.gray)
        axes[r, c].set_title(', '.join(map(lambda f: '{:.2f}'.format(f), yt[idx])))
        axes[r, c].set_axis_off()


def train_localizer_model(train_gen, val_gen, steps_per_epoch, validation_steps, batchsize=128,
                          initial_channels=32, multi_gpu=False, optimizer='Nadam'):
    if optimizer == 'AdamWithWeightnorm':
        optimizer = util.AdamWithWeightnorm(amsgrad=True)

    train_model = model.get_train_model(initial_channels=initial_channels)

    if multi_gpu:
        train_model = multi_gpu_model(train_model)

    train_model.compile(optimizer, loss='binary_crossentropy', metrics=['mae'])

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.25, patience=15, min_lr=0.00001, min_delta=0.0001, verbose=True)
    stopper = EarlyStopping(min_delta=0.0001, patience=20, restore_best_weights=True)
    history = History()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_model.fit_generator(train_gen, epochs=1000,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=val_gen,
                                  validation_steps=validation_steps,
                                  callbacks=[reduce_lr, stopper, history])

    return train_model
