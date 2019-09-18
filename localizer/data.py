import urllib
from itertools import count

import h5py
import imgaug
import numpy as np
import skimage
import tqdm
from imgaug import augmenters as iaa
from multiprocess import Pool, Semaphore, cpu_count
from scipy.ndimage import zoom

from localizer import const, util


def data_generator(X, Y, subset_slice, batchsize):
    from_idx = subset_slice.start
    slice_length = subset_slice.stop - subset_slice.start
    for idx in count(step=batchsize):
        from_idx = (idx % slice_length) + subset_slice.start
        to_idx = from_idx + batchsize
        x = X[from_idx:to_idx]
        y = Y[from_idx:to_idx]
        yield (x, y)


def extract_target(data):
    x, y = data
    return (x, y[:, y.shape[1] // 2, y.shape[2] // 2])


def get_augmenter():
    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        sometimes(
            iaa.Affine(
                scale={
                    "x": (0.9, 1.1),
                    "y": (0.9, 1.1)
                },
                rotate=(-45, 45),
                shear=(-16, 16),
                mode="reflect",
                order=3)),
        iaa.SomeOf((0, 3), [
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            sometimes(
                iaa.ElasticTransformation(
                    alpha=(0.5, 3.5), sigma=0.25, order=3)),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05), order=3)),
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ],
                   random_order=True)
    ])

    def activator_label(images, augmenter, parents, default):
        if augmenter.name in [
                "Sharpen", "ContrastNormalization", "AdditiveGaussianNoise",
                "GaussianBlur", "AverageBlur", "MedianBlur"
        ]:
            return False
        else:
            return default

    hooks_label = imgaug.HooksImages(activator=activator_label)

    return seq, hooks_label


def get_trivial_augmenter():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
    ])

    hooks_label = imgaug.HooksImages()

    return seq, hooks_label


def augment(images, labels, seq, zoom_factor, hooks_label):
    global semaphore
    semaphore.acquire()

    seq_det = seq.to_deterministic()
    images_aug = seq_det.augment_images(images)
    labels_aug = seq_det.augment_images(labels, hooks=hooks_label)

    images_aug = np.stack([zoom(i, zoom_factor, order=3)
                           for i in images_aug])[:, :, :, None]

    return images_aug.astype(np.float32) / 255, \
        labels_aug[:, labels_aug.shape[1] // 2, labels_aug.shape[2] //
                   2].astype(np.float32) / 255


def imgaug_generator(gen_, seq, hooks_label, zoom_factor=.32, max_prefetch=32):
    semaphore = Semaphore(max_prefetch)

    def init_child(lock_):
        global semaphore
        semaphore = lock_

    with Pool(
            cpu_count(), initializer=init_child,
            initargs=(semaphore, )) as pool:
        while True:
            gen = gen_()

            for images, labels in pool.imap_unordered(
                    lambda g: augment(g[0], g[1], seq, zoom_factor, hooks_label),
                    gen):
                semaphore.release()
                yield images, labels


def no_imgaug_generator(gen_, zoom_factor=.32):
    while True:
        gen = gen_()

        for images, labels in gen:
            images = np.stack([zoom(i, zoom_factor, order=3)
                               for i in images])[:, :, :, None]

            yield images.astype(np.float32) / 255, \
                labels[:, labels.shape[1] // 2, labels.shape[2] //
                       2].astype(np.float32) / 255


def load_labelbox_image(row):
    response = urllib.request.urlopen(row['Labeled Data'])
    image_data = response.read() 
    image = skimage.io.imread(image_data, as_gray=True, plugin='imageio')
    return image


def process_labelbox_files(df, X, Y, samples_per_file=20000, patch_size=128, padding=128, nobee_sample_rate=0.0005):
    idx = 0
    indices = list(range(samples_per_file * len(df)))
    np.random.shuffle(indices)

    pdf = util.get_normal_pdf(padding)
    
    for _, row in tqdm.tqdm_notebook(df.iterrows()):
        image = load_labelbox_image(row)
        image_padded = np.pad(image, padding, mode='edge')

        label_image = np.zeros((len(const.labels) + 1, image_padded.shape[0], image_padded.shape[1]))

        for label_idx, label in enumerate(const.labels):
            for entry in row.Label[label]:
                sx = slice(entry['geometry']['y'], entry['geometry']['y'] + 2 * padding)
                sy = slice(entry['geometry']['x'], entry['geometry']['x'] + 2 * padding)

                label_image[label_idx, sx, sy] = np.maximum(label_image[label_idx, sx, sy], pdf)

        label_image[-1, padding:-padding, padding:-padding] = nobee_sample_rate

        max_saliency = np.max(label_image)
        if max_saliency:
            label_image /= max_saliency

        sample_indices = np.random.choice(
            label_image.shape[1] * label_image.shape[2], 
            size=samples_per_file, 
            replace=False,                         
            p=label_image.sum(axis=0).flatten() / label_image.sum()
        )

        sample_coords = np.stack(np.unravel_index(sample_indices, (label_image.shape[1], label_image.shape[2])), axis=-1)

        samples = [image_padded[x-patch_size//2:x+patch_size//2, y-patch_size//2:y+patch_size//2] for x, y in sample_coords]
        sample_labels = [label_image[:, x-patch_size//2:x+patch_size//2, y-patch_size//2:y+patch_size//2] for x, y in sample_coords]

        total_num_of_idxs = len(indices)
        for x, y in tqdm.tqdm_notebook(zip(samples, sample_labels), total=samples_per_file):
            X[indices[idx], :, :] = (x * 255).astype(np.uint8)
            Y[indices[idx], :, :, :] = (np.transpose(y[:-1], (1, 2, 0)) * 255).astype(np.uint8)

            idx += 1


def create_data_hdf5(output_path, samples_per_file, train_indices, test_indices, patch_size=128):
    hf = h5py.File(output_path, 'w')

    g1 = hf.create_group('train')
    g1.create_dataset('X', shape=(samples_per_file * len(train_indices), patch_size, patch_size), dtype=np.uint8)
    g1.create_dataset('Y', shape=(samples_per_file * len(train_indices), patch_size, patch_size, len(const.labels)), dtype=np.uint8)
    Xtr = g1.get('X')
    Ytr = g1.get('Y')

    g2 = hf.create_group('test')
    g2.create_dataset('X', shape=(samples_per_file * len(test_indices), patch_size, patch_size), dtype=np.uint8)
    g2.create_dataset('Y', shape=(samples_per_file * len(test_indices), patch_size, patch_size, len(const.labels)), dtype=np.uint8)
    Xte = g2.get('X')
    Yte = g2.get('Y')

    return hf, (Xtr, Ytr), (Xte, Yte)
