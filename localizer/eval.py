import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import skimage.feature
import tqdm

from localizer import const, data

labels = const.labels
min_distance = 1


def match_positions(truth_positions, prediction_positions, max_distance=64):
    distance_matrix = scipy.spatial.distance_matrix(prediction_positions, truth_positions)

    assignments = []
    while len(assignments) < min(distance_matrix.shape) and distance_matrix.min() < max_distance:
        candidate = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

        if distance_matrix[candidate[0], candidate[1]] < max_distance:
            assignments.append(candidate)
            distance_matrix[candidate[0], :] = np.inf
            distance_matrix[:, candidate[1]] = np.inf
        else:
            assert False

    if len(assignments) > 0:
        assignments = np.stack(assignments)

    return assignments


def fbeta_score(recall, precision, beta=1):
    return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-8)


def calculate_metrics(num_true_positive, num_condition_positive, num_predicted_positive):
    recall = num_true_positive / (num_condition_positive + 1e-8)
    precision = num_true_positive / (num_predicted_positive + 1e-8)
    fone = fbeta_score(recall, precision, beta=1)
    ftwo = fbeta_score(recall, precision, beta=2)

    return recall, precision, fone, ftwo


def get_predictions(image, conv_model, padding=128):
    image_padded = np.pad(image, padding, mode='edge')
    predictions = conv_model.predict(image_padded[None, :, :, None])[0]

    return predictions


def get_subpixel_offsets(saliency, position, subpixel_range):
    sample = saliency[position[0]-subpixel_range:position[0]+subpixel_range,
                      position[1]-subpixel_range:position[1]+subpixel_range]

    M = skimage.measure.moments(sample)
    centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])

    return (centroid[0] - (subpixel_range - 0.5), centroid[1] - (subpixel_range - 0.5))


def get_predicted_positions(saliency, thresholds, min_distance=1, padding=128,
                            subpixel_precision=True, subpixel_range=3):
    predictions_positions = []
    for class_idx in range(len(thresholds)):
        class_saliency = saliency[:, :, class_idx]

        positions = skimage.feature.peak_local_max(
            class_saliency,
            min_distance=min_distance,
            threshold_abs=thresholds[class_idx]
        )

        if subpixel_precision:
            saliency_padded = np.pad(class_saliency, pad_width=subpixel_range, constant_values=0)
            subpixel_offsets = [
                get_subpixel_offsets(saliency_padded, p + subpixel_range, subpixel_range=subpixel_range) for p in positions]

            positions = positions.astype(np.float32)
            for idx in range(len(positions)):
                positions[idx, 0] += subpixel_offsets[idx][0]
                positions[idx, 1] += subpixel_offsets[idx][1]

        padded_positions = ((((positions + 5) * 2 + 1) * 2 + 1) * 2 + 1)

        predictions_positions.append(padded_positions - padding)

    return predictions_positions


def get_recall_precision(eval_metrics_by_image, labels, thresholds):
    plt.figure(figsize=(12, 6))

    recalls_per_class, precision_by_class = [], []
    for class_idx, class_name in enumerate(labels):
        recalls, precisions = [], []
        for threshold_idx, threshold in enumerate(thresholds):
            evals = [e[class_idx][threshold_idx] for e in eval_metrics_by_image]

            num_true_positive, num_condition_positive, num_predicted_positive = np.stack(evals).sum(axis=0)
            recall, precision, _, _ = calculate_metrics(
                num_true_positive, num_condition_positive, num_predicted_positive
            )
            recalls.append(recall)
            precisions.append(precision)

        recalls_per_class.append(recalls)
        precision_by_class.append(precisions)
        plt.plot(recalls, precisions, label=labels[class_idx], linestyle='--')

    plt.legend()
    plt.title('Precision-Recall curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([.5, 1.01])
    plt.ylim([.5, 1.01])
    plt.grid()

    return recalls_per_class, precision_by_class


def get_threshold_fone(eval_metrics_by_image, labels, thresholds):
    plt.figure(figsize=(12, 6))

    fones_by_class = []
    for class_idx, class_name in enumerate(labels):
        fones = []
        for threshold_idx, threshold in enumerate(thresholds):
            evals = [e[class_idx][threshold_idx] for e in eval_metrics_by_image]

            num_true_positive, num_condition_positive, num_predicted_positive = np.stack(evals).sum(axis=0)
            _, _, fone, _ = calculate_metrics(
                num_true_positive, num_condition_positive, num_predicted_positive
            )
            fones.append(fone)

        plt.plot(thresholds, fones, label=labels[class_idx], linestyle='--')
        fones_by_class.append(fones)

    plt.legend()
    plt.title('F1 curves')
    plt.xlabel('Threshold')
    plt.ylabel('F1')
    plt.grid()

    return fones_by_class


def get_threshold_ftwo(eval_metrics_by_image, labels, thresholds):
    plt.figure(figsize=(12, 6))

    ftwos_by_class = []
    for class_idx, class_name in enumerate(labels):
        ftwos = []
        for threshold_idx, threshold in enumerate(thresholds):
            evals = [e[class_idx][threshold_idx] for e in eval_metrics_by_image]

            num_true_positive, num_condition_positive, num_predicted_positive = np.stack(evals).sum(axis=0)
            recall, precision, _, ftwo = calculate_metrics(
                num_true_positive, num_condition_positive, num_predicted_positive
            )
            ftwos.append(ftwo)

        plt.plot(thresholds, ftwos, label=labels[class_idx], linestyle='--')
        ftwos_by_class.append(ftwos)

    plt.legend()
    plt.title('F2 curves')
    plt.xlabel('Threshold')
    plt.ylabel('F2')
    plt.grid()

    return ftwos_by_class


def get_best_thresholds(fones_by_class, ftwos_by_class, thresholds):
    fone_classes = ('UnmarkedBee', 'BeeInCell', 'UpsideDownBee')
    ftwo_classes = ('MarkedBee', )

    best_thresholds = \
        [(label, thresholds[np.argmax(fones_by_class[class_idx])]) for
         class_idx, label in enumerate(labels) if label in fone_classes] + \
        [(label, thresholds[np.argmax(ftwos_by_class[class_idx])]) for
         class_idx, label in enumerate(labels) if label in ftwo_classes]

    return best_thresholds


def evalulate_localizer(eval_df, get_predictions, get_predicted_positions, padding=128,
                        max_assignment_distance=64, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    eval_metrics_by_image = []
    for _, eval_row in tqdm.tqdm_notebook(eval_df.iterrows()):
        image = data.load_labelbox_image(eval_row)
        predictions = get_predictions(image, padding=padding)

        predictions_positions_by_threshold = []
        for saliency_threshold in thresholds:
            predicted_positions = get_predicted_positions(
                predictions.copy(),
                [saliency_threshold for _ in range(len(labels))],
                min_distance=min_distance,
                padding=padding
            )
            predictions_positions_by_threshold.append(predicted_positions)

        eval_metrics_by_class = []
        for class_idx in range(len(labels)):
            eval_metrics_by_threshold = []
            for threshold_idx, threshold in enumerate(thresholds):
                truth_positions = eval_row['Label'][labels[class_idx]]
                truth_positions = np.stack([np.array((l['geometry']['y'], l['geometry']['x']))
                                            for l in truth_positions])
                predictions_positions = predictions_positions_by_threshold[threshold_idx]
                assignments = match_positions(
                    truth_positions, predictions_positions[class_idx], max_distance=max_assignment_distance)

                eval_metrics_by_threshold.append(
                    (len(assignments), len(truth_positions), len(predictions_positions[class_idx]))
                )

            eval_metrics_by_class.append(eval_metrics_by_threshold)
        eval_metrics_by_image.append(eval_metrics_by_class)

    recalls_by_class, precisions_by_class = get_recall_precision(eval_metrics_by_image, labels, thresholds)
    plt.show()
    fones_by_class = get_threshold_fone(eval_metrics_by_image, labels, thresholds)
    plt.show()
    ftwos_by_class = get_threshold_ftwo(eval_metrics_by_image, labels, thresholds)
    plt.show()

    best_thresholds = get_best_thresholds(fones_by_class, ftwos_by_class, thresholds)

    for class_idx, label in enumerate(labels):
        threshold = [t for n, t in best_thresholds if n == label][0]
        threshold_idx = np.argwhere(thresholds == threshold)[0][0]

        print('{} | Threshold: {:.3f} | Recall: {:.3f} | Precision: {:.3f} | F1: {:.3f} | F2: {:.3f}'.format(
            label,
            threshold,
            recalls_by_class[class_idx][threshold_idx],
            precisions_by_class[class_idx][threshold_idx],
            fones_by_class[class_idx][threshold_idx],
            ftwos_by_class[class_idx][threshold_idx]
        ))
        print()


def calculate_iaa(df, max_distance=64):
    print(' ' * 20 + ('\t' * 2).join(('Recall', 'Precision', 'F1')))
    print()

    for class_idx, label in enumerate(labels):
        annotator_positions = []
        df_filtered = df[df['External ID'].apply(lambda s: s.startswith('Cam_0_2019-07-26T19:31:09.132765Z'))]
        for idx, row in df_filtered.iterrows():
            positions = row['Label'][labels[class_idx]]
            positions = np.stack([np.array((l['geometry']['y'], l['geometry']['x'])) for l in positions])

            annotator_positions.append(positions)

        iaa_metrics = []
        for positions_a, positions_b in itertools.combinations(annotator_positions, 2):
            assignments = match_positions(positions_a, positions_b, max_distance=max_distance)
            recall, precision, fone, _ = calculate_metrics(
                len(assignments), len(positions_a), len(positions_b)
            )
            iaa_metrics.append((recall, precision, fone))
        iaa_metrics = np.stack(iaa_metrics)

        print(label)
        print('     Mean:' + ' ' * 10 + ('\t' * 2).join(
            map(lambda f: '{:.3f}'.format(f), list(iaa_metrics.mean(axis=0))))
        )
        print('     Std:' + ' ' * 11 + ('\t' * 2).join(
            map(lambda f: '{:.3f}'.format(f), list(iaa_metrics.std(axis=0))))
        )
        print()
