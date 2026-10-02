"""Strict, dependency-free integrity checks for retained four-choice ARC rows.

Checks saved evidence only. Does not train, download data, or authorize test use.
"""
import math
from collections import Counter


def verify_rows(rows, expected_ids, expected_labels, *, tolerance=1e-5):
    if not rows or len(rows) != len(expected_ids) or len(rows) != len(expected_labels):
        raise ValueError('row count mismatch or empty input')
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError('duplicate canonical IDs')
    histogram = Counter()
    correct = 0
    maxima = []
    for row, item_id, label in zip(rows, expected_ids, expected_labels, strict=True):
        if row.get('id') != item_id:
            raise ValueError('row identity/order mismatch')
        if type(label) is not int or not 0 <= label < 4:
            raise ValueError('invalid canonical label')
        if type(row.get('label')) is not int or row['label'] != label:
            raise ValueError('label differs from canonical validation data')
        prediction = row.get('prediction')
        if type(prediction) is not int or not 0 <= prediction < 4:
            raise ValueError('invalid predicted class')
        probabilities = row.get('probabilities')
        if not isinstance(probabilities, list) or len(probabilities) != 4:
            raise ValueError('invalid probability vector')
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError('invalid probability value')
        if abs(sum(probabilities) - 1) > tolerance:
            raise ValueError('probabilities do not sum to one')
        if max(range(4), key=probabilities.__getitem__) != prediction:
            raise ValueError('prediction/probability mismatch')
        histogram[prediction] += 1
        correct += prediction == label
        maxima.append(max(probabilities))
    return {'n': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
            'prediction_support': len(histogram), 'prediction_histogram': dict(sorted(histogram.items())),
            'largest_predicted_class_share': max(histogram.values()) / len(rows),
            'mean_max_probability': sum(maxima) / len(rows)}
