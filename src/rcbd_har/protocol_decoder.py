from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

from rcbd_har.temporal_decoder import temporal_segments


@dataclass(frozen=True)
class ProtocolDecodeResult:
    predictions: np.ndarray
    class_biases: np.ndarray
    class_counts: np.ndarray
    constraints_satisfied: bool


@njit(cache=True)
def _biased_viterbi_segment(
    log_emissions: np.ndarray,
    log_transition: np.ndarray,
    class_biases: np.ndarray,
) -> np.ndarray:
    length, num_classes = log_emissions.shape
    previous = log_emissions[0] + class_biases
    backpointers = np.empty((length, num_classes), dtype=np.int16)
    for position in range(1, length):
        current = np.empty(num_classes, dtype=np.float64)
        for destination in range(num_classes):
            best_score = -np.inf
            best_source = 0
            for source in range(num_classes):
                score = previous[source] + log_transition[source, destination]
                if score > best_score:
                    best_score = score
                    best_source = source
            current[destination] = (
                log_emissions[position, destination]
                + class_biases[destination]
                + best_score
            )
            backpointers[position, destination] = best_source
        previous = current
    decoded = np.empty(length, dtype=np.int64)
    decoded[-1] = int(np.argmax(previous))
    for position in range(length - 1, 0, -1):
        decoded[position - 1] = backpointers[position, decoded[position]]
    return decoded


def _validate_inputs(
    probabilities: np.ndarray,
    frame: Any,
    transition_matrix: np.ndarray,
    minimum_counts: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    transition = np.asarray(transition_matrix, dtype=np.float64)
    bounds = np.asarray(minimum_counts, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(frame):
        raise ValueError("probabilities must be [rows, classes] and align with frame")
    if transition.shape != (values.shape[1], values.shape[1]):
        raise ValueError("transition_matrix does not align with probabilities")
    if bounds.shape != (values.shape[1],) or np.any(bounds < 0):
        raise ValueError("minimum_counts must contain one non-negative value per class")
    if strength < 0:
        raise ValueError("strength must be non-negative")
    if len(values) == 0:
        raise ValueError("probabilities cannot be empty")
    return values, transition, bounds


def protocol_constrained_decode(
    probabilities: np.ndarray,
    frame: Any,
    transition_matrix: np.ndarray,
    strength: float,
    minimum_counts: np.ndarray,
    *,
    max_rounds: int = 4,
    binary_steps: int = 10,
    initial_step: float = 0.25,
    max_bias: float = 16.0,
) -> ProtocolDecodeResult:
    """Decode with recording-level lower bounds using non-negative dual class biases.

    The constraints operate on unlabeled prediction counts. Labels are not accepted by this
    function, which makes accidental target-label leakage impossible at the decoder boundary.
    """
    values, transition, bounds = _validate_inputs(
        probabilities, frame, transition_matrix, minimum_counts, strength
    )
    if max_rounds <= 0 or binary_steps <= 0 or initial_step <= 0 or max_bias <= 0:
        raise ValueError("solver controls must be positive")
    num_classes = values.shape[1]
    segments = temporal_segments(frame)
    log_emissions = [
        np.log(np.clip(values[segment], 1e-12, 1.0)).astype(np.float64)
        for segment in segments
    ]
    log_transition = strength * np.log(np.clip(transition, 1e-12, 1.0))
    biases = np.zeros(num_classes, dtype=np.float64)

    def decode(current_biases: np.ndarray) -> np.ndarray:
        output = np.empty(len(values), dtype=np.int64)
        for segment, emissions in zip(segments, log_emissions, strict=True):
            output[segment] = _biased_viterbi_segment(
                emissions, log_transition, current_biases
            )
        return output

    predictions = decode(biases)
    for _ in range(max_rounds):
        counts = np.bincount(predictions, minlength=num_classes)
        unsatisfied = [
            class_index
            for class_index in range(num_classes)
            if bounds[class_index] > 0 and counts[class_index] < bounds[class_index]
        ]
        if not unsatisfied:
            break
        unsatisfied.sort(key=lambda index: counts[index] / bounds[index])
        for class_index in unsatisfied:
            counts = np.bincount(predictions, minlength=num_classes)
            if counts[class_index] >= bounds[class_index]:
                continue
            lower = float(biases[class_index])
            upper = max(initial_step, lower + initial_step)
            while upper < max_bias:
                candidate = biases.copy()
                candidate[class_index] = upper
                candidate_predictions = decode(candidate)
                if np.count_nonzero(candidate_predictions == class_index) >= bounds[class_index]:
                    break
                upper *= 2.0
            upper = min(upper, max_bias)
            for _ in range(binary_steps):
                midpoint = 0.5 * (lower + upper)
                candidate = biases.copy()
                candidate[class_index] = midpoint
                candidate_predictions = decode(candidate)
                if np.count_nonzero(candidate_predictions == class_index) >= bounds[class_index]:
                    upper = midpoint
                else:
                    lower = midpoint
            biases[class_index] = upper
            predictions = decode(biases)

    counts = np.bincount(predictions, minlength=num_classes)
    active = bounds > 0
    satisfied = bool(np.all(counts[active] >= bounds[active]))
    return ProtocolDecodeResult(predictions, biases, counts, satisfied)
