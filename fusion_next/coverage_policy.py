"""Conservative source replacement; coverage parameters are immutable."""
import numpy as np

GEOMETRY_COLUMNS = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13]

def replacement_gate(old_score, new_score, color_difference, evidence_distance,
                     depth_error, depth, confidence, minimum_gain=1.25, maximum_evidence_distance=.08):
    return ((new_score > np.maximum(old_score * minimum_gain, 1e-6))
            & (color_difference < .14) & (evidence_distance < maximum_evidence_distance)
            & (np.abs(depth_error) < .04 + .01 * depth) & (confidence > .8))

def assert_coverage_unchanged(before, after):
    if before.shape != after.shape or not np.array_equal(before[:, GEOMETRY_COLUMNS], after[:, GEOMETRY_COLUMNS]):
        raise ValueError('Coverage guard: point order, positions, opacity and covariance must remain identical')

def photometric_accept(delta, valid):
    mean=(delta*valid).sum(1)/np.maximum(valid.sum(1),1)
    worst=np.max(np.where(valid,delta,-np.inf),axis=1)
    return (valid.sum(1)>=2)&(mean<-.0005)&(worst<.005),mean,worst
