# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from .TADN import TADN, SingleBranchTransformer, DualBranchTransformer
from pydantic import BaseModel
from typing import Optional

class TrackerTransformerConfig(BaseModel):
    """Configuration for TADN transformer single-branch architecture"""

    type: str = "single"
    nhead: int = 4
    encoder_num_layers: int = 3
    decoder_num_layers: int = 3

    def get_transformer(self, d_model):
        """Initializes a single-branch TADN Transformer instance"""
        params_dict = self.dict()
        del params_dict["type"]
        return SingleBranchTransformer(d_model, **params_dict)



class TrackerEmbeddingConfig(BaseModel):
    """Configuration for TADN embedding parameters"""

    dim_multiplier: int = 2
    app_dim: int = 512
    app_embedding_dim: int = 64
    spatial_embedding_dim: int = 64
    spatial_memory_mask_weight: Optional[float] = None


class TrackerNullTargetConfig(BaseModel):
    """Configuration for TADN null-target options"""

    null_target_idx: int = -1


class Tracker:

    def __init__(self, metric, max_iou_distance=0.9, max_age=30, n_init=3, _lambda=0):

        self.embedding_params = TrackerEmbeddingConfig()


        self.null_target_params = TrackerNullTargetConfig()
        # self.transformer_params = TrackerTransformerConfig()
        self.transformer_params = {'nhead': 4, 'encoder_num_layers': 3, 'decoder_num_layers': 3}
        self.normalize_transformer_outputs = False
        self.TADNModel = self.get_tracker()
        # self.params = {
        # "transformer_params": {
        #  "type": "dual",
        #  "nhead": 2,
        #  "encoder_num_layers": 2,
        #  "decoder_num_layers": 2
        #     },
        # "null_target_params": {
        #  "null_target_idx": -1
        #     },
        # "normalize_transformer_outputs": False
        # }

    def get_tracker(self):
        """Initialize a TADN module given config options"""
        d_model = (
            self.embedding_params.app_embedding_dim
            + self.embedding_params.spatial_embedding_dim
        )
        # transformer_model = self.transformer_params.get_transformer(d_model)
        transformer_model = SingleBranchTransformer(d_model, **self.transformer_params)
        tracker_model = TADN(
            transformer_model=transformer_model,
            embedding_params=self.embedding_params.dict(),
            null_target_params=self.null_target_params.dict(),
            normalize_transformer_outputs=self.normalize_transformer_outputs,
        )
        return tracker_model

    def predict(self):


    def increment_ages(self):


    def update(self, detections, classes):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)
        test_match = self.TADNModel(self.tracks, detections)
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx], classes[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], classes[detection_idx].item())
        self.tracks = [t for t in self.tracks if not t.is_deleted()]        # confirm+tentative

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            if len(track.features) > 5:
                track.features.pop(0)
            # track.features = []
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)
        # for track in self.tracks:
        #     if track.is_confirmed():
        #         print('Test:', track.track_id, len(track.features[-1]))

    # pos+cos distance cost matrix including threshold
    def _full_cost_metric(self, tracks, dets, track_indices, detection_indices):
        """
        This implements the full lambda-based cost-metric. However, in doing so, it disregards
        the possibility to gate the position only which is provided by
        linear_assignment.gate_cost_matrix(). Instead, I gate by everything.
        Note that the Mahalanobis distance is itself an unnormalised metric. Given the cosine
        distance being normalised, we employ a quick and dirty normalisation based on the
        threshold: that is, we divide the positional-cost by the gating threshold, thus ensuring
        that the valid values range 0-1.
        Note also that the authors work with the squared distance. I also sqrt this, so that it
        is more intuitive in terms of values.
        """
        # Compute First the Position-based Cost Matrix
        pos_cost = np.empty([len(track_indices), len(detection_indices)])
        msrs = np.asarray([dets[i].to_xyah() for i in detection_indices])
        # pos_cost = np.sqrt(tracks[:].mean[:0] - msrs[:, 0])
        for row, track_idx in enumerate(track_indices):

            pos_cost[row, :] = np.sqrt(
                self.kf.gating_distance(
                    tracks[track_idx].mean, tracks[track_idx].covariance, msrs, False
                )
            ) / self.GATING_THRESHOLD

        pos_gate = pos_cost > 1.0
        # Now Compute the Appearance-based Cost Matrix
        app_cost = self.metric.distance(
            np.array([dets[i].feature for i in detection_indices]),
            np.array([tracks[i].track_id for i in track_indices]),
        )

        app_gate = app_cost > self.metric.matching_threshold
        # Now combine and threshold
        cost_matrix = self._lambda * pos_cost + (1 - self._lambda) * app_cost
        cost_matrix[np.logical_or(pos_gate, app_gate)] = linear_assignment.INFTY_COST
        # print("app_gate:", app_gate, app_gate.shape, "pos_gate:", pos_gate, pos_gate.shape, "\n")
        # print("pos_cost:", pos_cost, pos_cost.shape, "app_cost:", app_cost, app_cost.shape, "cost_matrix:", cost_matrix,
        #       cost_matrix.shape)
        # Return Matrix
        return cost_matrix

    def _match(self, detections):
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            self._full_cost_metric,
            linear_assignment.INFTY_COST - 1,  # no need for self.metric.matching_threshold here,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, class_id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, class_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1