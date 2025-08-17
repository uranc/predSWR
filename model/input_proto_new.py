# ----------------------------- Imports --------------------------------------
import os
import pdb
import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_dilation
from tensorflow.keras.utils import timeseries_dataset_from_array

# cnn-ripple utils from your repo
from model.cnn_ripple_utils import (
    load_lab_data, process_LFP, split_data,
)

from numpy.lib.stride_tricks import sliding_window_view


def sliding_window_zscore(x32, win=1250, eps=1e-8):
    x64 = x32.astype(np.float64)
    n, c = x64.shape
    cs   = np.cumsum(np.pad(x64, ((1,0),(0,0))), axis=0, dtype=np.float64)
    cs2  = np.cumsum(np.pad(x64**2, ((1,0),(0,0))), axis=0, dtype=np.float64)
    idx0 = np.clip(np.arange(n) - win + 1, 0, None)
    L    = (np.arange(n) - idx0 + 1).astype(np.float64)
    w_sum  = cs [1:] - cs [idx0]
    w_sum2 = cs2[1:] - cs2[idx0]
    mu  = w_sum / L[:, None]
    var = w_sum2 / L[:, None] - mu**2
    sig = np.sqrt(np.maximum(var, 0.0))
    return ((x64 - mu) / (sig + eps)).astype(np.float32)


# ----------------- Triplet-sampling helpers ---------------------------------
def _build_window_catalogue_2T(labels, win_len, grace, anchor_thr, pos_thr, min_neg_gap, multiplier=2):
    """
    Build window catalogue for nT extraction where:
    - Extract multiplier*win_len samples starting at returned indices
    - Overlap is computed only on the last win_len samples (label region)
    - Safety is checked for the entire multiplier*win_len window
    """
    total_win_len = multiplier * win_len

    # 1) grace mask used ONLY for overlap scoring (not for event identity)
    pos_mask = labels.astype(bool).copy()
    onsets = np.flatnonzero(np.diff(np.r_[0, labels]) == 1)
    for onset in onsets:
        pos_mask[max(0, onset - grace):onset] = True

    # 2) overlap on last win_len of the nT window
    max_start_idx = len(labels) - total_win_len
    if max_start_idx < 0:
        print(f"Warning: Data too short for {multiplier}T windows. Need {total_win_len}, got {len(labels)}")
        return np.array([]), np.array([]), np.array([])

    overlap_ratios = np.zeros(max_start_idx + 1, dtype=np.float32)
    for i in range(max_start_idx + 1):
        label_start = i + (multiplier - 1) * win_len
        label_end   = i + multiplier * win_len
        label_region = pos_mask[label_start:label_end]
        overlap_ratios[i] = np.sum(label_region) / win_len

    # 3) safe region for negatives: whole nT window must be safe
    safe_mask = ~binary_dilation(pos_mask, structure=np.ones(min_neg_gap * 2 + 1))
    safe_nT = np.zeros(max_start_idx + 1, dtype=bool)
    for i in range(max_start_idx + 1):
        safe_nT[i] = np.all(safe_mask[i:i + total_win_len])

    # 4) catalogues (positives include anchors; we’ll filter at sampling time)
    valid_indices = np.arange(max_start_idx + 1)
    anchor_idx   = valid_indices[overlap_ratios >= anchor_thr]
    positive_idx = valid_indices[overlap_ratios >= pos_thr]  # include anchors
    negative_idx = valid_indices[(overlap_ratios == 0) & safe_nT]

    return anchor_idx, positive_idx, negative_idx


# -------------------------- Data wrangling ----------------------------------
def rippleAI_prepare_training_data(train_LFPs, train_GTs,
                                   val_LFPs,   val_GTs,
                                   sf=1250, new_sf=1250,
                                   channels=np.arange(0, 8),
                                   zscore=True, process_online=False, use_band=None):
    """
    Down-sample, band-filter, z-score and concatenate training sessions;
    return processed training LFP, concatenated GT, list of processed
    validation sessions and their GT.
    """
    assert len(train_LFPs) == len(train_GTs)
    assert len(val_LFPs)   == len(val_GTs)
    assert len(train_LFPs) + len(val_LFPs) == len(sf)

    retrain_LFP, retrain_GT = [], []
    offset     = 0.0
    counter_sf = 0

    # ---- training sessions concatenated ------------------------------------
    for LFP, GT in zip(train_LFPs, train_GTs):
        print('Original training data shape:', LFP.shape, '| sf:', sf[counter_sf])
        if process_online:
            LFP = LFP[::int(sf[counter_sf] // new_sf), :].astype(np.float32)
            print('Sub-sampled data shape:', LFP.shape, '| new sf:', new_sf)
            aux = sliding_window_zscore(LFP, win=new_sf, eps=1e-8)
            print('Sliding-window z-scored data shape:', aux.shape, '| running window:', new_sf)
        else:
            aux = process_LFP(LFP, ch=channels, sf=sf[counter_sf],
                              new_sf=new_sf, use_zscore=False, use_band=use_band)
            if zscore:
                aux = (aux - aux.mean(0)) / aux.std(0)

        GT = GT + offset  # shift events by accumulated seconds

        if len(retrain_LFP) == 0:
            retrain_LFP = aux
            retrain_GT  = GT
        else:
            retrain_LFP = np.vstack([retrain_LFP, aux])
            retrain_GT  = np.vstack([retrain_GT, GT])

        offset     += len(aux) / new_sf
        counter_sf += 1

    # ---- validation sessions processed separately --------------------------
    norm_val_LFP = []
    for LFP in val_LFPs:
        if process_online:
            LFP = LFP[::int(sf[counter_sf] // new_sf), :].astype(np.float32)
            print('Sub-sampled data shape:', LFP.shape, '| new sf:', new_sf)
            tmp = sliding_window_zscore(LFP, win=new_sf, eps=1e-8)
            print('Sliding-window z-scored data shape:', tmp.shape, '| running window:', new_sf)
        else:        
            print('Original validation data shape:', LFP.shape, '| sf:', sf[counter_sf])
            tmp = process_LFP(LFP, ch=channels, sf=sf[counter_sf],
                            new_sf=new_sf, use_zscore=False, use_band=use_band)
            if zscore:
                tmp = (tmp - tmp.mean(0)) / tmp.std(0)
        norm_val_LFP.append(tmp)
        counter_sf += 1

    return retrain_LFP, retrain_GT, norm_val_LFP, val_GTs


# -------------------------- Top-level loader --------------------------------
def rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=False, use_band=None):
    """
    Build train / test tf.data.Datasets of triplets.
    """
    onset_params = {
        'USE_ONSET_SHIFTED_LABELS': False,
        'ONSET_SHIFT_SAMPLES': -10,
        'ONSET_EXTEND_SAMPLES': 30,

        # Triplet sampling
        'WINDOW_MULTIPLIER': 2,
        'JITTER': 20,
        'MIN_NEG_GAP': 20,
        'ANCHOR_THR': 0.85,
        'POS_THR': 0.5,
        'GRACE': 20,

        # Strict positive selection
        'POS_SAME_EVENT': True,
        'POS_EXCLUDE_ANCHORS': True,  # never allow P==A
        
        # NEW: onset/center positive mixing
        'POS_CENTER_PROB': 0.33,   # 33% of positives centered, rest onset-biased
        'POS_ONSET_PRE': 10,       # allow up to 10 samples before onset
        'POS_ONSET_POST': 30,      # allow up to 30 after onset
    }
    params.update(onset_params)

    # ---- sample-shift parsing for causal architectures ----------------------
    if 'TYPE_ARCH' in params and 'Shift' in params['TYPE_ARCH']:
        shift_ms     = float(params['TYPE_ARCH'].split('Shift')[1][:2])
        sample_shift = int(shift_ms / 1000 * params['SRATE'])
        print('Using Shift:', sample_shift, 'samples')
    else:
        sample_shift = 0

    # ---- gather raw sessions ------------------------------------------------
    train_LFPs, train_GTs, all_SFs = [], [], []
    if mode == 'train':
        for mouse, fig_id in [('Amigo2', '16847521'), ('Som2', '16856137')]:
            path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/',
                                'Downloaded_data', mouse, f'figshare_{fig_id}')
            LFP, GT = load_lab_data(path)
            train_LFPs.append(LFP)
            train_GTs.append(GT)
            all_SFs.append(30000)

    val_LFPs, val_GTs = [], []
    if mode == 'test':
        for mouse, fig_id in [('Dlx1', '14959449'),
                              ('Thy7', '14960085'),
                              ('Calb20', '20072252')]:
            path = os.path.join('/mnt/hpc/projects/OWVinckSWR/Dataset/rippl-AI-data/',
                                'Downloaded_data', mouse, f'figshare_{fig_id}')
            LFP, GT = load_lab_data(path)
            val_LFPs.append(LFP)
            val_GTs.append(GT)
            all_SFs.append(30000)

    # ---- preprocess ---------------------------------------------------------
    train_data, train_GT, val_data, val_GT = rippleAI_prepare_training_data(
        train_LFPs, train_GTs, val_LFPs, val_GTs,
        sf=all_SFs, new_sf=params['SRATE'],
        zscore=preprocess, process_online=process_online, use_band=use_band
    )

    if mode == 'test':
        print('Returning validation data only')
        val_data = [v.astype('float32') for v in val_data]
        return val_data, val_GT

    # ---- train / test split -------------------------------------------------
    train_data = train_data.astype('float32')
    te_x, ev_test, tr_x, ev_train = split_data(
        train_data, train_GT,
        sf=params['SRATE'], split=0.7
    )

    sf = params['SRATE']
    train_lab = np.zeros(len(tr_x), dtype=np.float32)
    for e in ev_train:
        train_lab[int(sf * e[0]):int(sf * e[1]) + sample_shift] = 1

    test_lab = np.zeros(len(te_x), dtype=np.float32)
    for e in ev_test:
        test_lab[int(sf * e[0]):int(sf * e[1]) + sample_shift] = 1

    label_ratio = train_lab.sum() / len(train_lab)

    # ---- Triplet dataset (Sequence) ----------------------------------------
    train_seq = TripletSequence(tr_x, train_lab, params)
    train_ds  = train_seq

    # ---- Validation windows (2T input, last T labels) ----------------------
    sample_length = int(params['NO_TIMEPOINTS'] * params.get('WINDOW_MULTIPLIER', 2))
    label_length  = int(params['NO_TIMEPOINTS'])
    label_skip    = int(sample_length - label_length)
    stride_step   = 8  # or params.get('STRIDE_STEP', 32)

    test_x = timeseries_dataset_from_array(
        te_x, None,
        sequence_length=sample_length,
        sequence_stride=stride_step,
        batch_size=None,
        shuffle=False
    )
    test_y = timeseries_dataset_from_array(
        test_lab[label_skip:].reshape(-1, 1), None,
        sequence_length=label_length,
        sequence_stride=stride_step,
        batch_size=None,
        shuffle=False
    )

    test_ds = tf.data.Dataset.zip((test_x, test_y))
    test_ds = test_ds.batch(params['BATCH_SIZE'] * 6, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    total_windows = (len(te_x) - sample_length) // stride_step + 1
    val_steps = total_windows // (params['BATCH_SIZE'] * 6)

    print(f"Validation: {len(te_x)} samples, {total_windows} windows")
    print(f"Validation steps: {val_steps} (dropping partial batch)")

    dataset_params = params.copy()
    dataset_params['VAL_STEPS'] = val_steps
    dataset_params['ESTIMATED_STEPS_PER_EPOCH'] = len(train_seq)
    return train_ds, test_ds, label_ratio, dataset_params


# --------------------- Convenience loader for Allen -------------------------
def load_allen(indices=np.int32(np.linspace(49, 62, 8))):
    raw = np.load('/cs/projects/OWVinckSWR/Carmen/LFP_extracted/sanity_check/raw_lfp_fc.npy')
    indices[::-1].sort()
    LFP = raw[:, indices]
    return process_LFP(LFP, sf=1250, channels=np.arange(0, 8))


# ------------------------ Onset-shifted labels ------------------------------
def _create_onset_shifted_labels(original_labels, shift_samples=-8, extend_samples=15):
    shifted_labels = original_labels.copy()
    onsets = np.where(np.diff(np.concatenate(([0], original_labels, [0]))) == 1)[0]
    if len(onsets) == 0:
        return shifted_labels
    for onset in onsets:
        new_start = max(0, onset + shift_samples)  # shift_samples can be negative
        new_end   = min(len(original_labels), onset + extend_samples)
        if new_start < onset and np.sum(original_labels[new_start:onset]) == 0:
            shifted_labels[new_start:new_end] = 1
    return shifted_labels


# -------------------------------------------------------
# TripletSequence (strict same-event positives)
# -------------------------------------------------------
class TripletSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence that:
      1) Rebuilds the anchor/positive/negative catalogue each epoch
      2) Samples triplets on the fly (with jitter, onset-shift, etc.)
      3) Enforces positive windows come from the SAME EVENT as the anchor (P != A)
    """
    def __init__(self, data, labels, params):
        self.data   = data
        self.labels = labels.astype(np.int8)  # raw binary labels (no grace)
        self.p      = params
        self.B      = params['BATCH_SIZE']
        self.rng    = np.random.default_rng()

        # Precompute event ids from raw labels
        self._evt_starts, self._evt_ends, self._evt_id = self._compute_events(self.labels)
        self._rebuild_catalogue()

    # --------- events ---------
    @staticmethod
    def _compute_events(labels):
        d = np.diff(np.r_[0, labels, 0])
        starts = np.flatnonzero(d == 1)
        ends   = np.flatnonzero(d == -1)
        eid = -np.ones(len(labels), np.int32)
        for k, (s, e) in enumerate(zip(starts, ends)):
            eid[s:e] = k
        return starts, ends, eid

    def _label_event_id_for_start(self, start_idx, W, mul):
        """Event id inside the LABEL region (last T of nT) based on TRUE labels (no grace)."""
        s = start_idx + (mul - 1) * W
        e = start_idx + mul * W
        seg = self._evt_id[s:e]
        seg = seg[seg >= 0]
        return int(seg[0]) if seg.size else -1

    def _jitter_within_event(self, start_idx, eid, W, mul, jitter):
        """Clamp jitter so [label_start,label_end) stays inside event eid."""
        if eid < 0 or jitter <= 0:
            return start_idx
        Wtot = W * mul
        lbl_s = start_idx + (mul - 1) * W
        lbl_e = start_idx + Wtot
        e_s   = int(self._evt_starts[eid]); e_e = int(self._evt_ends[eid])

        jmin  = e_s - lbl_s
        jmax  = e_e - lbl_e

        jmin = max(jmin, -jitter)
        jmax = min(jmax,  jitter)
        if jmin > jmax:
            j = 0
        else:
            j = int(self.rng.integers(jmin, jmax + 1))

        return max(0, min(start_idx + j, len(self.data) - Wtot))

    # --------- safe extract ---------
    def _safe_extract_window(self, idx, data, labels, win_total, win_label):
        if idx < 0 or idx + win_total > len(data):
            idx = max(0, min(idx, len(data) - win_total))
        dw = data[idx:idx+win_total]
        off = win_total - win_label
        lw = labels[idx+off:idx+win_total]
        return dw, lw

    # --------- catalogue (per epoch) ---------
    def _rebuild_catalogue(self):
        tf.print("Rebuilding triplet catalogue...")
        W   = self.p['NO_TIMEPOINTS']
        mul = self.p.get('WINDOW_MULTIPLIER', 2)

        a_idx_raw, p_all_raw, n_idx = _build_window_catalogue_2T(
            self.labels, W,
            self.p['GRACE'], self.p['ANCHOR_THR'],
            self.p['POS_THR'], self.p['MIN_NEG_GAP'],
            multiplier=mul
        )

        # Filter anchors to ensure a TRUE event (no grace-only anchors)
        a_idx = []
        for i in a_idx_raw:
            if self._label_event_id_for_start(int(i), W, mul) >= 0:
                a_idx.append(int(i))
        a_idx = np.array(a_idx, dtype=int)

        # Build positive pools by event (only indices with a TRUE event id)
        pos_by_event = {}
        for i in p_all_raw:
            eid = self._label_event_id_for_start(int(i), W, mul)
            if eid >= 0:
                pos_by_event.setdefault(eid, set()).add(int(i))

        # Enforce P != A
        if self.p.get('POS_EXCLUDE_ANCHORS', True):
            for eid, pool in pos_by_event.items():
                pos_by_event[eid] = set(pool)  # copy to be safe

        # Retain only anchors that actually have a same-event positive candidate (≠ anchor)
        anchors_with_pool = []
        for a in a_idx:
            eid = self._label_event_id_for_start(int(a), W, mul)
            pool = pos_by_event.get(eid, set())
            if self.p.get('POS_EXCLUDE_ANCHORS', True) and a in pool:
                pool = set(pool)
                pool.discard(a)
            if len(pool) > 0:
                anchors_with_pool.append(a)

        self.anchor_idx = np.array(anchors_with_pool, dtype=int)
        self.neg_idx    = n_idx
        self.pos_by_event = {eid: np.array(sorted(list(pool)), dtype=int) for eid, pool in pos_by_event.items()}

        if self.neg_idx.size == 0:
            raise RuntimeError(
                "TripletSequence: no valid negatives. Decrease MIN_NEG_GAP or thresholds."
            )

        if len(self.anchor_idx) == 0:
            raise RuntimeError(
                "TripletSequence: no anchors have a same-event positive. "
                "Relax thresholds (ANCHOR_THR/POS_THR), increase GRACE, "
                "or increase WINDOW_MULTIPLIER."
            )

        # Precompute anchor event ids for speed
        self.anchor_eids = np.array([self._label_event_id_for_start(i, W, mul) for i in self.anchor_idx], dtype=int)

        # Steps/epoch
        self.steps = max(1, len(self.anchor_idx) // self.B)

    def __len__(self):
        return self.steps

    def _event_bounds(self, eid):
        return int(self._evt_starts[eid]), int(self._evt_ends[eid])

    def _pick_positive_index(self, a_idx, pool, W, mul):
        """
        Pick a positive from the same-event pool, biased either to ONSET or CENTER.
        Falls back to a random pool pick if no candidates in the bias window.
        """
        if pool.size == 0:
            return None

        # event info
        eid = self._label_event_id_for_start(a_idx, W, mul)
        if eid < 0:
            return int(self.rng.choice(pool))

        e_s, e_e = self._event_bounds(eid)
        onset = e_s
        center = (e_s + e_e) // 2

        # label-region mapping for a start index i:
        # label_start = i + (mul-1)*W ; label_end = i + mul*W ; label_mid = label_start + W//2
        def label_mid(i):
            return i + (mul - 1) * W + (W // 2)

        # choose mode
        center_prob = float(self.p.get('POS_CENTER_PROB', 0.33))
        use_center  = (self.rng.random() < center_prob)

        if use_center:
            half_span = max(1, W // 6)
            lo, hi = center - half_span, center + half_span
        else:
            pre  = max(0, int(self.p.get('POS_ONSET_PRE', 10)))
            post = max(1, int(self.p.get('POS_ONSET_POST', 30)))  # ensure ≥1
            lo, hi = onset - pre, onset + post

        # clamp the target window inside the true event
        lo = max(lo, e_s)
        hi = min(hi, e_e)
        if lo > hi:
            lo, hi = e_s, e_e  # fall back to whole event band

        # candidates whose label midpoint falls in [lo, hi]
        mids = label_mid(pool)
        mask = (mids >= lo) & (mids <= hi)
        cand = pool[mask]
        if cand.size == 0:
            cand = pool  # graceful fallback

        # avoid picking the exact anchor index if requested
        if self.p.get('POS_EXCLUDE_ANCHORS', True) and cand.size > 1:
            cand = cand[cand != a_idx] if np.any(cand != a_idx) else pool

        return int(self.rng.choice(cand))


    def _sample_triplet(self):
        W   = self.p['NO_TIMEPOINTS']
        mul = self.p.get('WINDOW_MULTIPLIER', 2)
        Wtot = W * mul
        jitter = int(self.p.get('JITTER', 8))

        # --- sample anchor
        a = int(self.rng.choice(self.anchor_idx))
        a_eid = self._label_event_id_for_start(a, W, mul)

        # --- pool for same-event positives (P != A)
        pool = self.pos_by_event.get(a_eid, np.empty((0,), dtype=int))
        if pool.size == 0:
            raise RuntimeError("Invariant violated: empty positive pool for chosen anchor event.")

        # NEW: choose positive biased to onset or center
        p = self._pick_positive_index(a, pool, W, mul)
        if p is None:
            raise RuntimeError("No valid positive candidate after biasing.")

        # --- choose negative
        n = int(self.rng.choice(self.neg_idx))

        # --- jitter inside event for A and P (clamped)
        a = self._jitter_within_event(a, a_eid, W, mul, jitter)
        p_eid = self._label_event_id_for_start(p, W, mul)
        p = self._jitter_within_event(p, p_eid, W, mul, jitter)

        # Negatives can jitter freely
        if jitter > 0:
            n = max(0, min(n + int(self.rng.integers(-jitter, jitter+1)), len(self.data) - Wtot))

        # extract windows
        Ad, Al = self._safe_extract_window(a, self.data, self.labels, Wtot, W)
        Pd, Pl = self._safe_extract_window(p, self.data, self.labels, Wtot, W)
        Nd, Nl = self._safe_extract_window(n, self.data, self.labels, Wtot, W)

        # optional onset shift
        if self.p.get('USE_ONSET_SHIFTED_LABELS', False):
            shift, extend = self.p['ONSET_SHIFT_SAMPLES'], self.p['ONSET_EXTEND_SAMPLES']
            Al = _create_onset_shifted_labels(Al, shift, extend)
            Pl = _create_onset_shifted_labels(Pl, shift, extend)

        # ensure channel dim
        if Ad.ndim == 1: Ad = Ad[:, None]
        if Pd.ndim == 1: Pd = Pd[:, None]
        if Nd.ndim == 1: Nd = Nd[:, None]

        # hard check: same-event after jitter
        if self.p.get('POS_SAME_EVENT', True):
            ae = self._label_event_id_for_start(a, W, mul)
            pe = self._label_event_id_for_start(p, W, mul)
            if ae < 0 or pe != ae:
                raise RuntimeError("Invariant violated: positive not in same event as anchor after jitter.")

        return (Ad.astype(np.float32), Al.astype(np.float32)), \
               (Pd.astype(np.float32), Pl.astype(np.float32)), \
               (Nd.astype(np.float32), Nl.astype(np.float32))

    def __getitem__(self, idx):
        triplets = [self._sample_triplet() for _ in range(self.B)]
        A, P, N = zip(*triplets)
        Ad, Al = zip(*A)
        Pd, Pl = zip(*P)
        Nd, Nl = zip(*N)
        data_batch = np.concatenate([np.stack(Ad), np.stack(Pd), np.stack(Nd)], axis=0)
        labels_batch = np.concatenate([np.stack(Al)[:, :, None],
                                       np.stack(Pl)[:, :, None],
                                       np.stack(Nl)[:, :, None]], axis=0)
        return data_batch, labels_batch

    def on_epoch_end(self):
        self._rebuild_catalogue()
