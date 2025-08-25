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

# ----------------------------- Utilities ------------------------------------
def sliding_window_zscore(x32, win=1250, eps=1e-8):
    x64 = x32.astype(np.float64)
    n, c = x32.shape
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

def ms_to_samples(ms, sr):
    return int(round((ms / 1000.0) * sr))

# --------------- Window catalogue (ABS counts; last-T measured) --------------
def build_window_catalogue_2T_abs(
    labels,
    win_len,                 # T in samples
    grace=0,
    anchor_min_samp=40,      # absolute samples in last-T to be an anchor
    pos_min_samp=20,         # absolute samples in last-T to be a positive
    min_neg_gap_samp=20,     # dilation gap (samples) around events
    multiplier=2,
):
    """
    Returns (anchor_idx, positive_idx, negative_idx) start indices for nT windows.
    Overlap is measured ONLY on the last T in ABSOLUTE SAMPLES (no percentages).
    Negatives require last-T overlap == 0 and the FULL nT inside a dilated safe gap.
    """
    labels = np.asarray(labels).astype(bool)
    N = len(labels)
    W = int(win_len)
    mul = int(multiplier)
    total = mul * W

    max_start = N - total
    if max_start < 0:
        return np.array([], int), np.array([], int), np.array([], int)

    # grace for overlap scoring (does not change TRUE labels)
    pos_mask = labels.copy()
    onsets = np.flatnonzero(np.diff(np.r_[0, labels.view(np.int8)]) == 1)
    for onset in onsets:
        pos_mask[max(0, onset - grace):onset] = True

    idx = np.arange(max_start + 1, dtype=int)
    last_starts = idx + (mul - 1) * W

    # ABSOLUTE overlap on last T
    abs_ov = np.fromiter((np.sum(pos_mask[s:s + W]) for s in last_starts),
                         dtype=np.int32, count=idx.size)

    # negatives: full nT must be safe (dilated gap)
    if min_neg_gap_samp < 1:
        safe_mask = ~pos_mask
    else:
        safe_mask = ~binary_dilation(pos_mask, structure=np.ones(min_neg_gap_samp * 2 + 1, dtype=bool))
    safe2T = np.fromiter((np.all(safe_mask[i:i + total]) for i in idx),
                         dtype=bool, count=idx.size)

    anchor_idx   = idx[abs_ov >= int(anchor_min_samp)]
    positive_idx = idx[abs_ov >= int(pos_min_samp)]
    negative_idx = idx[(abs_ov == 0) & safe2T]
    return anchor_idx, positive_idx, negative_idx

# --------------------------- Augmentation Helper ----------------------------
def augment_window(x, rng, p):
    """
    Apply stochastic augmentations to a single window (2T x C).
    Used for anchors & positives; negatives remain untouched.
    """
    if not p.get('USE_AUGMENT', False):
        return x

    x = np.array(x, dtype=np.float32, copy=True)

    # 1) Per-channel amplitude scaling
    if rng.random() < p.get('AUG_SCALE_PROB', 0.5):
        scale = rng.uniform(
            p.get('AUG_SCALE_MIN', 0.8),
            p.get('AUG_SCALE_MAX', 1.2),
            size=(1, x.shape[1])
        ).astype(np.float32)
        x *= scale

    # 2) Additive Gaussian noise
    if rng.random() < p.get('AUG_NOISE_PROB', 0.5):
        sigma = p.get('AUG_NOISE_STD', 0.05) * np.std(x)
        noise = rng.normal(0, sigma, size=x.shape).astype(np.float32)
        x += noise

    # 3) Channel dropout / corruption
    if rng.random() < p.get('AUG_DROP_PROB', 0.2):
        ch = rng.integers(0, x.shape[1])
        mode = rng.choice(["zero", "gauss", "swap"])
        if mode == "zero":
            x[:, ch] = 0
        elif mode == "gauss":
            x[:, ch] = rng.normal(0, np.std(x[:, ch]), size=x.shape[0]).astype(np.float32)
        elif mode == "swap":
            ch2 = rng.integers(0, x.shape[1])
            if ch2 != ch:
                x[:, ch] = x[:, ch2]

    # 4) Transient artifact (short spike/jump)
    if rng.random() < p.get('AUG_ARTIFACT_PROB', 0.1):
        dur = rng.integers(2, p.get('AUG_ARTIFACT_MAXLEN', 10))
        t0 = rng.integers(0, x.shape[0] - dur)
        amp = rng.uniform(3, 6) * np.std(x)
        x[t0:t0+dur] += amp * np.sign(rng.normal())

    return x.astype(np.float32)

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
            step = int(sf[counter_sf] // new_sf)
            LFP = LFP[::step, :].astype(np.float32)
            print('Sub-sampled data shape:', LFP.shape, '| new sf:', new_sf)
            aux = sliding_window_zscore(LFP, win=new_sf, eps=1e-8)
            print('Sliding-window z-scored data shape:', aux.shape, '| running window:', new_sf)
        else:
            aux = process_LFP(LFP, ch=channels, sf=sf[counter_sf],
                              new_sf=new_sf, use_zscore=False, use_band=use_band)
            if zscore:
                aux = (aux - aux.mean(0)) / (aux.std(0) + 1e-8)

        GT = GT + offset  # shift event times by accumulated seconds

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
            step = int(sf[counter_sf] // new_sf)
            LFP = LFP[::step, :].astype(np.float32)
            print('Sub-sampled data shape:', LFP.shape, '| new sf:', new_sf)
            tmp = sliding_window_zscore(LFP, win=new_sf, eps=1e-8)
            print('Sliding-window z-scored data shape:', tmp.shape, '| running window:', new_sf)
        else:
            print('Original validation data shape:', LFP.shape, '| sf:', sf[counter_sf])
            tmp = process_LFP(LFP, ch=channels, sf=sf[counter_sf],
                              new_sf=new_sf, use_zscore=False, use_band=use_band)
            if zscore:
                tmp = (tmp - tmp.mean(0)) / (tmp.std(0) + 1e-8)
        norm_val_LFP.append(tmp)
        counter_sf += 1

    return retrain_LFP, retrain_GT, norm_val_LFP, val_GTs

# -------------------------- Top-level loader --------------------------------
def rippleAI_load_dataset(params, mode='train', preprocess=True, process_online=False, use_band=None):
    """
    Build train / test datasets of triplets. Labels are *never* altered.
    """
    onset_params = {
    'USE_ONSET_SHIFTED_LABELS': False,

    # Windowing
    'WINDOW_MULTIPLIER': 2,      # nT (n=2)
    'JITTER': 0,                 # keep 0
    'SRATE': params.get('SRATE', 2500),

    # Catalogue thresholds (define in ms, convert to samples below)
    'ANCHOR_MIN_MS': 20,   # e.g. ≥20 ms overlap
    'POS_MIN_MS': 8,       # e.g. ≥8 ms overlap
    'GRACE_MS': 0,

    # Negatives
    'NEG_GAP_MS': 20,      # dilation gap (ms)

    # Positives
    'POS_SAME_EVENT': True,
    'POS_EXCLUDE_ANCHORS': True,

    # Debug safety
    'ASSERT_LABEL_INDEXING': True,
    }
    params.update(onset_params)

    # ---- convert ms thresholds to samples ----
    SR = params['SRATE']
    params['ANCHOR_MIN_SAMPLES'] = ms_to_samples(params['ANCHOR_MIN_MS'], SR)
    params['POS_MIN_SAMPLES']    = ms_to_samples(params['POS_MIN_MS'],    SR)
    params['NEG_GAP_SAMPLES']    = ms_to_samples(params['NEG_GAP_MS'],    SR)
    params['GRACE']              = ms_to_samples(params.get('GRACE_MS',0), SR)

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

    label_ratio = float(train_lab.sum()) / float(len(train_lab) + 1e-8)

    # ---- Triplet dataset (Sequence) ----------------------------------------
    train_seq = TripletSequence(tr_x, train_lab, params)
    train_ds  = train_seq

    # ---- Validation windows (2T input, last T labels) ----------------------
    sample_length = int(params['NO_TIMEPOINTS'] * params.get('WINDOW_MULTIPLIER', 2))
    label_length  = int(params['NO_TIMEPOINTS'])
    label_skip    = int(sample_length - label_length)
    stride_step   = 8

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
    dataset_params['LABEL_RATIO'] = label_ratio
    return train_ds, test_ds, label_ratio, dataset_params

# --------------------- Convenience loader for Allen -------------------------
def load_allen(indices=np.int32(np.linspace(49, 62, 8))):
    raw = np.load('/cs/projects/OWVinckSWR/Carmen/LFP_extracted/sanity_check/raw_lfp_fc.npy')
    indices[::-1].sort()
    LFP = raw[:, indices]
    return process_LFP(LFP, sf=1250, channels=np.arange(0, 8))

# --------------------------- Jitter-free Triplet -----------------------------
# --------------------------- Jitter-free Triplet (fast) ----------------------

class TripletSequence(tf.keras.utils.Sequence):
    """
    Triplet sampler (fast, pool-based):
      • Input: 2T windows; targets: last T labels (unaltered).
      • Anchors: last-T has >= ANCHOR_MIN_SAMPLES positives.
      • Positives: any other start from the SAME event with >= POS_MIN_SAMPLES
                   in last-T. If POS_EXCLUDE_ANCHORS=True, remove A from its pool.
      • Negatives: last-T overlap == 0 and full 2T inside a dilated safe gap.
      • Augmentations: applied to A & P only (see augment_window()).
      • No jitter; labels and signal always aligned.
    """
    def __init__(self, x, y, p):
        self.x = np.asarray(x, np.float32)
        self.y = np.asarray(y, np.int8)   # true labels; NEVER modified
        self.p = dict(p)
        self.B = int(self.p['BATCH_SIZE'])
        self.W = int(self.p['NO_TIMEPOINTS'])
        self.mul = int(self.p.get('WINDOW_MULTIPLIER', 2))
        self.Wtot = self.W * self.mul
        self.rng = np.random.default_rng()

        # thresholds already converted to samples in loader
        self.anchor_min = int(self.p['ANCHOR_MIN_SAMPLES'])
        self.pos_min    = int(self.p['POS_MIN_SAMPLES'])
        self.neg_gap    = int(self.p['NEG_GAP_SAMPLES'])

        # event bookkeeping on TRUE labels
        self._evt_starts, self._evt_ends, self._eid = self._compute_events(self.y)
        self._next_true = self._build_next_true(self.y.astype(bool))

        # build catalogue & per-event pools
        self._build_catalogue_fast()

        self.steps = max(1, len(self.anchor_idx) // self.B)
        self.debug_last_batch = None

    # ---------- helpers ----------
    @staticmethod
    def _compute_events(labels):
        lab = np.asarray(labels).astype(bool).astype(np.int8)
        d = np.diff(np.r_[0, lab, 0])
        starts = np.flatnonzero(d == 1)
        ends   = np.flatnonzero(d == -1)
        eid = -np.ones(len(labels), np.int32)
        for k, (s, e) in enumerate(zip(starts, ends)):
            eid[s:e] = k
        return starts, ends, eid

    @staticmethod
    def _build_next_true(mask_bool):
        """next_true[i] = first j>=i with mask_bool[j]==True; else -1."""
        N = len(mask_bool)
        nxt = np.full(N, -1, np.int32)
        last = -1
        for i in range(N - 1, -1, -1):
            if mask_bool[i]:
                last = i
            nxt[i] = last
        return nxt

    def _event_bounds(self, eid):
        return int(self._evt_starts[eid]), int(self._evt_ends[eid])

    def _event_id_for_starts(self, starts):
        """
        Vectorized: event id inside last-T of each 2T window start.
        If no event in last-T, returns -1.
        """
        starts = np.asarray(starts, dtype=np.int32)
        s_last = starts + (self.mul - 1) * self.W
        j = self._next_true[s_last]
        has = (j != -1) & (j < (s_last + self.W))
        ev = np.where(has, self._eid[j], -1).astype(np.int32)
        return ev

    @staticmethod
    def _slice_window(s, x, y, Wtot, W):
        s = max(0, min(int(s), len(x) - Wtot))
        xw = x[s:s + Wtot]
        if xw.ndim == 1:
            xw = xw[:, None]
        yw = y[s + (Wtot - W): s + Wtot]
        return xw, yw

    # ---------- catalogue (vectorized; O(A+P)) ----------
    def _build_window_catalogue_2T_abs(self, labels, win_len, grace, anchor_min_samp,
                                       pos_min_samp, min_neg_gap_samp, multiplier):
        """Internal copy to avoid external calls; same logic as your helper."""
        labels = np.asarray(labels).astype(bool)
        N = len(labels)
        W = int(win_len)
        mul = int(multiplier)
        total = mul * W
        max_start = N - total
        if max_start < 0:
            return np.array([], int), np.array([], int), np.array([], int)

        # grace (overlap scoring only)
        pos_mask = labels.copy()
        onsets = np.flatnonzero(np.diff(np.r_[0, labels.view(np.int8)]) == 1)
        for onset in onsets:
            pos_mask[max(0, onset - grace):onset] = True

        idx = np.arange(max_start + 1, dtype=int)
        last_starts = idx + (mul - 1) * W

        # absolute overlap on last T
        abs_ov = np.fromiter((np.sum(pos_mask[s:s + W]) for s in last_starts),
                             dtype=np.int32, count=idx.size)

        # negatives: entire 2T safe
        if min_neg_gap_samp < 1:
            safe_mask = ~pos_mask
        else:
            safe_mask = ~binary_dilation(
                pos_mask, structure=np.ones(min_neg_gap_samp * 2 + 1, dtype=bool)
            )
        safe2T = np.fromiter((np.all(safe_mask[i:i + total]) for i in idx),
                             dtype=bool, count=idx.size)

        anchor_idx   = idx[abs_ov >= int(anchor_min_samp)]
        positive_idx = idx[abs_ov >= int(pos_min_samp)]
        negative_idx = idx[(abs_ov == 0) & safe2T]
        return anchor_idx, positive_idx, negative_idx

    def _build_catalogue_fast(self):
        # 1) catalogue by absolute counts
        a_idx, p_idx, n_idx = self._build_window_catalogue_2T_abs(
            self.y, self.W,
            grace=int(self.p.get('GRACE', 0)),
            anchor_min_samp=self.anchor_min,
            pos_min_samp=self.pos_min,
            min_neg_gap_samp=self.neg_gap,
            multiplier=self.mul
        )
        if a_idx.size == 0 or p_idx.size == 0 or n_idx.size == 0:
            raise RuntimeError("Empty catalogue: relax overlap mins or NEG_GAP_MS.")

        # 2) vectorized event-id mapping for union(A,P)
        union = np.unique(np.r_[a_idx, p_idx]).astype(np.int32)
        ev_union = self._event_id_for_starts(union)
        ev_map = dict(zip(union.tolist(), ev_union.tolist()))

        # 3) bucket positives by event id (vectorized)
        p_eids = np.array([ev_map[int(s)] for s in p_idx], dtype=np.int32)
        pos_by_eid = {}
        for s, eid in zip(p_idx, p_eids):
            if eid >= 0:
                pos_by_eid.setdefault(int(eid), []).append(int(s))

        # 4) per-anchor pools from same event (fast; no nested scans)
        keep_anchors = []
        pos_pools = {}
        exclude_self = bool(self.p.get('POS_EXCLUDE_ANCHORS', True))
        a_eids = np.array([ev_map.get(int(s), -1) for s in a_idx], dtype=np.int32)

        for a, ae in zip(a_idx, a_eids):
            if ae < 0:
                continue
            pool = pos_by_eid.get(int(ae), [])
            if exclude_self:
                # remove anchor if it also appears in positives
                pool = [p for p in pool if p != int(a)]
            if len(pool) > 0:
                keep_anchors.append(int(a))
                pos_pools[int(a)] = np.asarray(pool, dtype=np.int32)

        if len(keep_anchors) == 0:
            raise RuntimeError("No anchors with same-event positives under current settings.")

        self.anchor_idx = np.asarray(keep_anchors, dtype=np.int32)
        self.neg_idx    = n_idx.astype(np.int32)
        self.pos_pools  = pos_pools

        # randomize for this epoch
        self.rng.shuffle(self.anchor_idx)
        self.rng.shuffle(self.neg_idx)
        for k in self.pos_pools:
            self.rng.shuffle(self.pos_pools[k])

    # ---------- sampling ----------
    def _sample_triplet(self):
        a = int(self.rng.choice(self.anchor_idx))
        pool = self.pos_pools[a]
        p = int(self.rng.choice(pool))
        n = int(self.rng.choice(self.neg_idx))

        Ad, Al = self._slice_window(a, self.x, self.y, self.Wtot, self.W)
        Pd, Pl = self._slice_window(p, self.x, self.y, self.Wtot, self.W)
        Nd, Nl = self._slice_window(n, self.x, self.y, self.Wtot, self.W)

        # augment A/P only
        if self.p['TYPE_ARCH'].find('Aug')>-1 and self.p['mode']=='train':
            print('Applying augmentations to A/P')
            # print('Applying augmentations to A/P')
            Ad = augment_window(Ad, self.rng, self.p)
            Pd = augment_window(Pd, self.rng, self.p)
        else:
            print('No augmentations applied')

        return (Ad, Al[:, None].astype(np.float32)), \
               (Pd, Pl[:, None].astype(np.float32)), \
               (Nd, Nl[:, None].astype(np.float32))

    # ---------- Keras API ----------
    def __len__(self):
        return self.steps

    def __getitem__(self, _):
        triplets = [self._sample_triplet() for _ in range(self.B)]
        A, P, N = zip(*triplets)
        Ad, Al = zip(*A)
        Pd, Pl = zip(*P)
        Nd, Nl = zip(*N)
        X = np.concatenate([np.stack(Ad), np.stack(Pd), np.stack(Nd)], axis=0)
        Y = np.concatenate([np.stack(Al), np.stack(Pl), np.stack(Nl)], axis=0)
        return X, Y

    def on_epoch_end(self):
        # fresh entropy + rebuild & reshuffle pools each epoch
        self.rng = np.random.default_rng()        # <— optional: new seed per epoch
        self._build_catalogue_fast()
        self.steps = max(1, len(self.anchor_idx) // self.B)
