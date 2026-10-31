# HistGradientBoostingClassifier for a better and faster model 
# roc_auc_score make the model see the diffrence between the positive (char showed up) and negative (char havent showed up)
# average_precision_score calculate the mean under the curve the best for Imbalanced Datasets
# log loss analysing the accuracy of the probabilites in predections, when wrong punish the value with big values
# brier_score_loss calculate the mean of the square error MSE , analaysing the accuarcy of probabilites but punishs it by small value

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# HyperParameters & Configuration Constants for the monte carlo sampling 
# Monte Carlo Sampling method its a mathmetical method depends on trying millions on random simulations then calulate them all and calculate the mean
SEED = 42 # the random seed is const at 42 
N_FUTURE = 100 # define how many future episodes we need to predict 
N_SIMS = 200  # the number of monte carlo simulations     
N_SIMS_BT = 100  # simulations used in the backtests 
WINDOW = 300 # to look back at the last 300 episodes     
INPUT_CSV = "conan_episodes_full.csv"

META = ["episode_no", "title", "air_date", "is_canon"]

DEAD = [
    "jinpei_matsuda", "kenji_hagiwara", "wataru_date", "scotch", "ethan_hondou",
    "pisco", "tequila", "calvados", "irish", "atsushi_miyano", "elena_miyano",
    "akemi_miyano", "suguru_itakura", "koji_haneda", "rikumichi_kusuda",
]

# Officially dead, but fans (and the story) leave room for doubt
DOUBTFUL = {"akemi_miyano"}

GROUPS = {
    "heiji_kazuha": ["heiji", "kazuha"],
    "black_org": ["gin", "vodka", "vermouth", "bourbon_amuro", "chianti",
                "korn", "rum_wakita", "kir_rena"],
    "fbi_cia": ["akai_subaru", "jodie", "james_black", "camel"],
    "police": ["megure", "takagi", "sato", "shiratori", "chiba", "yumi"],
    "detective_boys": ["ayumi", "mitsuhiko", "genta", "haibara"],
    "mouri_ran_sonoko": ["ran", "kogoro", "sonoko"],
    "nagano_police": ["yamato", "yui_uehara", "takaaki_morofushi"],  # fixed name
    "kaito_kid_case": ["kaito_kid", "nakamori"],                      # fixed name
    "kudo_family": ["yusaku_kudo", "yukiko_kudo"],
    "sera_family": ["masumi_sera", "mary_sera", "shukichi_haneda"],
}

FEATURES = [
    "is_canon", "prev_canon", "same_run", "eps_since", "last1", "last2",
    "r10", "r20", "r50", "r100", "canon_rate", "filler_rate", "rate_cur_type",
    "group_recent",
]


# ----------------------------------------------------------------------------
# 1) Canon / filler schedule (semi-Markov run-length simulator)
# ----------------------------------------------------------------------------

# run legnths works for seeing how much manga epi in series and how many filler manga epi in series 
# itertools.groupby passes into the elemnts inside the list a and gives an output of k:keys the value itself , g:group iterator 
def run_lengths(a):
    return [(int(k), len(list(g))) for k, g in itertools.groupby(a)]


# the def schedule_dists will see from the last 150 epi the lengths of filler and manga
def schedule_dists(canon_hist, recent=150):
    """Empirical run-length distributions (canon runs / filler runs) from the
    recent past, lightly smoothed."""
    seg = [int(x) for x in canon_hist[-recent:]]
    runs = run_lengths(seg)[1:-1]  # drop truncated first run and ongoing last run
    lens = {0: [], 1: []}
    # the run_lengths gives a tuple of (k,g) k:1 or 0 the g is the len of iterations
    for v, l in runs: # v value: 1 0 , l length 
        lens[v].append(l) # if v = 1 it adds t to the 1 list if 0 to the list 0 
    out = {}
    for v in (0, 1): 
        max_len = max(lens[v] + [1]) + 3 # takes the max l like if we have [1,2,1] takes 2 + 3 = 5(3 for a better predection)
        counts = np.full(max_len, 0.25) # make a list of the 5 we got before in EX and make them all 0.25 [0.25,0.25,0.25,0.25,0.25]
        for l in lens[v]:
            counts[l - 1] += 1 # we supposed l is [1,2,1] it passes throught them find the num 1 iterates 2 times the num 2 iterates 1 time 
        # so now we get the [0.25,0.25,0.25,0.25,0.25] to [2.25,1.25,0.25,0.25,0.25] 
        out[v] = counts / counts.sum()
    return out


def simulate_schedule(canon_hist, n_future, n_sims, rng, recent=150):
    """Return an (n_sims, n_future) array of simulated canon(1)/filler(0)."""
    dists = schedule_dists(canon_hist, recent)
    hist = [int(x) for x in canon_hist]
    last_val = hist[-1]
    # read reversed to see how many episodes the arc been in series when it changes it stop (EX: [0,1,0,1,1,1] it reads 3 ones and stop)
    cur_len = len(list(itertools.takewhile(lambda x: x == last_val, reversed(hist))))
    out = np.zeros((n_sims, n_future), dtype=np.uint8)
    for s in range(n_sims):
        seq = []
        # how much of the ongoing run is still left?
        p = dists[last_val].copy()
        lens = np.arange(1, len(p) + 1) # make a matrix from one to len(p)
        p[lens < cur_len] = 0 # prevents an unlogical case by make it zero
        if p.sum() > 0:
            p /= p.sum() # probability nomrmalization 
            total_len = rng.choice(lens, p=p) # Weighted Random Choice choose randomly the len of this arc 
            seq += [last_val] * int(total_len - cur_len) # calculate how many episodes remains unitl the arc ends  
        v = 1 - last_val # Binary toggle , to start the new arc 
        while len(seq) < n_future:
            p = dists[v]
            length = rng.choice(np.arange(1, len(p) + 1), p=p)
            seq += [v] * int(length) # adding the new arc into the series 
            v = 1 - v # Binary toggle 
        out[s] = seq[:n_future] # so it realy stops at 100 episodes not 102 , 103 
    return out


def backtest_schedule(canon, cutoffs, rng): # cutoffs is to make a predection (EX:range(800,1120,40)) it will make a predection
    # from 800 to 900 then 840 to 940 ...
    print("\n=== Schedule backtest (100 episodes ahead, Brier score, lower = better) ===")
    rows = []
    for c in cutoffs:
        if c + N_FUTURE > len(canon):
            continue
        # Model Evaluation & Benchmarking and accuracy of the model
        sims = simulate_schedule(canon[:c], N_FUTURE, 300, rng) # generates 300 scenario for the next 100 epi
        p_model = sims.mean(0) # calculate the mean of the 300 scenario for the next 100 episodes
        actual = canon[c:c + N_FUTURE] # the real data not the predection for the 100 epi 
        p_base = np.full(N_FUTURE, canon[c - 100:c].mean()) # baseline / if the accuracy of p_model better means we succeeded 
        rows.append((c, brier_score_loss(actual, p_model), brier_score_loss(actual, p_base)))
    res = pd.DataFrame(rows, columns=["cutoff", "brier_model", "brier_baseline"])
    print(res.round(4).to_string(index=False))
    print(f"mean: model={res.brier_model.mean():.4f}  baseline(last-100 rate)={res.brier_baseline.mean():.4f}")


# ----------------------------------------------------------------------------
# 2) Character features (vectorised, identical for training and simulation)
# ----------------------------------------------------------------------------
def make_features(H, canon, t, group_idx): # H is history apperacnce with the 3D S T C  
    """H: (S, T, C) uint8 appearances, canon: (S, T) uint8.
    Features for episode t use only episodes < t, plus the TYPE of episode t.""" # Simulations n_sims
    # C character sum , T target epi , w window
    S, _, C = H.shape
    w0 = max(0, t - WINDOW) # calculate the start
    hw = H[:, w0:t, :].astype(np.float32)          # (S, w, C) take all scenarios and from the w0 to t-1 and all chars / takes data for the episodes in the range
    cw = canon[:, w0:t].astype(np.float32)         # (S, w) # all scenarios from the w0 to t-1 if its manga or filler
    cur = canon[:, t].astype(np.float32)           # (S,) the episode if its filler or manga 
    
    # see the chars appears just in manga or in both
    n_can = cw.sum(1)  # manga epis
    n_fil = (1.0 - cw).sum(1) # filler epis
    can_apps = np.einsum("swc,sw->sc", hw, cw) # the einstein * we make a matrix of hw,cw , we get the sum of the chars appearance in manga epis
    fil_apps = hw.sum(1) - can_apps # we get the sum of the chars appearance in filler epis
    can_rate = can_apps / np.maximum(n_can, 1)[:, None] # division of the chars appearance sum n the total manga epis 
    fil_rate = fil_apps / np.maximum(n_fil, 1)[:, None]
    rate_cur = np.where(cur[:, None] == 1, can_rate, fil_rate) # the rate of each char appearance 

    # the last appearance of each char
    rev = hw[:, ::-1, :]
    has = rev.max(1) > 0
    eps_since = np.where(has, rev.argmax(1) + 1, WINDOW + 1).astype(np.float32)

    def roll(n):
        return hw[:, -n:, :].sum(1) / n # takes the last n number of epis in reverse to see the latset seen of the chars
    # then divide it with n to see the precentage of its apperance 

    # how many consecutive previous episodes have the same type as episode t
    same = np.zeros(S, dtype=np.float32) # matrix of zeros to store the epis in series in it
    alive = np.ones(S, dtype=bool) # boolean matrix to stop counting when the type change
    for k in range(1, 9):
        if t - k < 0:
            break
        alive &= canon[:, t - k] == canon[:, t] # see the last 8 epis 
        same += alive # add 1 everytime it find the type the same

    # see the appearances of the group in the last 3 epis
    grp = np.zeros((S, C), dtype=np.float32) 
    for members in group_idx:
        if not members:
            continue
        any_recent = hw[:, -3:, :][:, :, members].max(2)  # (S, 3) take only the members of the group / max(2) so it gives one if only 1 char or more appeared and zero if none
        grp[:, members] = (any_recent.sum(1) / 3.0)[:, None]

    ones = np.ones((S, C), dtype=np.float32)
    feats = [
        cur[:, None] * ones, # epi type
        canon[:, t - 1].astype(np.float32)[:, None] * ones, # previous epi type
        same[:, None] * ones, # the serie of epis type
        eps_since, # num of epis since the last apperance 
        hw[:, -1, :], # does the char appears in the last 1 epi?
        hw[:, -2, :], # does the char appears in the last 2 epi?
        roll(10), roll(20), roll(50), roll(100), # apperance of the char in the last 10,20,50,100 epis
        can_rate, fil_rate, rate_cur, # can_rate:char appearance in the manga , fil_rate:char appearance in the filler,rate_cur:appearance of the chars in the epis
        grp, # the appearance of the group
    ]
    return np.stack(feats, axis=-1)  # (S, C, n_features) / matrix for the 14 features 

# Data Flattening so we fit them into just one model (HistGradientBoostingClassifier)
def build_training(Hreal, canon_real, t_end, group_idx, t_start=50):
    Hx, cx = Hreal[None], canon_real[None] # Hreal (Episodes,C) / canon_real(,Episodes )
    X, y = [], []
    for t in range(t_start, t_end): # from epi 50 and on 
        X.append(make_features(Hx, cx, t, group_idx)[0]) # calcualte the 14 features for the all chars 
        y.append(Hreal[t]) 
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)


def new_model():
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=5,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
    )

# Autoregressive Monte Carlo Simulation to predict the appearance of the chars in the next epis also take random otputs and make it Feedback Loop
def simulate(model, Hreal, canon_real, T0, canon_future, group_idx, rng, canon_only_mask): # model : the model
    # T0: the begining point/rng:random number generator/canon_only_mask: mask for the chars that only appears in manga 
    """Monte Carlo: predict probabilities, SAMPLE appearances, feed samples back."""
    S, n = canon_future.shape # S:monte carlo simulations number / n :number of next episodes predection 
    C = Hreal.shape[1] # C : the number of the chars (89)
    H = np.zeros((S, T0 + n, C), dtype=np.uint8) # matrix of zeros with S and total of rpis + n and C
    H[:, :T0] = Hreal[:T0] # copy the apperance of the chars in the last epis from n to T0 it still zeros
    can = np.zeros((S, T0 + n), dtype=np.uint8) # matrix of zeros for the epis type
    can[:, :T0] = canon_real[:T0] # takes the epis real data and put them in the matrix 
    can[:, T0:] = canon_future # for the epis from T0
    P = np.zeros((n, S, C), dtype=np.float32)
    for step in range(n):
        t = T0 + step
        F = make_features(H, can, t, group_idx) # training on the 14 features for all chars at the t 
        p = model.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(S, C) # transforming the matrix to binary 2d for the sickit learn
        # transform from(S,C,14) to (S*C,14)
        # characters that never appeared in a filler episode cannot appear in one
        p = np.where((can[:, t][:, None] == 0) & canon_only_mask[None, :], 0.0, p)
        P[step] = p # stores the output
        H[:, t, :] = rng.random(p.shape) < p # generates random numbers between 0 and 1 / if the num < p means the char appeared 1
        # if the number > p means the char didnt appear 0 
    return H[:, T0:, :], P 

# mange characters 
def canon_only_mask_from(Hreal, canon_real, upto): # upto cutoff index for the historical data
    filler_apps = Hreal[:upto][canon_real[:upto] == 0].sum(0) # Hreal[:upto] takes all the epis / [canon_real[:upto] == 0] takes only the filler epis
    total_apps = Hreal[:upto].sum(0) 
    return (filler_apps == 0) & (total_apps > 0)


def backtest_characters(Hreal, canon_real, group_idx, cutoff, rng):
    print(f"\n--- character backtest: train on episodes < index {cutoff}, forecast next {N_FUTURE} ---")
    # training on the data before the cutoffs 
    X, y = build_training(Hreal, canon_real, cutoff, group_idx)
    model = new_model().fit(X, y)
    mask = canon_only_mask_from(Hreal, canon_real, cutoff)
    # the scenario on the epis before the cutoffs 
    sched = simulate_schedule(canon_real[:cutoff], N_FUTURE, N_SIMS_BT, rng)
    _, P = simulate(model, Hreal, canon_real, cutoff, sched, group_idx, rng, mask)
    p_model = P.mean(1)                                    # (100, C)
    # baseline predictor calculate the mean of the chars appearance in the last 300 epi and work on the same value 
    actual = Hreal[cutoff:cutoff + N_FUTURE]
    p_base = np.tile(Hreal[cutoff - WINDOW:cutoff].mean(0), (N_FUTURE, 1))

    # takes the real data to see the real accuaracy 
    true_sched = np.tile(canon_real[cutoff:cutoff + N_FUTURE], (N_SIMS_BT, 1)).astype(np.uint8)
    _, P2 = simulate(model, Hreal, canon_real, cutoff, true_sched, group_idx, rng, mask)
    p_oracle = P2.mean(1)

    # ravel() Flattening the data (turns the 3D and 2D matrixs into 1D matrix)
    yt = actual.ravel() # yt : the appearance of the chars in epis 1 or 0 
    out = {}
    for name, p in [("model (simulated schedule)", p_model),
                    ("model (true schedule)", p_oracle),
                    ("baseline (char rate, last 300)", p_base)]:
        pc = np.clip(p.ravel(), 1e-4, 1 - 1e-4) # np.clip for the log loss it makes it between 0.000...1 and 0.999../and pc is the value of the predection for char appearance 
        out[name] = [roc_auc_score(yt, pc), average_precision_score(yt, pc),
                    log_loss(yt, pc), brier_score_loss(yt, pc)]
    res = pd.DataFrame(out, index=["ROC-AUC", "PR-AUC", "LogLoss", "Brier"]).T
    print(res.round(4).to_string())
    exp_count = p_model.sum(1).mean()
    print(f"expected chars/episode: model={exp_count:.2f}  actual={actual.sum(1).mean():.2f}")
    return res


# ----------------------------------------------------------------------------
# 3) Main
# ----------------------------------------------------------------------------
def next_window(hits, max_width=5, trim=0.10):
    """hits: (S, n) 0/1 simulated appearances of ONE character.
    Returns the narrowest window (<= max_width episodes) that is the most likely place
    for the NEXT appearance, as (first_index, last_index, probability)."""
    S, n = hits.shape # hitsis the (S,n) 1/0 appearnce
    first = np.where(hits.any(1), hits.argmax(1), n)  # n = no appearance in 100 / hits.any(1) it gives true when find a 1 
    # hits.argmax(1) the first appearance in the n epis 
    dist = np.bincount(first, minlength=n + 1)[:n] / S          # P(next appearance = episode i)
    # bincount take a matrix like[0,1,1,3,2,1,0] -> [2,3,1,1] / minlength=n make the matrix of n elemnts and others is zeros
    # [:n] / S it makes the predection for the next appearance mathmetically
    best = max(range(n - max_width + 1), key=lambda a: dist[a:a + max_width].sum()) # first index to best probabilites
    # Edge Trimming 
    lo, hi = best, best + max_width - 1 # lo:low boundary (best) / hi: high boundary  
    mass = dist[lo:hi + 1].sum() # 
    while lo < hi and dist[lo] < trim * mass:                   # drop negligible edges
        lo += 1
    while hi > lo and dist[hi] < trim * mass:
        hi -= 1
    return lo, hi, float(dist[lo:hi + 1].sum())


# the def fmt if for writing the name and its 5 like conan 85% 
def fmt(names, probs, lo, hi):
    order = np.argsort(-probs) # np.argsort :sorting ascending the - will make it descending 
    items = [f"{names[i]} {probs[i] * 100:.0f}%" for i in order if lo <= probs[i] < hi]
    return ", ".join(items)


def main():
    rng = np.random.default_rng(SEED) # for the same results everytime 
    df = pd.read_csv(INPUT_CSV).sort_values("episode_no").reset_index(drop=True) 
    all_chars = [c for c in df.columns if c not in META] # take the chars
    model_chars = [c for c in all_chars if c not in DEAD and df[c].sum() > 0]
    no_data = [c for c in all_chars if df[c].sum() == 0]
    print(f"{len(all_chars)} characters | {len(model_chars)} modelled | "
        f"no data at all (check scraping slug): {no_data}")

    canon = df["is_canon"].values.astype(np.uint8)
    Hreal = df[model_chars].values.astype(np.uint8)
    T0 = len(df)
    col = {c: i for i, c in enumerate(model_chars)} # connect each char with a number 
    group_idx = [[col[m] for m in members if m in col] for members in GROUPS.values()]

    # --- backtests (how good is this, honestly?) ---
    backtest_schedule(canon, cutoffs=range(800, 1120, 40), rng=rng)
    for cutoff in (1000, 1100):
        backtest_characters(Hreal, canon, group_idx, cutoff, rng)

    # --- final model on all data ---
    print("\nTraining final model on all episodes...")
    X, y = build_training(Hreal, canon, T0, group_idx)
    model = new_model().fit(X, y)
    mask = canon_only_mask_from(Hreal, canon, T0)
    print(f"{int(mask.sum())} modelled characters never appeared in filler -> forced to 0 in filler episodes")

    sched = simulate_schedule(canon, N_FUTURE, N_SIMS, rng)
    Hf, P = simulate(model, Hreal, canon, T0, sched, group_idx, rng, mask)
    p_marg = P.mean(1)                    # (100, C) marginal probabilities
    p_canon = sched.mean(0)

    # one representative scenario: simulation whose canon count is closest to the mean
    rep = int(np.argmin(np.abs(sched.sum(1) - sched.sum(1).mean()))) # np.abs : Absolute deviation
    start_ep = int(df["episode_no"].max()) + 1
    names = np.array(model_chars)

    rows = []
    for i in range(N_FUTURE):
        p_i = P[i, rep]
        rows.append({
            "episode_no": start_ep + i,
            "p_canon_%": round(p_canon[i] * 100, 1), #make it % 
            "type_in_scenario": "Manga (Canon)" if sched[rep, i] == 1 else "Filler (TV Original)",
            "expected_chars": round(float(p_i.sum()), 1), # num of chars expected to appear 
            "likely_characters(>=50%)": fmt(names, p_i, 0.5, 1.01), 
            "possible_characters(25-50%)": fmt(names, p_i, 0.25, 0.5),
        })
    out = pd.DataFrame(rows)
    out.to_csv("conan_future_100_episodes.csv", index=False)

    wide = pd.DataFrame(np.round(p_marg, 4), columns=model_chars)
    wide.insert(0, "p_canon", np.round(p_canon, 4))
    wide.insert(0, "episode_no", np.arange(start_ep, start_ep + N_FUTURE))
    wide.to_csv("conan_future_probs.csv", index=False)

    # --- one row per character: raw material for the cards ---
    first_canon = sched[:, 0] == 1
    cards = []
    for c in all_chars:
        s = df[c].values
        seen = np.where(s == 1)[0]
        row = {
            "character": c,
            "status": "dead_doubtful" if c in DOUBTFUL else ("dead" if c in DEAD else "alive"),
            "total_appearances": int(s.sum()),
            "appearances_last_50": int(s[-50:].sum()),
            "last_seen_episode": int(df["episode_no"].iloc[seen[-1]]) if len(seen) else None,
        }
        if c in col:
            j = col[c]
            w_lo, w_hi, w_p = next_window(Hf[:, :, j]) 
            row.update({
                "next_window_start": start_ep + w_lo,
                "next_window_end": start_ep + w_hi,
                "next_window_prob_%": round(w_p * 100, 1),
                "next_ep_prob_%": round(p_marg[0, j] * 100, 1),
                "next_ep_prob_if_canon_%": round(P[0, first_canon, j].mean() * 100, 1) if first_canon.any() else None,
                "next_ep_prob_if_filler_%": round(P[0, ~first_canon, j].mean() * 100, 1) if (~first_canon).any() else None,
                "expected_appearances_in_100": round(float(Hf[:, :, j].sum(1).mean()), 1),
                "prob_at_least_once_in_10_%": round(float((Hf[:, :10, j].sum(1) > 0).mean() * 100), 1),
                "prob_at_least_once_in_100_%": round(float((Hf[:, :, j].sum(1) > 0).mean() * 100), 1),
            })
        else:
            reason = "dead" if c in DEAD else "no data"
            row.update({"next_ep_prob_%": 0.0, "expected_appearances_in_100": 0.0,
                        "prob_at_least_once_in_100_%": 0.0, "note": reason})
        cards.append(row)
    pd.DataFrame(cards).sort_values("next_ep_prob_%", ascending=False).to_csv(
        "conan_character_cards.csv", index=False)

    print("\n=== Forecast summary ===")
    print(f"expected manga share in next {N_FUTURE}: {p_canon.mean() * 100:.1f}% "
          f"(recent 300 eps: {canon[-300:].mean() * 100:.1f}%)")
    print(f"scenario: {int(sched[rep].sum())} manga / {N_FUTURE - int(sched[rep].sum())} filler episodes")
    print(f"distinct 'likely' sets in scenario: {out['likely_characters(>=50%)'].nunique()} of {N_FUTURE}")
    print(out.head(15).to_string(index=False))
    print("\nSaved: conan_future_100_episodes.csv, conan_future_probs.csv, conan_character_cards.csv")


if __name__ == "__main__":
    main()
