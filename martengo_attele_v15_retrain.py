# ============================================================
# MARTENGO — Réentraînement ATTELÉ V15 (toutes les 2 semaines)
# Ce script :
#   1. Charge l'historique attelé à jour
#   2. Recalcule toutes les features V15
#   3. Réentraîne le modèle XGBoost Ranking
#   4. Exporte attele_snapshots.json.gz (< 5 Mo pour GitHub)
#   5. Sauvegarde model_pmu_v15_attele.pkl sur Drive
# Durée estimée : 45-60 min
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import pickle
import json
import gzip
import os
from xgboost import XGBRanker
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.isotonic import IsotonicRegression

BASE = '/content/drive/MyDrive/martengo_v8'

# ════════════════════════════════════════════════════════════
# BLOC 0 — Chargement
# ════════════════════════════════════════════════════════════

df = pd.read_csv(f'{BASE}/historique_enrichi.csv')
df['date']   = pd.to_datetime(df['date'])
df           = df.sort_values(['date','r_num','c_num','numero']).reset_index(drop=True)
df['top3']   = (df['rang_arrivee'] <= 3).astype(int)

prior_global = df['top3'].mean()
k_bayes      = 10
fallback     = prior_global * k_bayes / (k_bayes + 1)
date_ref     = df['date'].max()

print(f"✅ Historique ATTELÉ : {len(df):,} lignes")
print(f"   Dates   : {df['date'].min().date()} → {date_ref.date()}")
print(f"   Courses : {df.groupby(['date','r_num','c_num']).ngroups:,}")
print(f"   prior   : {prior_global:.4f}")

# Cible ranking
def calc_rank_score(g):
    max_r = g['rang_arrivee'].max()
    return (max_r - g['rang_arrivee']).clip(lower=0)

df['target_rank'] = (df.groupby(['date','r_num','c_num'], group_keys=False)
                       .apply(calc_rank_score, include_groups=False)
                       .fillna(0).astype(int))


# ════════════════════════════════════════════════════════════
# BLOC 1 — Features simples
# ════════════════════════════════════════════════════════════

print("\n=== BLOC 1 — Features simples ===")

df['ratio_victoires']  = df['nb_victoires']  / (df['nb_courses'] + 1)
df['ratio_places']     = df['nb_places']      / (df['nb_courses'] + 1)
df['gains_par_course'] = df['gains_carriere'] / (df['nb_courses'] + 1)
df['ratio_gains_rec']  = df['gains_annee']    / (df['gains_carriere'] + 1)
df['log_montant_prix'] = np.log1p(df['montant_prix'])
df['nb_partants_c']    = df['nb_partants']

fallback_rk = {'court':76000,'moyen':75100,'long':76000,'tres_long':76500}
df['tranche_distance'] = pd.cut(df['distance'],
    bins=[0,1600,2100,2700,9999],
    labels=['court','moyen','long','tres_long']).astype(str)
df['reduction_km_v2'] = df.apply(
    lambda r: r['reduction_km']
    if r['reduction_km'] > 0 and r['reduction_km'] != 72600
    else fallback_rk.get(r['tranche_distance'], 76100), axis=1)

# ── Catégorie ferrure (3 classes) ──────────────────────────────────────
def cat_ferrure(f):
    f = str(f)
    if 'ANTERIEURS_POSTERIEURS' in f and 'PROTEGE' not in f: return 'DEFERRE_TOTAL'
    if 'DEFERRE' in f: return 'DEFERRE_PARTIEL'
    return 'FERRE'
df['cat_ferrure'] = df['deferre'].apply(cat_ferrure)

# deferre_4 et corde_avantage
if 'deferre_4' in df.columns:
    df['deferre_4'] = (df['deferre_4'] == 1).astype(float)
else:
    df['deferre_4'] = 0.0

if 'place_corde_sc' in df.columns:
    df['place_corde_sc'] = df['place_corde_sc'].fillna(0)
    def avantage_corde(c):
        if c<=0: return 0.5
        elif c<=2: return 1.0
        elif c<=4: return 0.8
        elif c<=7: return 0.5
        elif c<=10: return 0.3
        else: return 0.1
    df['corde_avantage'] = df['place_corde_sc'].apply(avantage_corde)
else:
    df['corde_avantage'] = 0.5

print("  ✅ OK")


# ════════════════════════════════════════════════════════════
# BLOC 2 — Features peloton (calculées en temps réel en prod)
# ════════════════════════════════════════════════════════════

print("\n=== BLOC 2 — Features peloton ===")

df = df.sort_values(['date','r_num','c_num','numero']).reset_index(drop=True)

def features_peloton(g):
    n      = len(g)
    result = pd.DataFrame(index=g.index)
    rk     = g['reduction_km_v2'].values.astype(float)
    rk_clean = np.where((rk>0)&(rk<100000), rk, np.nan)
    rk_med   = float(np.nanmedian(rk_clean)) if np.any(~np.isnan(rk_clean)) else 76000
    rk_fill  = np.where(np.isnan(rk_clean), rk_med, rk_clean)

    rk_rank      = pd.Series(rk_fill).rank(ascending=True).values
    result['rang_rk_peloton']    = 1-(rk_rank-1)/max(n-1,1)
    rk_min = float(np.nanmin(rk_clean)) if np.any(~np.isnan(rk_clean)) else rk_med
    ecart  = rk_fill - rk_min
    result['ecart_meilleur_rk']  = 1-(ecart/ecart.max()) if ecart.max()>0 else 0.5

    gains     = g['gains_carriere'].values.astype(float)
    gains_moy = gains.mean()
    result['ratio_gains_peloton'] = (
        (gains/(gains_moy+1)).clip(0,5)/5 if gains_moy>0 else 0.5)

    tv     = g['ratio_victoires'].values.astype(float) if 'ratio_victoires' in g.columns \
             else (g['nb_victoires']/(g['nb_courses']+1)).values.astype(float)
    tv_med = np.median(tv); tv_std = tv.std()
    result['ratio_victoires_peloton'] = (
        ((tv-tv_med)/(tv_std*2+1e-9)).clip(-1,1)*0.5+0.5 if tv_std>1e-6 else 0.5)

    age     = g['age'].values.astype(float)
    age_med = np.median(age); age_std = age.std()
    result['ratio_age_peloton'] = (
        ((age-age_med)/(age_std*2+1e-9)).clip(-1,1)*0.5+0.5 if age_std>1e-6 else 0.5)

    return result

peloton = (df.groupby(['date','r_num','c_num'], group_keys=False)
             .apply(features_peloton, include_groups=False))
for col in peloton.columns:
    df[col] = peloton[col]

print("  ✅ OK")
# NOTE : rang_rk_peloton et ecart_meilleur_rk seront recalculés après BLOC 3b
# avec reduction_km_v2_ferrure (anti-leakage) au lieu de reduction_km_v2


# ════════════════════════════════════════════════════════════
# BLOC 3 — Stats glissantes anti-leakage
# ════════════════════════════════════════════════════════════

print("\n=== BLOC 3 — Stats glissantes ===")

def calc_rate_glissant(g, col='top3', k=10, prior=None, min_obs=3):
    if prior is None: prior = prior_global
    fb = prior*k/(k+1)
    vals = g[col].values
    res = np.full(len(g), fb)
    for i in range(len(g)):
        past = vals[:i]
        if len(past) >= min_obs:
            res[i] = (past.mean()*len(past)+prior*k)/(len(past)+k)
    return pd.Series(res, index=g.index)

def calc_rate_Nj(g, col='top3', days=90, k=5, prior=None):
    if prior is None: prior = prior_global
    fb = prior*k/(k+1)
    vals = g[col].values
    dates = g['date'].values
    res = np.full(len(g), fb)
    for i in range(len(g)):
        mask = (dates<dates[i])&(dates>=dates[i]-np.timedelta64(days,'D'))
        past = vals[mask]
        if len(past) >= 2:
            res[i] = (past.mean()*len(past)+prior*k)/(len(past)+k)
    return pd.Series(res, index=g.index)

print("  driver stats…")
df = df.sort_values(['driver','date']).reset_index(drop=True)
df['driver_win_rate_bayes'] = (
    df.groupby('driver',group_keys=False)
    .apply(calc_rate_glissant,include_groups=False).fillna(fallback))
df['driver_n'] = (
    df.groupby('driver',group_keys=False)
    .apply(lambda g: pd.Series(np.arange(len(g),dtype=float),index=g.index),
           include_groups=False).fillna(0))
df['driver_win_rate_90j'] = (
    df.groupby('driver',group_keys=False)
    .apply(lambda g: calc_rate_Nj(g,days=90),include_groups=False).fillna(fallback))

print("  duo stats…")
df = df.sort_values(['nom','driver','date']).reset_index(drop=True)
df['duo_win_rate_bayes'] = (
    df.groupby(['nom','driver'],group_keys=False)
    .apply(calc_rate_glissant,include_groups=False).fillna(fallback))

print("  entraîneur stats…")
df = df.sort_values(['entraineur','date']).reset_index(drop=True)
df['entr_win_rate_bayes'] = (
    df.groupby('entraineur',group_keys=False)
    .apply(calc_rate_glissant,include_groups=False).fillna(fallback))
df['entr_win_rate_30j'] = (
    df.groupby('entraineur',group_keys=False)
    .apply(lambda g: calc_rate_Nj(g,days=30),include_groups=False).fillna(fallback))

print("  rang_driver_peloton…")
df = df.sort_values(['date','r_num','c_num']).reset_index(drop=True)
def rang_driver(g):
    drv_wr = g['driver_win_rate_bayes'].values
    n = len(drv_wr)
    if n<=1: return pd.Series(0.5,index=g.index)
    ranks = pd.Series(drv_wr).rank(ascending=False).values
    return pd.Series(1-(ranks-1)/max(n-1,1),index=g.index)
df['rang_driver_peloton'] = (
    df.groupby(['date','r_num','c_num'],group_keys=False)
    .apply(rang_driver,include_groups=False).fillna(0.5))

print("  ✅ OK")


# ════════════════════════════════════════════════════════════
# BLOC 3b — reduction_km_v2_ferrure (chrono filtré par ferrure)
# ════════════════════════════════════════════════════════════

# ── Anti-leakage strict : pour chaque course, uniquement les chronos passés ──
# Tri par (nom, cat_ferrure, date) pour traiter en ordre chronologique
print("  Calcul reduction_km_v2_ferrure en anti-leakage strict...")
df = df.sort_values(['nom','cat_ferrure','date']).reset_index(drop=True)
df['nom_upper'] = df['nom'].str.upper().str.strip()

def weighted_rk_antileakage(g):
    rk_vals = g['reduction_km'].values
    result = np.full(len(g), np.nan)
    for i in range(len(g)):
        # Uniquement les courses AVANT la course i (anti-leakage strict)
        past = [r for j, r in enumerate(rk_vals) if j < i and 60000 < r < 90000]
        if past:
            recent = past[-5:]  # 5 dernières valides
            n = len(recent)
            poids = np.array([n - k for k in range(n)], dtype=float)
            result[i] = float(np.sum(np.array(recent) * poids) / poids.sum())
    return pd.Series(result, index=g.index)

rk_ferrure_al = (df.groupby(['nom_upper','cat_ferrure'], group_keys=False)
    .apply(weighted_rk_antileakage, include_groups=False))
df['reduction_km_v2_ferrure'] = rk_ferrure_al

# Fallback sur reduction_km_v2 si pas assez d'historique avec cette ferrure
df['reduction_km_v2_ferrure'] = df['reduction_km_v2_ferrure'].fillna(df['reduction_km_v2'])
df = df.drop(columns=['nom_upper'], errors='ignore')

# Retrier par ordre course pour la suite
df = df.sort_values(['date','r_num','c_num','numero']).reset_index(drop=True)

corr_f = df['reduction_km_v2_ferrure'].corr(df['top3'].astype(float))
print(f"  reduction_km_v2_ferrure (anti-leakage) : corr={corr_f:+.4f}")
print(f"  Non-null : {df['reduction_km_v2_ferrure'].notna().sum():,}")

# ── Recalcul rang_rk_peloton et ecart_meilleur_rk avec reduction_km_v2_ferrure ──
# Maintenant qu'on a reduction_km_v2_ferrure en anti-leakage, on recalcule
# les features peloton pour éliminer le leakage de reduction_km_v2
print("  Recalcul rang_rk_peloton et ecart_meilleur_rk (anti-leakage)...")
def features_peloton_al(g):
    n = len(g)
    result = pd.DataFrame(index=g.index)
    rk = g['reduction_km_v2_ferrure'].values.astype(float)
    rk_clean = np.where((rk > 60000) & (rk < 90000), rk, np.nan)
    rk_med = float(np.nanmedian(rk_clean)) if np.any(~np.isnan(rk_clean)) else 76000
    rk_fill = np.where(np.isnan(rk_clean), rk_med, rk_clean)
    rk_rank = pd.Series(rk_fill).rank(ascending=True).values
    result['rang_rk_peloton'] = 1 - (rk_rank - 1) / max(n - 1, 1)
    ecart = rk_fill - rk_fill.min()
    result['ecart_meilleur_rk'] = 1 - ecart / ecart.max() if ecart.max() > 0 else 0.5
    return result

peloton_al = (df.groupby(['date','r_num','c_num'], group_keys=False)
    .apply(features_peloton_al, include_groups=False))
for col in peloton_al.columns:
    df[col] = peloton_al[col]

corr_rk = df['rang_rk_peloton'].corr(df['top3'].astype(float))
corr_ec = df['ecart_meilleur_rk'].corr(df['top3'].astype(float))
print(f"  rang_rk_peloton (anti-leakage) : corr={corr_rk:+.4f}")
print(f"  ecart_meilleur_rk (anti-leakage) : corr={corr_ec:+.4f}")

# ── taux_completion_ferrure — fiabilité du chrono par ferrure ──────────
# Taux de courses terminées avec chrono valide / total courses avec cette ferrure
# Fallback : taux toutes ferrures si première fois avec cette ferrure
df_all_f = df[['nom','date','reduction_km','cat_ferrure']].copy()
df_all_f['nom'] = df_all_f['nom'].str.upper().str.strip()
df_all_f = df_all_f.sort_values(['nom','cat_ferrure','date'], ascending=[True,True,False])

def calc_completion(g):
    g5 = g.head(5)
    n_total = len(g5)
    n_valide = ((g5['reduction_km'] > 60000) & (g5['reduction_km'] < 90000)).sum()
    return n_valide / n_total if n_total > 0 else 0.5

# Taux par (nom, cat_ferrure)
completion_ferrure = (df_all_f.groupby(['nom','cat_ferrure'])
    .apply(calc_completion, include_groups=False)
    .reset_index(name='taux_completion_ferrure'))

# Taux global toutes ferrures — fallback si première fois avec cette ferrure
completion_global = (df_all_f.groupby('nom')
    .apply(calc_completion, include_groups=False)
    .reset_index(name='taux_completion_global'))

df['nom_upper'] = df['nom'].str.upper().str.strip()
df = df.merge(completion_ferrure, left_on=['nom_upper','cat_ferrure'],
              right_on=['nom','cat_ferrure'], how='left', suffixes=('','_comp'))
df = df.drop(columns=['nom_comp'], errors='ignore')
df = df.merge(completion_global, left_on='nom_upper',
              right_on='nom', how='left', suffixes=('','_glob'))
df = df.drop(columns=['nom_glob','nom_upper'], errors='ignore')

# Fallback : taux ferrure → taux global → prior
prior_completion = (df_all_f['reduction_km'].between(60000,90000)).mean()
df['taux_completion_ferrure'] = (df['taux_completion_ferrure']
    .fillna(df['taux_completion_global'])
    .fillna(prior_completion))
df = df.drop(columns=['taux_completion_global'], errors='ignore')

corr_f   = df['reduction_km_v2_ferrure'].corr(df['top3'].astype(float))
corr_tc  = df['taux_completion_ferrure'].corr(df['top3'].astype(float))
print(f"  reduction_km_v2_ferrure   : corr={corr_f:+.4f}")
print(f"  taux_completion_ferrure   : corr={corr_tc:+.4f}")
print(f"  prior_completion          : {prior_completion:.3f}")

# ════════════════════════════════════════════════════════════
# BLOC 4 — Features forme
# ════════════════════════════════════════════════════════════

print("\n=== BLOC 4 — Features forme ===")

print("  duo_momentum_3…")
df = df.sort_values(['nom','driver','date']).reset_index(drop=True)

# Moyenne pondérée récence sur 10 courses — plus stable et plus prédictif
# Poids : course la plus récente = 10, la plus ancienne = 1
def duo_momentum_pondere(s, n=10, k_bayes=10, prior=None):
    if prior is None: prior = fallback
    vals = s.shift(1).values  # anti-leakage
    res = np.full(len(vals), prior)
    for i in range(len(vals)):
        past = [v for v in vals[max(0,i-n):i] if not np.isnan(v)]
        if len(past) >= 2:
            n_past = len(past)
            poids = np.array([n_past - k for k in range(n_past)], dtype=float)
            wmean = np.sum(np.array(past) * poids) / poids.sum()
            res[i] = (wmean * n_past + prior * k_bayes) / (n_past + k_bayes)
    return pd.Series(res, index=s.index)

df['duo_momentum_3'] = (
    df.groupby(['nom','driver'], group_keys=False)['top3']
    .apply(duo_momentum_pondere)
    .fillna(fallback))

print("  top3_3courses…")
df = df.sort_values(['nom','date']).reset_index(drop=True)

# Moyenne pondérée récence sur 10 courses — même approche que duo_momentum
def top3_pondere(s, n=10, k_bayes=10):
    vals = s.shift(1).values  # anti-leakage
    res = np.full(len(vals), prior_global)
    for i in range(len(vals)):
        past = [v for v in vals[max(0,i-n):i] if not np.isnan(v)]
        if len(past) >= 2:
            n_past = len(past)
            poids = np.array([n_past - j for j in range(n_past)], dtype=float)
            wmean = float(np.sum(np.array(past) * poids) / poids.sum())
            res[i] = (wmean * n_past + prior_global * k_bayes) / (n_past + k_bayes)
    return pd.Series(res, index=s.index)

df['top3_3courses'] = (
    df.groupby('nom', group_keys=False)['top3']
    .apply(top3_pondere)
    .fillna(prior_global))

print("  top3_60j…")
def top3_60j_glissant(g):
    vals=g['top3'].values; dates=g['date'].values
    res=np.full(len(g),prior_global)
    for i in range(len(g)):
        mask=(dates<dates[i])&(dates>=dates[i]-np.timedelta64(60,'D'))
        n=mask.sum()
        if n>=2: res[i]=(vals[mask].mean()*n+prior_global*3)/(n+3)
    return pd.Series(res,index=g.index)
df['top3_60j'] = (
    df.groupby('nom',group_keys=False)
    .apply(top3_60j_glissant,include_groups=False).fillna(prior_global))

print("  niveau_course…")
def calc_niveau_habituel(g):
    vals=g['montant_prix'].values
    res=np.full(len(g),np.median(vals))
    for i in range(len(g)):
        if i>=2: res[i]=np.mean(vals[:i])
    return pd.Series(res,index=g.index)
df['niveau_habituel'] = (
    df.groupby('nom',group_keys=False)
    .apply(calc_niveau_habituel,include_groups=False)
    .fillna(df['montant_prix'].median()))
df['ratio_niveau'] = (df['montant_prix']/(df['niveau_habituel']+1)).clip(0,5)

def avantage_numero(n):
    try:
        n=int(n)
        if n<=2: return 1.0
        elif n<=5: return 0.8
        elif n<=8: return 0.5
        elif n<=12: return 0.3
        else: return 0.1
    except: return 0.5
df['place_avantage'] = df['numero'].apply(avantage_numero)

# ── Feature flag_disq_recente — disqualifications récentes ──
# Combinaison : disq en dernière course + taux disq global élevé
# mus_derniere_place=15 → disq/non classé en dernière course
df['flag_disq_recente'] = (
    (df['mus_derniere_place'] == 15).astype(int) * 0.6 +
    (df['mus_taux_disq'] > 0.3).astype(int) * 0.4
).clip(0, 1)
corr_disq = df['flag_disq_recente'].corr(df['top3'].astype(float))
print(f"  flag_disq_recente : corr={corr_disq:+.4f}")

# ── nb_jours_absence — jours depuis la dernière course ────────────────
# Calculé sur copie triée par (nom, date) sans toucher à l'ordre du df principal
df_abs = df[['nom','date']].copy()
df_abs = df_abs.sort_values(['nom','date']).reset_index()  # garde l'index original
df_abs['date_prec'] = df_abs.groupby('nom')['date'].shift(1)
df_abs['nb_jours_absence'] = (df_abs['date'] - df_abs['date_prec']).dt.days.fillna(30).clip(0,400)
def _score_absence_v2(j):
    j = max(0, min(400, float(j)))
    if j <= 7:   return 0.944 + (1.000 - 0.944) * (j / 7)
    if j <= 21:  return 0.944 + (1.000 - 0.944) * ((j-7) / 14)
    if j <= 45:  return 1.000 - (1.000 - 0.918) * ((j-21) / 24)
    if j <= 75:  return 0.918 - (0.918 - 0.762) * ((j-45) / 30)
    if j <= 135: return 0.762 - (0.762 - 0.673) * ((j-75) / 60)
    if j <= 270: return 0.673 - (0.673 - 0.647) * ((j-135) / 135)
    return       0.647 - (0.647 - 0.587) * ((j-270) / 130)
df_abs['score_absence'] = df_abs['nb_jours_absence'].apply(_score_absence_v2)
# Réaligner sur l'index original du df — pas de tri du df principal
df['score_absence'] = df_abs.set_index('index')['score_absence']
df['score_absence'] = df['score_absence'].fillna(0.73)  # prior
corr_abs = df['score_absence'].corr(df['top3'].astype(float))
print(f"  score_absence     : corr={corr_abs:+.4f}")

print("  ✅ OK")


# ════════════════════════════════════════════════════════════
# BLOC 5 — Préparation ranking
# ════════════════════════════════════════════════════════════

print("\n=== BLOC 5 — Préparation ranking ===")

FEATURES_V15 = [
    'rang_rk_peloton', 'ecart_meilleur_rk',
    'ratio_gains_peloton', 'ratio_victoires_peloton',
    'ratio_age_peloton', 'rang_driver_peloton',
    'deferre_4', 'corde_avantage',
    'duo_momentum_3', 'top3_3courses', 'top3_60j',
    'mus_score_pondere', 'mus_tendance', 'mus_regularite',
    'mus_nb_disq',                        # nb disq total historique
    'flag_disq_recente',                  # disq recente : derniere course + taux élevé
    'score_absence',                      # pénalité absence longue durée ← NOUVEAU
    'driver_win_rate_bayes', 'driver_n',
    'duo_win_rate_bayes', 'driver_win_rate_90j',
    'entr_win_rate_bayes', 'entr_win_rate_30j', 'avis_entraineur',
    'reduction_km_v2_ferrure', 'taux_completion_ferrure', 'age',
    'ratio_victoires', 'ratio_places',
    'gains_par_course', 'gains_annee', 'ratio_gains_rec',
    'ratio_niveau', 'nb_partants_c', 'log_montant_prix', 'place_avantage',
]
FEATURES_V15 = [f for f in FEATURES_V15
                if f in df.columns and df[f].std()>1e-6]

print(f"  {len(FEATURES_V15)} features")

# Split temporel glissant — 3 derniers mois en validation
split_date = date_ref - pd.Timedelta(days=90)
df_clean = df[FEATURES_V15+['target_rank','top3','date','r_num','c_num']].dropna()
df_clean = df_clean.sort_values(['date','r_num','c_num']).reset_index(drop=True)
train = df_clean[df_clean['date'] < split_date]
val   = df_clean[df_clean['date'] >= split_date]

X_train,y_train = train[FEATURES_V15],train['target_rank']
X_val,y_val     = val[FEATURES_V15],val['target_rank']
train_groups = train.groupby(['date','r_num','c_num'],sort=False).size().values
val_groups   = val.groupby(['date','r_num','c_num'],sort=False).size().values

print(f"  Split : < {split_date.date()} (train) / >= {split_date.date()} (val)")
print(f"  Train : {len(train):,} · {len(train_groups):,} courses")
print(f"  Val   : {len(val):,} · {len(val_groups):,} courses")


# ════════════════════════════════════════════════════════════
# BLOC 6 — Entraînement
# ════════════════════════════════════════════════════════════

print("\n=== BLOC 6 — Entraînement XGBoost Ranking V15 ===")

model_v15 = XGBRanker(
    objective='rank:pairwise', n_estimators=600,
    max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, eval_metric='ndcg',
    random_state=42, n_jobs=-1, early_stopping_rounds=40,
)
model_v15.fit(X_train,y_train,group=train_groups,
              eval_set=[(X_val,y_val)],eval_group=[val_groups],verbose=50)

scores_val = model_v15.predict(X_val)
auc_v15    = round(roc_auc_score(val['top3'],scores_val),4)
ndcg_scores=[]; idx=0
for g in val_groups:
    sc=scores_val[idx:idx+g]; gt=y_val.values[idx:idx+g]
    if len(sc)>=2 and gt.max()>0: ndcg_scores.append(ndcg_score([gt],[sc]))
    idx+=g
ndcg_v15 = round(np.mean(ndcg_scores),4)

imp = dict(zip(FEATURES_V15,model_v15.feature_importances_.round(4)))
print(f"\n  AUC  : {auc_v15}")
print(f"  NDCG : {ndcg_v15}")
print(f"\n  === Importances ===")
for feat,v in sorted(imp.items(),key=lambda x:x[1],reverse=True)[:15]:
    print(f"  {feat:<32} {v*100:5.1f}%  {'█'*int(v*100)}")


# ════════════════════════════════════════════════════════════
# BLOC 7 — Calibration
# ════════════════════════════════════════════════════════════

print("\n=== BLOC 7 — Calibration ===")

n_cal  = len(X_val)//2
iso_v15 = IsotonicRegression(out_of_bounds='clip', increasing=True)
iso_v15.fit(scores_val[:n_cal], val['top3'].values[:n_cal])
probas_all = iso_v15.predict(scores_val)
auc_cal    = round(roc_auc_score(val['top3'].values[n_cal:],
                                  iso_v15.predict(scores_val[n_cal:])),4)
p_min = round(float(np.percentile(probas_all,2)),3)
p_max = round(float(np.percentile(probas_all,98)),3)
plages=[]; idx=0
for g in val_groups:
    sc=scores_val[idx:idx+g]; plages.append(sc.max()-sc.min()); idx+=g
plages=np.array(plages)
p25=round(float(np.percentile(plages,25)),3)
p50=round(float(np.percentile(plages,50)),3)
p75=round(float(np.percentile(plages,75)),3)
print(f"  AUC calibré     : {auc_cal}")
print(f"  Seuils confiance: faible<{p25} · moyen={p50} · fort>{p75}")

# ════════════════════════════════════════════════════════════
# BLOC 7b — Backtest sur val set (sans data leakage)
# ════════════════════════════════════════════════════════════

print("\n=== BLOC 7b — Backtest val set ===")

R_GAG=5.0; R_PLA=2.2; R_CGC=12.0; R_CPC=4.5; R_TRIO=15.0

val_bt = val.copy()
val_bt['score_brut'] = scores_val
# Récupérer rang_arrivee depuis df original (absent de val car exclu par dropna)
val_bt['rang_arrivee'] = df['rang_arrivee'].reindex(val_bt.index)
val_bt = val_bt[val_bt['rang_arrivee'].notna()].copy()
print(f"  val_bt avec rang_arrivee : {len(val_bt):,} lignes")
val_bt['rang_predit'] = val_bt.groupby(['date','r_num','c_num'])['score_brut'].rank(
    ascending=False, method='first').astype(int)

# Calcul gap sans groupby apply pour éviter perte des colonnes
gap_by_course = (val_bt.groupby(['date','r_num','c_num'])['score_brut']
    .apply(lambda x: sorted(x.values)[-1]-sorted(x.values)[-2] if len(x)>=2 else 0))
val_bt = val_bt.join(gap_by_course.rename('gap'), on=['date','r_num','c_num'])

# Recalibrer seuils sur les gaps du val set
p25_bt = float(np.percentile(val_bt['gap'].dropna().values, 25))
p50_bt = float(np.percentile(val_bt['gap'].dropna().values, 50))
p75_bt = float(np.percentile(val_bt['gap'].dropna().values, 75))
print(f"  Seuils gaps val set : p25={p25_bt:.3f} · p50={p50_bt:.3f} · p75={p75_bt:.3f}")
val_bt['conf'] = val_bt['gap'].apply(
    lambda x: 'fort' if x>=p75_bt else ('moyen' if x>=p50_bt else 'faible'))
print(f"  Distribution : {val_bt.groupby(['date','r_num','c_num'])['conf'].first().value_counts().to_dict()}")

resultats_bt = []
for (date,r,c),g in val_bt.groupby(['date','r_num','c_num']):
    if len(g)<3: continue
    gs = g.sort_values('score_brut', ascending=False)
    p1,p2,p3 = gs.iloc[0],gs.iloc[1],gs.iloc[2]
    r1,r2,r3 = p1['rang_arrivee'],p2['rang_arrivee'],p3['rang_arrivee']
    # CG combiné 3 paires
    cg12 = 1 if r1==1 and r2==2 else 0
    cg13 = 1 if r1==1 and r3==2 else 0
    cg23 = 1 if r2==1 and r3==2 else 0
    resultats_bt.append({
        'conf':      p1['conf'],
        'gagnant':   1 if r1==1 else 0,
        'place':     1 if r1<=3 else 0,
        'cg_couple': cg12,
        'cp_couple': 1 if r1<=3 and r2<=3 else 0,
        'trio':      1 if all(x<=3 for x in [r1,r2,r3]) else 0,
        'cg12': cg12, 'cg13': cg13, 'cg23': cg23,
        'any_cg': 1 if (cg12 or cg13 or cg23) else 0,
    })

df_bt = pd.DataFrame(resultats_bt)
print(f"  {len(df_bt):,} courses · {split_date.date()} → {date_ref.date()}")
print(f"\n{'Conf':<8}{'N':>6}{'Gag%':>7}{'Plac%':>7}{'CG%':>6}{'CP%':>6}{'Trio%':>7}{'AnyCG%':>8}{'ROI':>9}")
print('─'*70)
for conf in ['faible','moyen','fort','ALL']:
    sub = df_bt if conf=='ALL' else df_bt[df_bt['conf']==conf]
    if len(sub)==0: continue
    n=len(sub); mise=n*4.5
    gains=(sub['gagnant'].sum()*R_GAG + sub['place'].sum()*R_PLA +
           sub['cg_couple'].sum()*R_CGC + sub['cp_couple'].sum()*R_CPC +
           sub['trio'].sum()*R_TRIO*0.5)
    roi=(gains-mise)/mise*100
    label=conf.upper() if conf!='ALL' else 'TOUTES'
    print(f"{label:<8}{n:>6,}{sub['gagnant'].mean()*100:>6.1f}%"
          f"{sub['place'].mean()*100:>6.1f}%{sub['cg_couple'].mean()*100:>5.1f}%"
          f"{sub['cp_couple'].mean()*100:>5.1f}%{sub['trio'].mean()*100:>6.1f}%"
          f"{sub['any_cg'].mean()*100:>7.1f}%{roi:>+8.1f}%")

# Détail CG combiné 3 paires
print(f"\n{'='*65}\nDÉTAIL CG COMBINÉ 3 PAIRES\n{'='*65}")
for conf in ['moyen','fort','ALL']:
    sub = df_bt if conf=='ALL' else df_bt[df_bt['conf']==conf]
    if len(sub)==0: continue
    n=len(sub)
    label=conf.upper() if conf!='ALL' else 'TOUTES'
    print(f"\n  {label} ({n:,} courses) :")
    print(f"    CG #1-#2 gagne : {sub['cg12'].mean()*100:.1f}% ({sub['cg12'].sum()}/{n})")
    print(f"    CG #1-#3 gagne : {sub['cg13'].mean()*100:.1f}% ({sub['cg13'].sum()}/{n})")
    print(f"    CG #2-#3 gagne : {sub['cg23'].mean()*100:.1f}% ({sub['cg23'].sum()}/{n})")
    print(f"    Au moins 1 CG  : {sub['any_cg'].mean()*100:.1f}% ({sub['any_cg'].sum()}/{n})")


# ════════════════════════════════════════════════════════════
# BLOC 8 — Snapshots
# ════════════════════════════════════════════════════════════

print("\n=== BLOC 8 — Snapshots ===")

# Driver stats
driver_stats = df.groupby('driver').agg(
    driver_n=('top3','count'),driver_wins=('top3','sum')).reset_index()
driver_stats['driver_win_rate_bayes'] = (
    (driver_stats['driver_wins']+prior_global*k_bayes)
    /(driver_stats['driver_n']+k_bayes))
drv_90j = (df[df['date']>=date_ref-pd.Timedelta(days=90)]
    .groupby('driver').agg(n=('top3','count'),wins=('top3','sum')).reset_index())
drv_90j['driver_win_rate_90j'] = (drv_90j['wins']+prior_global*5)/(drv_90j['n']+5)
driver_stats = driver_stats.merge(
    drv_90j[['driver','driver_win_rate_90j']], on='driver', how='left')
driver_stats['driver_win_rate_90j'] = driver_stats['driver_win_rate_90j'].fillna(fallback)

# Duo stats
duo_stats = df.groupby(['nom','driver']).agg(
    duo_n=('top3','count'),duo_wins=('top3','sum')).reset_index()
duo_stats['duo_win_rate_bayes'] = (
    (duo_stats['duo_wins']+prior_global*k_bayes)/(duo_stats['duo_n']+k_bayes))

# Entraîneur stats
entr_stats = df.groupby('entraineur').agg(
    entr_n=('top3','count'),entr_wins=('top3','sum')).reset_index()
entr_stats['entr_win_rate_bayes'] = (
    (entr_stats['entr_wins']+prior_global*k_bayes)/(entr_stats['entr_n']+k_bayes))
entr_30j = (df[df['date']>=date_ref-pd.Timedelta(days=30)]
    .groupby('entraineur').agg(n=('top3','count'),wins=('top3','sum')).reset_index())
entr_30j['entr_win_rate_30j'] = (entr_30j['wins']+prior_global*5)/(entr_30j['n']+5)
entr_stats = entr_stats.merge(
    entr_30j[['entraineur','entr_win_rate_30j']], on='entraineur', how='left')
entr_stats['entr_win_rate_30j'] = entr_stats['entr_win_rate_30j'].fillna(fallback)

# Forme récente
# Snap duo_momentum — pondéré récence sur 10 dernières courses
def duo_mom_snap_pondere(s, n=10, k_bayes=10):
    vals = s.tail(n).values
    n_past = len(vals)
    if n_past < 2:
        return fallback
    poids = np.array([n_past - k for k in range(n_past)], dtype=float)
    wmean = float(np.sum(vals * poids) / poids.sum())
    return (wmean * n_past + fallback * k_bayes) / (n_past + k_bayes)

duo_mom_snap = (df.sort_values(['nom','driver','date'])
    .groupby(['nom','driver'], group_keys=False)['top3']
    .apply(duo_mom_snap_pondere).reset_index()
    .rename(columns={'top3':'duo_momentum_3'}))

# Snap top3_3courses — pondéré récence sur 10 dernières courses
def top3_snap_pondere(s, n=10, k_bayes=10):
    vals = s.tail(n).values
    n_past = len(vals)
    if n_past < 2:
        return prior_global
    poids = np.array([n_past - j for j in range(n_past)], dtype=float)
    wmean = float(np.sum(vals * poids) / poids.sum())
    return (wmean * n_past + prior_global * k_bayes) / (n_past + k_bayes)

top3_3c_snap = (df.sort_values(['nom','date'])
    .groupby('nom', group_keys=False)['top3']
    .apply(top3_snap_pondere).reset_index()
    .rename(columns={'top3':'top3_3courses'}))

top3_60j_snap = (df[df['date']>=date_ref-pd.Timedelta(days=60)]
    .groupby('nom').agg(n60=('top3','count'),t60=('top3','mean')).reset_index())
top3_60j_snap['top3_60j'] = (
    (top3_60j_snap['t60']*top3_60j_snap['n60']+prior_global*3)
    /(top3_60j_snap['n60']+3))
top3_60j_snap = top3_60j_snap[['nom','top3_60j']]

# Niveau
niveau_snap = df.groupby('nom').agg(
    niveau_habituel=('montant_prix','mean')).reset_index()

print(f"  driver_stats   : {len(driver_stats):,}")
print(f"  duo_stats      : {len(duo_stats):,}")
print(f"  entr_stats     : {len(entr_stats):,}")
print(f"  top3_3courses  : {len(top3_3c_snap):,} chevaux")
print(f"  top3_60j       : {len(top3_60j_snap):,} chevaux")


# ════════════════════════════════════════════════════════════
# BLOC 9 — Sauvegarde pkl + JSON
# ════════════════════════════════════════════════════════════

print("\n=== BLOC 9 — Sauvegarde ===")

bundle_v15 = {
    'model':               model_v15,
    'calibrator':          iso_v15,
    'model_type':          'ranking',
    'features':            FEATURES_V15,
    'importances':         imp,
    'version':             'v15_ranking',
    'auc_val':             auc_cal,
    'ndcg_val':            ndcg_v15,
    'n_train':             len(train),
    'n_val':               len(val),
    'date_split':          str(split_date.date()),
    'date_ref':            str(date_ref.date()),
    'prior_win':           float(prior_global),
    'k_bayes':             k_bayes,
    'proba_min':           p_min,
    'proba_max':           p_max,
    'fallback_rk':         fallback_rk,
    'confiance_seuils':    {'faible':float(p25),'moyen':float(p50),'fort':float(p75)},
    # Snapshots embarqués (fallback si JSON absent)
    'driver_stats':        driver_stats,
    'duo_stats':           duo_stats,
    'entr_stats':          entr_stats,
    'duo_momentum_snap':   duo_mom_snap,
    'top3_3courses_snap':  top3_3c_snap,
    'top3_60j_snap':       top3_60j_snap,
    'niveau_snap':         niveau_snap,
}

pkl_out = f'{BASE}/model_pmu_v15_attele.pkl'
with open(pkl_out,'wb') as f:
    pickle.dump(bundle_v15, f)
taille_pkl = round(os.path.getsize(pkl_out)/1024/1024,1)
print(f"✅ model_pmu_v15_attele.pkl ({taille_pkl} Mo)")

# ── Chrono cache — indexé par (nom, cat_ferrure) ──────────────────────
print("Construction chrono_cache…")
df_rk = df[['nom','date','reduction_km','cat_ferrure']].copy()
df_rk = df_rk[(df_rk['reduction_km'] > 60000) & (df_rk['reduction_km'] < 90000)]
df_rk['nom'] = df_rk['nom'].str.upper().str.strip()
df_rk = df_rk.sort_values(['nom','date'], ascending=[True, False])

# Date dernière course — depuis df complet (pas filtré sur chrono valide)
# Pour calculer nb_jours_absence en production
df_date_last = df[['nom','date']].copy()
df_date_last['nom'] = df_date_last['nom'].str.upper().str.strip()
df_date_last = df_date_last.sort_values(['nom','date'], ascending=[True,False])
date_derniere_map = df_date_last.groupby('nom')['date'].first().dt.strftime('%Y-%m-%d').to_dict()

chrono_cache = {}
# Index par nom seul (toutes ferrures confondues) — pour flag_chrono global
for nom, grp in df_rk.groupby('nom'):
    rk_recents = grp.head(3)['reduction_km'].values.tolist()
    chrono_cache[nom] = {
        'min':           float(min(rk_recents)),
        'last':          float(rk_recents[0]),
        'history':       [float(r) for r in rk_recents],
        'date_derniere': date_derniere_map.get(nom, ''),
    }
# Ajouter les chevaux sans chrono valide (absents longs) — date uniquement
for nom, date_str in date_derniere_map.items():
    if nom not in chrono_cache:
        chrono_cache[nom] = {
            'min': 76000.0, 'last': 76000.0, 'history': [],
            'date_derniere': date_str,
        }

# Index par (nom, cat_ferrure) — pour reduction_km_v2_ferrure en prod
chrono_cache_ferrure = {}
for (nom, ferrure), grp in df_rk.groupby(['nom','cat_ferrure']):
    rk_recents = grp.head(5)['reduction_km'].values.tolist()
    key = f"{nom}||{ferrure}"
    chrono_cache_ferrure[key] = {
        'min':     float(min(rk_recents)),
        'last':    float(rk_recents[0]),
        'history': [float(r) for r in rk_recents],
    }

print(f"✅ chrono_cache : {len(chrono_cache)} chevaux")
print(f"✅ chrono_cache_ferrure : {len(chrono_cache_ferrure)} entrées (nom×ferrure)")
ex = list(chrono_cache.items())[0]
print(f"   Exemple '{ex[0]}' : {ex[1]}")
ex_f = list(chrono_cache_ferrure.items())[0]
print(f"   Exemple ferrure '{ex_f[0]}' : {ex_f[1]}")

# Export JSON compressé pour GitHub
snapshots_json = {
    'driver_stats':       driver_stats.to_dict(orient='records'),
    'duo_stats':          duo_stats.to_dict(orient='records'),
    'entr_stats':         entr_stats.to_dict(orient='records'),
    'duo_momentum_snap':  duo_mom_snap.to_dict(orient='records'),
    'top3_3courses_snap': top3_3c_snap.to_dict(orient='records'),
    'top3_60j_snap':      top3_60j_snap.to_dict(orient='records'),
    'niveau_snap':        niveau_snap.to_dict(orient='records'),
    'chrono_cache':         chrono_cache,
    'chrono_cache_ferrure': chrono_cache_ferrure,
    '_date_ref':          str(date_ref.date()),
    '_auc':               str(auc_cal),
    '_n_lignes':          len(df),
}

json_out = f'{BASE}/attele_snapshots.json.gz'
with gzip.open(json_out,'wt',encoding='utf-8') as f:
    json.dump(snapshots_json, f)
taille_json = round(os.path.getsize(json_out)/1024/1024,1)
print(f"✅ attele_snapshots.json.gz ({taille_json} Mo)")

print(f"\n  AUC  : {auc_cal}")
print(f"  NDCG : {ndcg_v15}")
print(f"  Date ref : {date_ref.date()}")
print(f"  {len(FEATURES_V15)} features · split {split_date.date()}")

from google.colab import files
files.download(pkl_out)
files.download(json_out)
print("\n✅ Téléchargements lancés")
print("\n📋 Fichiers à pousser sur GitHub :")
print("   - model_pmu_v15_attele.pkl")
print("   - attele_snapshots.json.gz")
