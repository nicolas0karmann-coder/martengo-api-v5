# api.py — Backend Flask pour Martengo Prediction
# Déploiement : Railway / Render / Heroku

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from io import StringIO
import os
import re
import pickle
import threading
import requests as http_requests
from sklearn.ensemble import HistGradientBoostingClassifier

app = Flask(__name__)
CORS(app)

# ============================================================
# DONNÉES HISTORIQUES
# ============================================================
# historique_notes.csv : versionné dans le repo (117k lignes, source de vérité)
# historique_courses.csv : courses ajoutées via /ajouter
#
# PERSISTANCE :
#   - Si DATABASE_URL est défini (PostgreSQL sur Render/Railway) → stockage SQL
#   - Sinon → fallback CSV local (éphémère sur Render Free, acceptable en dev)
#
# Pour activer Postgres : ajoutez DATABASE_URL dans les variables d'env Render.
# La table est créée automatiquement au démarrage.
# ============================================================
HISTORIQUE_PATH = "historique_notes.csv"
CSV_PATH        = "historique_courses.csv"

# ── Connexion PostgreSQL optionnelle ──────────────────────────
_pg_conn = None

def _get_pg():
    """Retourne une connexion PostgreSQL si DATABASE_URL est défini, sinon None."""
    global _pg_conn
    if _pg_conn is not None:
        try:
            _pg_conn.cursor().execute("SELECT 1")
            return _pg_conn
        except Exception:
            _pg_conn = None
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return None
    try:
        import psycopg2
        # Render fournit parfois "postgres://" → psycopg2 veut "postgresql://"
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        _pg_conn = psycopg2.connect(db_url, sslmode="require")
        _pg_conn.autocommit = True
        # Créer la table si elle n'existe pas
        with _pg_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS courses_manuelles (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    numero INTEGER NOT NULL,
                    note FLOAT,
                    rapport FLOAT,
                    rang_arrivee INTEGER,
                    score_cible INTEGER DEFAULT 0
                )
            """)
        print("✅ PostgreSQL connecté et table prête")
        return _pg_conn
    except Exception as e:
        print(f"⚠️  PostgreSQL indisponible ({e}) — fallback CSV")
        return None


def _lire_courses_manuelles():
    """Lit les courses ajoutées manuellement (PG ou CSV)."""
    conn = _get_pg()
    if conn:
        try:
            return pd.read_sql("SELECT date, numero, note, rapport, rang_arrivee, score_cible FROM courses_manuelles", conn)
        except Exception as e:
            print(f"⚠️  Lecture PG échouée ({e}) — fallback CSV")
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame(columns=['date','numero','note','rapport','rang_arrivee','score_cible'])


def _ecrire_courses_manuelles(df_new_rows):
    """Persiste de nouvelles lignes (PG prioritaire, sinon CSV)."""
    conn = _get_pg()
    if conn:
        try:
            from psycopg2.extras import execute_values
            with conn.cursor() as cur:
                execute_values(cur,
                    "INSERT INTO courses_manuelles (date, numero, note, rapport, rang_arrivee, score_cible) VALUES %s",
                    [(row['date'], int(row['numero']), float(row['note']),
                      float(row['rapport']), int(row['rang_arrivee']), int(row.get('score_cible', 0)))
                     for _, row in df_new_rows.iterrows()]
                )
            print(f"✅ {len(df_new_rows)} lignes écrites en PostgreSQL")
            return
        except Exception as e:
            print(f"⚠️  Écriture PG échouée ({e}) — fallback CSV")
    # Fallback CSV (éphémère sur Render Free)
    df_all = _lire_courses_manuelles()
    df_all = pd.concat([df_all, df_new_rows], ignore_index=True).drop_duplicates()
    df_all.to_csv(CSV_PATH, index=False)
    print(f"⚠️  Données écrites dans {CSV_PATH} (éphémère — seront perdues au redémarrage)")


FEATURES = [
    'note','rapport','log_rapport',
    'note_normalisee','inverse_rapport',
    'score_valeur','rapport_over_10','valeur_brute'
]
FEATURES_ABSOLU = FEATURES + ['ratio_note_rapport']

# ============================================================
# LOGIQUE ML (identique aux blocs 3 & 5)
# ============================================================
def _enrichir(df_in, nm=None, ns=None):
    d = df_in.copy()
    d['log_rapport']   = np.log1p(d['rapport'])
    if nm is None:
        nm = d['note'].mean()
        ns = d['note'].std(); ns = ns if ns != 0 else 1.0
    d['note_normalisee']    = (d['note'] - nm) / ns
    d['inverse_rapport']    = 1.0 / (1.0 + d['rapport'])
    d['score_valeur']       = d['note'] / (1.0 + d['log_rapport'])
    d['rapport_over_10']    = np.maximum(0, d['rapport'] - 10)
    d['valeur_brute']       = d['note'] * d['log_rapport']
    d['ratio_note_rapport'] = d['note'] / (d['rapport'] + 1)
    return d, nm, ns


def _entrainer(df_source, features, target_col):
    d = df_source.sort_values('date')
    last = d['date'].max()
    train = d[d['date'] < last]
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6,
        max_iter=800, l2_regularization=0.5, random_state=42
    )
    clf.fit(train[features], train[target_col])
    return clf


def initialiser():
    global df, model, note_mean, note_std, modele_abs, note_mean_a, note_std_a

    # Charger historique complet (9575 courses)
    if os.path.exists(HISTORIQUE_PATH):
        df = pd.read_csv(HISTORIQUE_PATH)
        print(f"✅ Historique chargé : {len(df)} lignes / {df['date'].nunique()} dates")
    elif os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"✅ CSV manuel chargé : {len(df)} lignes")
    else:
        print("⚠️  Aucun fichier historique trouvé — données vides")
        df = pd.DataFrame(columns=['date','r_num','c_num','numero','nom','note','rapport','rang_arrivee'])

    # Fusionner avec courses ajoutées manuellement (PG ou CSV)
    df_manual = _lire_courses_manuelles()
    if len(df_manual) > 0:
        common_cols = ['date', 'numero', 'note', 'rapport', 'rang_arrivee']
        df_manual = df_manual[[c for c in common_cols if c in df_manual.columns]]
        df_base   = df[[c for c in common_cols if c in df.columns]]
        df = pd.concat([df_base, df_manual], ignore_index=True).drop_duplicates()
        print(f"✅ {len(df_manual)} lignes manuelles fusionnées")

    df['date'] = pd.to_datetime(df['date'])
    # Assurer colonne score_cible pour compatibilité
    if 'score_cible' not in df.columns:
        df['score_cible'] = 0

    # Modèle principal (cote >= 10)
    df_p = df.copy()
    df_p['target'] = ((df_p['rapport'] >= 10) & (df_p['rang_arrivee'] <= 3)).astype(int)
    df_p, note_mean, note_std = _enrichir(df_p)
    model = _entrainer(df_p, FEATURES, 'target')

    # Modèle absolu
    df_a = df.copy()
    df_a['target_absolu'] = (df_a['rang_arrivee'] <= 3).astype(int)
    df_a, note_mean_a, note_std_a = _enrichir(df_a)
    modele_abs = _entrainer(df_a, FEATURES_ABSOLU, 'target_absolu')

    print(f"✅ Modèles entraînés sur {len(df)} lignes / {df['date'].nunique()} courses")


# ============================================================
# ROUTES API
# ============================================================
@app.route('/programme', methods=['GET'])
def programme():
    """Proxy vers l'API PMU — retourne le programme du jour."""
    date_str = request.args.get('date', '')
    if not date_str:
        from datetime import datetime
        date_str = datetime.now().strftime('%d%m%Y')
    try:
        url  = f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme/{date_str}"
        resp = http_requests.get(url, timeout=10)
        if resp.status_code != 200:
            return jsonify({"error": f"API PMU {resp.status_code}"}), resp.status_code
        data = resp.json()
        courses = []
        for reunion in data.get('programme', {}).get('reunions', []):
            r_num = reunion.get('numOfficiel') or reunion.get('numReunion')
            lieu  = (reunion.get('hippodrome') or {}).get('libelleCourt', '') or \
                    (reunion.get('hippodrome') or {}).get('libelleLong', '')
            for course in reunion.get('courses', []):
                c_num   = course.get('numOrdre') or course.get('numExterne')
                heure_ts = course.get('heureDepart', 0) or 0
                if heure_ts:
                    from datetime import datetime as dt, timedelta
                    heure = (dt.fromtimestamp(heure_ts/1000) + timedelta(hours=1)).strftime('%H:%M')
                else:
                    heure = '—'
                courses.append({
                    'r_num':   r_num,
                    'c_num':   c_num,
                    'disc':    course.get('discipline', ''),
                    'dist':    course.get('distance', 0) or 0,
                    'lieu':    lieu,
                    'heure':   heure,
                    'libelle': course.get('libelle', '') or course.get('libelleCourt', ''),
                })
        return jsonify({'date': date_str, 'courses': courses})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route('/health', methods=['GET'])
def health():
    if df is not None and 'r_num' in df.columns and 'c_num' in df.columns:
        nb_courses = int(df.groupby(['date','r_num','c_num']).ngroups)
    elif df is not None:
        nb_courses = int(df['date'].nunique())
    else:
        nb_courses = 0
    return jsonify({"status": "ok", "courses": nb_courses})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Body JSON : { "chevaux": [{"numero":1,"note":15,"rapport":12.5}, ...] }
    """
    data = request.get_json()
    chevaux = data.get('chevaux', [])
    if not chevaux:
        return jsonify({"error": "Aucun cheval fourni"}), 400

    df_nc = pd.DataFrame(chevaux)
    df_nc, _, _ = _enrichir(df_nc, note_mean, note_std)

    # Modèle principal
    df_nc['proba_principal'] = model.predict_proba(df_nc[FEATURES])[:, 1]

    # Modèle absolu
    df_nc['proba_absolu'] = modele_abs.predict_proba(df_nc[FEATURES_ABSOLU])[:, 1]

    # Top 3 principal (cote > 10)
    candidats = df_nc[df_nc['rapport'] > 10].copy()
    candidats = candidats.sort_values(['proba_principal','rapport'], ascending=[False,False])
    top3_principal = candidats.head(3)['numero'].tolist()

    # Top 3 absolu
    tous = df_nc.sort_values(['proba_absolu','rapport'], ascending=[False,False])
    top3_absolu = tous.head(3)['numero'].tolist()

    # ── Top features par cheval : rang relatif dans le peloton ──
    # Pour chaque cheval, on calcule son rang sur 4 dimensions clés
    # et on l'exprime en percentile (100 = meilleur du peloton)
    n = len(tous)

    # Résultat complet trié par proba absolu
    tous_list = []
    for i, (_, row) in enumerate(tous.iterrows()):
        rang_note    = int((tous['note']              > row['note']).sum())              + 1
        rang_rapport = int((tous['rapport']           < row['rapport']).sum())            + 1
        rang_valeur  = int((tous['score_valeur']      > row['score_valeur']).sum())      + 1
        rang_ratio   = int((tous['ratio_note_rapport']> row['ratio_note_rapport']).sum())+ 1

        def pct(rang): return round((1 - (rang - 1) / n) * 100)

        top_features = sorted([
            {"feature": "Note",          "valeur": float(row['note']),                       "score": pct(rang_note),    "rang": rang_note,    "total": n},
            {"feature": "Favori (cote)", "valeur": float(row['rapport']),                    "score": pct(rang_rapport), "rang": rang_rapport, "total": n},
            {"feature": "Valeur",        "valeur": round(float(row['score_valeur']), 2),     "score": pct(rang_valeur),  "rang": rang_valeur,  "total": n},
            {"feature": "Ratio note/cote","valeur": round(float(row['ratio_note_rapport']),3),"score": pct(rang_ratio),   "rang": rang_ratio,   "total": n},
        ], key=lambda x: x['score'], reverse=True)

        tous_list.append({
            "numero":          int(row['numero']),
            "note":            float(row['note']),
            "rapport":         float(row['rapport']),
            "proba_principal": round(float(row['proba_principal']) * 100, 1),
            "proba_absolu":    round(float(row['proba_absolu']) * 100, 1),
            "top3_principal":  int(row['numero']) in top3_principal,
            "top3_absolu":     int(row['numero']) in top3_absolu,
            "top_features":    top_features,
        })
    return jsonify({
        "tous": tous_list,
        "top3_principal": top3_principal,
        "top3_absolu":    top3_absolu,
    })


@app.route('/ajouter', methods=['POST'])
def ajouter():
    """
    Body JSON : {
      "date": "2026-03-01",
      "chevaux": [{"numero":1,"note":15,"rapport":12.5,"rang_arrivee":2}, ...]
    }

    Le réentraînement des modèles est lancé en tâche de fond (thread) pour
    ne pas bloquer la réponse HTTP — sur 117k lignes, l'entraînement prend
    plusieurs secondes et bloquerait le client sinon.
    """
    global df

    data    = request.get_json()
    date    = data.get('date')
    chevaux = data.get('chevaux', [])

    if not date or not chevaux:
        return jsonify({"error": "date et chevaux requis"}), 400

    # ── 1. Construire les nouvelles lignes ──
    rows = []
    for c in chevaux:
        rows.append({
            "date":         pd.to_datetime(date),
            "numero":       c['numero'],
            "note":         c['note'],
            "rapport":      c['rapport'],
            "rang_arrivee": c['rang_arrivee'],
            "score_cible":  0,
        })
    df_new = pd.DataFrame(rows)

    # ── 2. Persister immédiatement (PG ou CSV) ──
    _ecrire_courses_manuelles(df_new)

    # ── 3. Mettre à jour le DataFrame en mémoire ──
    df = pd.concat([df, df_new], ignore_index=True)

    # ── 4. Réentraînement asynchrone ──
    # On répond tout de suite au client, le réentraînement tourne en arrière-plan.
    # Les modèles restent utilisables avec les anciennes valeurs pendant ce temps.
    def _reentrainer_bg():
        global model, note_mean, note_std, modele_abs, note_mean_a, note_std_a
        try:
            print(f"🔄 Réentraînement en arrière-plan sur {len(df)} lignes…")
            df_p = df.copy()
            df_p['target'] = ((df_p['rapport'] >= 10) & (df_p['rang_arrivee'] <= 3)).astype(int)
            df_p, nm, ns = _enrichir(df_p)
            m = _entrainer(df_p, FEATURES, 'target')

            df_a = df.copy()
            df_a['target_absolu'] = (df_a['rang_arrivee'] <= 3).astype(int)
            df_a, nma, nsa = _enrichir(df_a)
            ma = _entrainer(df_a, FEATURES_ABSOLU, 'target_absolu')

            # Mise à jour atomique des globals
            model, note_mean, note_std = m, nm, ns
            modele_abs, note_mean_a, note_std_a = ma, nma, nsa
            print(f"✅ Réentraînement terminé ({len(df)} lignes / {df['date'].nunique()} courses)")
        except Exception as e:
            print(f"❌ Erreur réentraînement : {e}")

    threading.Thread(target=_reentrainer_bg, daemon=True).start()

    return jsonify({
        "message":   f"Course du {date} ajoutée — réentraînement en cours",
        "nb_lignes":  len(df),
        "nb_courses": int(df['date'].nunique()),
        "stockage":   "postgresql" if _get_pg() else "csv_ephemere",
    })


# ============================================================
# MODELE PMU — Chargement
# ============================================================
# MODELE PMU — Globals
# ============================================================
_model_pmu           = None
_features_pmu        = None
_le_driver           = None
_le_entr             = None
_driver_stats        = None
_entr_stats          = None
_duo_stats           = None
_spec_dist           = None
_spec_disc           = None
_prior_pmu           = None
_k_bayes_pmu         = None
_target_mean_pmu     = None
_target_std_pmu      = None
_ferrage_map_pmu     = None
_avis_map_pmu        = {'POSITIF': 1, 'NEUTRE': 0, 'NEGATIF': -1}
_mediane_rapport_ref = 18.0
_hist_snapshot       = None
_seuils_notes        = None

PMU_MODEL_PATH      = "model_pmu_v5.pkl"
PMU_V7_PATH         = "model_pmu_v9_attele.pkl"   # XGBoost trot attelé V9

# Modèles galop
GALOP_MODEL_PATHS = {
    'PLAT':  "model_pmu_plat.pkl",
    'HAIE':  "model_pmu_haie.pkl",
    'MONTE': "model_pmu_monte.pkl",
}
DISCIPLINES_TROT  = ('ATTELE', 'TROT_ATTELE')
DISCIPLINES_GALOP = ('PLAT', 'HAIE', 'MONTE')
DISCIPLINES_SKIP  = ('STEEPLECHASE', 'CROSS')

# Globals V9 (trot attelé)
_model_v7              = None
_bundle_v7             = {}
_use_v7                = False
# Snapshots V9 — forme récente et momentum duo
_duo_momentum_snap     = None
_top3_3courses_snap    = None
_top3_60j_snap         = None
_fallback_rk_v9        = {'court': 76000, 'moyen': 75100,
                           'long': 76000, 'tres_long': 76500}
_duo_fiable_seuil_v9   = 5

# Globals galop
_models_galop       = {}   # {'PLAT': bundle, 'HAIE': bundle, 'MONTE': bundle}
_jockey_stats_galop = None # stats jockey pour PLAT et HAIE

DISC_MUSIQUE_MAP = {'a': 0, 'm': 1, 'p': 2, 'h': 3, 's': 4, 'c': 5}
DISCIPLINE_MAP   = {'TROT_ATTELE': 0, 'TROT_MONTE': 1, 'PLAT': 2, 'OBSTACLE': 3}
CORDE_MAP        = {'CORDE_A_GAUCHE': 0, 'CORDE_A_DROITE': 1}
SEXE_MAP         = {'MALES': 0, 'FEMELLES': 1, 'MIXTE': 2}


def _calculer_hist_snapshot():
    """Calcule les stats historiques par cheval depuis historique_notes.csv."""
    global _hist_snapshot
    hist = pd.read_csv(HISTORIQUE_PATH, usecols=['date','nom','rang_arrivee','rapport'])
    hist['date'] = pd.to_datetime(hist['date'])
    hist = hist.dropna(subset=['rang_arrivee']).sort_values(['nom','date'])

    # Calcul vectorisé
    g = hist.groupby('nom')
    nb       = g['rang_arrivee'].count()
    taux_top3= (hist['rang_arrivee'] <= 3).groupby(hist['nom']).mean()
    moy_cl   = g['rang_arrivee'].mean().round(2)
    moy_cote = g['rapport'].mean().round(2)

    # Tendance : moy 3 dernières - moy 3 premières (vectorisé)
    def _tendance(x):
        n = len(x)
        rec = x.iloc[-3:].mean()
        anc = x.iloc[:3].mean() if n >= 6 else x.mean()
        return round(float(anc - rec), 2)

    tendance = g['rang_arrivee'].apply(_tendance)

    # Courses dans les 60 derniers jours
    date_max  = hist['date'].max()
    date_60j  = date_max - pd.Timedelta(days=60)
    hist_rec  = hist[hist['date'] >= date_60j]
    courses_60j = hist_rec.groupby('nom')['rang_arrivee'].count().rename('courses_60j')

    _hist_snapshot = pd.DataFrame({
        'nom':                nb.index,
        'hist_nb':            nb.values,
        'hist_taux_top3':     taux_top3.values.round(3),
        'hist_moy_classement':moy_cl.values,
        'hist_tendance':      tendance.values,
        'hist_moy_cote':      moy_cote.values,
    }).join(courses_60j, on='nom').fillna({'courses_60j': 0})
    _hist_snapshot['courses_60j'] = _hist_snapshot['courses_60j'].astype(int)
    print(f"✅ hist_snapshot calculé : {len(_hist_snapshot)} chevaux")


def _charger_modele_pmu():
    global _model_pmu, _features_pmu, _le_driver, _le_entr
    global _driver_stats, _entr_stats, _duo_stats, _spec_dist, _spec_disc
    global _prior_pmu, _k_bayes_pmu
    global _target_mean_pmu, _target_std_pmu, _ferrage_map_pmu, _mediane_rapport_ref
    global _hist_snapshot, _seuils_notes

    if not os.path.exists(PMU_MODEL_PATH):
        print("⚠️  model_pmu.pkl introuvable — endpoint /notes_pmu désactivé")
        return False
    try:
        with open(PMU_MODEL_PATH, 'rb') as f:
            pmu = pickle.load(f)
        _model_pmu           = pmu['model']
        _features_pmu        = pmu['features']
        _le_driver           = pmu.get('le_driver')
        _le_entr             = pmu.get('le_entr')
        _driver_stats        = pmu.get('driver_stats')
        _entr_stats          = pmu.get('entr_stats')
        _duo_stats           = pmu.get('duo_stats')
        _spec_dist           = pmu.get('spec_dist')
        _spec_disc           = pmu.get('spec_disc')
        _prior_pmu           = pmu['prior']
        _k_bayes_pmu         = pmu['k_bayes']
        _target_mean_pmu     = pmu.get('target_mean')
        _target_std_pmu      = pmu.get('target_std')
        _ferrage_map_pmu     = pmu['ferrage_map']
        _mediane_rapport_ref = pmu.get('mediane_rapport_ref', 18.0)
        _hist_snapshot       = pmu.get('hist_snapshot')
        _seuils_notes        = pmu.get('seuils_notes')
        v = pmu.get('version', 1)
        nb_duos = len(_duo_stats) if _duo_stats is not None else 0
        print(f"✅ Modèle PMU v{v} chargé ({len(_features_pmu)} features, {nb_duos} duos)")

        # ── Calcul hist_snapshot depuis historique si absent du pkl ──
        if _hist_snapshot is None and os.path.exists(HISTORIQUE_PATH):
            try:
                _calculer_hist_snapshot()
            except Exception as e:
                print(f"⚠️  hist_snapshot non calculé : {e}")

        return True
    except Exception as e:
        print(f"❌ Erreur chargement model_pmu.pkl : {e}")
        return False


# ── Parseur musique v3 (format réel : position + discipline) ─
def _parser_musique_api(musique):
    from collections import Counter
    if not musique:
        return {
            'mus_nb_courses': 0, 'mus_nb_victoires': 0, 'mus_nb_podiums': 0,
            'mus_moy_classement': 99, 'mus_derniere_place': 99, 'mus_regularite': 0,
            'mus_nb_disq': 0, 'mus_taux_disq': 0.0,
            'mus_nb_tombes': 0, 'mus_nb_arretes': 0,
            'mus_tendance': 0.0, 'mus_score_pondere': 0.0,
            'mus_disc_principale': -1, 'mus_nb_disciplines': 0,
        }
    clean   = re.sub(r'\(\d+\)', '', musique).strip()
    tokens  = re.findall(r'[0-9DATRdat][amphsc]', clean)
    if not tokens:
        return {
            'mus_nb_courses': 0, 'mus_nb_victoires': 0, 'mus_nb_podiums': 0,
            'mus_moy_classement': 99, 'mus_derniere_place': 99, 'mus_regularite': 0,
            'mus_nb_disq': 0, 'mus_taux_disq': 0.0,
            'mus_nb_tombes': 0, 'mus_nb_arretes': 0,
            'mus_tendance': 0.0, 'mus_score_pondere': 0.0,
            'mus_disc_principale': -1, 'mus_nb_disciplines': 0,
        }
    entries, nb_disq, nb_tombes, nb_arretes = [], 0, 0, 0
    for tok in tokens[:10]:
        pos, disc = tok[0], tok[1].lower()
        if pos.isdigit():
            place = 10 if pos == '0' else int(pos)
        elif pos.upper() == 'D':
            place = 15; nb_disq += 1
        elif pos.upper() == 'T':
            place = 15; nb_tombes += 1
        elif pos.upper() == 'A':
            place = 15; nb_arretes += 1
        elif pos.upper() == 'R':
            place = 12
        else:
            continue
        entries.append((place, disc))
    if not entries:
        return {
            'mus_nb_courses': 0, 'mus_nb_victoires': 0, 'mus_nb_podiums': 0,
            'mus_moy_classement': 99, 'mus_derniere_place': 99, 'mus_regularite': 0,
            'mus_nb_disq': 0, 'mus_taux_disq': 0.0,
            'mus_nb_tombes': 0, 'mus_nb_arretes': 0,
            'mus_tendance': 0.0, 'mus_score_pondere': 0.0,
            'mus_disc_principale': -1, 'mus_nb_disciplines': 0,
        }
    places      = [e[0] for e in entries]
    disciplines = [e[1] for e in entries]
    nb          = len(places)
    recentes    = places[:3]
    anciennes   = places[-3:] if nb >= 6 else places
    tendance    = round(float(np.mean(anciennes) - np.mean(recentes)), 2)
    poids       = [1.0 / (i + 1) for i in range(nb)]
    score_p     = round(sum(p*(10-min(pl,10)) for p,pl in zip(poids,places))/sum(poids), 3)
    disc_counter     = Counter(disciplines)
    disc_principale  = DISC_MUSIQUE_MAP.get(disc_counter.most_common(1)[0][0], -1)
    return {
        'mus_nb_courses':      nb,
        'mus_nb_victoires':    sum(1 for p in places if p == 1),
        'mus_nb_podiums':      sum(1 for p in places if p <= 3),
        'mus_moy_classement':  round(sum(places) / nb, 2),
        'mus_derniere_place':  places[0],
        'mus_regularite':      round(sum(1 for p in places if p <= 5) / nb, 2),
        'mus_nb_disq':         nb_disq,
        'mus_taux_disq':       round(nb_disq / nb, 2),
        'mus_nb_tombes':       nb_tombes,
        'mus_nb_arretes':      nb_arretes,
        'mus_tendance':        tendance,
        'mus_score_pondere':   score_p,
        'mus_disc_principale': disc_principale,
        'mus_nb_disciplines':  len(disc_counter),
    }


def _perf_vide():
    return {
        'perf_nb': 0, 'perf_moy_classement': 99, 'perf_derniere_place': 99,
        'perf_nb_top3': 0, 'perf_taux_top3': 0.0,
        'perf_moy_rk': 0.0, 'perf_moy_gains': 0.0, 'perf_regularite': 0.0,
    }


def _fetch_performances(date_str, r_num, c_num):
    url = (f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
           f"/{date_str}/R{r_num}/C{c_num}/performances-detaillees")
    try:
        resp = http_requests.get(url, timeout=5)
        if resp.status_code in (400, 404, 204):
            return {}
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}
    result = {}
    for cheval in data.get('performancesDetaillees', []):
        num_pmu = cheval.get('numPmu')
        perfs   = cheval.get('performances', [])[:5]
        if not perfs:
            result[num_pmu] = _perf_vide(); continue
        classements, temps_list, gains_list = [], [], []
        for perf in perfs:
            cl = perf.get('ordreArrivee') or perf.get('classement')
            if cl and cl <= 15:
                classements.append(cl)
            t = perf.get('tempsObtenu') or perf.get('reductionKilometrique')
            if t and t > 0: temps_list.append(t)
            g = perf.get('gainsCourse') or perf.get('gains') or 0
            if g: gains_list.append(g)
        nb = len(classements)
        result[num_pmu] = {
            'perf_nb':             nb,
            'perf_moy_classement': round(sum(classements)/nb, 2) if nb > 0 else 99,
            'perf_derniere_place': classements[0] if classements else 99,
            'perf_nb_top3':        sum(1 for c in classements if c <= 3),
            'perf_taux_top3':      round(sum(1 for c in classements if c<=3)/nb,2) if nb>0 else 0.0,
            'perf_moy_rk':         round(sum(temps_list)/len(temps_list),1) if temps_list else 0.0,
            'perf_moy_gains':      round(sum(gains_list)/len(gains_list),1) if gains_list else 0.0,
            'perf_regularite':     round(sum(1 for c in classements if c<=5)/nb,2) if nb>0 else 0.0,
        }
    return result


def _fetch_conditions(date_str, r_num, c_num):
    url = f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme/{date_str}"
    try:
        resp = http_requests.get(url, timeout=5)
        if resp.status_code in (400, 404, 204):
            return _cond_vides()
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return _cond_vides()
    for reunion in data.get('programme', {}).get('reunions', []):
        if reunion.get('numOfficiel') == r_num or reunion.get('numReunion') == r_num:
            for course in reunion.get('courses', []):
                if course.get('numOrdre') == c_num or course.get('numExterne') == c_num:
                    penet = course.get('penetrometre', {}) or {}
                    try:
                        terrain_val = float(str(penet.get('valeurMesure','3')).replace(',','.'))
                    except:
                        terrain_val = 3.0
                    terrain_label = penet.get('intitule', '') or ''
                    return {
                        'distance':       course.get('distance', 0) or 0,
                        'montant_prix':   course.get('montantPrix', 0) or 0,
                        'discipline':     DISCIPLINE_MAP.get(course.get('discipline',''), 0),
                        'discipline_raw': course.get('discipline', ''),
                        'corde':          CORDE_MAP.get(course.get('corde',''), 0),
                        'condition_sexe': SEXE_MAP.get(course.get('conditionSexe',''), 2),
                        'nb_partants':    course.get('nombreDeclaresPartants', 0) or 0,
                        'terrain_val':    terrain_val,
                        'terrain_label':  terrain_label,
                    }
    return _cond_vides()


def _cond_vides():
    return {'distance': 0, 'montant_prix': 0, 'discipline': 0, 'discipline_raw': '',
            'corde': 0, 'condition_sexe': 2, 'nb_partants': 0,
            'terrain_val': 3.0, 'terrain_label': ''}


# Seuils par défaut si bundle ne les contient pas (utilisé pour V6 fallback)
_SEUILS_DEFAUT = [
    (0.05, 1), (0.10, 2), (0.15, 3), (0.20, 4), (0.25, 5),
    (0.30, 6), (0.35, 7), (0.40, 8), (0.45, 9), (0.50, 10),
    (0.55, 11), (0.60, 12), (0.65, 13), (0.70, 14), (0.75, 15),
    (0.80, 16), (0.85, 17), (0.90, 18), (0.95, 19), (1.01, 20),
]

def _proba_to_note_api(proba_series):
    """
    Conversion V6 (fallback) : seuils fixes sur probabilités absolues.
    """
    seuils = _seuils_notes if _seuils_notes is not None else _SEUILS_DEFAUT
    def _convert(p):
        for seuil, note in seuils:
            if p < seuil:
                return note
        return 20
    return pd.Series(proba_series).apply(_convert)


def _proba_to_note_v7(proba_series, proba_min_ref=None, proba_max_ref=None):
    """
    Conversion V9 : seuils fixes calibrés sur la plage réelle du modèle.

    Principe :
      - On étale les 20 niveaux sur la plage [proba_min_ref, proba_max_ref]
        du modèle (lue depuis le pkl si disponible, sinon estimée sur le peloton).
      - Deux chevaux avec des probas proches (53% vs 52%) → notes proches ou égales.
      - Un cheval dominant (53% vs 25%) → écart de notes significatif.
      - Le #1 n'est PAS forcément 20 ni le dernier forcément 1 :
        si toute la course est regroupée en [0.25, 0.35], les notes
        seront toutes dans la même tranche (ex: 5 à 7).
    """
    s = pd.Series(proba_series).reset_index(drop=True)
    n = len(s)

    if n == 1:
        notes = pd.Series([10], index=s.index)
        notes.index = proba_series.index if hasattr(proba_series, 'index') else notes.index
        return notes

    # Bornes de référence — depuis le pkl si disponibles, sinon peloton courant
    # Le pkl V9 stocke proba_min=0.06 et proba_max=0.81
    if proba_min_ref is None:
        proba_min_ref = _bundle_v7.get('proba_min', s.min())
    if proba_max_ref is None:
        proba_max_ref = _bundle_v7.get('proba_max', s.max())

    plage = proba_max_ref - proba_min_ref
    if plage < 1e-6:
        notes = pd.Series(10, index=s.index)
        notes.index = proba_series.index if hasattr(proba_series, 'index') else notes.index
        return notes

    # Conversion linéaire sur la plage de référence → notes 1-20
    # note = (proba - proba_min_ref) / plage * 19 + 1
    notes_float = (s - proba_min_ref) / plage * 19 + 1
    notes = notes_float.round().clip(1, 20).astype(int)

    # Remettre l'index original
    notes.index = proba_series.index if hasattr(proba_series, 'index') else notes.index
    return notes





def _norm_g(s, lo, hi):
    return ((s.clip(lo, hi) - lo) / (hi - lo + 1e-9)).clip(0, 1)

def _norm_rel_g(s):
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-9: return pd.Series(0.5, index=s.index)
    return ((s - mn) / (mx - mn)).clip(0, 1)

def _norm_mix_g(s, lo, hi, rel=0.5):
    return (_norm_g(s, lo, hi) * (1-rel) + _norm_rel_g(s) * rel).clip(0, 1)

def _notes_pmu_galop(df_nc, discipline_raw, date_str, r_num, c_num):
    """Pipeline de scoring et prédiction pour les disciplines galop."""
    prior    = 0.15
    k_bayes  = 10
    fallback = prior * k_bayes / (k_bayes + 1)

    # ── Features dérivées ────────────────────────────────────
    df_nc['ratio_victoires']  = df_nc['nb_victoires'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places']     = df_nc['nb_places']    / (df_nc['nb_courses'] + 1)
    df_nc['gains_par_course'] = df_nc['gains_carriere'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_gains_rec']  = df_nc['gains_annee'] / (df_nc['gains_carriere'] + 1)
    df_nc['ratio_places_second']    = df_nc['nb_places_second']    / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places_troisieme'] = df_nc['nb_places_troisieme'] / (df_nc['nb_courses'] + 1)
    df_nc['rang_cote_course'] = df_nc['rapport_ref'].rank(ascending=True, method='min')
    nb_ch = len(df_nc)
    df_nc['rang_cote_norme']  = (df_nc['rang_cote_course'] - 1) / (nb_ch - 1 + 1e-8)
    df_nc['rang_poids']   = df_nc['handicap_poids'].rank(ascending=True, method='min')
    df_nc['rang_poids_n'] = (df_nc['rang_poids'] - 1) / (nb_ch - 1 + 1e-8)

    # ── Score forme ──────────────────────────────────────────
    s_score_p  = _norm_mix_g(df_nc['mus_score_pondere'], 0, 9)
    s_derniere = _norm_mix_g(15 - df_nc['mus_derniere_place'], 0, 14)
    s_podiums  = _norm_mix_g(df_nc['mus_nb_podiums'], 0, 5)
    s_disq     = 1 - _norm_g(df_nc['mus_taux_disq'], 0, 0.3)
    s_chutes   = 1 - _norm_g(df_nc['mus_nb_tombes'], 0, 3)
    s_age      = 1 - _norm_g(df_nc['age'].fillna(5), 3, 12)
    df_nc['score_forme'] = (s_score_p*0.30 + s_derniere*0.25 + s_podiums*0.15 +
                            s_disq*0.10 + s_chutes*0.10 + s_age*0.10).clip(0, 1)

    # ── Score duo / jockey selon discipline ────────────────────
    if discipline_raw in ('PLAT', 'HAIE'):
        # PLAT et HAIE : score jockey seul depuis historique_galop.csv
        df_nc['jockey_win_rate_bayes'] = fallback
        df_nc['jockey_n']              = 0
        jockey_source = _jockey_stats_galop if _jockey_stats_galop is not None else (
            _driver_stats if _driver_stats is not None and 'driver_win_rate_bayes' in _driver_stats.columns else None
        )
        if jockey_source is not None:
            df_nc['_driver_clean'] = df_nc['driver'].str.strip().str.upper()
            jockey_merge = jockey_source.copy()
            jockey_merge['driver'] = jockey_merge['driver'].str.strip().str.upper()
            df_nc = df_nc.merge(
                jockey_merge[['driver','driver_win_rate_bayes','driver_n']],
                left_on='_driver_clean', right_on='driver', how='left',
                suffixes=('','_jk'))
            df_nc['jockey_win_rate_bayes'] = df_nc['driver_win_rate_bayes'].fillna(fallback)
            df_nc['jockey_n']              = df_nc['driver_n'].fillna(0)
            df_nc.drop(columns=['_driver_clean','driver_jk'], errors='ignore', inplace=True)
        df_nc['jockey_win_rate_bayes'] = df_nc['jockey_win_rate_bayes'].fillna(fallback)
        df_nc['jockey_n']              = df_nc['jockey_n'].fillna(0)
        df_nc['jockey_fiable']         = (df_nc['jockey_n'] >= 5).astype(float)
        df_nc['score_jockey'] = (_norm_mix_g(df_nc['jockey_win_rate_bayes'], fallback*0.8, 0.35)*0.60 +
                                  df_nc['jockey_fiable']*0.25 +
                                  _norm_mix_g(df_nc['jockey_n'], 5, 50)*0.15).clip(0, 1)
        df_nc['score_duo'] = df_nc['score_jockey']  # alias
    else:
        # MONTE : score duo cheval+jockey classique
        if _duo_stats is not None:
            df_nc = df_nc.merge(_duo_stats[['nom','driver','duo_win_rate_bayes','duo_n']],
                                on=['nom','driver'], how='left')
        if 'duo_win_rate_bayes' not in df_nc.columns:
            df_nc['duo_win_rate_bayes'] = fallback
            df_nc['duo_n'] = 0
        df_nc['duo_win_rate_bayes'] = df_nc['duo_win_rate_bayes'].fillna(fallback)
        df_nc['duo_n']              = df_nc['duo_n'].fillna(0)
        df_nc['duo_fiable']         = (df_nc['duo_n'] >= 2).astype(float)
        df_nc['score_duo'] = (_norm_mix_g(df_nc['duo_win_rate_bayes'], fallback*0.8, 0.65)*0.60 +
                              df_nc['duo_fiable']*0.25 +
                              _norm_mix_g(df_nc['duo_n'], 1, 15)*0.15).clip(0, 1)
        df_nc['score_jockey'] = df_nc['score_duo']  # alias

    # ── Score historique ─────────────────────────────────────
    if _hist_snapshot is not None:
        hist_cols = [c for c in ['nom','hist_nb','hist_taux_top3','hist_moy_classement',
                                  'hist_tendance','hist_moy_cote'] if c in _hist_snapshot.columns]
        df_nc = df_nc.merge(_hist_snapshot[hist_cols], on='nom', how='left')
    for col, val in [('hist_nb',0),('hist_taux_top3',fallback),('hist_moy_classement',8),
                     ('hist_tendance',0),('hist_moy_cote',15)]:
        if col not in df_nc.columns: df_nc[col] = val
        df_nc[col] = df_nc[col].fillna(val)
    hist_cote_med = df_nc['hist_moy_cote'].median()
    df_nc['score_historique'] = (
        _norm_mix_g(df_nc['hist_taux_top3'], 0, 0.7)*0.35 +
        _norm_mix_g(10 - df_nc['hist_moy_classement'], -5, 9)*0.25 +
        _norm_g(df_nc['hist_nb'], 0, 20)*0.10 +
        _norm_mix_g(df_nc['hist_tendance'], -3, 3)*0.20 +
        (1 - _norm_g(df_nc['hist_moy_cote'].fillna(hist_cote_med).fillna(15), 2, 30))*0.10
    ).clip(0, 1)

    # ── Score gains ──────────────────────────────────────────
    df_nc['score_gains'] = (
        _norm_mix_g(df_nc['ratio_victoires'],  0, 0.4)*0.30 +
        _norm_mix_g(df_nc['gains_par_course'], 0, 8000)*0.25 +
        _norm_mix_g(df_nc['gains_annee'],      0, 150000)*0.20 +
        _norm_mix_g(df_nc['ratio_gains_rec'],  0, 0.5)*0.15 +
        _norm_mix_g(df_nc['ratio_places'],     0, 0.6)*0.10
    ).clip(0, 1)

    # ── Score handicap ───────────────────────────────────────
    s_hcap_val   = _norm_mix_g(df_nc['handicap_valeur'].fillna(df_nc['handicap_valeur'].median()), 30, 70)
    s_poids_rang = df_nc['rang_poids_n']
    s_poids_abs  = 1 - _norm_g(df_nc['handicap_poids'], 520, 640)
    df_nc['score_handicap'] = (s_hcap_val*0.50 + s_poids_rang*0.30 + s_poids_abs*0.20).clip(0, 1)

    # ── Score cote ───────────────────────────────────────────
    s_cote_rang   = 1 - df_nc['rang_cote_norme']
    s_ecart       = _norm_g(-df_nc['ecart_cotes'].abs(), -10, 0)
    s_cote_direct = 1 - _norm_mix_g(df_nc['rapport_direct'].fillna(df_nc['rapport_ref']), 2, 50)
    df_nc['score_cote'] = (s_cote_rang*0.50 + s_cote_direct*0.35 +
                           s_ecart*0.15).clip(0, 1).fillna(s_cote_rang.clip(0, 1))

    # ── Scoring final via XGBoost galop ──────────────────────
    bundle_galop = _models_galop[discipline_raw]
    features_5   = bundle_galop['features']
    poids_cote   = bundle_galop.get('poids_cote_fixe', 0.15)
    poids_xgb    = bundle_galop.get('poids_xgb', 0.85)

    # ── Features PLAT V8 brutes ───────────────────────────────
    if discipline_raw == 'PLAT':
        fallback_v8 = bundle_galop.get('prior_win', 0.098) * bundle_galop.get('k_bayes', 10) / (bundle_galop.get('k_bayes', 10) + 1)

        # Stats driver depuis bundle
        driver_stats = bundle_galop.get('driver_stats')
        if driver_stats is not None:
            df_nc = df_nc.merge(driver_stats[['driver','driver_win_rate_bayes','driver_n']],
                                on='driver', how='left')
        df_nc['driver_win_rate_bayes'] = df_nc.get('driver_win_rate_bayes', pd.Series([fallback_v8]*len(df_nc))).fillna(fallback_v8)
        df_nc['driver_n']              = df_nc.get('driver_n', pd.Series([0]*len(df_nc))).fillna(0)

        # Stats duo depuis bundle
        duo_stats = bundle_galop.get('duo_stats')
        if duo_stats is not None:
            df_nc = df_nc.merge(duo_stats[['nom','driver','duo_win_rate_bayes']],
                                on=['nom','driver'], how='left')
        df_nc['duo_win_rate_bayes'] = df_nc.get('duo_win_rate_bayes', pd.Series([fallback_v8]*len(df_nc))).fillna(fallback_v8)

        # Stats entraineur depuis bundle
        entr_stats = bundle_galop.get('entr_stats')
        if entr_stats is not None:
            df_nc = df_nc.merge(entr_stats[['entraineur','entr_win_rate_bayes']],
                                on='entraineur', how='left')
        df_nc['entr_win_rate_bayes'] = df_nc.get('entr_win_rate_bayes', pd.Series([fallback_v8]*len(df_nc))).fillna(fallback_v8)

        # Spécialisation distance
        df_nc['tranche_dist'] = pd.cut(df_nc['distance'],
            bins=[0,1600,2100,2700,9999], labels=['court','moyen','long','tres_long']).astype(str)
        spec_dist = bundle_galop.get('spec_dist')
        if spec_dist is not None:
            spec = spec_dist.copy()
            if 'tranche_distance' in spec.columns:
                spec = spec.rename(columns={'tranche_distance':'tranche_dist'})
            df_nc = df_nc.merge(spec[['nom','tranche_dist','spec_dist_rate']],
                                on=['nom','tranche_dist'], how='left')
        df_nc['spec_dist_rate'] = df_nc.get('spec_dist_rate', pd.Series([fallback_v8]*len(df_nc))).fillna(fallback_v8)

        # Features numériques
        df_nc['log_montant_prix'] = np.log1p(df_nc['montant_prix'])
        df_nc['nb_partants_c']    = len(df_nc)

        # Stalle de départ
        df_nc['place_corde_norm'] = df_nc['place_corde'].replace(0, np.nan).fillna(len(df_nc)/2)
        df_nc['stalle_avantage']  = df_nc['place_corde'].apply(
            lambda x: 1.0 if 3<=x<=5 else (0.7 if x in [1,2,6,7] else 0.3)
            if pd.notna(x) and x > 0 else 0.5
        )

        # Terrain
        _terrain_map = {
            'Lourd': 1.0, 'Collant': 1.5, 'Très souple': 2.0,
            'Souple': 2.5, 'Bon souple': 3.0, 'Bon léger': 3.5,
            'Bon': 4.0, 'Léger': 4.5,
            'PSF LENTE': 3.0, 'PSF': 3.5, 'PSF STANDARD': 3.5,
            'PSF RAPIDE': 4.0, 'PSF TRES RAPIDE': 4.5,
        }
        terrain_label = str(df_nc['terrain_label'].iloc[0]) if 'terrain_label' in df_nc.columns and len(df_nc) > 0 else ''
        df_nc['terrain_num'] = _terrain_map.get(terrain_label, bundle_galop.get('median_terrain_num', 3.5))
        df_nc['terrain_psf'] = 1 if 'PSF' in terrain_label else 0

    df_input = pd.DataFrame(index=df_nc.index)
    for feat in features_5:
        df_input[feat] = df_nc[feat] if feat in df_nc.columns else bundle_galop.get(f'median_{feat}', 0.5)

    probas = bundle_galop['model'].predict_proba(df_input[features_5])[:, 1]

    # PLAT V8 : pipeline brut sans score_cote (comme ATTELÉ)
    if discipline_raw == 'PLAT':
        proba_min   = bundle_galop.get('proba_min', 0.19)
        proba_max   = bundle_galop.get('proba_max', 0.70)
        score_final = pd.Series(probas, index=df_nc.index)
        df_nc['note_pmu']  = np.round((score_final - proba_min) / (proba_max - proba_min) * 19 + 1).clip(1, 20).astype(int)
        df_nc['proba_pmu'] = score_final
    else:
        # HAIE/MONTE : pipeline brut sans score_cote (comme ATTELÉ et PLAT)
        poids_cote  = bundle_galop.get('poids_cote_fixe', 0.0)   # fallback 0.0 — cote désactivée
        poids_xgb   = bundle_galop.get('poids_xgb', 1.0)         # fallback 1.0
        if poids_cote > 0:
            score_final = pd.Series(poids_xgb * probas + poids_cote * df_nc['score_cote'].values,
                                    index=df_nc.index)
        else:
            score_final = pd.Series(probas, index=df_nc.index)
        g_min = bundle_galop.get('proba_min')
        g_max = bundle_galop.get('proba_max')
        df_nc['note_pmu']  = _proba_to_note_v7(score_final, proba_min_ref=g_min, proba_max_ref=g_max)
        df_nc['proba_pmu'] = score_final

    # ── Résultat JSON ─────────────────────────────────────────
    result = []
    for _, row in df_nc.sort_values('note_pmu', ascending=False).iterrows():
        result.append({
            "numero":    int(row['numero']),
            "nom":       str(row['nom']),
            "note_pmu":  int(row['note_pmu']),
            "proba_pmu": round(float(row['proba_pmu']) * 100, 1) if pd.notna(row['proba_pmu']) else 0,
            "driver":    str(row['driver']),
            "cote":      float(row['_cote_app']) if pd.notna(row['_cote_app']) else None,
            "avis":      int(row['avis_entraineur']) if pd.notna(row.get('avis_entraineur')) else 0,
            "scores": {
                "forme":      int(round(float(row['score_forme'])      * 100)) if pd.notna(row['score_forme'])      else 0,
                "duo":        int(round(float(row['score_duo'])        * 100)) if pd.notna(row['score_duo'])        else 0,
                "jockey":     int(round(float(row['score_jockey'])     * 100)) if pd.notna(row.get('score_jockey')) else 0,
                "historique": int(round(float(row['score_historique']) * 100)) if pd.notna(row['score_historique']) else 0,
                "gains":      int(round(float(row['score_gains'])      * 100)) if pd.notna(row['score_gains'])      else 0,
                "handicap":   int(round(float(row['score_handicap'])   * 100)) if pd.notna(row['score_handicap'])   else 0,
                "cote":       int(round(float(row['score_cote'])       * 100)) if pd.notna(row['score_cote'])       else 0,
            },
            "taux_disq":    round(float(row['mus_taux_disq']) * 100, 1) if pd.notna(row.get('mus_taux_disq')) else 0,
            "musique":      str(row.get('musique', '')) if row.get('musique') else '',
            "handicap_poids":  int(row.get('handicap_poids', 0)),
            "handicap_valeur": float(row.get('handicap_valeur', 0)),
        })

    return jsonify({
        "date":       date_str,
        "reunion":    r_num,
        "course":     c_num,
        "discipline": discipline_raw,
        "version":    f"v2_galop_{discipline_raw.lower()}",
        "chevaux":    result,
    })


@app.route('/notes_pmu', methods=['GET'])
def notes_pmu():
    """
    Calcule les notes PMU pour une course donnée.
    Paramètres GET : date (DDMMYYYY), reunion (int), course (int)
    Ex: /notes_pmu?date=05032026&reunion=1&course=5
    """
    if _model_pmu is None:
        return jsonify({"error": "Modèle PMU non disponible"}), 503

    date_str = request.args.get('date', '')
    r_num    = request.args.get('reunion', '')
    c_num    = request.args.get('course', '')

    if not date_str or not r_num or not c_num:
        return jsonify({"error": "Paramètres requis : date, reunion, course"}), 400
    try:
        r_num = int(r_num); c_num = int(c_num)
    except ValueError:
        return jsonify({"error": "reunion et course doivent être des entiers"}), 400

    # ── Conditions de course & performances détaillées ───────
    conditions    = _fetch_conditions(date_str, r_num, c_num)
    perfs_map     = _fetch_performances(date_str, r_num, c_num)
    discipline_raw = conditions.get('discipline_raw', '')

    # ── Routing par discipline ────────────────────────────────
    # STEEPLECHASE et CROSS : non supportés
    if discipline_raw in DISCIPLINES_SKIP:
        return jsonify({"error": f"Discipline {discipline_raw} non supportée"}), 400
    # Galop : PLAT, HAIE, MONTE
    is_galop = discipline_raw in DISCIPLINES_GALOP
    if is_galop and discipline_raw not in _models_galop:
        return jsonify({"error": f"Modèle {discipline_raw} non disponible"}), 503

    # ── Participants ──────────────────────────────────────────
    url = (f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
           f"/{date_str}/R{r_num}/C{c_num}/participants")
    try:
        resp = http_requests.get(url, timeout=8)
        resp.raise_for_status()
        participants = resp.json().get('participants', [])
    except Exception as e:
        return jsonify({"error": f"Erreur API PMU : {str(e)}"}), 502

    if not participants:
        return jsonify({"error": "Aucun participant trouvé"}), 404

    # Médiane rapport de référence
    rapports_course = [
        p['dernierRapportReference'].get('rapport')
        for p in participants
        if p.get('dernierRapportReference') and p.get('statut') != 'NON_PARTANT'
    ]
    rapports_course  = [r for r in rapports_course if r]
    mediane_rr       = float(np.median(rapports_course)) if rapports_course else _mediane_rapport_ref

    rows = []
    for p in participants:
        if p.get('statut') == 'NON_PARTANT' or p.get('incident') == 'NON_PARTANT':
            continue
        mus        = _parser_musique_api(p.get('musique', ''))
        musique_brute = p.get('musique', '')
        gains      = p.get('gainsParticipant', {}) or {}
        rk         = p.get('reductionKilometrique', 0) or 0
        num_pmu    = p.get('numPmu')
        nb_courses = p.get('nombreCourses', 0) or 0
        driver_nom = (p.get('driver', {}).get('nom', '')
                      if isinstance(p.get('driver'), dict) else str(p.get('driver', '')))
        entr_nom   = (p.get('entraineur', {}).get('nom', '')
                      if isinstance(p.get('entraineur'), dict) else str(p.get('entraineur', '')))
        nb_victoires = p.get('nombreVictoires', 0) or 0
        nb_places    = p.get('nombrePlaces', 0) or 0
        gains_car    = gains.get('gainsCarriere', 0) or 0
        gains_ann    = gains.get('gainsAnneeEnCours', 0) or 0

        rapport_ref = None
        if p.get('dernierRapportReference'):
            rapport_ref = p['dernierRapportReference'].get('rapport')
        if rapport_ref is None:
            rapport_ref = mediane_rr

        # Cote en temps réel (rapport direct)
        cote_app = None
        if p.get('dernierRapportDirect'):
            cote_app = p['dernierRapportDirect'].get('rapport')
        if cote_app is None and p.get('dernierRapportReference'):
            cote_app = p['dernierRapportReference'].get('rapport')

        perf = perfs_map.get(num_pmu, _perf_vide())

        row = {
            'numero':            num_pmu,
            'nom':               p.get('nom', ''),
            # Conditions course
            'distance':          conditions['distance'],
            'montant_prix':      conditions['montant_prix'],
            'discipline':        conditions['discipline'],
            'corde':             conditions['corde'],
            'condition_sexe':    conditions['condition_sexe'],
            'nb_partants':       conditions['nb_partants'],
            # Cheval
            'age':               p.get('age', 0) or 0,
            'deferre':           _ferrage_map_pmu.get(p.get('deferre', 'FERRE'), 0),
            'oeilleres':         1 if p.get('oeilleres') else 0,
            'driver':            driver_nom,
            'entraineur':        entr_nom,
            'nb_courses':        nb_courses,
            'nb_victoires':      nb_victoires,
            'nb_places':         nb_places,
            'gains_carriere':    gains_car,
            'gains_annee':       gains_ann,
            'reduction_km_corr': rk if rk > 0 else 72600,
            'avis_entraineur':   _avis_map_pmu.get(p.get('avisEntraineur', 'NEUTRE'), 0),
            'rapport_ref':       float(rapport_ref),
            'rapport_direct':    float(cote_app) if cote_app else float(rapport_ref),
            'ecart_cotes':       float(cote_app - rapport_ref) if cote_app else 0.0,
            'log_rapport_ref':   float(np.log1p(rapport_ref)),
            'nb_places_second':  p.get('nombrePlacesSecond', 0) or 0,
            'nb_places_troisieme': p.get('nombrePlacesTroisieme', 0) or 0,
            'temps_obtenu':      float(p.get('tempsObtenu', 0) or 0),
            'handicap_distance': float(p.get('handicapDistance', 0) or conditions['distance'] or 0),
            'handicap_poids':    float(p.get('handicapPoids', 0) or 0),
            'handicap_valeur':   float(p.get('handicapValeur', 0) or 0),
            '_cote_app':         cote_app,
            'place_corde':       float(p.get('placeCorde', 0) or 0),
            'terrain_label':     conditions.get('terrain_label', ''),
            'terrain_val':       float(conditions.get('terrain_val', 3.0)),
        }
        row['musique'] = musique_brute
        row.update(mus)
        row.update(perf)
        rows.append(row)

    df_nc = pd.DataFrame(rows)

    # ════════════════════════════════════════════════════════════
    # ROUTING GALOP — si discipline galop, pipeline dédié
    # ════════════════════════════════════════════════════════════
    if is_galop:
        return _notes_pmu_galop(df_nc, discipline_raw, date_str, r_num, c_num)

    # ════════════════════════════════════════════════════════════
    # ARCHITECTURE V6 — DEUX ÉTAPES
    # Étape 1 : scores métier déterministes (6 dimensions)
    # Étape 2 : XGBoost sur ces 6 scores uniquement
    # ════════════════════════════════════════════════════════════

    # ── Features dérivées brutes (inchangées) ─────────────────
    df_nc['ratio_victoires']        = df_nc['nb_victoires'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places']           = df_nc['nb_places']    / (df_nc['nb_courses'] + 1)
    df_nc['gains_par_course']       = df_nc['gains_carriere'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_gains_rec']        = df_nc['gains_annee'] / (df_nc['gains_carriere'] + 1)
    df_nc['ratio_places_second']    = df_nc['nb_places_second']    / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places_troisieme'] = df_nc['nb_places_troisieme'] / (df_nc['nb_courses'] + 1)
    df_nc['temps_norme'] = df_nc.apply(
        lambda r: round(r['temps_obtenu'] / r['handicap_distance'], 4)
        if r['handicap_distance'] > 0 and r['temps_obtenu'] > 0 else np.nan, axis=1
    )
    df_nc['log_distance']     = np.log1p(df_nc['distance'])
    df_nc['log_montant_prix'] = np.log1p(df_nc['montant_prix'])
    df_nc['rang_cote_course'] = df_nc['rapport_ref'].rank(ascending=True, method='min')
    nb_ch = len(df_nc)
    df_nc['rang_cote_norme']  = (df_nc['rang_cote_course'] - 1) / (nb_ch - 1 + 1e-8)
    df_nc['tranche_distance'] = pd.cut(
        df_nc['distance'], bins=[0, 1600, 2100, 2700, 9999],
        labels=['court', 'moyen', 'long', 'tres_long']
    ).astype(str)

    _fallback = _prior_pmu * _k_bayes_pmu / (_k_bayes_pmu + 1) if _prior_pmu else 0.094

    # ── Driver stats ──────────────────────────────────────────
    if _le_driver is not None and _driver_stats is not None:
        top_drivers = set(_le_driver.classes_)
        df_nc['driver_enc'] = df_nc['driver'].apply(lambda x: x if x in top_drivers else 'AUTRE')
        df_nc['driver_id']  = _le_driver.transform(df_nc['driver_enc'])
        d_cols = ['driver', 'driver_win_rate_bayes', 'driver_n']
        if 'driver_place_rate_bayes' in _driver_stats.columns:
            d_cols += ['driver_place_rate_bayes', 'driver_disq']
        df_nc = df_nc.merge(_driver_stats[d_cols], on='driver', how='left')
        df_nc['driver_win_rate_bayes']   = df_nc['driver_win_rate_bayes'].fillna(_fallback)
        df_nc['driver_n']                = df_nc['driver_n'].fillna(0)
        if 'driver_place_rate_bayes' in df_nc.columns:
            df_nc['driver_place_rate_bayes'] = df_nc['driver_place_rate_bayes'].fillna(_fallback)
            df_nc['driver_disq']             = df_nc['driver_disq'].fillna(0)
    elif _driver_stats is not None:
        df_nc = df_nc.merge(
            _driver_stats[['driver', 'driver_win_rate_bayes', 'driver_n']],
            on='driver', how='left')
    if 'driver_win_rate_bayes' not in df_nc.columns:
        df_nc['driver_win_rate_bayes'] = _fallback
        df_nc['driver_n']              = 0
    df_nc['driver_win_rate_bayes'] = df_nc['driver_win_rate_bayes'].fillna(_fallback)
    df_nc['driver_n']              = df_nc['driver_n'].fillna(0)

    if _le_entr is not None:
        top_entrs = set(_le_entr.classes_)
        df_nc['entraineur_enc'] = df_nc['entraineur'].apply(lambda x: x if x in top_entrs else 'AUTRE')
        df_nc['entraineur_id']  = _le_entr.transform(df_nc['entraineur_enc'])
    if _entr_stats is not None:
        df_nc = df_nc.merge(
            _entr_stats[['entraineur', 'entr_win_rate_bayes', 'entr_n']],
            on='entraineur', how='left')
    if 'entr_win_rate_bayes' not in df_nc.columns:
        df_nc['entr_win_rate_bayes'] = _fallback
        df_nc['entr_n'] = 0
    df_nc['entr_win_rate_bayes'] = df_nc['entr_win_rate_bayes'].fillna(_fallback)
    df_nc['entr_n']              = df_nc['entr_n'].fillna(0)

    if _duo_stats is not None:
        df_nc = df_nc.merge(
            _duo_stats[['nom', 'driver', 'duo_win_rate_bayes', 'duo_n']],
            on=['nom', 'driver'], how='left')
    if 'duo_win_rate_bayes' not in df_nc.columns:
        df_nc['duo_win_rate_bayes'] = _fallback
        df_nc['duo_n'] = 0
    df_nc['duo_win_rate_bayes'] = df_nc['duo_win_rate_bayes'].fillna(_fallback)
    df_nc['duo_n']              = df_nc['duo_n'].fillna(0)
    df_nc['duo_fiable']         = (df_nc['duo_n'] >= 2).astype(int)
    df_nc['duo_fiable_v2']      = (df_nc['duo_n'] >= _duo_fiable_seuil_v9).astype(int)

    if _spec_dist is not None:
        spec_cols_dist = 'tranche_dist' if 'tranche_dist' in _spec_dist.columns else 'tranche_distance'
        df_nc = df_nc.merge(
            _spec_dist[['nom', spec_cols_dist, 'spec_dist_rate']].rename(
                columns={spec_cols_dist: 'tranche_distance'}),
            on=['nom', 'tranche_distance'], how='left')
    if 'spec_dist_rate' not in df_nc.columns:
        df_nc['spec_dist_rate'] = _fallback
        df_nc['spec_n'] = 0
    df_nc['spec_dist_rate'] = df_nc['spec_dist_rate'].fillna(_fallback)
    df_nc['spec_n']         = df_nc['spec_n'].fillna(0) if 'spec_n' in df_nc.columns else 0

    if _spec_disc is not None:
        df_nc = df_nc.merge(
            _spec_disc[['nom', 'discipline', 'spec_disc_rate']],
            on=['nom', 'discipline'], how='left')
    if 'spec_disc_rate' not in df_nc.columns:
        df_nc['spec_disc_rate'] = _fallback
    df_nc['spec_disc_rate'] = df_nc['spec_disc_rate'].fillna(_fallback)

    if _hist_snapshot is not None:
        hist_cols_dispo = [c for c in ['nom', 'hist_nb', 'hist_moy_classement', 'hist_nb_top3',
                            'hist_taux_top3', 'hist_moy_temps', 'hist_tendance', 'hist_moy_cote',
                            'courses_60j']
                           if c in _hist_snapshot.columns]
        df_nc = df_nc.merge(_hist_snapshot[hist_cols_dispo], on='nom', how='left')
    if 'courses_60j' not in df_nc.columns:
        df_nc['courses_60j'] = 0
    df_nc['courses_60j'] = df_nc['courses_60j'].fillna(0).astype(int)
    for col in ['hist_nb', 'hist_nb_top3']:
        if col not in df_nc.columns:
            df_nc[col] = 0
        df_nc[col] = df_nc[col].fillna(0)
    for col in ['hist_moy_classement', 'hist_taux_top3', 'hist_moy_temps',
                'hist_tendance', 'hist_moy_cote']:
        if col not in df_nc.columns:
            df_nc[col] = np.nan

    # ── Features V9 : forme récente & momentum duo ────────────
    prior_v9 = _prior_pmu if _prior_pmu else 0.309

    if _duo_momentum_snap is not None:
        df_nc = df_nc.merge(
            _duo_momentum_snap[['nom', 'driver', 'duo_momentum_3']],
            on=['nom', 'driver'], how='left')
    if 'duo_momentum_3' not in df_nc.columns:
        df_nc['duo_momentum_3'] = prior_v9
    df_nc['duo_momentum_3'] = df_nc['duo_momentum_3'].fillna(prior_v9)

    if _top3_3courses_snap is not None:
        df_nc = df_nc.merge(
            _top3_3courses_snap[['nom', 'top3_3courses']],
            on='nom', how='left')
    if 'top3_3courses' not in df_nc.columns:
        df_nc['top3_3courses'] = prior_v9
    df_nc['top3_3courses'] = df_nc['top3_3courses'].fillna(prior_v9)

    if _top3_60j_snap is not None:
        col_60j = 'top3_60j' if 'top3_60j' in _top3_60j_snap.columns else 'top3_60j_snap'
        df_nc = df_nc.merge(
            _top3_60j_snap[['nom', col_60j]].rename(columns={col_60j: 'top3_60j'}),
            on='nom', how='left')
    if 'top3_60j' not in df_nc.columns:
        df_nc['top3_60j'] = prior_v9
    df_nc['top3_60j'] = df_nc['top3_60j'].fillna(prior_v9)

    # ── reduction_km_v2 (fallback corrigé par tranche) ────────
    def _get_rk_v2(row):
        rk = row.get('reduction_km_corr', 0)
        if rk and rk > 0 and rk != 72600:
            return rk
        return _fallback_rk_v9.get(str(row.get('tranche_distance', 'long')), 76100)
    df_nc['reduction_km_v2'] = df_nc.apply(_get_rk_v2, axis=1)
    df_nc['reduction_km']    = df_nc['reduction_km_v2']

    # ════════════════════════════════════════════════════════════
    # ÉTAPE 1 — SCORES MÉTIER (0.0 → 1.0 chacun)
    # Chaque score résume une dimension indépendante.
    # Aucune dimension ne peut écraser les autres.
    # ════════════════════════════════════════════════════════════

    def _norm(series, low, high):
        """Clip + normalise linéairement entre 0 et 1 (bornes absolues)."""
        return ((series.clip(low, high) - low) / (high - low + 1e-9)).clip(0, 1)

    def _norm_rel(series):
        """Normalise relativement au peloton : min→0, max→1.
        Si tous les chevaux sont identiques, retourne 0.5 pour tous."""
        mn, mx = series.min(), series.max()
        if mx - mn < 1e-9:
            return pd.Series(0.5, index=series.index)
        return ((series - mn) / (mx - mn)).clip(0, 1)

    def _norm_mix(series, low, high, rel_weight=0.5):
        """Mix normalisation absolue + relative au peloton."""
        abs_norm = _norm(series, low, high)
        rel_norm = _norm_rel(series)
        return (abs_norm * (1 - rel_weight) + rel_norm * rel_weight).clip(0, 1)

    # ── Score 1 : Forme / Musique ─────────────────────────────
    s_score_p   = _norm_mix(df_nc['mus_score_pondere'],  0, 9)
    s_derniere  = _norm_mix(15 - df_nc['mus_derniere_place'], 0, 14)
    s_podiums   = _norm_mix(df_nc['mus_nb_podiums'],     0, 5)
    s_disq      = 1 - _norm(df_nc['mus_taux_disq'], 0, 0.3)
    s_temps_mus = _norm_mix(df_nc['hist_moy_temps'].fillna(df_nc['hist_moy_temps'].median()).fillna(100), 60, 100)
    s_age       = 1 - _norm(df_nc['age'].fillna(5), 3, 12)        # jeune = meilleur potentiel
    s_deferre   = df_nc['deferre'].fillna(0).astype(float)        # déferré = signal positif

    df_nc['score_forme'] = (
        s_score_p   * 0.30 +
        s_derniere  * 0.25 +
        s_podiums   * 0.15 +
        s_disq      * 0.10 +
        s_temps_mus * 0.10 +
        s_age       * 0.05 +
        s_deferre   * 0.05
    ).clip(0, 1)

    # ── Score 2 : Duo cheval/driver ───────────────────────────
    s_winrate = _norm_mix(df_nc['duo_win_rate_bayes'], _fallback * 0.8, 0.65)
    s_fiable  = df_nc['duo_fiable'].astype(float)
    s_duo_n   = _norm_mix(df_nc['duo_n'], 1, 15)

    df_nc['score_duo'] = (
        s_winrate * 0.60 +
        s_fiable  * 0.25 +
        s_duo_n   * 0.15
    ).clip(0, 1)

    # ── Score 3 : Historique cheval ───────────────────────────
    hist_taux   = df_nc['hist_taux_top3'].fillna(_fallback)
    hist_class  = df_nc['hist_moy_classement'].fillna(8)
    hist_fiable = _norm(df_nc['hist_nb'], 0, 20)
    hist_tend   = df_nc['hist_tendance'].fillna(0)
    hist_cote   = df_nc['hist_moy_cote'].fillna(df_nc['hist_moy_cote'].median()).fillna(15)

    s_taux_top3  = _norm_mix(hist_taux,        0, 0.7)
    s_classement = _norm_mix(10 - hist_class, -5, 9)
    s_h_fiable   = hist_fiable
    s_tendance   = _norm_mix(hist_tend, -3, 3)                    # tendance positive = en forme
    s_hist_cote  = 1 - _norm(hist_cote, 2, 30)                   # cote historique basse = bon cheval

    df_nc['score_historique'] = (
        s_taux_top3  * 0.35 +
        s_classement * 0.25 +
        s_h_fiable   * 0.10 +
        s_tendance   * 0.20 +
        s_hist_cote  * 0.10
    ).clip(0, 1)

    # ── Score 4 : Gains / Palmarès carrière ───────────────────
    s_ratio_vic   = _norm_mix(df_nc['ratio_victoires'],  0, 0.4)
    s_gains_c     = _norm_mix(df_nc['gains_par_course'], 0, 8000)
    s_gains_ann   = _norm_mix(df_nc['gains_annee'],      0, 150000)
    s_ratio_gains = _norm_mix(df_nc['ratio_gains_rec'],  0, 0.5)
    s_ratio_pl    = _norm_mix(df_nc['ratio_places'],     0, 0.6)  # régularité dans le peloton

    df_nc['score_gains'] = (
        s_ratio_vic   * 0.30 +
        s_gains_c     * 0.25 +
        s_gains_ann   * 0.20 +
        s_ratio_gains * 0.15 +
        s_ratio_pl    * 0.10
    ).clip(0, 1)

    # ── Score 5 : Spécialisation / Adéquation course ─────────
    s_spec_dist  = _norm_mix(df_nc['spec_dist_rate'],       _fallback * 0.8, 0.65)
    s_spec_disc  = _norm_mix(df_nc['spec_disc_rate'],       _fallback * 0.8, 0.65)
    s_entr       = _norm_mix(df_nc['entr_win_rate_bayes'],  _fallback * 0.8, 0.55)
    s_avis       = _norm(df_nc['avis_entraineur'].astype(float), -1, 1)

    df_nc['score_adequation'] = (
        s_spec_dist * 0.35 +
        s_spec_disc * 0.25 +
        s_entr      * 0.25 +
        s_avis      * 0.15
    ).clip(0, 1)

    # ── Score 6 : Cote & marché ───────────────────────────────
    s_cote_rang   = 1 - df_nc['rang_cote_norme']                        # rang inversé : favori = 1
    s_ecart       = _norm(-df_nc['ecart_cotes'].abs(), -10, 0)          # petite déviation live/ref = bon signe
    _med_temps    = df_nc['temps_norme'].median()
    _med_temps    = _med_temps if pd.notna(_med_temps) else 0.0
    s_temps       = _norm(1 / (1 + df_nc['temps_norme'].fillna(_med_temps)), 0, 1)
    s_cote_direct = 1 - _norm_mix(df_nc['rapport_direct'].fillna(df_nc['rapport_ref']), 2, 50)  # cote live basse = favori

    df_nc['score_cote'] = (
        s_cote_rang   * 0.40 +
        s_cote_direct * 0.30 +
        s_ecart       * 0.15 +
        s_temps       * 0.15
    ).clip(0, 1).fillna(s_cote_rang.clip(0, 1))

    # ════════════════════════════════════════════════════════════
    # SCORING FINAL
    #
    # Les 6 scores (FORME, DUO, PALMARES, GAINS, ADEQUATION, COTE)
    # sont déjà calculés ci-dessus depuis les données PMU en temps réel.
    #
    # Priorité :
    #   1. XGBoost V8  → entraîné sur les 6 vrais scores (reconstruct_and_train_v8.py)
    #   2. Fallback V6 → somme pondérée fixe (si V8 non disponible)
    # ════════════════════════════════════════════════════════════

    SCORES_6 = ['score_forme', 'score_duo', 'score_historique',
                'score_gains', 'score_adequation', 'score_cote']
    POIDS_V6 = [0.21, 0.17, 0.14, 0.08, 0.26, 0.11]

    # Somme pondérée V6 (toujours calculée — sert de fallback)
    score_metier = sum(df_nc[s] * p for s, p in zip(SCORES_6, POIDS_V6))
    df_nc['score_metier'] = score_metier

    if _use_v7 and _model_v7 is not None:
        try:
            # Utiliser les features telles qu'enregistrées dans le pkl
            # (SCORES_6 pour V8, autres features pour V7.x antérieurs)
            features_modele = _bundle_v7.get('features', SCORES_6)

            # Construire le DataFrame d'input — toutes les features
            # doivent être disponibles dans df_nc (calculées ci-dessus)
            df_input = pd.DataFrame(index=df_nc.index)
            for feat in features_modele:
                if feat in df_nc.columns:
                    df_input[feat] = df_nc[feat]
                else:
                    # Feature inconnue → valeur neutre
                    print(f"⚠️  Feature '{feat}' absente — remplacée par 0.5")
                    df_input[feat] = 0.5

            probas = _model_v7.predict_proba(df_input[features_modele])[:, 1]

            # Combinaison hybride : XGBoost + poids fixe score_cote
            # Si le pkl définit poids_cote_fixe (modèle V8), on l'applique.
            # Sinon on utilise les probas brutes (comportement V7 classique).
            poids_cote = _bundle_v7.get('poids_cote_fixe', 0.0)
            poids_xgb  = _bundle_v7.get('poids_xgb', 1.0)
            if poids_cote > 0:
                score_final = pd.Series(
                    poids_xgb * probas + poids_cote * df_nc['score_cote'].values,
                    index=df_nc.index
                )
            else:
                score_final = pd.Series(probas, index=df_nc.index)

            # Conversion en notes : bornes lues depuis le pkl V9
            p_min = _bundle_v7.get('proba_min')
            p_max = _bundle_v7.get('proba_max')
            df_nc['note_pmu'] = _proba_to_note_v7(score_final,
                                                   proba_min_ref=p_min,
                                                   proba_max_ref=p_max)

            version_utilisee = _bundle_v7.get('version', 'v9')

        except Exception as e:
            print(f"⚠️  XGBoost predict_proba échoué ({e}) — fallback V6")
            score_final      = score_metier
            df_nc['note_pmu'] = _proba_to_note_api(score_final)
            version_utilisee = "v6_fallback"
    else:
        # V6 : somme pondérée fixe
        score_final      = score_metier
        df_nc['note_pmu'] = _proba_to_note_api(score_final)
        version_utilisee = "v6"

    df_nc['proba_pmu'] = score_final


    # ── Résultat JSON (scores détaillés inclus) ───────────────
    result = []
    for _, row in df_nc.sort_values('note_pmu', ascending=False).iterrows():
        result.append({
            "numero":    int(row['numero']),
            "nom":       str(row['nom']),
            "note_pmu":  int(row['note_pmu']),
            "proba_pmu": round(float(row['proba_pmu']) * 100, 1) if pd.notna(row['proba_pmu']) else 0,
            "driver":    str(row['driver']),
            "cote":      float(row['_cote_app']) if pd.notna(row['_cote_app']) else None,
            "avis":      int(row['avis_entraineur']) if pd.notna(row['avis_entraineur']) else 0,
            # ✨ V6 — scores détaillés par dimension (0-100)
            "scores": {
                "forme":      int(round(float(row['score_forme'])      * 100)) if pd.notna(row['score_forme'])      else 0,
                "duo":        int(round(float(row['score_duo'])        * 100)) if pd.notna(row['score_duo'])        else 0,
                "historique": int(round(float(row['score_historique']) * 100)) if pd.notna(row['score_historique']) else 0,
                "gains":      int(round(float(row['score_gains'])      * 100)) if pd.notna(row['score_gains'])      else 0,
                "adequation": int(round(float(row['score_adequation']) * 100)) if pd.notna(row['score_adequation']) else 0,
                "cote":       int(round(float(row['score_cote'])       * 100)) if pd.notna(row['score_cote'])       else 0,
            },
            "taux_disq":    round(float(row['mus_taux_disq']) * 100, 1) if pd.notna(row.get('mus_taux_disq')) else 0,
            "musique":      str(row.get('musique', '')) if row.get('musique') else '',
            "courses_60j":  int(row['courses_60j']) if pd.notna(row.get('courses_60j')) else 0,
        })

    return jsonify({
        "date":     date_str,
        "reunion":  r_num,
        "course":   c_num,
        "version":  version_utilisee,
        "chevaux":  result,
    })


# ============================================================
# TÉLÉCHARGEMENT CSV
# ============================================================
@app.route('/download_historique', methods=['GET'])
def download_historique():
    from flask import send_file
    if os.path.exists(HISTORIQUE_PATH):
        return send_file(
            HISTORIQUE_PATH,
            mimetype='text/csv',
            as_attachment=True,
            download_name='historique_notes.csv'
        )
    else:
        return jsonify({"error": "Fichier non trouvé"}), 404

# ============================================================
# DEBUG — liste des features V5
# ============================================================
@app.route('/features', methods=['GET'])
def get_features():
    if _features_pmu is None:
        return jsonify({"error": "Modèle non chargé"}), 503
    return jsonify({
        "nb_features": len(_features_pmu),
        "features": list(_features_pmu)
    })

# ============================================================
# DEBUG — infos stockage
# ============================================================
@app.route('/storage_info', methods=['GET'])
def storage_info():
    """Indique quel backend de stockage est actif pour /ajouter."""
    conn = _get_pg()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM courses_manuelles")
                nb = cur.fetchone()[0]
            return jsonify({"stockage": "postgresql", "nb_lignes_manuelles": nb, "ephemere": False})
        except Exception as e:
            return jsonify({"stockage": "postgresql_erreur", "detail": str(e)})
    nb_csv = 0
    if os.path.exists(CSV_PATH):
        try:
            nb_csv = len(pd.read_csv(CSV_PATH))
        except Exception:
            pass
    return jsonify({
        "stockage": "csv_local",
        "ephemere": True,
        "avertissement": "Les données /ajouter seront perdues au redémarrage. Ajoutez DATABASE_URL pour activer PostgreSQL.",
        "nb_lignes_csv": nb_csv,
    })

# ============================================================
# ENTRAÎNEMENT XGBoost V7 — sur les 6 scores métier
# ============================================================
def _calculer_scores_historique(df_hist):
    """
    Applique le même pipeline de scoring V6 sur le DataFrame historique
    (qui contient rang_arrivee) pour produire les features d'entraînement V7.
    Retourne un DataFrame avec les 6 scores + la cible top3.
    """
    d = df_hist.dropna(subset=['rang_arrivee', 'note', 'rapport']).copy()
    if len(d) < 200:
        return None

    # ── Features de base disponibles dans historique_notes.csv ──
    # note, rapport, rang_arrivee sont toujours présents.
    # Les features PMU détaillées (musique, gains, driver…) ne sont PAS
    # dans l'historique CSV — on calcule des scores simplifiés mais cohérents.

    def _norm(s, lo, hi):
        return ((s.clip(lo, hi) - lo) / (hi - lo + 1e-9)).clip(0, 1)

    def _norm_rel(s):
        mn, mx = s.min(), s.max()
        if mx - mn < 1e-9:
            return pd.Series(0.5, index=s.index)
        return ((s - mn) / (mn)) .clip(0, 1) if False else ((s - mn) / (mx - mn)).clip(0, 1)

    def _norm_mix(s, lo, hi):
        return (_norm(s, lo, hi) * 0.5 + _norm_rel(s) * 0.5).clip(0, 1)

    # Score forme — basé sur note (proxy musique)
    d['score_forme']      = _norm_mix(d['note'], 0, 20)

    # Score duo — non disponible dans l'historique CSV → valeur neutre 0.5
    d['score_duo']        = 0.5

    # Score historique — taux top3 par cheval calculé sur l'historique glissant
    grp = d.groupby('nom')['rang_arrivee'].apply(lambda x: (x <= 3).mean()).rename('hist_top3')
    d = d.join(grp, on='nom')
    d['score_historique'] = _norm_mix(d['hist_top3'].fillna(0.3), 0, 0.7)

    # Score gains — non disponible → valeur neutre
    d['score_gains']      = 0.5

    # Score adéquation — non disponible → valeur neutre
    d['score_adequation'] = 0.5

    # Score cote — inversement proportionnel au rapport (favori = score élevé)
    d['score_cote']       = _norm_mix(1.0 / (1.0 + d['rapport']), 0, 0.5)

    d['target_top3'] = (d['rang_arrivee'] <= 3).astype(int)

    SCORES = ['score_forme', 'score_duo', 'score_historique',
              'score_gains', 'score_adequation', 'score_cote']
    return d[SCORES + ['target_top3', 'date']].dropna()


def _entrainer_v7():
    """
    Charge le modèle V9 attelé depuis model_pmu_v9_attele.pkl.
    Contient le modèle XGBoost + snapshots pour l'inférence.
    Pas de réentraînement au démarrage — uniquement chargement.
    """
    global _model_v7, _bundle_v7, _use_v7
    global _duo_momentum_snap, _top3_3courses_snap, _top3_60j_snap
    global _fallback_rk_v9, _duo_fiable_seuil_v9
    global _driver_stats, _entr_stats, _duo_stats, _spec_dist
    global _prior_pmu, _k_bayes_pmu

    if not os.path.exists(PMU_V7_PATH):
        print(f"⚠️  {PMU_V7_PATH} introuvable — modèle attelé V9 désactivé")
        _use_v7 = False
        return

    try:
        with open(PMU_V7_PATH, 'rb') as f:
            bundle = pickle.load(f)

        _model_v7  = bundle['model']
        _bundle_v7 = bundle
        _use_v7    = True

        # Snapshots V9
        _duo_momentum_snap  = bundle.get('duo_momentum_snap')
        _top3_3courses_snap = bundle.get('top3_3courses_snap')
        _top3_60j_snap      = bundle.get('top3_60j_snap')

        # Stats (priorité V9)
        if bundle.get('driver_stats') is not None:
            _driver_stats = bundle['driver_stats']
        if bundle.get('entr_stats') is not None:
            _entr_stats   = bundle['entr_stats']
        if bundle.get('duo_stats') is not None:
            _duo_stats    = bundle['duo_stats']
        if bundle.get('spec_dist') is not None:
            _spec_dist    = bundle['spec_dist']

        _fallback_rk_v9      = bundle.get('fallback_rk',
                                           {'court': 76000, 'moyen': 75100,
                                            'long': 76000, 'tres_long': 76500})
        _duo_fiable_seuil_v9 = bundle.get('duo_fiable_seuil', 5)
        if bundle.get('prior_win') is not None and _prior_pmu is None:
            _prior_pmu   = bundle['prior_win']
            _k_bayes_pmu = bundle.get('k_bayes', 10)

        # ── Bornes de probabilité pour la conversion en notes ──
        # Si le pkl ne les contient pas (V9 Colab), on les estime
        # depuis prior_win : le modèle produit des probas dans
        # [prior/3, prior*2.5] environ pour un XGBoost calibré
        if 'proba_min' not in bundle:
            prior = bundle.get('prior_win', 0.309)
            bundle['proba_min'] = round(prior * 0.25, 3)   # ≈ 0.077
            bundle['proba_max'] = round(min(prior * 2.5, 0.90), 3)  # ≈ 0.773
            print(f"ℹ️  proba_min/max estimés : [{bundle['proba_min']}, {bundle['proba_max']}]"
                  f" (prior={prior:.3f})")

        ver  = bundle.get('version', 'v9')
        auc  = bundle.get('auc_val', '?')
        nf   = len(bundle.get('features', []))
        n_dm = len(_duo_momentum_snap)  if _duo_momentum_snap  is not None else 0
        n_t3 = len(_top3_3courses_snap) if _top3_3courses_snap is not None else 0
        n_60 = len(_top3_60j_snap)      if _top3_60j_snap      is not None else 0
        print(f"✅ XGBoost {ver} chargé (AUC:{auc} · {nf} features · "
              f"duo_momentum:{n_dm} · top3_3c:{n_t3} · top3_60j:{n_60})")

    except Exception as e:
        print(f"❌ Erreur chargement {PMU_V7_PATH} : {e}")
        _use_v7 = False


# ============================================================
# STATS JOCKEY GALOP
# ============================================================
def _charger_jockey_stats_galop():
    """Charge les stats jockey depuis les pkl PLAT et HAIE."""
    global _jockey_stats_galop
    # Les stats sont embarquées dans model_pmu_plat.pkl
    for pkl_path in [GALOP_MODEL_PATHS.get('PLAT'), GALOP_MODEL_PATHS.get('HAIE')]:
        if not pkl_path or not os.path.exists(pkl_path):
            continue
        try:
            with open(pkl_path, 'rb') as f:
                bundle = pickle.load(f)
            if 'jockey_stats' in bundle:
                _jockey_stats_galop = bundle['jockey_stats']
                n = len(_jockey_stats_galop)
                mx = _jockey_stats_galop['driver_win_rate_bayes'].max()
                print(f"✅ Stats jockey galop chargées depuis {pkl_path} : {n} jockeys (max={mx:.3f})")
                return
        except Exception as e:
            print(f"⚠️  Erreur lecture jockey_stats depuis {pkl_path} : {e}")
    print("⚠️  jockey_stats non trouvées dans les pkl — score_jockey au fallback")


# ============================================================
# CHARGEMENT MODÈLES GALOP
# ============================================================
def _charger_modeles_galop():
    """Charge les modèles XGBoost galop depuis les pkl."""
    global _models_galop
    for disc, path in GALOP_MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️  {path} introuvable — {disc} désactivé")
            continue
        try:
            with open(path, 'rb') as f:
                bundle = pickle.load(f)
            _models_galop[disc] = bundle
            auc = bundle.get('auc_final_val', '?')
            print(f"✅ Modèle galop {disc} chargé (AUC final: {auc})")
        except Exception as e:
            print(f"❌ Erreur chargement {path} : {e}")
    print(f"✅ {len(_models_galop)} modèles galop chargés")


# ============================================================
# DÉMARRAGE
# ============================================================
_charger_modele_pmu()
initialiser()
_entrainer_v7()
_charger_modeles_galop()
_charger_jockey_stats_galop()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
