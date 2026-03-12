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
import requests as http_requests
from sklearn.ensemble import HistGradientBoostingClassifier

app = Flask(__name__)
CORS(app)

# ============================================================
# DONNÉES HISTORIQUES
# ============================================================
# historique_notes.csv : généré depuis dataset_pmu_v4 + ordre_arrivees
# colonnes : date, r_num, c_num, numero, nom, note, rapport, rang_arrivee
HISTORIQUE_PATH = "historique_notes.csv"
CSV_PATH        = "historique_courses.csv"  # courses ajoutées manuellement via /ajouter


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

    # Fusionner avec courses ajoutées manuellement
    if os.path.exists(CSV_PATH):
        df_manual = pd.read_csv(CSV_PATH)
        if len(df_manual) > 0:
            # Garder seulement les colonnes communes
            common_cols = ['date', 'numero', 'note', 'rapport', 'rang_arrivee']
            df_manual = df_manual[[c for c in common_cols if c in df_manual.columns]]
            df_base   = df[[c for c in common_cols if c in df.columns]]
            df = pd.concat([df_base, df_manual], ignore_index=True).drop_duplicates()

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
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "courses": int(df['date'].nunique()) if df is not None else 0})


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
    """
    global df, model, note_mean, note_std, modele_abs, note_mean_a, note_std_a

    data    = request.get_json()
    date    = data.get('date')
    chevaux = data.get('chevaux', [])

    if not date or not chevaux:
        return jsonify({"error": "date et chevaux requis"}), 400

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

    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    # Réentraînement
    df_p = df.copy()
    df_p['target'] = ((df_p['rapport'] >= 10) & (df_p['rang_arrivee'] <= 3)).astype(int)
    df_p, note_mean, note_std = _enrichir(df_p)
    model = _entrainer(df_p, FEATURES, 'target')

    df_a = df.copy()
    df_a['target_absolu'] = (df_a['rang_arrivee'] <= 3).astype(int)
    df_a, note_mean_a, note_std_a = _enrichir(df_a)
    modele_abs = _entrainer(df_a, FEATURES_ABSOLU, 'target_absolu')

    return jsonify({
        "message":  f"Course du {date} ajoutée et modèles réentraînés",
        "nb_lignes": len(df),
        "nb_courses": int(df['date'].nunique()),
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

PMU_MODEL_PATH = "model_pmu_v5.pkl"

DISC_MUSIQUE_MAP = {'a': 0, 'm': 1, 'p': 2, 'h': 3, 's': 4, 'c': 5}
DISCIPLINE_MAP   = {'TROT_ATTELE': 0, 'TROT_MONTE': 1, 'PLAT': 2, 'OBSTACLE': 3}
CORDE_MAP        = {'CORDE_A_GAUCHE': 0, 'CORDE_A_DROITE': 1}
SEXE_MAP         = {'MALES': 0, 'FEMELLES': 1, 'MIXTE': 2}


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
                    return {
                        'distance':       course.get('distance', 0) or 0,
                        'montant_prix':   course.get('montantPrix', 0) or 0,
                        'discipline':     DISCIPLINE_MAP.get(course.get('discipline',''), 0),
                        'corde':          CORDE_MAP.get(course.get('corde',''), 0),
                        'condition_sexe': SEXE_MAP.get(course.get('conditionSexe',''), 2),
                        'nb_partants':    course.get('nombreDeclaresPartants', 0) or 0,
                    }
    return _cond_vides()


def _cond_vides():
    return {'distance': 0, 'montant_prix': 0, 'discipline': 0,
            'corde': 0, 'condition_sexe': 2, 'nb_partants': 0}


# Seuils par défaut si bundle ne les contient pas
_SEUILS_DEFAUT = [
    (0.05, 1), (0.10, 2), (0.15, 3), (0.20, 4), (0.25, 5),
    (0.30, 6), (0.35, 7), (0.40, 8), (0.45, 9), (0.50, 10),
    (0.55, 11), (0.60, 12), (0.65, 13), (0.70, 14), (0.75, 15),
    (0.80, 16), (0.85, 17), (0.90, 18), (0.95, 19), (1.01, 20),
]

def _proba_to_note_api(proba_series):
    """
    Convertit les probabilités en notes 1-20 via seuils asymétriques.
    Utilise les seuils du bundle si disponibles, sinon les seuils par défaut.
    """
    seuils = _seuils_notes if _seuils_notes is not None else _SEUILS_DEFAUT
    def _convert(p):
        for seuil, note in seuils:
            if p < seuil:
                return note
        return 20
    return pd.Series(proba_series).apply(_convert)


def _scores_to_notes(score_series):
    """
    Convertit les scores métier (0-1) en notes 1-20 par rang relatif dans la course.
    Garantit une bonne différenciation même quand les scores sont proches.
    Le meilleur cheval obtient toujours une note élevée, le moins bon une note basse.
    """
    s = pd.Series(score_series)
    n = len(s)
    if n == 0:
        return s

    # Rang : 1 = meilleur score
    rangs = s.rank(ascending=False, method='min')

    # Étirement linéaire : rang 1 → note max, rang n → note min
    # Note max dépend du nombre de partants (plus il y a de chevaux, plus le favori se démarque)
    note_max = min(20, 12 + max(0, n - 6))   # 6 chevaux → 18, 16 chevaux → 20
    note_min = max(1,  note_max - n - 2)      # écart suffisant entre premier et dernier

    notes = note_max - (rangs - 1) * (note_max - note_min) / (n - 1 + 1e-9)
    return notes.round().astype(int).clip(1, 20)


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
    conditions = _fetch_conditions(date_str, r_num, c_num)
    perfs_map  = _fetch_performances(date_str, r_num, c_num)

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
            '_cote_app':         cote_app,
        }
        row.update(mus)
        row.update(perf)
        rows.append(row)

    df_nc = pd.DataFrame(rows)

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

    _fallback = _prior_pmu * _k_bayes_pmu / (_k_bayes_pmu + 1)

    # Merge stats externes (inchangé)
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

    if _spec_dist is not None:
        df_nc = df_nc.merge(
            _spec_dist[['nom', 'tranche_distance', 'spec_dist_rate', 'spec_n']],
            on=['nom', 'tranche_distance'], how='left')
    if 'spec_dist_rate' not in df_nc.columns:
        df_nc['spec_dist_rate'] = _fallback
        df_nc['spec_n'] = 0
    df_nc['spec_dist_rate'] = df_nc['spec_dist_rate'].fillna(_fallback)
    df_nc['spec_n']         = df_nc['spec_n'].fillna(0)

    if _spec_disc is not None:
        df_nc = df_nc.merge(
            _spec_disc[['nom', 'discipline', 'spec_disc_rate']],
            on=['nom', 'discipline'], how='left')
    if 'spec_disc_rate' not in df_nc.columns:
        df_nc['spec_disc_rate'] = _fallback
    df_nc['spec_disc_rate'] = df_nc['spec_disc_rate'].fillna(_fallback)

    if _hist_snapshot is not None:
        df_nc = df_nc.merge(
            _hist_snapshot[['nom', 'hist_nb', 'hist_moy_classement', 'hist_nb_top3',
                            'hist_taux_top3', 'hist_moy_temps', 'hist_tendance', 'hist_moy_cote']],
            on='nom', how='left')
    for col in ['hist_nb', 'hist_nb_top3']:
        if col not in df_nc.columns:
            df_nc[col] = 0
        df_nc[col] = df_nc[col].fillna(0)
    for col in ['hist_moy_classement', 'hist_taux_top3', 'hist_moy_temps',
                'hist_tendance', 'hist_moy_cote']:
        if col not in df_nc.columns:
            df_nc[col] = np.nan

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

    df_nc['score_forme'] = (
        s_score_p  * 0.40 +
        s_derniere * 0.30 +
        s_podiums  * 0.20 +
        s_disq     * 0.10
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

    s_taux_top3 = _norm_mix(hist_taux,        0, 0.7)
    s_classement= _norm_mix(10 - hist_class, -5, 9)
    s_h_fiable  = hist_fiable

    df_nc['score_historique'] = (
        s_taux_top3  * 0.50 +
        s_classement * 0.35 +
        s_h_fiable   * 0.15
    ).clip(0, 1)

    # ── Score 4 : Gains / Palmarès carrière ───────────────────
    s_ratio_vic  = _norm_mix(df_nc['ratio_victoires'],  0, 0.4)
    s_gains_c    = _norm_mix(df_nc['gains_par_course'], 0, 8000)
    s_gains_ann  = _norm_mix(df_nc['gains_annee'],      0, 150000)
    s_ratio_gains= _norm_mix(df_nc['ratio_gains_rec'],  0, 0.5)

    df_nc['score_gains'] = (
        s_ratio_vic   * 0.35 +
        s_gains_c     * 0.30 +
        s_gains_ann   * 0.20 +
        s_ratio_gains * 0.15
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
    # Signal du marché PMU — présent mais non dominant
    s_cote_rang  = 1 - df_nc['rang_cote_norme']               # rang inversé : favori = 1
    s_ecart      = _norm(-df_nc['ecart_cotes'].abs(), -10, 0) # petite déviation live/ref = bon signe
    _med_temps   = df_nc['temps_norme'].median()
    _med_temps   = _med_temps if pd.notna(_med_temps) else 0.0
    s_temps      = _norm(1 / (1 + df_nc['temps_norme'].fillna(_med_temps)), 0, 1)

    df_nc['score_cote'] = (
        s_cote_rang * 0.60 +
        s_ecart     * 0.25 +
        s_temps     * 0.15
    ).clip(0, 1).fillna(s_cote_rang.clip(0, 1))  # fallback sur rang seul si NaN

    # ════════════════════════════════════════════════════════════
    # ÉTAPE 2 — XGBOOST sur les 6 scores équilibrés
    # Le modèle V5 existant reste utilisé pour compatibilité,
    # mais on injecte aussi les scores comme features supplémentaires
    # pour le réentraînement futur (V6).
    # ════════════════════════════════════════════════════════════

    # Prédiction V5 (conservé pour debug/comparaison)
    probas_v5 = _model_pmu.predict_proba(df_nc[_features_pmu])[:, 1]

    # V6 : note basée uniquement sur les 6 scores métier équilibrés
    SCORES_V6 = ['score_forme', 'score_duo', 'score_historique',
                 'score_gains', 'score_adequation', 'score_cote']

    score_metier = df_nc[SCORES_V6].mean(axis=1)  # moyenne équipondérée des 6

    df_nc['proba_pmu']    = score_metier
    df_nc['proba_pmu_v5'] = probas_v5
    df_nc['note_pmu']     = _proba_to_note_api(score_metier)

    # ── Plancher note par cote (inchangé) ─────────────────────
    def _plancher_cote(cote):
        if cote is None: return 1
        if cote < 3:     return 12
        elif cote < 5:   return 10
        elif cote < 8:   return 8
        elif cote < 15:  return 6
        elif cote < 25:  return 4
        else:            return 1

    df_nc['plancher'] = df_nc['_cote_app'].apply(_plancher_cote)
    df_nc['note_pmu'] = df_nc[['note_pmu', 'plancher']].max(axis=1)

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
        })

    return jsonify({
        "date":     date_str,
        "reunion":  r_num,
        "course":   c_num,
        "version":  "v6",
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
# DÉMARRAGE
# ============================================================
_charger_modele_pmu()
initialiser()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
