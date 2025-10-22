# streamlit run gw2tp.py
import base64, struct, json, sqlite3, time
import numpy as np
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# ---------- Constants ----------
TP_NET = 0.85
TIMEOUT = (5, 20)
GW2TP_BULK_ITEMS = "https://api.gw2tp.com/1/bulk/items.json"
GW2TP_BULK_NAMES = "https://api.gw2tp.com/1/bulk/items-names.json"
DB_PATH = "gw2tp_history.sqlite"
SNAPSHOT_BUCKET_SECONDS = 5 * 60  # un snapshot max toutes les 5 min / item

# ---------- Emoji pièces ----------
EMO_G = "🟡"  # or
EMO_S = "⚪"  # argent
EMO_C = "🟠"  # cuivre

# ---------- i18n (FR/EN/DE/ES) ----------
LANG_FLAGS = {"fr":"🇫🇷","en":"🇬🇧","de":"🇩🇪","es":"🇪🇸"}

I18N = {
    # En-têtes / titres / captions
    "🛠️ escarbeille.4281 · Discord : escarmouche": {
        "en": "🛠️ escarbeille.4281 · Discord: escarmouche",
        "de": "🛠️ escarbeille.4281 · Discord: escarmouche",
        "es": "🛠️ escarbeille.4281 · Discord: escarmouche",
    },
    "Flips Trading Post (source: GW2TP)": {
        "en": "Trading Post Flips (source: GW2TP)",
        "de": "Trading-Post-Flips (Quelle: GW2TP)",
        "es": "Flips del Trading Post (fuente: GW2TP)",
    },
    "Dernière mise à jour : ": {
        "en": "Last update: ",
        "de": "Letztes Update: ",
        "es": "Última actualización: ",
    },
    "Langue": {"en": "Language", "de": "Sprache", "es": "Idioma"},

    # Sidebar
    "Actualisation": {"en": "Refresh", "de": "Aktualisierung", "es": "Actualización"},
    "Auto-refresh": {"en": "Auto-refresh", "de": "Auto-Refresh", "es": "Autoactualización"},
    "Intervalle (minutes)": {"en": "Interval (minutes)", "de": "Intervall (Minuten)", "es": "Intervalo (minutos)"},
    "Un snapshot est enregistré à chaque refresh (5 min max par item).": {
        "en": "A snapshot is saved on each refresh (max once/5 min per item).",
        "de": "Bei jeder Aktualisierung wird ein Snapshot gespeichert (max alle 5 Min pro Item).",
        "es": "Se guarda una instantánea en cada actualización (máx. una cada 5 min por objeto).",
    },

    "Filtres - Profits": {"en": "Filters – Profit", "de": "Filter – Profit", "es": "Filtros – Beneficio"},
    "Profit net min (or)": {"en": "Min net profit (g)", "de": "Min Nettogewinn (g)", "es": "Beneficio neto mín (o)"},
    "Profit net max (or, 0 = illimité)": {"en": "Max net profit (g, 0 = unlimited)", "de": "Max. Nettogewinn (g, 0 = unbegrenzt)", "es": "Beneficio neto máx (o, 0 = ilimitado)"},
    "ROI min (%)": {"en": "Min ROI (%)", "de": "Min. ROI (%)", "es": "ROI mín (%)"},

    # --- Nettoyage / Outliers (nouveaux libellés) ---
    "Nettoyage / Outliers": {"en": "Cleaning / Outliers", "de": "Bereinigung / Ausreißer", "es": "Limpieza / Atípicos"},
    "Ignorer si prix d'achat < (cuivre)": {"en": "Ignore if buy price < (copper)", "de": "Ignoriere, wenn Kaufpreis < (Kupfer)", "es": "Ignorar si precio de compra < (cobre)"},
    "Exclure ROI négatifs": {"en": "Exclude negative ROI", "de": "Negative ROI ausschließen", "es": "Excluir ROI negativos"},
    "Seuil ROI max (%) (0 = auto)": {"en": "Max ROI (%) threshold (0 = auto)", "de": "Max. ROI (%) (0 = auto)", "es": "ROI máx (%) (0 = auto)"},
    "Méthode auto des outliers (si ROI max = 0)": {"en": "Auto outlier method (if ROI max = 0)", "de": "Autom. Ausreißermethode (wenn ROI max = 0)", "es": "Método auto de atípicos (si ROI máx = 0)"},
    "IQR (1.5×)": {"en": "IQR (1.5×)", "de": "IQR (1,5×)", "es": "IQR (1,5×)"},
    "Percentile 99.5%": {"en": "99.5th percentile", "de": "99,5. Perzentil", "es": "Percentil 99,5%"},

    "Volume": {"en": "Volume", "de": "Volumen", "es": "Volumen"},
    "Quantité min (min(demand, supply))": {"en": "Min quantity (min(demand, supply))", "de": "Mindestmenge (min(Nachfrage, Angebot))", "es": "Cantidad mín (min(demanda, oferta))"},
    "Quantité max (min(demand, supply), 0 = illimité)": {"en": "Max quantity (min(demand, supply), 0 = unlimited)", "de": "Max. Menge (min(Nachfrage, Angebot), 0 = unbegrenzt)", "es": "Cantidad máx (min(demanda, oferta), 0 = ilimitado)"},
    "Ventes min sur la période": {"en": "Min sales over period", "de": "Min. Verkäufe im Zeitraum", "es": "Ventas mín en el periodo"},

    "Prix (or)": {"en": "Price (gold)", "de": "Preis (Gold)", "es": "Precio (oro)"},
    "Prix d'achat minimum": {"en": "Min buy price", "de": "Min. Kaufpreis", "es": "Precio de compra mín"},
    "Prix d'achat maximum (0 = illimité)": {"en": "Max buy price (0 = unlimited)", "de": "Max. Kaufpreis (0 = unbegrenzt)", "es": "Precio de venta máx (0 = ilimitado)"},
    "Prix de vente min (net)": {"en": "Min sell price (net)", "de": "Min. Verkaufspreis (netto)", "es": "Precio de venta mín (neto)"},
    "Prix de vente max (net, 0 = illimité)": {"en": "Max sell price (net, 0 = unlimited)", "de": "Max. Verkaufspreis (netto, 0 = unbegrenzt)", "es": "Precio de venta máx (neto, 0 = ilimitado)"},

    "Nom / Cache": {"en": "Name / Cache", "de": "Name / Cache", "es": "Nombre / Caché"},
    "Recherche par nom (contient)": {"en": "Search by name (contains)", "de": "Suche nach Name (enthält)", "es": "Buscar por nombre (contiene)"},
    "Durée du cache (sec)": {"en": "Cache duration (sec)", "de": "Cache-Dauer (Sek.)", "es": "Duración de caché (s)"},
    "Stratégie": {"en": "Strategy", "de": "Strategie", "es": "Estrategia"},
    "Preset": {"en": "Preset", "de": "Voreinstellung", "es": "Preajuste"},
    "Personnalisé": {"en": "Custom", "de": "Benutzerdefiniert", "es": "Personalizado"},
    "Prudent": {"en": "Cautious", "de": "Vorsichtig", "es": "Prudente"},
    "Équilibré": {"en": "Balanced", "de": "Ausgewogen", "es": "Equilibrado"},
    "Agressif": {"en": "Aggressive", "de": "Aggressiv", "es": "Agresivo"},
    "Profil de risque (0 = prudent, 100 = agressif)": {"en": "Risk profile (0 = cautious, 100 = aggressive)", "de": "Risikoprofil (0 = vorsichtig, 100 = aggressiv)", "es": "Perfil de riesgo (0 = prudente, 100 = agresivo)"},

    "Historique en local": {"en": "Local history", "de": "Lokale Historie", "es": "Histórico local"},
    "Snapshots 5 min → estimations (période) + tendances.": {"en": "5-min snapshots → estimates (window) + trends.", "de": "5-Min-Snapshots → Schätzungen (Zeitraum) + Trends.", "es": "Instantáneas cada 5 min → estimaciones (periodo) + tendencias."},
    "Activer le suivi (SQLite local)": {"en": "Enable tracking (local SQLite)", "de": "Tracking aktivieren (lokales SQLite)", "es": "Activar seguimiento (SQLite local)"},
    "Durée d'analyse (heures)": {"en": "Analysis window (hours)", "de": "Analysezeitraum (Stunden)", "es": "Ventana de análisis (horas)"},
    "Fenêtre tendance ΔSupply/Demand (heures)": {"en": "Trend window ΔSupply/Demand (hours)", "de": "Trendfenster ΔAngebot/Nachfrage (Stunden)", "es": "Ventana de tendencia ΔOferta/Demanda (horas)"},
    "Échelle logarithmique (graphe Offre/Demande)": {"en": "Log scale (Supply/Demand chart)", "de": "Log-Skala (Angebot/Nachfrage-Graph)", "es": "Escala logarítmica (gráfico Oferta/Demanda)"},

    # Corps / tableaux
    "Aucun flip avec ces filtres.": {"en": "No flips with these filters.", "de": "Keine Flips mit diesen Filtern.", "es": "No hay flips con estos filtros."},
    "objets — période historique : ": {"en": "items — history window: ", "de": "Objekte — Historienzeitraum: ", "es": "objetos — ventana histórica: "},
    " | ΔSupply/Demand : ": {"en": " | ΔSupply/Demand: ", "de": " | ΔAngebot/Nachfrage: ", "es": " | ΔOferta/Demanda: "},

    # Colonnes
    "Nom": {"en": "Name", "de": "Name", "es": "Nombre"},
    "Profit Net (PO)": {"en": "Net Profit (g)", "de": "Nettogewinn (g)", "es": "Beneficio neto (o)"},
    "ROI (%)": {"en": "ROI (%)", "de": "ROI (%)", "es": "ROI (%)"},
    "Score": {"en": "Score", "de": "Score", "es": "Puntuación"},
    "Prix Achat (PO)": {"en": "Buy Price (g)", "de": "Kaufpreis (g)", "es": "Precio compra (o)"},
    "Prix Vente Net (PO)": {"en": "Net Sell Price (g)", "de": "Netto-Verkaufspreis (g)", "es": "Precio venta neto (o)"},
    "Quantité (min)": {"en": "Quantity (min)", "de": "Menge (min)", "es": "Cantidad (mín)"},
    "Supply": {"en": "Supply", "de": "Angebot", "es": "Oferta"},
    "Demand": {"en": "Demand", "de": "Nachfrage", "es": "Demanda"},
    "ID": {"en": "ID", "de": "ID", "es": "ID"},
    "ChatCode": {"en": "ChatCode", "de": "ChatCode", "es": "ChatCode"},

    # Section prix g/s/c
    "Affichage prix (g/s/c)": {"en": "Price display (g/s/c)", "de": "Preisanzeige (g/s/c)", "es": "Vista de precios (g/s/c)"},
    "Lignes à afficher (prix g/s/c)": {"en": "Rows to show (g/s/c prices)", "de": "Zeilen anzeigen (g/s/c-Preise)", "es": "Filas a mostrar (precios g/s/c)"},
    "Profit Net": {"en": "Net Profit", "de": "Nettogewinn", "es": "Beneficio neto"},
    "Prix Achat": {"en": "Buy Price", "de": "Kaufpreis", "es": "Precio de compra"},
    "Vente nette (85%)": {"en": "Net sell (85%)", "de": "Nettoverkauf (85%)", "es": "Venta neta (85%)"},
    "Copier": {"en": "Copy", "de": "Kopieren", "es": "Copiar"},

    # Graphes / labels
    "Métrique du graphique": {"en": "Chart metric", "de": "Diagrammmetrik", "es": "Métrica del gráfico"},
    "Score (profil de risque)": {"en": "Score (risk profile)", "de": "Score (Risikoprofil)", "es": "Puntuación (perfil de riesgo)"},
    "Top 20 par Profit Net": {"en": "Top 20 by Net Profit", "de": "Top 20 nach Nettogewinn", "es": "Top 20 por beneficio neto"},
    "Top 20 (GW2TP) — Profit Net": {"en": "Top 20 (GW2TP) — Net Profit", "de": "Top 20 (GW2TP) — Nettogewinn", "es": "Top 20 (GW2TP) — Beneficio neto"},
    "Top 20 par Score (profil de risque)": {"en": "Top 20 by Score (risk profile)", "de": "Top 20 nach Score (Risikoprofil)", "es": "Top 20 por puntuación (perfil de riesgo)"},
    "Top 20 (GW2TP) — Score": {"en": "Top 20 (GW2TP) — Score", "de": "Top 20 (GW2TP) — Score", "es": "Top 20 (GW2TP) — Puntuación"},

    # Historique / courbes
    "Évolution Offre / Demande sur la période": {"en": "Supply / Demand over the period", "de": "Angebot / Nachfrage über den Zeitraum", "es": "Oferta / Demanda en el periodo"},
    "Aucun item à tracer.": {"en": "No item to plot.", "de": "Kein Objekt zu zeichnen.", "es": "No hay objeto para trazar."},
    "Choisir un objet": {"en": "Choose an item", "de": "Objekt auswählen", "es": "Elige un objeto"},
    "Pas encore assez d'historique pour tracer (laisse l'app tourner).": {"en": "Not enough history to plot yet (leave the app running).", "de": "Noch nicht genug Historie zum Plotten (App weiterlaufen lassen).", "es": "Aún no hay historial suficiente para graficar (deja la app ejecutándose)."},
    "Offre / Demande — ": {"en": "Supply / Demand — ", "de": "Angebot / Nachfrage — ", "es": "Oferta / Demanda — "},
    "Temps": {"en": "Time", "de": "Zeit", "es": "Tiempo"},
    "Quantités": {"en": "Quantities", "de": "Mengen", "es": "Cantidades"},
    "Évolution des prix (ultra clean)": {"en": "Price evolution (ultra clean)", "de": "Preisentwicklung (ultra clean)", "es": "Evolución de precios (ultra clean)"},
    "Achat (PO)": {"en": "Buy (g)", "de": "Kauf (g)", "es": "Compra (o)"},
    "Vente nette 85% (PO)": {"en": "Net sell 85% (g)", "de": "Nettoverkauf 85% (g)", "es": "Venta neta 85% (o)"},
    "Prix — ": {"en": "Price — ", "de": "Preis — ", "es": "Precio — "},
    "Prix (g)": {"en": "Price (g)", "de": "Preis (g)", "es": "Precio (o)"},

    # Divers
    "Acheté période": {"en": "Bought in period", "de": "Gekauft im Zeitraum", "es": "Comprado en el periodo"},
    "Vendu période": {"en": "Sold in period", "de": "Verkauft im Zeitraum", "es": "Vendido en el periodo"},
    "ΔSupply": {"en": "ΔSupply", "de": "ΔAngebot", "es": "ΔOferta"},
    "ΔDemand": {"en": "ΔDemand", "de": "ΔNachfrage", "es": "ΔDemanda"},

    # Erreurs
    "Erreur enregistrement historique : ": {"en": "History save error: ", "de": "Fehler beim Speichern der Historie: ", "es": "Error al guardar el histórico: "},

    # CSV
    "Télécharger CSV (résultats filtrés)": {"en": "Download CSV (filtered results)", "de": "CSV herunterladen (gefilterte Ergebnisse)", "es": "Descargar CSV (resultados filtrados)"},
    # >>> NEW <<<
    "Télécharger CSV — Liste de courses optimisée (max 1000 unités)": {
        "en": "Download CSV — Optimized shopping list (max 1000 units)",
        "de": "CSV herunterladen — Optimierte Einkaufsliste (max. 1000 Einheiten)",
        "es": "Descargar CSV — Lista de compras optimizada (máx. 1000 unidades)",
    },

    # -------- Ajouts OPTI --------
    "Optimisation d'achat": {"en":"Buy optimization", "de":"Kaufoptimierung", "es":"Optimización de compra"},
    "Budget (or, 0 = illimité)": {"en":"Budget (g, 0 = unlimited)", "de":"Budget (g, 0 = unbegrenzt)", "es":"Presupuesto (o, 0 = ilimitado)"},
    "Horizon de revente (heures)": {"en":"Resale horizon (hours)", "de":"Wiederverkaufshorizont (Stunden)", "es":"Horizonte de reventa (horas)"},
    "Marge de sécurité (%)": {"en":"Safety margin (%)", "de":"Sicherheitsmarge (%)", "es":"Margen de seguridad (%)"},
    "Qté optimisée": {"en":"Opt. qty", "de":"Optim. Menge", "es":"Cant. óptima"},
    "Profit net optimisé (PO)": {"en":"Optimized net profit (g)", "de":"Optimierter Nettogewinn (g)", "es":"Beneficio neto optimizado (o)"},
    "Profit net optimisé": {"en":"Optimized net profit", "de":"Optimierter Nettogewinn", "es":"Beneficio neto optimizado"},
}

def T(fr_text: str) -> str:
    lang = st.session_state.get("lang", "fr")
    if lang == "fr":
        return fr_text
    trans = I18N.get(fr_text, {})
    return trans.get(lang, fr_text)

def language_switcher():
    st.markdown("### " + T("Langue"))
    cols = st.columns(len(LANG_FLAGS))
    for i, (code, flag) in enumerate(LANG_FLAGS.items()):
        if cols[i].button(flag, use_container_width=True):
            st.session_state["lang"] = code

# Valeurs par défaut langue AVANT set_page_config
if "lang" not in st.session_state:
    st.session_state["lang"] = "fr"

# ---------- HTTP session ----------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "User-Agent": "GW2TP-Flips/1.0"
    })
    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=(429,500,502,503,504), allowed_methods=frozenset(["GET"]))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()

# ---------- DB ----------
def db_connect():
    conn = sqlite3.connect(DB_PATH, isolation_level=None, timeout=30)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER NOT NULL,
        ts INTEGER NOT NULL,
        bucket INTEGER NOT NULL,
        buy INTEGER, sell INTEGER, supply INTEGER, demand INTEGER,
        PRIMARY KEY (id, bucket)
    )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(ts)")
    return conn

def persist_snapshot(df_bulk: pd.DataFrame):
    if df_bulk.empty:
        return
    now = int(time.time())
    bucket = now // SNAPSHOT_BUCKET_SECONDS
    rows = [(int(r.id), now, bucket, int(r.buy), int(r.sell),
             int(r.supply), int(r.demand)) for r in df_bulk.itertuples()]
    with db_connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO snapshots (id, ts, bucket, buy, sell, supply, demand) VALUES (?,?,?,?,?,?,?)",
            rows
        )

def fetch_metrics_for_ids(ids, hours_window=24, trend_hours=1):
    if not ids:
        return {}, {}, {}, {}
    with db_connect() as conn:
        now = int(time.time())
        since_window = now - hours_window * 3600
        q_marks = ",".join("?" for _ in ids)
        cur = conn.execute(
            f"SELECT id, ts, buy, sell, supply, demand FROM snapshots "
            f"WHERE id IN ({q_marks}) AND ts >= ? ORDER BY id, ts",
            (*ids, since_window)
        )
        rows = cur.fetchall()

    data = {}
    for id_, ts, buy, sell, supply, demand in rows:
        data.setdefault(id_, []).append((ts, buy, sell, supply, demand))

    sold, bought, dSup, dDem = {}, {}, {}, {}
    now = int(time.time())
    since_trend = now - trend_hours * 3600
    for id_, arr in data.items():
        df = pd.DataFrame(arr, columns=["ts","buy","sell","supply","demand"]).sort_values("ts")
        dsup = df["supply"].diff()
        ddem = df["demand"].diff()
        sold[id_]   = int(np.maximum(-dsup, 0).sum(skipna=True)) if len(df)>1 else 0
        bought[id_] = int(np.maximum(-ddem, 0).sum(skipna=True)) if len(df)>1 else 0
        idx = df["ts"].searchsorted(since_trend, side="left")
        sup_now = int(df["supply"].iloc[-1]); dem_now = int(df["demand"].iloc[-1])
        if idx < len(df):
            sup_t = int(df["supply"].iloc[idx]); dem_t = int(df["demand"].iloc[idx])
        else:
            sup_t = int(df["supply"].iloc[0]); dem_t = int(df["demand"].iloc[0])
        dSup[id_] = sup_now - sup_t
        dDem[id_] = dem_now - dem_t
    return sold, bought, dSup, dDem

def fetch_timeseries_for_id(item_id: int, hours_window: int) -> pd.DataFrame:
    with db_connect() as conn:
        since = int(time.time()) - hours_window * 3600
        cur = conn.execute(
            "SELECT ts, buy, sell, supply, demand FROM snapshots WHERE id = ? AND ts >= ? ORDER BY ts",
            (int(item_id), since)
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts","buy","sell","supply","demand"])
    df = pd.DataFrame(rows, columns=["ts","buy","sell","supply","demand"])
    df["dt"] = pd.to_datetime(df["ts"], unit="s")
    return df

# ---------- Page config ----------
st.set_page_config(page_title=T("Flips Trading Post (source: GW2TP)"), layout="wide")
st.caption(T("🛠️ escarbeille.4281 · Discord : escarmouche"))
st.title(T("Flips Trading Post (source: GW2TP)"))

with st.sidebar:
    # Sélecteur de langue
    language_switcher()
    st.divider()

    st.header(T("Actualisation"))
    auto_refresh = st.checkbox(T("Auto-refresh"), value=True)
    refresh_min = st.slider(T("Intervalle (minutes)"), 1, 30, 5)
    st.caption(T("Un snapshot est enregistré à chaque refresh (5 min max par item)."))

    # --- Budget (placé sous "Actualisation") ---
    st.header(T("Optimisation d'achat"))
    budget_gold = st.number_input(T("Budget (or, 0 = illimité)"), 0.0, 1e9, 0.0, 0.5, key="budget_top")
    st.divider()

    st.header(T("Filtres - Profits"))
    min_profit = st.number_input(T("Profit net min (or)"), 0.0, 1e6, 1.0, 0.5)
    max_profit = st.number_input(T("Profit net max (or, 0 = illimité)"), 0.0, 1e6, 0.0, 0.5)
    min_roi = st.number_input(T("ROI min (%)"), 0.0, 1000.0, 10.0, 1.0)

    # --- Nettoyage / Outliers ---
    st.header(T("Nettoyage / Outliers"))
    min_buy_copper_filter = st.number_input(
        T("Ignorer si prix d'achat < (cuivre)"), 0, 10_000, 10, 1
    )
    exclude_neg_roi = st.checkbox(T("Exclure ROI négatifs"), value=False)
    roi_cap_manual = st.number_input(T("Seuil ROI max (%) (0 = auto)"), 0.0, 1_000_000.0, 0.0, 10.0)

    outlier_modes_fr = ["IQR (1.5×)", "Percentile 99.5%"]
    outlier_modes_display = [T(x) for x in outlier_modes_fr]
    outlier_choice_display = st.selectbox(
        T("Méthode auto des outliers (si ROI max = 0)"),
        outlier_modes_display, index=0
    )
    outlier_choice_fr = outlier_modes_fr[outlier_modes_display.index(outlier_choice_display)]

    st.header(T("Volume"))
    min_quantity_user = st.number_input(T("Quantité min (min(demand, supply))"), 0, 10_000_000, 10, 5)
    max_quantity_user = st.number_input(T("Quantité max (min(demand, supply), 0 = illimité)"), 0, 10_000_000, 0, 5)
    min_sold = st.number_input(T("Ventes min sur la période"), 0, 10_000_000, 0, 100)

    st.header(T("Prix (or)"))
    min_buy = st.number_input(T("Prix d'achat minimum"), 0.0, 1e9, 0.0, 0.5)
    max_buy = st.number_input(T("Prix d'achat maximum (0 = illimité)"), 0.0, 1e9, 0.0, 0.5)
    min_sell = st.number_input(T("Prix de vente min (net)"), 0.0, 1e9, 0.0, 0.5)
    max_sell = st.number_input(T("Prix de vente max (net, 0 = illimité)"), 0.0, 1e9, 0.0, 0.5)

    st.header(T("Nom / Cache"))
    name_query = st.text_input(T("Recherche par nom (contient)"), "")
    cache_ttl = st.slider(T("Durée du cache (sec)"), 60, 1800, 300, 30)

    st.header(T("Stratégie"))
    strategy_options_fr = ["Personnalisé", "Prudent", "Équilibré", "Agressif"]
    strategy_options_display = [T(x) for x in strategy_options_fr]
    strategy_choice_display = st.selectbox(T("Preset"), strategy_options_display, 0)
    strategy_choice_fr = strategy_options_fr[strategy_options_display.index(strategy_choice_display)]
    risk_slider = st.slider(T("Profil de risque (0 = prudent, 100 = agressif)"), 0, 100, 40)

    st.header(T("Historique en local"))
    st.caption(T("Snapshots 5 min → estimations (période) + tendances."))
    show_history = st.checkbox(T("Activer le suivi (SQLite local)"), value=False)
    hist_hours = st.slider(T("Durée d'analyse (heures)"), 1, 168, 24, 1)
    trend_hours = st.slider(T("Fenêtre tendance ΔSupply/Demand (heures)"), 1, 48, 1, 1)
    log_scale_od = st.checkbox(T("Échelle logarithmique (graphe Offre/Demande)"), value=False)

    # --------- OPTI : contrôles (sans budget ici) ---------
    st.subheader(T("Optimisation d'achat"))
    horizon_sell_h = st.slider(T("Horizon de revente (heures)"), 1, 168, 24, 1)
    safety_pct = st.slider(T("Marge de sécurité (%)"), 10, 100, 60, 5)

# Auto-refresh
if auto_refresh:
    st_autorefresh(interval=refresh_min * 60 * 1000, key="auto_refresh_key")

st.caption(T("Dernière mise à jour : ") + time.strftime("%Y-%m-%d %H:%M:%S"))
PRESET_TO_RISK = {"Prudent": 20, "Équilibré": 50, "Agressif": 80}
risk_level = PRESET_TO_RISK.get(strategy_choice_fr, risk_slider)

# ---------- Conversions ----------
def gsc_emoji_from_copper(copper_value: int) -> str:
    sign = "-" if copper_value < 0 else ""
    c = abs(int(round(copper_value)))
    g = c // 10000; s = (c % 10000) // 100; cc = c % 100
    parts = []
    if g > 0: parts.append(f"{g}{EMO_G}")
    if s > 0 or g > 0: parts.append(f"{s}{EMO_S}")
    parts.append(f"{cc}{EMO_C}")
    return sign + " ".join(parts)

def make_session() -> requests.Session:
    return SESSION

def make_item_chat_code(item_id: int, qty: int = 1) -> str:
    q = max(1, min(255, int(qty)))
    data = bytearray([0x02, q]) + struct.pack("<I", int(item_id)) + bytearray([0x00])
    return "[&" + base64.b64encode(bytes(data)).decode("ascii") + "]"

# ---------- Fetchers ----------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_bulk_items():
    for url in (GW2TP_BULK_ITEMS, GW2TP_BULK_ITEMS.replace("https://", "http://")):
        resp = SESSION.get(url, timeout=TIMEOUT)
        if resp.ok:
            obj = resp.json()
            return pd.DataFrame(obj.get("items", []), columns=obj.get("columns", []))
    return pd.DataFrame(columns=["id","buy","sell","supply","demand"])

@st.cache_data(ttl=300, show_spinner=False)
def fetch_item_names():
    for url in (GW2TP_BULK_NAMES, GW2TP_BULK_NAMES.replace("https://", "http://")):
        resp = SESSION.get(url, timeout=TIMEOUT)
        if resp.ok:
            items = resp.json().get("items", [])
            return {int(i[0]): i[1] for i in items if len(i) >= 2}
    return {}

# ---------- Build dataset ----------
def build_flips_df(df_bulk: pd.DataFrame, id2name: dict):
    if df_bulk.empty:
        return pd.DataFrame(columns=[
            "Nom","ID","Prix Achat (PO)","Prix Vente Net (PO)","Profit Net (PO)","ROI (%)",
            "Quantité (min)","Supply","Demand","ChatCode",
            "Prix Achat (c)","Prix Vente Net (c)","Profit Net (c)",
            "Prix Achat (gsc)","Prix Vente Net (gsc)","Profit Net (gsc)"
        ])
    df = df_bulk.copy()
    df["Nom"] = df["id"].map(id2name).fillna("")
    df["Prix Achat (c)"] = df["buy"].astype("Int64")
    df["Prix Vente Net (c)"] = (df["sell"] * TP_NET).round().astype("Int64")
    df["Profit Net (c)"] = (df["Prix Vente Net (c)"] - df["Prix Achat (c)"]).astype("Int64")
    df["Prix Achat (gsc)"] = df["Prix Achat (c)"].fillna(0).astype(int).apply(gsc_emoji_from_copper)
    df["Prix Vente Net (gsc)"] = df["Prix Vente Net (c)"].fillna(0).astype(int).apply(gsc_emoji_from_copper)
    df["Profit Net (gsc)"] = df["Profit Net (c)"].fillna(0).astype(int).apply(gsc_emoji_from_copper)
    df["Prix Achat (PO)"] = (df["Prix Achat (c)"] / 10000.0).round(2)
    df["Prix Vente Net (PO)"] = (df["Prix Vente Net (c)"] / 10000.0).round(2)
    df["Profit Net (PO)"] = (df["Profit Net (c)"] / 10000.0).round(2)

    # --- ROI robuste ---
    buy_po = df["Prix Achat (PO)"].replace(0, np.nan)
    df["ROI (%)"] = ((df["Profit Net (PO)"] / buy_po) * 100) \
        .replace([np.inf, -np.inf], np.nan).fillna(0).round(2)

    df["Quantité (min)"] = df[["supply","demand"]].min(axis=1).fillna(0).astype(int)
    df["ChatCode"] = df["id"].apply(lambda x: make_item_chat_code(int(x)))
    df.rename(columns={"id":"ID","supply":"Supply","demand":"Demand"}, inplace=True)
    return df[[
        "Nom","ID","Prix Achat (PO)","Prix Vente Net (PO)","Profit Net (PO)","ROI (%)",
        "Quantité (min)","Supply","Demand","ChatCode",
        "Prix Achat (c)","Prix Vente Net (c)","Profit Net (c)",
        "Prix Achat (gsc)","Prix Vente Net (gsc)","Profit Net (gsc)"
    ]]

@st.cache_data(ttl=300, show_spinner=False)
def get_bulk_and_df():
    bulk = fetch_bulk_items()
    names = fetch_item_names()
    return bulk, build_flips_df(bulk, names)

# ---------- Risk ----------
def add_risk_score(df, risk_level):
    if df.empty: return df
    w_roi = risk_level / 100.0; w_vol = 1 - w_roi
    roi = df["ROI (%)"].clip(lower=0).fillna(0)
    vol = np.log10(df["Quantité (min)"].fillna(0) + 1)
    def norm(s): return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s*0
    df["Score"] = (w_roi * norm(roi) + w_vol * norm(vol)).round(3)
    return df

# ============================= RUN =============================
bulk, df_all = get_bulk_and_df()
df_all_raw = df_all.copy()  # >>> NEW <<< base non filtrée pour la liste de courses

# Enregistrement snapshot si historique activé
if show_history:
    try: persist_snapshot(bulk)
    except Exception as e: st.warning(T("Erreur enregistrement historique : ") + str(e))

# Filtrage principal
mask = (df_all["Profit Net (PO)"] >= min_profit) & (df_all["ROI (%)"] >= min_roi)
if max_profit > 0: mask &= df_all["Profit Net (PO)"] <= max_profit
mask &= df_all["Quantité (min)"] >= min_quantity_user
if max_quantity_user > 0: mask &= df_all["Quantité (min)"] <= max_quantity_user
if min_buy > 0: mask &= df_all["Prix Achat (PO)"] >= min_buy
if max_buy > 0: mask &= df_all["Prix Achat (PO)"] <= max_buy
if min_sell > 0: mask &= df_all["Prix Vente Net (PO)"] >= min_sell
if max_sell > 0: mask &= df_all["Prix Vente Net (PO)"] <= max_sell
if name_query: mask &= df_all["Nom"].str.lower().str.contains(name_query.lower(), na=False)
df_all = df_all[mask].reset_index(drop=True)

# --- Nettoyage / Outliers (après filtrage de base, avant score/tri) ---
if not df_all.empty:
    df_all = df_all[df_all["Prix Achat (c)"].fillna(0) >= int(min_buy_copper_filter)]
    df_all = df_all[df_all["Prix Vente Net (c)"] > df_all["Prix Achat (c)"]]
    if exclude_neg_roi:
        df_all = df_all[df_all["ROI (%)"] >= 0]
    if roi_cap_manual > 0:
        df_all = df_all[df_all["ROI (%)"] <= roi_cap_manual]
    else:
        roi_series = df_all["ROI (%)"].astype(float)
        q1, q3 = roi_series.quantile(0.25), roi_series.quantile(0.75)
        iqr = (q3 - q1) if pd.notna(q3) and pd.notna(q1) else 0.0
        if iqr > 0:
            max_auto = q3 + 1.5 * iqr
            df_all = df_all[roi_series <= max_auto]
        else:
            p995 = roi_series.quantile(0.995)
            if pd.notna(p995):
                df_all = df_all[roi_series <= p995]

# Historique (ventes cumulées & tendances)
if show_history and not df_all.empty:
    ids = df_all["ID"].tolist()
    sold, bought, dSup, dDem = fetch_metrics_for_ids(ids, hist_hours, trend_hours)
    df_all["Vendu période"] = df_all["ID"].map(lambda i: sold.get(i, 0)).astype(int)
    df_all["Acheté période"] = df_all["ID"].map(lambda i: bought.get(i, 0)).astype(int)
    df_all["ΔSupply"] = df_all["ID"].map(lambda i: dSup.get(i, 0)).astype(int)
    df_all["ΔDemand"] = df_all["ID"].map(lambda i: dDem.get(i, 0)).astype(int)
    if min_sold > 0:
        df_all = df_all[df_all["Vendu période"] >= min_sold].reset_index(drop=True)

# Score & tri
df_all = add_risk_score(df_all, risk_level)
df_all = df_all.sort_values(["Score","Profit Net (PO)"], ascending=[False,False])

# --------- OPTI : quantité/profit optimisés ---------
def compute_optimal_qty(row, budget_gold, horizon_sell_h, safety_pct, show_history_flag):
    buy_c = int(row["Prix Achat (c)"]) if pd.notna(row["Prix Achat (c)"]) else 0
    profit_c = int(row["Profit Net (c)"]) if pd.notna(row["Profit Net (c)"]) else 0
    if buy_c <= 0 or profit_c <= 0:
        return 0, 0

    supply = int(row["Supply"]) if pd.notna(row["Supply"]) else 0
    demand = int(row["Demand"]) if pd.notna(row["Demand"]) else 0

    if budget_gold and budget_gold > 0:
        budget_c = int(round(budget_gold * 10000))
        cap_budget = budget_c // buy_c
    else:
        cap_budget = 10**9
    cap_budget = max(0, cap_budget)

    if show_history_flag and "Vendu période" in row and hist_hours > 0:
        sold_rate_per_h = max(0.0, float(row["Vendu période"])) / float(hist_hours)
        sell_capacity = int(sold_rate_per_h * float(horizon_sell_h))
    else:
        sell_capacity = int(demand)

    sell_capacity = int(sell_capacity * (max(10, min(100, safety_pct)) / 100.0))

    qty = max(0, min(supply, sell_capacity, cap_budget))
    total_profit_c = qty * profit_c
    return qty, total_profit_c

if not df_all.empty:
    qopts, popts_c = [], []
    for _, r in df_all.iterrows():
        q, p_c = compute_optimal_qty(r, budget_gold, horizon_sell_h, safety_pct, show_history)
        qopts.append(int(q)); popts_c.append(int(p_c))
    df_all["Qté optimisée"] = qopts
    df_all["Profit net optimisé (c)"] = popts_c
    df_all["Profit net optimisé (PO)"] = (df_all["Profit net optimisé (c)"] / 10000.0).round(2)
    df_all["Profit net optimisé (gsc)"] = df_all["Profit net optimisé (c)"].fillna(0).astype(int).apply(gsc_emoji_from_copper)

# >>> NEW <<< -----------------------------------------------------------------
# Liste de courses optimisée indépendante des paramètres UI (max_units=1000)
def generate_optimal_shopping_df(df_source: pd.DataFrame, max_units: int = 1000) -> pd.DataFrame:
    if df_source is None or df_source.empty:
        return pd.DataFrame(columns=[
            "Nom","ID","Qté à acheter","Profit unitaire (PO)","Profit total (PO)",
            "ROI (%)","Prix Achat (PO)","Prix Vente Net (PO)","Supply","Demand","ChatCode"
        ])
    df = df_source.copy()

    # Nettoyage minimal et outliers auto (indépendant des contrôles UI)
    df = df[df["Prix Achat (c)"].fillna(0) >= 10]
    df = df[df["Prix Vente Net (c)"] > df["Prix Achat (c)"]]
    df = df[df["Profit Net (c)"] > 0]
    roi_series = df["ROI (%)"].astype(float)
    q1, q3 = roi_series.quantile(0.25), roi_series.quantile(0.75)
    iqr = (q3 - q1) if pd.notna(q3) and pd.notna(q1) else 0.0
    if iqr > 0:
        max_auto = q3 + 1.5 * iqr
        df = df[roi_series <= max_auto]
    else:
        p995 = roi_series.quantile(0.995)
        if pd.notna(p995):
            df = df[roi_series <= p995]

    # Capacité "immédiate" = min(supply, demand)
    df["q_avail"] = df[["Supply","Demand"]].min(axis=1).fillna(0).astype(int)
    df = df[df["q_avail"] > 0]

    # Tri par profit unitaire décroissant (contrainte = nb d'unités total)
    df = df.sort_values("Profit Net (c)", ascending=False)

    units_left = int(max(0, max_units))
    picks = []
    for _, r in df.iterrows():
        if units_left <= 0:
            break
        qty = int(min(int(r["q_avail"]), units_left))
        if qty <= 0:
            continue
        profit_unit_po = float(r["Profit Net (PO)"]) if pd.notna(r["Profit Net (PO)"]) else 0.0
        total_profit_po = round(qty * profit_unit_po, 2)
        picks.append({
            "Nom": r["Nom"],
            "ID": int(r["ID"]),
            "Qté à acheter": qty,
            "Profit unitaire (PO)": round(profit_unit_po, 2),
            "Profit total (PO)": total_profit_po,
            "ROI (%)": float(r["ROI (%)"]),
            "Prix Achat (PO)": float(r["Prix Achat (PO)"]),
            "Prix Vente Net (PO)": float(r["Prix Vente Net (PO)"]),
            "Supply": int(r["Supply"]),
            "Demand": int(r["Demand"]),
            "ChatCode": make_item_chat_code(int(r["ID"]), min(qty, 255)),
        })
        units_left -= qty

    cols = ["Nom","ID","Qté à acheter","Profit unitaire (PO)","Profit total (PO)",
            "ROI (%)","Prix Achat (PO)","Prix Vente Net (PO)","Supply","Demand","ChatCode"]
    return pd.DataFrame(picks, columns=cols)

# Génération immédiate + bouton de téléchargement (toujours visible)
opt_df_1000 = generate_optimal_shopping_df(df_all_raw, max_units=1000)
st.download_button(
    T("Télécharger CSV — Liste de courses optimisée (max 1000 unités)"),
    data=opt_df_1000.to_csv(index=False),
    file_name="shopping_list_opt_1000.csv",
    mime="text/csv",
    key="download_csv_opt_1000",
    disabled=opt_df_1000.empty
)
# ---------------------------------------------------------------------------

# ---------- Display ----------
if df_all.empty:
    st.warning(T("Aucun flip avec ces filtres."))
else:
    st.caption(f"{len(df_all):,} {T('objets — période historique : ')}{hist_hours} h{T(' | ΔSupply/Demand : ')}{trend_hours} h")

    # ----- Table lisible g/s/c avec copier + scroll -----
    st.subheader(T("Affichage prix (g/s/c)"))
    if not df_all.empty:
        max_rows_emo = st.slider(T("Lignes à afficher (prix g/s/c)"), 10, 500, 100, 10)
        view_gsc = df_all.head(max_rows_emo)[[
            "Nom","Profit Net (gsc)","ROI (%)","Prix Achat (gsc)","Prix Vente Net (gsc)",
            "Quantité (min)","Supply","Demand","Qté optimisée","Profit net optimisé (gsc)","ID","ChatCode"
        ]].rename(columns={
            "Nom": T("Nom"),
            "Profit Net (gsc)": T("Profit Net"),
            "ROI (%)": T("ROI (%)"),
            "Prix Achat (gsc)": T("Prix Achat"),
            "Prix Vente Net (gsc)": T("Vente nette (85%)"),
            "Quantité (min)": T("Quantité (min)"),
            "Supply": T("Supply"),
            "Demand": T("Demand"),
            "Qté optimisée": T("Qté optimisée"),
            "Profit net optimisé (gsc)": T("Profit net optimisé"),
            "ID": T("ID"),
            "ChatCode": T("ChatCode"),
        })

        view_raw = df_all.head(max_rows_emo)[[
            "Profit Net (c)","Prix Achat (c)","Prix Vente Net (c)","ROI (%)",
            "Quantité (min)","Supply","Demand","Qté optimisée","Profit net optimisé (c)","ID"
        ]]

        records = []
        for disp_row, raw_row in zip(view_gsc.to_dict("records"), view_raw.to_dict("records")):
            rec = dict(disp_row)
            rec["_profit_c"] = int(raw_row["Profit Net (c)"]) if pd.notna(raw_row["Profit Net (c)"]) else 0
            rec["_buy_c"]    = int(raw_row["Prix Achat (c)"])   if pd.notna(raw_row["Prix Achat (c)"])   else 0
            rec["_sell_c"]   = int(raw_row["Prix Vente Net (c)"]) if pd.notna(raw_row["Prix Vente Net (c)"]) else 0
            rec["_roi"]      = float(raw_row["ROI (%)"]) if pd.notna(raw_row["ROI (%)"]) else 0.0
            rec["_qmin"]     = int(raw_row["Quantité (min)"]) if pd.notna(raw_row["Quantité (min)"]) else 0
            rec["_supply"]   = int(raw_row["Supply"]) if pd.notna(raw_row["Supply"]) else 0
            rec["_demand"]   = int(raw_row["Demand"]) if pd.notna(raw_row["Demand"]) else 0
            rec["_qopt"]     = int(raw_row["Qté optimisée"]) if pd.notna(raw_row["Qté optimisée"]) else 0
            rec["_popt_c"]   = int(raw_row["Profit net optimisé (c)"]) if pd.notna(raw_row["Profit net optimisé (c)"]) else 0
            rec["_id"]       = int(raw_row["ID"]) if pd.notna(raw_row["ID"]) else 0
            records.append(rec)

        items_json2 = json.dumps(records, ensure_ascii=True)

        keys = {
            "NAME": T("Nom"),
            "PROFIT": T("Profit Net"),
            "ROI": T("ROI (%)"),
            "BUY": T("Prix Achat"),
            "SELL": T("Vente nette (85%)"),
            "QMIN": T("Quantité (min)"),
            "SUP": T("Supply"),
            "DEM": T("Demand"),
            "QOPT": T("Qté optimisée"),
            "POPT": T("Profit net optimisé"),
            "ID": T("ID"),
            "CODE": T("ChatCode"),
            "COPY": T("Copier"),
        }
        keys_json = json.dumps(keys, ensure_ascii=False)

        components.html(f'''
        <style>
          .gsc-frame {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; border:1px solid #e5e7eb;border-radius:10px;overflow:hidden; }}
          .gsc-scroller {{ max-height: 620px; overflow-y: auto; }}
          .gsc-head, .gsc-row {{
            display:grid;
            grid-template-columns: 1.5fr 0.9fr 0.7fr 1.0fr 1.0fr 0.9fr 0.8fr 0.8fr 0.9fr 1.1fr 0.7fr 1.2fr 0.8fr;
            gap:8px; align-items:center;
          }}
          .gsc-head {{ position: sticky; top: 0; z-index: 5; background:#f8fafc; padding:10px 12px; font-weight:600; border-bottom:1px solid #eef2f7 }}
          .gsc-head button {{ all:unset; cursor:pointer; display:flex; align-items:center; gap:6px; }}
          .gsc-head .arrow {{ font-size:12px; color:#6b7280 }}
          .gsc-row {{ padding:10px 12px; border-bottom:1px dashed #eef2f7; background:#fff }}
          .gsc-row:nth-child(odd) {{ background:#fcfcfd }}
          .gsc-row:last-child {{ border-bottom:none }}
          .gsc-code {{ cursor:pointer;background:#f3f4f6;padding:3px 6px;border-radius:6px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis }}
          .gsc-code.ok {{ background:#dcfce7 }}
          .gsc-btn {{ padding:6px 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff;cursor:pointer;white-space:nowrap }}
          .gsc-btn:hover {{ background:#f9fafb }}
          .gsc-muted {{ color:#6b7280 }}
          .gsc-title {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis }}
        </style>

        <div class="gsc-frame">
          <div class="gsc-scroller" id="gsc-scroll">
            <div class="gsc-head" id="gsc-head">
              <button data-key="name"><span>{T("Nom")}</span><span class="arrow" id="arr-name"></span></button>
              <button data-key="profit"><span>{T("Profit Net")}</span><span class="arrow" id="arr-profit"></span></button>
              <button data-key="roi"><span>{T("ROI (%)")}</span><span class="arrow" id="arr-roi"></span></button>
              <button data-key="buy"><span>{T("Prix Achat")}</span><span class="arrow" id="arr-buy"></span></button>
              <button data-key="sell"><span>{T("Vente nette (85%)")}</span><span class="arrow" id="arr-sell"></span></button>
              <button data-key="qmin"><span>{T("Quantité (min)")}</span><span class="arrow" id="arr-qmin"></span></button>
              <button data-key="supply"><span>{T("Supply")}</span><span class="arrow" id="arr-supply"></span></button>
              <button data-key="demand"><span>{T("Demand")}</span><span class="arrow" id="arr-demand"></span></button>
              <button data-key="qopt"><span>{T("Qté optimisée")}</span><span class="arrow" id="arr-qopt"></span></button>
              <button data-key="popt"><span>{T("Profit net optimisé")}</span><span class="arrow" id="arr-popt"></span></button>
              <button data-key="id"><span>{T("ID")}</span><span class="arrow" id="arr-id"></span></button>
              <div>{T("ChatCode")}</div>
              <div class="gsc-muted">{T("Copier")}</div>
            </div>
            <div id="gsc-list"></div>
          </div>
        </div>

        <script>/* ... (inchangé) ... */</script>
        ''', height=680)

    # ----- CSV (haut) -----
    # (conserve le bouton "résultats filtrés" ; la liste de courses optimisée est déjà plus haut et indépendante)
    st.download_button(T("Télécharger CSV (résultats filtrés)"),
        data=df_all.to_csv(index=False), file_name="flips_gw2tp.csv",
        mime="text/csv", key="download_csv_top")

    # ====== GRAPHE MEILLEURS ARTICLES ======
    chart_options_fr = ["Profit Net (PO)", "Score (profil de risque)"]
    chart_options_display = [T(x) for x in chart_options_fr]
    chart_choice_display = st.radio(T("Métrique du graphique"), chart_options_display, index=0, horizontal=True)
    chart_metric_fr = chart_options_fr[chart_options_display.index(chart_choice_display)]

    if chart_metric_fr == "Profit Net (PO)":
        st.subheader(T("Top 20 par Profit Net"))
        top20 = df_all.sort_values("Profit Net (PO)", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(11,5))
        ax.bar(range(len(top20)), top20["Profit Net (PO)"])
        ax.set_xticks(range(len(top20)))
        ax.set_xticklabels(top20["Nom"], rotation=45, ha="right")
        ax.set_ylabel(T("Profit Net (PO)"))
        ax.set_title(T("Top 20 (GW2TP) — Profit Net"))
        st.pyplot(fig, clear_figure=True)
    else:
        st.subheader(T("Top 20 par Score (profil de risque)"))
        top20 = df_all.sort_values("Score", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(11,5))
        ax.bar(range(len(top20)), top20["Score"])
        ax.set_xticks(range(len(top20)))
        ax.set_xticklabels(top20["Nom"], rotation=45, ha="right")
        ax.set_ylabel(T("Score"))
        ax.set_title(T("Top 20 (GW2TP) — Score"))
        st.pyplot(fig, clear_figure=True)

    # ====== TABLEAU TOP 20 (triable) ======
    if chart_metric_fr == "Profit Net (PO)":
        top20_tbl = df_all.sort_values("Profit Net (PO)", ascending=False).head(20)
        default_sort_key = "profit"
        initial_key = "profit"
    else:
        top20_tbl = df_all.sort_values("Score", ascending=False).head(20)
        default_sort_key = "popt"
        initial_key = "popt"

    # ... (tableau Top 20 et CSV bas inchangés) ...

    # ----- CSV (bas) -----
    st.download_button(T("Télécharger CSV (résultats filtrés)"),
        data=df_all.to_csv(index=False), file_name="flips_gw2tp.csv",
        mime="text/csv", key="download_csv_bottom")

    # ====== COURBES HISTORIQUES (si suivi activé) ======
    # ... (inchangé) ...
