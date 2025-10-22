# -*- coding: utf-8 -*-
# GW2TP Flips — v2 (UI simplifiée + esthétique)
# -------------------------------------------------
# Dépendances : streamlit, requests, pandas, numpy, matplotlib
# Exécution : streamlit run gw2tp_v2.py
# -------------------------------------------------

import base64, struct, json, sqlite3, time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ========================= Constantes =========================
TP_NET = 0.85
TIMEOUT = (5, 20)
GW2TP_BULK_ITEMS = "https://api.gw2tp.com/1/bulk/items.json"
GW2TP_BULK_NAMES = "https://api.gw2tp.com/1/bulk/items-names.json"
DB_PATH = "gw2tp_history.sqlite"
SNAPSHOT_BUCKET_SECONDS = 5 * 60

EMO_G, EMO_S, EMO_C = "🟡", "⚪", "🟠"
LANG_FLAGS = {"fr":"🇫🇷","en":"🇬🇧","de":"🇩🇪","es":"🇪🇸"}

I18N = {
    "title": {
        "fr": "Flips Trading Post (source: GW2TP)",
        "en": "Trading Post Flips (source: GW2TP)",
        "de": "Trading-Post-Flips (Quelle: GW2TP)",
        "es": "Flips del Trading Post (fuente: GW2TP)",
    },
    "byline": {
        "fr": "🛠️ escarbeille.4281 · Discord : escarmouche",
        "en": "🛠️ escarbeille.4281 · Discord: escarmouche",
        "de": "🛠️ escarbeille.4281 · Discord: escarmouche",
        "es": "🛠️ escarbeille.4281 · Discord: escarmouche",
    },
    "last_update": {"fr":"Dernière mise à jour : ","en":"Last update: ","de":"Letztes Update: ","es":"Última actualización: "},
    "tab_flips": {"fr":"Flips","en":"Flips","de":"Flips","es":"Flips"},
    "tab_history": {"fr":"Historique","en":"History","de":"Historie","es":"Histórico"},
    "tab_advanced": {"fr":"Paramètres avancés","en":"Advanced","de":"Erweitert","es":"Avanzado"},
    "tab_about": {"fr":"À propos","en":"About","de":"Info","es":"Acerca de"},
    "refresh": {"fr":"Auto-refresh","en":"Auto-refresh","de":"Auto-Refresh","es":"Autoactualización"},
    "interval": {"fr":"Intervalle (min)","en":"Interval (min)","de":"Intervall (Min)","es":"Intervalo (min)"},
    "essentials": {"fr":"Essentiels","en":"Essentials","de":"Essentials","es":"Esenciales"},
    "filters": {"fr":"Filtres","en":"Filters","de":"Filter","es":"Filtros"},
    "budget": {"fr":"Budget (or, 0 = illimité)","en":"Budget (g, 0 = unlimited)","de":"Budget (g, 0 = unbegrenzt)","es":"Presupuesto (o, 0 = ilimitado)"},
    "min_profit": {"fr":"Profit net min (or)","en":"Min net profit (g)","de":"Min. Nettogewinn (g)","es":"Beneficio neto mín (o)"},
    "min_roi": {"fr":"ROI min (%)","en":"Min ROI (%)","de":"Min. ROI (%)","es":"ROI mín (%)"},
    "min_qty": {"fr":"Quantité min","en":"Min quantity","de":"Mindestmenge","es":"Cantidad mín"},
    "search": {"fr":"Recherche nom (contient)","en":"Search name (contains)","de":"Suche Name (enthält)","es":"Buscar nombre (contiene)"},
    "risk": {"fr":"Profil de risque","en":"Risk profile","de":"Risikoprofil","es":"Perfil de riesgo"},
    "preset": {"fr":"Preset","en":"Preset","de":"Preset","es":"Preset"},
    "custom": {"fr":"Personnalisé","en":"Custom","de":"Benutzerdefiniert","es":"Personalizado"},
    "cautious": {"fr":"Prudent","en":"Cautious","de":"Vorsichtig","es":"Prudente"},
    "balanced": {"fr":"Équilibré","en":"Balanced","de":"Ausgewogen","es":"Equilibrado"},
    "aggressive": {"fr":"Agressif","en":"Aggressive","de":"Aggressiv","es":"Agresivo"},
    "download_csv": {"fr":"Télécharger CSV","en":"Download CSV","de":"CSV herunterladen","es":"Descargar CSV"},
    "no_rows": {"fr":"Aucun flip avec ces filtres.","en":"No flips with these filters.","de":"Keine Flips mit diesen Filtern.","es":"No hay flips con estos filtros."},
    "optimized": {"fr":"Optimisation d'achat","en":"Buy optimization","de":"Kaufoptimierung","es":"Optimización de compra"},
    "horizon": {"fr":"Horizon de revente (h)","en":"Resale horizon (h)","de":"Wiederverkaufshorizont (h)","es":"Horizonte de reventa (h)"},
    "safety": {"fr":"Marge de sécurité (%)","en":"Safety margin (%)","de":"Sicherheitsmarge (%)","es":"Margen de seguridad (%)"},
    "history_enable": {"fr":"Activer le suivi (SQLite local)","en":"Enable tracking (local SQLite)","de":"Tracking aktivieren (lokales SQLite)","es":"Activar seguimiento (SQLite local)"},
    "history_window": {"fr":"Fenêtre d'analyse (h)","en":"Analysis window (h)","de":"Analysezeitraum (h)","es":"Ventana de análisis (h)"},
    "trend_window": {"fr":"Fenêtre tendance ΔSupply/Demand (h)","en":"Trend window ΔSupply/Demand (h)","de":"Trendfenster ΔAngebot/Nachfrage (h)","es":"Ventana de tendencia ΔOferta/Demanda (h)"},
}

# ========================= i18n =========================
if "lang" not in st.session_state:
    st.session_state["lang"] = "fr"

def T(key: str) -> str:
    lang = st.session_state.get("lang", "fr")
    d = I18N.get(key)
    if isinstance(d, dict):
        return d.get(lang, d.get("fr", key))
    return key

# ========================= HTTP session =========================
@st.cache_resource(show_spinner=False)
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "User-Agent": "GW2TP-Flips/2.0"
    })
    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=(429,500,502,503,504), allowed_methods=frozenset(["GET"]))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

SESSION = make_session()

# ========================= DB (historique local) =========================
@st.cache_resource(show_spinner=False)
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

# ========================= Helpers =========================
def gsc_emoji_from_copper(c: int) -> str:
    sign = "-" if c < 0 else ""
    c = abs(int(c)); g = c // 10000; s = (c % 10000) // 100; cc = c % 100
    parts = []
    if g > 0: parts.append(f"{g}{EMO_G}")
    if s > 0 or g > 0: parts.append(f"{s}{EMO_S}")
    parts.append(f"{cc}{EMO_C}")
    return sign + " ".join(parts)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_bulk_items() -> pd.DataFrame:
    for url in (GW2TP_BULK_ITEMS, GW2TP_BULK_ITEMS.replace("https://","http://")):
        r = SESSION.get(url, timeout=TIMEOUT)
        if r.ok:
            obj = r.json()
            return pd.DataFrame(obj.get("items", []), columns=obj.get("columns", []))
    return pd.DataFrame(columns=["id","buy","sell","supply","demand"])

@st.cache_data(ttl=300, show_spinner=False)
def fetch_item_names() -> Dict[int, str]:
    for url in (GW2TP_BULK_NAMES, GW2TP_BULK_NAMES.replace("https://","http://")):
        r = SESSION.get(url, timeout=TIMEOUT)
        if r.ok:
            items = r.json().get("items", [])
            return {int(x[0]): x[1] for x in items if len(x) >= 2}
    return {}

@st.cache_data(ttl=300, show_spinner=False)
def get_bulk_and_df():
    bulk = fetch_bulk_items()
    names = fetch_item_names()
    df = build_flips_df(bulk, names)
    return bulk, df

# ========================= Dataset =========================
def make_item_chat_code(item_id: int, qty: int = 1) -> str:
    q = max(1, min(255, int(qty)))
    data = bytearray([0x02, q]) + struct.pack("<I", int(item_id)) + bytearray([0x00])
    return "[&" + base64.b64encode(bytes(data)).decode("ascii") + "]"


def build_flips_df(df_bulk: pd.DataFrame, id2name: Dict[int, str]) -> pd.DataFrame:
    if df_bulk.empty:
        return pd.DataFrame(columns=[
            "Nom","ID","Prix Achat (PO)","Prix Vente Net (PO)","Profit Net (PO)","ROI (%)",
            "Quantité (min)","Supply","Demand","ChatCode",
            "Prix Achat (c)","Prix Vente Net (c)","Profit Net (c)",
            "Profit Net (gsc)"
        ])

    df = df_bulk.copy()
    df["Nom"] = df["id"].map(id2name).fillna("")
    df["Prix Achat (c)"] = df["buy"].astype("Int64")
    df["Prix Vente Net (c)"] = (df["sell"] * TP_NET).round().astype("Int64")
    df["Profit Net (c)"] = (df["Prix Vente Net (c)"] - df["Prix Achat (c)"]).astype("Int64")

    df["Prix Achat (PO)"] = (df["Prix Achat (c)"] / 10000.0).round(2)
    df["Prix Vente Net (PO)"] = (df["Prix Vente Net (c)"] / 10000.0).round(2)
    df["Profit Net (PO)"] = (df["Profit Net (c)"] / 10000.0).round(2)

    buy_po = df["Prix Achat (PO)"].replace(0, np.nan)
    df["ROI (%)"] = ((df["Profit Net (PO)"] / buy_po) * 100).replace([np.inf, -np.inf], np.nan).fillna(0).round(2)

    df["Quantité (min)"] = df[["supply","demand"]].min(axis=1).fillna(0).astype(int)
    df["ChatCode"] = df["id"].apply(lambda x: make_item_chat_code(int(x)))

    df.rename(columns={"id":"ID","supply":"Supply","demand":"Demand"}, inplace=True)
    df["Profit Net (gsc)"] = df["Profit Net (c)"].fillna(0).astype(int).apply(gsc_emoji_from_copper)

    return df[[
        "Nom","ID","Prix Achat (PO)","Prix Vente Net (PO)","Profit Net (PO)","ROI (%)",
        "Quantité (min)","Supply","Demand","ChatCode",
        "Prix Achat (c)","Prix Vente Net (c)","Profit Net (c)","Profit Net (gsc)"
    ]]

# ========================= Scoring & opti =========================
PRESET_TO_RISK = {"Prudent": 20, "Équilibré": 50, "Agressif": 80}


def add_risk_score(df: pd.DataFrame, risk_level: int) -> pd.DataFrame:
    if df.empty:
        return df
    w_roi = risk_level / 100.0
    w_vol = 1 - w_roi
    roi = df["ROI (%)"].clip(lower=0).fillna(0)
    vol = np.log10(df["Quantité (min)"].fillna(0) + 1)

    def norm(s):
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

    df = df.copy()
    df["Score"] = (w_roi * norm(roi) + w_vol * norm(vol)).round(3)
    return df


def compute_optimal_qty(row, budget_gold: float, horizon_h: int, safety_pct: int, hist_window_h: int, sold_in_period: int = None):
    buy_c = int(row.get("Prix Achat (c)", 0) or 0)
    profit_c = int(row.get("Profit Net (c)", 0) or 0)
    if buy_c <= 0 or profit_c <= 0:
        return 0, 0

    supply = int(row.get("Supply", 0) or 0)
    demand = int(row.get("Demand", 0) or 0)

    # Cap budget
    cap_budget = 10**9
    if budget_gold and budget_gold > 0:
        budget_c = int(round(budget_gold * 10000))
        cap_budget = max(0, budget_c // buy_c)

    # Cap revente (simple : demande), optionnellement basé sur historique
    sell_capacity = int(demand)
    if (sold_in_period is not None) and (hist_window_h or 0) > 0:
        rate = max(0.0, float(sold_in_period)) / float(hist_window_h)
        sell_capacity = int(rate * float(horizon_h))

    sell_capacity = int(sell_capacity * (max(10, min(100, safety_pct)) / 100.0))

    qty = max(0, min(supply, sell_capacity, cap_budget))
    total_profit_c = qty * profit_c
    return qty, total_profit_c

# ========================= Historique (SQLite) =========================

def persist_snapshot(df_bulk: pd.DataFrame):
    if df_bulk.empty:
        return
    now = int(time.time())
    bucket = now // SNAPSHOT_BUCKET_SECONDS
    rows = [(int(r.id), now, bucket, int(r.buy), int(r.sell), int(r.supply), int(r.demand)) for r in df_bulk.itertuples()]
    with db_connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO snapshots (id, ts, bucket, buy, sell, supply, demand) VALUES (?,?,?,?,?,?,?)",
            rows,
        )


def fetch_metrics_for_ids(ids: List[int], hours_window=24, trend_hours=1):
    if not ids:
        return {}, {}, {}, {}
    with db_connect() as conn:
        now = int(time.time())
        since_window = now - hours_window * 3600
        q_marks = ",".join("?" for _ in ids)
        cur = conn.execute(
            f"SELECT id, ts, buy, sell, supply, demand FROM snapshots WHERE id IN ({q_marks}) AND ts >= ? ORDER BY id, ts",
            (*ids, since_window),
        )
        rows = cur.fetchall()

    data: Dict[int, List[Tuple[int,int,int,int,int]]] = {}
    for id_, ts, buy, sell, supply, demand in rows:
        data.setdefault(id_, []).append((ts, buy, sell, supply, demand))

    sold, bought, dSup, dDem = {}, {}, {}, {}
    now = int(time.time())
    since_trend = now - trend_hours * 3600
    for id_, arr in data.items():
        df = pd.DataFrame(arr, columns=["ts","buy","sell","supply","demand"]).sort_values("ts")
        dsup = df["supply"].diff(); ddem = df["demand"].diff()
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
            (int(item_id), since),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts","buy","sell","supply","demand"])
    df = pd.DataFrame(rows, columns=["ts","buy","sell","supply","demand"])
    df["dt"] = pd.to_datetime(df["ts"], unit="s")
    return df

# ========================= UI =========================
st.set_page_config(
    page_title=T("title"),
    page_icon="📈",
    layout="wide",
)

# ---- Header ----
left, mid, right = st.columns([1.4, 1, 1])
with left:
    st.title(T("title"))
    st.caption(T("byline"))
with mid:
    st.metric(T("last_update"), time.strftime("%Y-%m-%d %H:%M:%S"))
with right:
    st.write("")
    cols = st.columns(len(LANG_FLAGS))
    for i, (code, flag) in enumerate(LANG_FLAGS.items()):
        if cols[i].button(flag, use_container_width=True):
            st.session_state["lang"] = code

# ---- Chargement données ----
bulk, df_all = get_bulk_and_df()

# ---- Barre d'actions rapides ----
a1, a2, a3, a4 = st.columns([1, 1, 1, 2])
with a1:
    auto_refresh = st.toggle(T("refresh"), value=True)
with a2:
    refresh_min = st.slider(T("interval"), 1, 30, 5)
with a3:
    preset = st.selectbox(T("preset"), [T("custom"), T("cautious"), T("balanced"), T("aggressive")], index=2)
with a4:
    search_name = st.text_input(T("search"), "", placeholder="rune, inscription, ecto…")

# ---- Essentiels & filtres ----
with st.expander(T("essentials"), expanded=True):
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        budget_gold = st.number_input(T("budget"), 0.0, 1e9, 0.0, 0.5)
    with c2:
        min_profit = st.number_input(T("min_profit"), 0.0, 1e6, 1.0, 0.5)
    with c3:
        min_roi = st.number_input(T("min_roi"), 0.0, 1000.0, 10.0, 1.0)
    with c4:
        min_qty = st.number_input(T("min_qty"), 0, 10_000_000, 10, 5)
    with c5:
        risk_slider = st.slider(T("risk"), 0, 100, 50)

# Preset → risk
_preset_map = {T("cautious"): 20, T("balanced"): 50, T("aggressive"): 80}
if preset in _preset_map:
    risk_level = _preset_map[preset]
else:
    risk_level = risk_slider

# ---- Avancé ----
with st.expander(T("filters"), expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        max_profit = st.number_input("Max profit (g, 0 = ∞)", 0.0, 1e9, 0.0, 0.5)
    with c2:
        min_buy = st.number_input("Min buy (g)", 0.0, 1e9, 0.0, 0.5)
    with c3:
        max_buy = st.number_input("Max buy (g, 0 = ∞)", 0.0, 1e9, 0.0, 0.5)

# ---- Historique & optimisation ----
hist_tab, trend_tab = st.columns([1, 1])
with hist_tab:
    enable_history = st.checkbox(T("history_enable"), value=False)
    hist_hours = st.slider(T("history_window"), 1, 168, 24, 1)
    trend_hours = st.slider(T("trend_window"), 1, 48, 1, 1)
with trend_tab:
    st.subheader(T("optimized"))
    horizon_h = st.slider(T("horizon"), 1, 168, 24, 1)
    safety_pct = st.slider(T("safety"), 10, 100, 60, 5)

# ---- Auto refresh (Streamlit native) ----
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_min * 60 * 1000, key="auto_refresh_key_v2")
    except Exception:
        pass

# ========================= Pipeline =========================
# Filtrage de base
mask = (df_all["Profit Net (PO)"] >= min_profit) & (df_all["ROI (%)"] >= min_roi) & (df_all["Quantité (min)"] >= min_qty)
if max_profit > 0: mask &= df_all["Profit Net (PO)"] <= max_profit
if min_buy > 0: mask &= df_all["Prix Achat (PO)"] >= min_buy
if max_buy > 0: mask &= df_all["Prix Achat (PO)"] <= max_buy
if search_name: mask &= df_all["Nom"].str.contains(search_name, case=False, na=False)

view = df_all[mask].reset_index(drop=True)

# Historique : snapshot + métriques
if enable_history:
    try:
        persist_snapshot(bulk)
    except Exception as e:
        st.warning("History error: " + str(e))

    if not view.empty:
        ids = view["ID"].tolist()
        sold, bought, dSup, dDem = fetch_metrics_for_ids(ids, hist_hours, trend_hours)
        view["Vendu période"] = view["ID"].map(lambda i: sold.get(i, 0)).astype(int)
        view["Acheté période"] = view["ID"].map(lambda i: bought.get(i, 0)).astype(int)
        view["ΔSupply"] = view["ID"].map(lambda i: dSup.get(i, 0)).astype(int)
        view["ΔDemand"] = view["ID"].map(lambda i: dDem.get(i, 0)).astype(int)

# Score & tri par défaut
view = add_risk_score(view, risk_level)
view = view.sort_values(["Score","Profit Net (PO)"], ascending=[False, False])

# Optimisation
if not view.empty:
    qty_opt, profit_opt_c = [], []
    for _, r in view.iterrows():
        sold_period = int(r.get("Vendu période", 0)) if enable_history else None
        q, p_c = compute_optimal_qty(r, budget_gold, horizon_h, safety_pct, hist_hours, sold_period)
        qty_opt.append(int(q)); profit_opt_c.append(int(p_c))
    view["Qté optimisée"] = qty_opt
    view["Profit net optimisé (c)"] = profit_opt_c
    view["Profit net optimisé (PO)"] = (view["Profit net optimisé (c)"] / 10000.0).round(2)

# ========================= Tabs =========================
TAB1, TAB2, TAB3, TAB4 = st.tabs([T("tab_flips"), T("tab_history"), T("tab_advanced"), T("tab_about")])

with TAB1:
    st.subheader("KPIs")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Items", f"{len(view):,}")
    with k2:
        st.metric("Top Profit (g)", f"{(view['Profit Net (PO)'].max() if not view.empty else 0):,.2f}")
    with k3:
        st.metric("Top ROI (%)", f"{(view['ROI (%)'].max() if not view.empty else 0):,.2f}")
    with k4:
        st.metric("Profit optimisé (g)", f"{(view['Profit net optimisé (PO)'].sum() if 'Profit net optimisé (PO)' in view else 0):,.2f}")

    if view.empty:
        st.info(T("no_rows"))
    else:
        # Vue compacte/détaillée
        compact = st.toggle("Compact", value=True)
        cols_compact = [
            "Nom","Profit Net (gsc)","ROI (%)","Prix Achat (PO)","Prix Vente Net (PO)",
            "Quantité (min)","Supply","Demand","Qté optimisée","Profit net optimisé (PO)","ID","ChatCode"
        ]
        cols_full = cols_compact + ["ΔSupply","ΔDemand","Vendu période","Acheté période"] if enable_history else cols_compact
        cols = cols_compact if compact else cols_full

        # Formatage
        df_disp = view[cols].copy()
        # Télécharger CSV
        st.download_button(T("download_csv"), data=df_disp.to_csv(index=False), file_name="flips_gw2tp_v2.csv", mime="text/csv")

        # Tableau interactif
        st.dataframe(
            df_disp,
            use_container_width=True,
            hide_index=True,
        )

        # Graphique Top 20 par profit
        st.subheader("Top 20 — Profit Net (g)")
        top20 = view.sort_values("Profit Net (PO)", ascending=False).head(20)
        if not top20.empty:
            fig, ax = plt.subplots(figsize=(11,5))
            ax.bar(range(len(top20)), top20["Profit Net (PO)"])
            ax.set_xticks(range(len(top20)))
            ax.set_xticklabels(top20["Nom"], rotation=45, ha="right")
            ax.set_ylabel("g")
            st.pyplot(fig, clear_figure=True)

with TAB2:
    st.caption("Visualise l'historique d'un item (si suivi activé)")
    if not enable_history:
        st.info("Active l'historique dans les réglages au-dessus.")
    else:
        options = view[["ID","Nom"]].copy() if not view.empty else df_all[["ID","Nom"]]
        if options.empty:
            st.info("Aucun item disponible.")
        else:
            options["label"] = options.apply(lambda r: f"{r['Nom']} (ID {r['ID']})", axis=1)
            choice = st.selectbox("Choisir un objet", options["label"].tolist())
            try:
                chosen_id = int(choice.rsplit("ID", 1)[1].strip(" )"))
            except Exception:
                chosen_id = int(options["ID"].iloc[0])

            ts_df = fetch_timeseries_for_id(chosen_id, hist_hours)
            if ts_df.empty or len(ts_df) < 2:
                st.info("Pas encore assez d'historique pour tracer.")
            else:
                # Offre/Demande
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(ts_df["dt"], ts_df["supply"].replace(0, np.nan), label="Supply")
                ax.plot(ts_df["dt"], ts_df["demand"].replace(0, np.nan), label="Demand")
                ax.set_xlabel("Temps"); ax.set_ylabel("Quantités"); ax.legend()
                st.pyplot(fig, clear_figure=True)

                # Prix
                price_df = ts_df.copy()
                price_df["buy_po"] = (price_df["buy"] / 100.0).round(2)
                price_df["sell_net_po"] = (price_df["sell"] * TP_NET / 100.0).round(2)
                fig2, ax2 = plt.subplots(figsize=(11, 4))
                ax2.plot(price_df["dt"], price_df["buy_po"], label="Buy (g)")
                ax2.plot(price_df["dt"], price_df["sell_net_po"], label="Sell 85% (g)")
                ax2.set_xlabel("Temps"); ax2.set_ylabel("Prix (g)"); ax2.legend()
                st.pyplot(fig2, clear_figure=True)

with TAB3:
    st.markdown("""
    **Conseils rapides**
    - Réduis le nombre de filtres au strict nécessaire.
    - Utilise *Recherche nom* pour cibler des familles (ex: *Rune*, *Sceau*, *Inscription*).
    - La *Qté optimisée* tient compte du budget, de la demande et, si possible, du rythme de ventes historique.
    """)
    st.markdown("""
    **Astuces UI**
    - Bascule *Compact* pour une vue plus lisible.
    - Le CSV exporte exactement les colonnes affichées.
    """)

with TAB4:
    st.write("GW2TP Flips v2 — UI simplifiée. Inspiré de la version originale d'escarbeille.")
    st.write("Open-source friendly. N'hésite pas à modifier/adapter.")
    st.caption("Made with ❤️ and Streamlit.")
