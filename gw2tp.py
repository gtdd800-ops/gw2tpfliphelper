# -*- coding: utf-8 -*-
# GW2TP Flips — v5
# UI simplifiée + ROI robuste + Watchlist + Profit/heure + Rotation rapide + Alertes locales
# + Optimisation panier (budget) + Export watchlist (CSV/JSON) + Heatmap ROI/Profit
# 4 langues (FR/EN/DE/ES) — Sélecteur par drapeaux
# Exécution : streamlit run gw2tp_v5.py

import base64, struct, json, sqlite3, time, io
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
STAR = "⭐"
LANG_FLAGS = {"fr":"🇫🇷","en":"🇬🇧","de":"🇩🇪","es":"🇪🇸"}

I18N = {
    # titres / en-têtes
    "title": {"fr": "Flips Trading Post (source: GW2TP)", "en": "Trading Post Flips (source: GW2TP)", "de": "Trading-Post-Flips (Quelle: GW2TP)", "es": "Flips del Trading Post (fuente: GW2TP)"},
    "byline": {"fr": "🔨 escarbeille.4281 · 💬 Discord: escarmouche", "en": "🔨 escarbeille.4281 · 💬 Discord: escarmouche", "de": "🔨 escarbeille.4281 · 💬 Discord: escarmouche", "es": "🔨 escarbeille.4281 · 💬 Discord: escarmouche"},
    "last_update": {"fr":"Dernière mise à jour : ","en":"Last update: ","de":"Letztes Update: ","es":"Última actualización: "},
    "tab_flips": {"fr":"Flips","en":"Flips","de":"Flips","es":"Flips"},
    "tab_history": {"fr":"Historique","en":"History","de":"Historie","es":"Histórico"},
    "tab_advanced": {"fr":"Paramètres avancés","en":"Advanced","de":"Erweitert","es":"Avanzado"},
    "tab_about": {"fr":"À propos","en":"About","de":"Info","es":"Acerca de"},

    # actions rapides / essentiels
    "refresh": {"fr":"Auto-refresh","en":"Auto-refresh","de":"Auto-Refresh","es":"Autoactualización"},
    "interval": {"fr":"Intervalle (min)","en":"Interval (min)","de":"Intervall (Min)","es":"Intervalo (min)"},
    "preset": {"fr":"Preset","en":"Preset","de":"Preset","es":"Preset"},
    "custom": {"fr":"Personnalisé","en":"Custom","de":"Benutzerdefiniert","es":"Personalizado"},
    "cautious": {"fr":"Prudent","en":"Cautious","de":"Vorsichtig","es":"Prudente"},
    "balanced": {"fr":"Équilibré","en":"Balanced","de":"Ausgewogen","es":"Equilibrado"},
    "aggressive": {"fr":"Agressif","en":"Aggressive","de":"Aggressiv","es":"Agresivo"},
    "search": {"fr":"Recherche nom (contient)","en":"Search name (contains)","de":"Suche Name (enthält)","es":"Buscar nombre (contiene)"},

    # blocs
    "essentials": {"fr":"Essentiels","en":"Essentials","de":"Essentials","es":"Esenciales"},
    "filters": {"fr":"Filtres","en":"Filters","de":"Filter","es":"Filtros"},

    # champs essentiels
    "budget": {"fr":"Budget (or, 0 = illimité)","en":"Budget (g, 0 = unlimited)","de":"Budget (g, 0 = unbegrenzt)","es":"Presupuesto (o, 0 = ilimitado)"},
    "min_profit": {"fr":"Profit net min (or)","en":"Min net profit (g)","de":"Min. Nettogewinn (g)","es":"Beneficio neto mín (o)"},
    "min_roi": {"fr":"ROI min (%)","en":"Min ROI (%)","de":"Min. ROI (%)","es":"ROI mín (%)"},
    "min_qty": {"fr":"Quantité min","en":"Min quantity","de":"Mindestmenge","es":"Cantidad mín"},
    "risk": {"fr":"Profil de risque","en":"Risk profile","de":"Risikoprofil","es":"Perfil de riesgo"},

    # avancé
    "max_profit": {"fr":"Profit net max (g, 0 = ∞)","en":"Max net profit (g, 0 = ∞)","de":"Max. Nettogewinn (g, 0 = ∞)","es":"Beneficio neto máx (o, 0 = ∞)"},
    "min_buy_g": {"fr":"Prix d'achat min (g)","en":"Min buy (g)","de":"Min. Kauf (g)","es":"Compra mín (o)"},
    "max_buy_g": {"fr":"Prix d'achat max (g, 0 = ∞)","en":"Max buy (g, 0 = ∞)","de":"Max. Kauf (g, 0 = ∞)","es":"Compra máx (o, 0 = ∞)"},
    "min_buy_c": {"fr":"Achat min (cuivre) pris en compte","en":"Min buy (copper) to consider","de":"Min. Kauf (Kupfer) berücksicht.","es":"Compra mín (cobre) a considerar"},
    "cap_roi": {"fr":"Cap ROI (%) (0 = auto 1000)","en":"Cap ROI (%) (0 = auto 1000)","de":"ROI-Limit (%) (0 = auto 1000)","es":"Tope ROI (%) (0 = auto 1000)"},

    # histoire & opti
    "history_enable": {"fr":"Activer le suivi (SQLite local)","en":"Enable tracking (local SQLite)","de":"Tracking aktivieren (lokales SQLite)","es":"Activar seguimiento (SQLite local)"},
    "history_window": {"fr":"Fenêtre d'analyse (h)","en":"Analysis window (h)","de":"Analysezeitraum (h)","es":"Ventana de análisis (h)"},
    "trend_window": {"fr":"Fenêtre tendance ΔSupply/Demand (h)","en":"Trend window ΔSupply/Demand (h)","de":"Trendfenster ΔAngebot/Nachfrage (h)","es":"Ventana de tendencia ΔOferta/Demanda (h)"},
    "optimized": {"fr":"Optimisation d'achat","en":"Buy optimization","de":"Kaufoptimierung","es":"Optimización de compra"},
    "horizon": {"fr":"Horizon de revente (h)","en":"Resale horizon (h)","de":"Wiederverkaufshorizont (h)","es":"Horizonte de reventa (h)"},
    "safety": {"fr":"Marge de sécurité (%)","en":"Safety margin (%)","de":"Sicherheitsmarge (%)","es":"Margen de seguridad (%)"},
    "fast_mode": {"fr":"Rotation rapide (priorise profit/h)","en":"Fast rotation (prioritize profit/h)","de":"Schnelle Rotation (Profit/h)","es":"Rotación rápida (prioriza beneficio/h)"},

    # watchlist
    "watchlist": {"fr":"Watchlist","en":"Watchlist","de":"Watchlist","es":"Watchlist"},
    "wl_hint": {"fr":"Ajoute/retire des IDs, filtre la vue ou ajoute depuis la liste courante.","en":"Add/remove IDs, filter the view or add from current list.","de":"IDs hinzufügen/entfernen, Ansicht filtern oder aus Liste hinzufügen.","es":"Añade/elimina IDs, filtra la vista o añade desde la lista actual."},
    "add_ids": {"fr":"Ajouter ID(s) (séparés par ,)","en":"Add ID(s) (comma separated)","de":"IDs hinzufügen (durch Komma)","es":"Añadir ID(s) (separados por comas)"},
    "btn_add": {"fr":"➕ Ajouter","en":"➕ Add","de":"➕ Hinzufügen","es":"➕ Añadir"},
    "remove": {"fr":"Retirer","en":"Remove","de":"Entfernen","es":"Quitar"},
    "btn_remove": {"fr":"🗑️ Retirer","en":"🗑️ Remove","de":"🗑️ Entfernen","es":"🗑️ Quitar"},
    "only_wl": {"fr":"Afficher uniquement la watchlist","en":"Show only watchlist","de":"Nur Watchlist anzeigen","es":"Mostrar solo la watchlist"},
    "add_sel_to_wl": {"fr":"⭐ Ajouter sélection → watchlist","en":"⭐ Add selection → watchlist","de":"⭐ Auswahl → Watchlist","es":"⭐ Añadir selección → watchlist"},
    "export_wl_csv": {"fr":"Exporter watchlist (CSV)","en":"Export watchlist (CSV)","de":"Watchlist exportieren (CSV)","es":"Exportar watchlist (CSV)"},
    "export_wl_json": {"fr":"Exporter watchlist (JSON)","en":"Export watchlist (JSON)","de":"Watchlist exportieren (JSON)","es":"Exportar watchlist (JSON)"},

    # alertes
    "alerts": {"fr":"Alertes (local)","en":"Alerts (local)","de":"Alarme (lokal)","es":"Alertas (local)"},
    "alert_profit": {"fr":"Seuil Profit Net (g)","en":"Threshold Net Profit (g)","de":"Schwellwert Nettogewinn (g)","es":"Umbral beneficio neto (o)"},
    "alert_roi": {"fr":"Seuil ROI (%)","en":"Threshold ROI (%)","de":"Schwellwert ROI (%)","es":"Umbral ROI (%)"},

    # affichage / exports
    "download_csv": {"fr":"Télécharger CSV","en":"Download CSV","de":"CSV herunterladen","es":"Descargar CSV"},
    "no_rows": {"fr":"Aucun flip avec ces filtres.","en":"No flips with these filters.","de":"Keine Flips mit diesen Filtern.","es":"No hay flips con estos filtros."},
    "compact": {"fr":"Compact","en":"Compact","de":"Kompakt","es":"Compacto"},

    # colonnes
    "STAR": {"fr":"⭐","en":"⭐","de":"⭐","es":"⭐"},
    "Name": {"fr":"Nom","en":"Name","de":"Name","es":"Nombre"},
    "NetProfit_gsc": {"fr":"Profit Net (g/s/c)","en":"Net Profit (g/s/c)","de":"Nettogewinn (g/s/c)","es":"Beneficio neto (g/s/c)"},
    "ROI_col": {"fr":"ROI (%)","en":"ROI (%)","de":"ROI (%)","es":"ROI (%)"},
    "Buy_g": {"fr":"Prix Achat (g)","en":"Buy Price (g)","de":"Kaufpreis (g)","es":"Precio compra (o)"},
    "Sell_g": {"fr":"Prix Vente Net (g)","en":"Net Sell (g)","de":"Netto-Verkauf (g)","es":"Venta neta (o)"},
    "Qmin": {"fr":"Quantité (min)","en":"Quantity (min)","de":"Menge (min)","es":"Cantidad (mín)"},
    "Supply": {"fr":"Supply","en":"Supply","de":"Supply","es":"Supply"},
    "Demand": {"fr":"Demand","en":"Demand","de":"Demand","es":"Demand"},
    "ProfitPerHour": {"fr":"Profit/h (est)","en":"Profit/h (est)","de":"Profit/h (gesch.)","es":"Beneficio/h (est)"},
    "QtyOpt": {"fr":"Qté optimisée","en":"Opt. qty","de":"Opt. Menge","es":"Cant. óptima"},
    "ProfitOpt": {"fr":"Profit net optimisé (g)","en":"Optimized net profit (g)","de":"Optimierter Nettogewinn (g)","es":"Beneficio neto optimizado (o)"},
    "ID": {"fr":"ID","en":"ID","de":"ID","es":"ID"},
    "ChatCode": {"fr":"ChatCode","en":"ChatCode","de":"ChatCode","es":"ChatCode"},

    # KPIs & graph
    "KPIs": {"fr":"KPIs","en":"KPIs","de":"KPIs","es":"KPIs"},
    "Items": {"fr":"Objets","en":"Items","de":"Objekte","es":"Objetos"},
    "TopProfit": {"fr":"Top Profit (g)","en":"Top Profit (g)","de":"Top Profit (g)","es":"Top Beneficio (o)"},
    "TopROI": {"fr":"Top ROI (%)","en":"Top ROI (%)","de":"Top ROI (%)","es":"Top ROI (%)"},
    "TotalProfitH": {"fr":"Profit/h total (est)","en":"Total Profit/h (est)","de":"Gesamt Profit/h (gesch.)","es":"Beneficio/h total (est)"},
    "Top20Profit": {"fr":"Top 20 — Profit Net (g)","en":"Top 20 — Net Profit (g)","de":"Top 20 — Nettogewinn (g)","es":"Top 20 — Beneficio neto (o)"},
    "HeatmapTitle": {"fr":"Heatmap ROI vs Profit","en":"Heatmap ROI vs Profit","de":"Heatmap ROI vs Profit","es":"Heatmap ROI vs Profit"},
    "Buy_g_axis": {"fr":"Achat (g)","en":"Buy (g)","de":"Kauf (g)","es":"Compra (o)"},
    "Sell_g_axis": {"fr":"Vente 85% (g)","en":"Sell 85% (g)","de":"Verkauf 85% (g)","es":"Venta 85% (o)"},

    # historique UI
    "hist_hint": {"fr":"Visualise l'historique d'un item (si suivi activé)","en":"View item history (if tracking enabled)","de":"Historie anzeigen (wenn Tracking aktiv)","es":"Ver historial (si el seguimiento está activo)"},
    "enable_hist_first": {"fr":"Active l'historique dans les réglages au-dessus.","en":"Enable history in the settings above.","de":"Historie oben in den Einstellungen aktivieren.","es":"Activa el historial en los ajustes de arriba."},
    "no_item": {"fr":"Aucun item disponible.","en":"No item available.","de":"Kein Item verfügbar.","es":"No hay objeto disponible."},
    "choose_item": {"fr":"Choisir un objet","en":"Choose an item","de":"Objekt auswählen","es":"Elige un objeto"},
    "not_enough_hist": {"fr":"Pas encore assez d'historique pour tracer.","en":"Not enough history to plot yet.","de":"Noch nicht genug Historie zum Plotten.","es":"Aún no hay historial suficiente para graficar."},

    # Conseils
    "TipsTitle": {"fr":"Conseils rapides","en":"Quick tips","de":"Schnelle Tipps","es":"Consejos rápidos"},
    "TipLines": {
        "fr": """- Privilégie la **rotation** (Profit/h) plutôt que la marge unitaire.
- Utilise la **watchlist** pour suivre 4–8 flips actifs.
- Fixe un **seuil cuivre min** (>10–25c) pour éviter les ROI aberrants.
- Utilise les **alertes** pour repérer instantanément les opportunités.
- Diversifie et reviens souvent : le TP bouge toute la journée.""",
        "en": """- Favor **rotation** (profit/h) over unit margin.
- Use **watchlist** to track 4–8 active flips.
- Set a **min copper** threshold (>10–25c) to avoid aberrant ROI.
- Use **alerts** to catch opportunities instantly.
- Diversify and revisit often: TP moves all day.""",
        "de": """- Bevorzuge **Rotation** (Profit/h) statt Stückmarge.
- Nutze die **Watchlist** für 4–8 aktive Flips.
- Setze eine **Kupfer-Untergrenze** (>10–25c) gegen Ausreißer-ROI.
- Nutze **Alarme** für schnelle Chancen.
- Diversifiziere und prüfe oft: TP bewegt sich ständig.""",
        "es": """- Prioriza la **rotación** (beneficio/h) sobre el margen unitario.
- Usa **watchlist** para 4–8 flips activos.
- Fija un **mín de cobre** (>10–25c) para evitar ROI aberrante.
- Usa **alertas** para detectar oportunidades al instante.
- Diversifica y vuelve a menudo: el TP se mueve todo el día."""
    },

    # Optimisation panier (libellés manquants ajoutés)
    "BasketOpt": {"fr":"Optimisation panier (budget)","en":"Basket optimization (budget)","de":"Warenkorb-Optimierung (Budget)","es":"Optimización de cesta (presupuesto)"},
    "BasketBudget": {"fr":"Budget panier (g)","en":"Basket budget (g)","de":"Warenkorb-Budget (g)","es":"Presupuesto de la cesta (o)"},
    "BasketTarget": {"fr":"Objectif panier","en":"Basket target","de":"Warenkorb-Ziel","es":"Objetivo de la cesta"},
    "TargetProfitH": {"fr":"Maximiser Profit/h","en":"Maximize Profit/h","de":"Profit/h maximieren","es":"Maximizar beneficio/h"},
    "TargetProfit": {"fr":"Maximiser Profit net","en":"Maximize Net Profit","de":"Nettogewinn maximieren","es":"Maximizar beneficio neto"},
    "BasketN": {"fr":"Top N à considérer","en":"Top N to consider","de":"Top N berücksichtigen","es":"Top N a considerar"},
    "RunBasket": {"fr":"Lancer optimisation","en":"Run optimization","de":"Optimierung starten","es":"Ejecutar optimización"},
    "BasketResult": {"fr":"Allocation proposée","en":"Proposed allocation","de":"Vorgeschlagene Zuteilung","es":"Asignación propuesta"},
    "DownloadBasketCSV": {"fr":"Télécharger panier (CSV)","en":"Download basket (CSV)","de":"Warenkorb herunterladen (CSV)","es":"Descargar cesta (CSV)"}
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
        "User-Agent": "GW2TP-Flips/5.0"
    })
    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=(429,500,502,503,504), allowed_methods=frozenset(["GET"]))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

SESSION = make_session()

# ========================= DB init (création tables) =========================
with sqlite3.connect(DB_PATH, isolation_level=None, timeout=30, check_same_thread=False) as _conn:
    _conn.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            bucket INTEGER NOT NULL,
            buy INTEGER, sell INTEGER, supply INTEGER, demand INTEGER,
            PRIMARY KEY (id, bucket)
        )
        """
    )
    _conn.execute("CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(ts)")
    _conn.execute("CREATE TABLE IF NOT EXISTS watchlist (id INTEGER PRIMARY KEY)")

# ========================= Helpers DB =========================
def wl_get_all() -> List[int]:
    with sqlite3.connect(DB_PATH, isolation_level=None, timeout=30, check_same_thread=False) as conn:
        cur = conn.execute("SELECT id FROM watchlist ORDER BY id")
        return [int(r[0]) for r in cur.fetchall()]

def wl_add(ids: List[int]):
    if not ids: return
    with sqlite3.connect(DB_PATH, isolation_level=None, timeout=30, check_same_thread=False) as conn:
        conn.executemany("INSERT OR IGNORE INTO watchlist(id) VALUES(?)", [(int(x),) for x in ids])

def wl_remove(ids: List[int]):
    if not ids: return
    with sqlite3.connect(DB_PATH, isolation_level=None, timeout=30, check_same_thread=False) as conn:
        conn.executemany("DELETE FROM watchlist WHERE id = ?", [(int(x),) for x in ids])

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

    # --- ROI robuste (anti-aberrations) ---
    buy_po = df["Prix Achat (PO)"].replace(0, np.nan)
    roi = (df["Profit Net (PO)"] / buy_po) * 100
    roi = roi.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=-1000, upper=1000)
    df["ROI (%)"] = roi.round(2)

    df["Quantité (min)"] = df[["supply","demand"]].min(axis=1).fillna(0).astype(int)
    df["ChatCode"] = df["id"].apply(lambda x: make_item_chat_code(int(x)))

    # Retire profits impossibles
    df = df[df["Prix Vente Net (c)"] > df["Prix Achat (c)"]]

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
    # Cap revente
    sell_capacity = int(demand)
    if (sold_in_period is not None) and (hist_window_h or 0) > 0:
        rate = max(0.0, float(sold_in_period)) / float(hist_window_h)
        sell_capacity = int(rate * float(horizon_h))
    sell_capacity = int(sell_capacity * (max(10, min(100, safety_pct)) / 100.0))
    qty = max(0, min(supply, sell_capacity, cap_budget))
    total_profit_c = qty * profit_c
    return qty, total_profit_c

# Profit/heure estimé
def compute_profit_per_hour(df: pd.DataFrame, hist_hours: int, safety_pct: int, history_enabled: bool) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    profit_unit_g = df["Profit Net (PO)"]
    if history_enabled and "Vendu période" in df.columns and hist_hours > 0:
        sold_rate = df["Vendu période"].fillna(0) / float(hist_hours)
    else:
        sold_rate = df["Demand"].fillna(0) * 0.10
    sold_rate = sold_rate * (max(10, min(100, safety_pct)) / 100.0)
    return (profit_unit_g * sold_rate).round(4)

# ========================= Historique (SQLite) =========================

def persist_snapshot(df_bulk: pd.DataFrame):
    if df_bulk.empty:
        return
    now = int(time.time())
    bucket = now // SNAPSHOT_BUCKET_SECONDS
    rows = [(int(r.id), now, bucket, int(r.buy), int(r.sell), int(r.supply), int(r.demand)) for r in df_bulk.itertuples()]
    with sqlite3.connect(DB_PATH, isolation_level=None, timeout=30, check_same_thread=False) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO snapshots (id, ts, bucket, buy, sell, supply, demand) VALUES (?,?,?,?,?,?,?)",
            rows,
        )


def fetch_metrics_for_ids(ids: List[int], hours_window=24, trend_hours=1):
    if not ids:
        return {}, {}, {}, {}
    with sqlite3.connect(DB_PATH, isolation_level=None, timeout=30, check_same_thread=False) as conn:
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
    with sqlite3.connect(DB_PATH, isolation_level=None, timeout=30, check_same_thread=False) as conn:
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
st.set_page_config(page_title=T("title"), page_icon="📈", layout="wide")

# ---- Header ----
left, mid, right = st.columns([1.4, 1, 1])
with left:
    st.title(T("title"))
    st.caption(T("byline"))
with mid:
    st.metric(T("last_update"), time.strftime("%Y-%m-%d %H:%M:%S"))
with right:
    # Sélecteur de langue par drapeaux uniquement
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
    with c1: budget_gold = st.number_input(T("budget"), 0.0, 1e9, 0.0, 0.5)
    with c2: min_profit = st.number_input(T("min_profit"), 0.0, 1e6, 1.0, 0.5)
    with c3: min_roi    = st.number_input(T("min_roi"), 0.0, 1000.0, 10.0, 1.0)
    with c4: min_qty    = st.number_input(T("min_qty"), 0, 10_000_000, 10, 5)
    with c5: risk_slider= st.slider(T("risk"), 0, 100, 50)

# Preset → risk
_preset_map = {T("cautious"): 20, T("balanced"): 50, T("aggressive"): 80}
risk_level = _preset_map.get(preset, risk_slider)

# ---- Avancé ----
with st.expander(T("filters"), expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1: max_profit = st.number_input(T("max_profit"), 0.0, 1e9, 0.0, 0.5)
    with c2: min_buy    = st.number_input(T("min_buy_g"), 0.0, 1e9, 0.0, 0.5)
    with c3: max_buy    = st.number_input(T("max_buy_g"), 0.0, 1e9, 0.0, 0.5)
    c4, c5 = st.columns(2)
    with c4: min_buy_copper_filter = st.number_input(T("min_buy_c"), 0, 10_000, 10, 1)
    with c5: roi_cap = st.number_input(T("cap_roi"), 0, 1_000_000, 0, 50)

# ---- Historique & optimisation ----
hist_tab, trend_tab, watch_tab = st.columns([1, 1, 1])
with hist_tab:
    enable_history = st.checkbox(T("history_enable"), value=False)
    hist_hours = st.slider(T("history_window"), 1, 168, 24, 1)
    trend_hours = st.slider(T("trend_window"), 1, 48, 1, 1)
with trend_tab:
    st.subheader(T("optimized"))
    horizon_h = st.slider(T("horizon"), 1, 168, 24, 1)
    safety_pct = st.slider(T("safety"), 10, 100, 60, 5)
    fast_mode = st.toggle(T("fast_mode"), value=False)
with watch_tab:
    st.subheader(T("watchlist"))
    wl_ids = wl_get_all()
    st.caption(T("wl_hint"))
    w1, w2 = st.columns(2)
    with w1:
        new_ids_str = st.text_input(T("add_ids"), "")
        if st.button(T("btn_add")):
            try:
                ids_add = [int(x.strip()) for x in new_ids_str.split(',') if x.strip()]
                wl_add(ids_add); st.toast(f"+ {ids_add}")
            except Exception as e:
                st.warning(f"IDs invalides: {e}")
    with w2:
        rm_ids = st.multiselect(T("remove"), wl_ids, [])
        if st.button(T("btn_remove")) and rm_ids:
            wl_remove(list(map(int, rm_ids))); st.toast(f"- {rm_ids}")
    show_only_wl = st.toggle(T("only_wl"), value=False)
    # exports watchlist
    col_wl1, col_wl2 = st.columns(2)
    with col_wl1:
        wl_csv = io.StringIO(); pd.Series(wl_get_all(), name='ID').to_csv(wl_csv, index=False)
        st.download_button(T("export_wl_csv"), data=wl_csv.getvalue(), file_name='watchlist.csv', mime='text/csv')
    with col_wl2:
        wl_json = json.dumps({"watchlist": wl_get_all()}, ensure_ascii=False)
        st.download_button(T("export_wl_json"), data=wl_json, file_name='watchlist.json', mime='application/json')

# ---- Alertes locales ----
with st.expander(T("alerts"), expanded=False):
    alert_profit = st.number_input(T("alert_profit"), 0.0, 1e6, 5.0, 0.5)
    alert_roi = st.number_input(T("alert_roi"), 0.0, 10000.0, 25.0, 1.0)

# ---- Auto refresh ----
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_min * 60 * 1000, key="auto_refresh_key_v5")
    except Exception:
        pass

# ========================= Pipeline =========================
mask = (
    (df_all["Profit Net (PO)"] >= min_profit)
    & (df_all["ROI (%)"] >= min_roi)
    & (df_all["Quantité (min)"] >= min_qty)
)
if max_profit > 0: mask &= df_all["Profit Net (PO)"] <= max_profit
if min_buy > 0:   mask &= df_all["Prix Achat (PO)"] >= min_buy
if max_buy > 0:   mask &= df_all["Prix Achat (PO)"] <= max_buy
if min_buy_copper_filter > 0: mask &= df_all["Prix Achat (c)"].fillna(0) >= int(min_buy_copper_filter)
if roi_cap > 0:  mask &= df_all["ROI (%)"] <= float(roi_cap)
if search_name:   mask &= df_all["Nom"].str.contains(search_name, case=False, na=False)

view = df_all[mask].reset_index(drop=True)

# Watchlist badge + filtre
wl_set = set(wl_get_all())
view["⭐"] = view["ID"].apply(lambda i: STAR if int(i) in wl_set else "")
if show_only_wl:
    view = view[view["ID"].isin(wl_set)].reset_index(drop=True)

# Historique
if enable_history:
    try: persist_snapshot(bulk)
    except Exception as e: st.warning("History error: " + str(e))
    if not view.empty:
        ids = view["ID"].tolist()
        sold, bought, dSup, dDem = fetch_metrics_for_ids(ids, hist_hours, trend_hours)
        view["Vendu période"] = view["ID"].map(lambda i: sold.get(i, 0)).astype(int)
        view["Acheté période"] = view["ID"].map(lambda i: bought.get(i, 0)).astype(int)
        view["ΔSupply"] = view["ID"].map(lambda i: dSup.get(i, 0)).astype(int)
        view["ΔDemand"] = view["ID"].map(lambda i: dDem.get(i, 0)).astype(int)

# Score & profit/h
view = add_risk_score(view, risk_level)
view["Profit/h (est)"] = compute_profit_per_hour(view, hist_hours, safety_pct, enable_history)

# Tri
if fast_mode:
    view = view.sort_values(["Profit/h (est)", "Profit Net (PO)"], ascending=[False, False])
else:
    view = view.sort_values(["Score","Profit Net (PO)"], ascending=[False, False])

# Optimisation unitaire
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
    st.subheader(T("KPIs"))
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric(T("Items"), f"{len(view):,}")
    with k2: st.metric(T("TopProfit"), f"{(view['Profit Net (PO)'].max() if not view.empty else 0):,.2f}")
    with k3: st.metric(T("TopROI"), f"{(view['ROI (%)'].max() if not view.empty else 0):,.2f}")
    with k4: st.metric(T("TotalProfitH"), f"{(view['Profit/h (est)'].sum() if 'Profit/h (est)' in view else 0):,.2f}")

    # Alertes visuelles
    if not view.empty:
        hits = view[(view["Profit Net (PO)"] >= alert_profit) | (view["ROI (%)"] >= alert_roi)]
        if not hits.empty:
            st.success(f"🎯 {len(hits)}")

    if view.empty:
        st.info(T("no_rows"))
    else:
        compact = st.toggle(T("compact"), value=True)
        cols_compact = [
            "⭐", "Nom", "Profit Net (gsc)", "ROI (%)", "Prix Achat (PO)", "Prix Vente Net (PO)",
            "Quantité (min)", "Supply", "Demand", "Profit/h (est)", "Qté optimisée", "Profit net optimisé (PO)", "ID", "ChatCode"
        ]
        cols_full = cols_compact + ["ΔSupply","ΔDemand","Vendu période","Acheté période"] if enable_history else cols_compact
        cols = cols_compact if compact else cols_full
        df_disp = view[cols].copy()

        add_sel = st.multiselect(T("add_sel_to_wl"), df_disp["ID"].tolist(), [])
        c_add, c_csv = st.columns([1,1])
        with c_add:
            if st.button(T("add_sel_to_wl")) and add_sel:
                wl_add(list(map(int, add_sel)))
                st.toast("Watchlist ✨")
        with c_csv:
            st.download_button(T("download_csv"), data=df_disp.to_csv(index=False), file_name="flips_gw2tp_v5.csv", mime="text/csv")

        st.dataframe(df_disp, use_container_width=True, hide_index=True)

        # -------- Optimisation panier (budget) --------
        st.subheader(T("BasketOpt"))
        bcol1, bcol2, bcol3 = st.columns([1,1,1])
        with bcol1: basket_budget = st.number_input(T("BasketBudget"), 0.0, 1e9, float(max(0.0, budget_gold)), 0.5)
        with bcol2: basket_target = st.selectbox(T("BasketTarget"), [T("TargetProfitH"), T("TargetProfit")], index=0)
        with bcol3: basket_topn = st.number_input(T("BasketN"), 0, 2000, 50, 1)
        if st.button(T("RunBasket")) and not view.empty:
            df_pool = view.copy()
            if basket_topn > 0:
                if basket_target == T("TargetProfitH"):
                    df_pool = df_pool.sort_values(["Profit/h (est)", "Profit Net (PO)"], ascending=[False, False]).head(basket_topn)
                else:
                    df_pool = df_pool.sort_values("Profit Net (PO)", ascending=False).head(basket_topn)
            budget_c = int(round(basket_budget * 10000)) if basket_budget > 0 else 10**12
            alloc_rows = []
            remaining = budget_c
            for _, r in df_pool.iterrows():
                buy_c = int(r["Prix Achat (c)"]) if pd.notna(r["Prix Achat (c)"]) else 0
                if buy_c <= 0: continue
                limit_qty = int(r.get("Qté optimisée", 0) or 0)
                if limit_qty <= 0:
                    limit_qty = int(r["Quantité (min)"]) if pd.notna(r["Quantité (min)"]) else 0
                if limit_qty <= 0: continue
                if basket_target == T("TargetProfitH"):
                    score = float(r["Profit/h (est)"]) / max(limit_qty,1)
                else:
                    score = float(r["Profit Net (PO)"])
                max_afford = remaining // buy_c
                take = int(min(limit_qty, max_afford))
                if take <= 0: continue
                cost_c = take * buy_c
                remaining -= cost_c
                alloc_rows.append({
                    "ID": int(r["ID"]),
                    "Nom": r["Nom"],
                    "Prix Achat (g)": round(buy_c/10000.0,2),
                    "Qté": take,
                    "Coût (g)": round(cost_c/10000.0,2),
                    "Profit unitaire (g)": float(r["Profit Net (PO)"]),
                    "Profit/h item": float(r["Profit/h (est)"]),
                    "ChatCode": r["ChatCode"],
                })
                if remaining <= 0: break
            if alloc_rows:
                basket_df = pd.DataFrame(alloc_rows)
                st.caption(T("BasketResult"))
                st.dataframe(basket_df, use_container_width=True, hide_index=True)
                st.download_button(T("DownloadBasketCSV"), data=basket_df.to_csv(index=False), file_name="basket.csv", mime="text/csv")
            else:
                st.info(T("no_rows"))

with TAB2:
    st.caption(T("hist_hint"))
    if not enable_history:
        st.info(T("enable_hist_first"))
    else:
        options = view[["ID","Nom"]].copy() if not view.empty else df_all[["ID","Nom"]]
        if options.empty:
            st.info(T("no_item"))
        else:
            options["label"] = options.apply(lambda r: f"{r['Nom']} (ID {r['ID']})", axis=1)
            choice = st.selectbox(T("choose_item"), options["label"].tolist())
            try:
                chosen_id = int(choice.rsplit("ID", 1)[1].strip(" )"))
            except Exception:
                chosen_id = int(options["ID"].iloc[0])

            ts_df = fetch_timeseries_for_id(chosen_id, hist_hours)
            if ts_df.empty or len(ts_df) < 2:
                st.info(T("not_enough_hist"))
            else:
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(ts_df["dt"], ts_df["supply"].replace(0, np.nan), label="Supply")
                ax.plot(ts_df["dt"], ts_df["demand"].replace(0, np.nan), label="Demand")
                ax.set_xlabel("Time"); ax.set_ylabel("Qty"); ax.legend()
                st.pyplot(fig, clear_figure=True)

                price_df = ts_df.copy()
                price_df["buy_po"] = (price_df["buy"] / 100.0).round(2)
                price_df["sell_net_po"] = (price_df["sell"] * TP_NET / 100.0).round(2)
                fig2, ax2 = plt.subplots(figsize=(11, 4))
                ax2.plot(price_df["dt"], price_df["buy_po"], label=T("Buy_g_axis"))
                ax2.plot(price_df["dt"], price_df["sell_net_po"], label=T("Sell_g_axis"))
                ax2.set_xlabel("Time"); ax2.set_ylabel("g"); ax2.legend()
                st.pyplot(fig2, clear_figure=True)

with TAB3:
    st.markdown(f"""**{T('TipsTitle')}**

{T('TipLines')}""")

with TAB4:
    # À propos — texte exact demandé (non traduit)
    st.write("GW2TP Flips v3")
    st.write("Open-source friendly. N'hésite pas à modifier/adapter.")
    st.write("Utilisation de la base de données de gw2tp.com")
    st.caption("🔨 escarbeille.4281 · 💬 Discord: escarmouche")
