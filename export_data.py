"""
export_data.py — génère data.js pour le site GitHub Pages
Usage : python export_data.py  (depuis le dossier florian-kraus-site/)
        ou : python export_data.py --app-dir "C:/Users/fkraus/Desktop/DASHBOARD CRYPTO"
"""
import sys, os, json, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--app-dir", default=r"C:\Users\fkraus\Desktop\DASHBOARD CRYPTO")
args = parser.parse_args()

APP = Path(args.app_dir)
sys.path.insert(0, str(APP))

import pandas as pd
import numpy as np
from pipelines.crypto_stability import load_stability

DATA   = APP / "data" / "cache"
RAW    = APP / "data" / "raw"
FINE   = RAW / "buckets" / "aggTrades_monthly_fine.csv"
CRIX   = RAW / "CRIX_data.csv"

out = {}

# ── 1. Buckets — all 4 assets, default grouping ─────────────────────────────
ASSETS   = ["BTC", "ETH", "XRP", "LTC"]
SYMS     = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT", "LTC": "LTCUSDT"}
FINE_COLS= ["<$1k","$1k-$5k","$5k-$10k","$10k-$20k","$20k-$30k","$30k-$40k",
             "$40k-$50k","$50k-$60k","$60k-$70k","$70k-$80k","$80k-$90k","$90k-$100k",
             "$100k-$200k","$200k-$300k","$300k-$400k","$400k-$500k","$500k-$1M",">$1M"]
# default cuts: $10k, $100k → 3 groups
GROUPS = {
    "< $10k":       ["<$1k","$1k-$5k","$5k-$10k"],
    "$10k – $100k": ["$10k-$20k","$20k-$30k","$30k-$40k","$40k-$50k",
                     "$50k-$60k","$60k-$70k","$70k-$80k","$80k-$90k","$90k-$100k"],
    "> $100k":      ["$100k-$200k","$200k-$300k","$300k-$400k","$400k-$500k","$500k-$1M",">$1M"],
}
COLORS = {"< $10k": "#818cf8", "$10k – $100k": "#6366f1", "> $100k": "#3730a3"}

df_fine = pd.read_csv(FINE, encoding="utf-8-sig") if FINE.exists() else None
buckets = {}
for asset in ASSETS:
    sym = SYMS[asset]
    if df_fine is None or sym not in df_fine["symbol"].values:
        continue
    df = df_fine[df_fine["symbol"] == sym].sort_values("ym").copy()
    tot = df["total_usd_volume"]
    asset_data = {"months": df["ym"].tolist(), "total_bn": [round(v/1e9,2) for v in tot]}
    for g_label, g_cols in GROUPS.items():
        valid = [c for c in g_cols if c in df.columns]
        share = df[valid].sum(axis=1) / tot
        asset_data[g_label] = [round(max(v, 0), 4) for v in share]
    # last-month stats
    last = df[df["ym"] == df["ym"].max()].iloc[0]
    t = last["total_usd_volume"]
    small = sum(last[c] for c in ["<$1k","$1k-$5k","$5k-$10k"] if c in last.index)
    large = sum(last[c] for c in ["$100k-$200k","$200k-$300k","$300k-$400k",
                                   "$400k-$500k","$500k-$1M",">$1M"] if c in last.index)
    whale = last[">$1M"] if ">$1M" in last.index else 0
    asset_data["last_ym"]      = last["ym"]
    asset_data["small_pct"]    = round(small/t*100, 1) if t else 0
    asset_data["large_pct"]    = round(large/t*100, 1) if t else 0
    asset_data["whale_pct"]    = round(whale/t*100, 1) if t else 0
    asset_data["n_months"]     = len(df)
    buckets[asset] = asset_data
out["buckets"] = buckets
print(f"Buckets: {list(buckets.keys())}")

# ── 2. Common factor + CRIX ──────────────────────────────────────────────────
stab   = load_stability()
factor = stab["factor"].copy()
factor.index = pd.to_datetime(factor.index)
factor_cum = factor.cumsum()

out["factor"] = {
    "dates":   [d.strftime("%Y-%m-%d") for d in factor.index],
    "pc1":     [round(float(v), 4) for v in factor.values],
    "pc1_cum": [round(float(v), 4) for v in factor_cum.values],
}
print(f"Factor: {len(out['factor']['dates'])} days")

crix = {}
if CRIX.exists():
    df_crix = pd.read_csv(CRIX, sep=";", parse_dates=["date"]).dropna(subset=["price"]).sort_values("date")
    crix = {
        "dates": [d.strftime("%Y-%m-%d") for d in df_crix["date"]],
        "log_price": [round(float(np.log(p)), 4) for p in df_crix["price"]],
    }
    print(f"CRIX: {len(crix['dates'])} days")
out["crix"] = crix

# ── 3. FI/FF share + portfolio ───────────────────────────────────────────────
share = stab["fi_ff_share"].copy()
share["date"] = pd.to_datetime(share["date"])
last = share.iloc[-1]
n_total = int(last["n_assets"])

out["fifff"] = {
    "months":   share["date"].dt.strftime("%Y-%m").tolist(),
    "fi":       [round(v, 4) for v in share["fi_share"]],
    "ff":       [round(v, 4) for v in share["ff_share"]],
    "stable":   [round(max(1 - fi - ff, 0), 4) for fi, ff in zip(share["fi_share"], share["ff_share"])],
    "n_assets": share["n_assets"].astype(int).tolist(),
    "last_fi":  round(float(last["fi_share"]), 4),
    "last_ff":  round(float(last["ff_share"]), 4),
    "n_total":  n_total,
    "last_win": share["date"].max().strftime("%Y-%m"),
}

# Portfolio
portfolio = {}
try:
    broad_path = DATA / "crypto_prices_broad.parquet"
    if broad_path.exists():
        from app import _compute_portfolio
        _broad = pd.read_parquet(broad_path)
        fi_ff_df = stab["fi_ff"].copy()
        cum, perf = _compute_portfolio(fi_ff_df, _broad)
        if cum is not None:
            portfolio = {
                "dates": cum["date"].dt.strftime("%Y-%m-%d").tolist(),
                "FI":     [round(v, 2) for v in cum["FI"]],
                "FF":     [round(v, 2) for v in cum["FF"]],
                "Stable": [round(v, 2) for v in cum["Stable"]],
                "Top10":  [round(v, 2) for v in cum["Top10"]],
            }
            print(f"Portfolio: {len(portfolio['dates'])} months")
except Exception as e:
    print(f"Portfolio skipped: {e}")
out["portfolio"] = portfolio

# ── 4. CBDC ──────────────────────────────────────────────────────────────────
cbdc_m = pd.read_parquet(DATA / "cbdc_monthly.parquet")
cbdc_m["month"] = pd.to_datetime(cbdc_m["month"])
cbdc_m = cbdc_m[cbdc_m["month"] >= "2018-01-01"].sort_values("month").copy()

MA = 3
cbdc_m["jev_stance"] = (cbdc_m["jev_stance_pro_pct"] - cbdc_m["jev_stance_anti_pct"]).rolling(MA, min_periods=1).mean()
cbdc_m["jev_sent"]   = (cbdc_m["jev_sentiment_pos_pct"] - cbdc_m["jev_sentiment_neg_pct"]).rolling(MA, min_periods=1).mean()

out["cbdc"] = {
    "months":      cbdc_m["month"].dt.strftime("%Y-%m").tolist(),
    "stance":      [round(float(v), 4) for v in cbdc_m["jev_stance"]],
    "sentiment":   [round(float(v), 4) for v in cbdc_m["jev_sent"]],
    "pro_pct":     [round(float(v), 4) for v in cbdc_m["jev_stance_pro_pct"]],
    "anti_pct":    [round(float(v), 4) for v in cbdc_m["jev_stance_anti_pct"]],
    "wait_pct":    [round(float(v), 4) for v in cbdc_m["jev_stance_wait_pct"]],
    # type distribution (% of sentences)
    "type_retail":     [round(float(v), 4) for v in cbdc_m["jev_type_retail_pct"]],
    "type_wholesale":  [round(float(v), 4) for v in cbdc_m["jev_type_wholesale_pct"]],
    "type_general":    [round(float(v), 4) for v in cbdc_m["jev_type_general_pct"]],
    # discourse distribution (% of sentences)
    "disc_feat":  [round(float(v), 4) for v in cbdc_m["jev_discourse_feat_pct"]],
    "disc_proc":  [round(float(v), 4) for v in cbdc_m["jev_discourse_proc_pct"]],
    "disc_risk":  [round(float(v), 4) for v in cbdc_m["jev_discourse_risk_pct"]],
    "n_speeches":  cbdc_m["n_speeches"].astype(int).tolist(),
    "last_stance": round(float(cbdc_m["jev_stance"].iloc[-1]), 4),
    "last_month":  cbdc_m["month"].max().strftime("%Y-%m"),
    "total_speeches": int(cbdc_m["n_speeches"].sum()),
}
print(f"CBDC: {len(out['cbdc']['months'])} months, {out['cbdc']['total_speeches']} speeches")

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "data.js"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("const CHART_DATA = " + json.dumps(out, ensure_ascii=False, default=str) + ";")
print(f"\nSaved → {out_path}")
