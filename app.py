"""
Aplikacja Streamlit: prognoza ceny złota — GC=F (USD/oz lub USD/g) lub GLD (USD/udział)

Źródło: Yahoo Finance (yfinance)
Model: SARIMA (statsmodels.SARIMAX bez zmiennych zewnętrznych, szybka wersja)
"""

import warnings
import os
import os, certifi, tempfile, shutil


#kod wstawiony przez Sławomira Kobyłko, wymagane jest do działania na jego komputerze (problemy z certyfikatem)
# 1) skopiuj cacert.pem do ścieżki bez diakrytyków (temp)
_src = certifi.where()
_dst = os.path.join(tempfile.gettempdir(), "cacert.pem")
try:
    if not os.path.exists(_dst) or os.path.getsize(_dst) != os.path.getsize(_src):
        shutil.copyfile(_src, _dst)
except Exception:
    # jeśli coś pójdzie nie tak – trudno, zostawimy oryginał
    _dst = _src

# 2) ustaw wszystkie relevantne zmienne
os.environ["SSL_CERT_FILE"] = _dst
os.environ["REQUESTS_CA_BUNDLE"] = _dst
os.environ["CURL_CA_BUNDLE"] = _dst          # ważne dla curl / curl_cffi
os.environ["YF_USE_CURL_CFFI"] = "0"         # wyłącz curl_cffi w yfinance (na wszelki wypadek)
os.environ["YFINANCE_USE_CURL_CFFI"] = "0"   # alias – też ustawiamy
warnings.filterwarnings("ignore")


import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


st.set_page_config(page_title="Prognoza ceny złota — Yahoo", page_icon="💰", layout="centered")

#kod wstawiony przez Miłosz Furman
st.markdown("""
<style>
html, body, [data-testid="stApp"] {
  background: radial-gradient(1100px 600px at 10% 0%, #0f172a 0%, #0b1220 45%, #0a0f1a 100%) !important;
  color: #e5e7eb;
}
h1, h2, h3 { color: #f8fafc; }
.block-container { padding-top: 2rem; }
.stButton button {
  background: linear-gradient(135deg, #334155, #0ea5e9);
  color: #f8fafc; border: 0; border-radius: 12px; padding: .6rem 1rem; font-weight: 600;
  box-shadow: 0 6px 18px rgba(14,165,233,.25);
}
.stButton button:hover { filter: brightness(1.08); transform: translateY(-1px); }
.stSelectbox > div, .stDateInput > div {
  background: rgba(30,41,59,.6); border:1px solid rgba(148,163,184,.25); border-radius:12px;
}
[data-testid="stMetric"] {
  background: rgba(2,6,23,.55); border:1px solid rgba(148,163,184,.2); border-radius:16px; padding: 16px;
}
[data-testid="stExpander"] {
  background: rgba(2,6,23,.55); border:1px solid rgba(148,163,184,.2); border-radius:14px;
}
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

st.title("💰 Prognoza ceny złota — GoldPredict")
st.caption("Źródło: Yahoo Finance. Model: **SARIMA**.")

#kod wstawiony przez Wojciech Czarnecki (zmodyfikowany potem przez chatGPT)
colA, colB = st.columns([2, 1])
with colA:
    target_date = st.date_input("📅 Wybierz datę prognozy", value=dt.date.today() + relativedelta(years=1))
with colB:
    history_start = st.date_input(
        "⏳ Pobierz historię od",
        value=(dt.date.today() - relativedelta(years=10)),
        help="Początek zakresu danych historycznych"
    )

col1, col2, col3 = st.columns([2, 1.5, 1])
with col1:
    symbol_map = {
        "GC=F": "Cena rynkowa złota (USD/oz)",
        "GLD": "Fundusz inwestycyjny GLD (ETF w złoto)"
    }
    choice = st.selectbox("Źródło ceny", list(symbol_map.values()), index=0)
    symbol = next(k for k, v in symbol_map.items() if v == choice)
with col2:
    unit = st.selectbox(
        "Jednostka ceny",
        ["USD za 1 uncję (oz)", "USD za 1 gram"],
        index=0 if symbol == "GC=F" else 0,
        disabled=(symbol == "GLD"),
        help="Dla GLD jednostka zawsze: USD za 1 udział ETF"
    )
with col3:
    btn = st.button("Prognozuj")

# --- dodatkowe przełączniki ---
fast_mode = st.toggle("⚡ Tryb szybki (zalecany)", value=True,
                      help="Kilka sprawdzonych konfiguracji SARIMA zamiast dużej siatki (dużo szybciej).")
use_weekly = st.toggle("📉 Dane tygodniowe (szybciej)", value=True,
                       help="Resampluj do W-FRI (ok. 5× mniej punktów → 5× szybciej).")

# ============ FUNKCJE ============

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_series(sym: str, start_date: dt.date) -> pd.Series:
    """Pobierz serię dzienną, uporządkuj indeks (bez TZ), usuń NaN/inf, tylko wartości > 0."""
    df = yf.download(sym, start=start_date, interval="1d",
                     progress=False, auto_adjust=False, group_by="column")
    if df is None or df.empty:
        return pd.Series(dtype=float)

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].copy()

    # indeks = czysta data (bez TZ), posortowany
    s.index = pd.to_datetime(pd.Index(s.index)).tz_localize(None).normalize()
    s = s.sort_index().replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    s.name = sym
    return s

#kod wygenerowany przez ChatGPT. Jest to implementacja modelu SARIMA do prognozowania cen złota
def select_sarima_fast(series: pd.Series, fast: bool = True):
    """
    Szybki SARIMA: mało kandydatów, L-BFGS, maxiter=60.
    Sezonowość 52 dla tygodni, 252 dla dni roboczych. Historia przycięta (~12 lat).
    """
    inferred = pd.infer_freq(series.index)
    is_weekly = bool(inferred and inferred.startswith("W"))

    if is_weekly:
        series = series.asfreq("W-FRI").ffill().iloc[-600:]   # ~11–12 lat tygodni
        season_len = 52
    else:
        series = series.asfreq("B").ffill().iloc[-3000:]      # ~12 lat dni roboczych
        season_len = 252 if len(series) >= 400 else 0

    # UCZĘ NA LOG-POZIOMACH (nie na różnicach!)
    y = np.log(series)

    if fast:
        candidates = [
            ((1, 1, 1), (0, 1, 1, season_len)),
            ((2, 1, 1), (0, 1, 1, season_len)),
            ((1, 1, 0), (0, 1, 1, season_len)),
        ]
    else:
        ps, qs = range(0, 2), range(0, 2)
        candidates = [((p, 1, q), (0, 1, 1, season_len)) for p in ps for q in qs]

    best = {"aic": np.inf, "order": None, "seasonal_order": None, "model": None}

    def fit_one(order, sorder):
        if season_len == 0:
            sorder = (0, 0, 0, 0)
        # KLUCZ: simple_differencing=False → prognozy wracają do skali log-poziomów
        return SARIMAX(
            y,
            order=order,
            seasonal_order=sorder,
            simple_differencing=False,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, method="lbfgs", maxiter=60)

    for order, sorder in candidates:
        try:
            m = fit_one(order, sorder)
            if best["model"] is None or m.aic < best["aic"] - 2:
                best = {"aic": m.aic, "order": order, "seasonal_order": sorder, "model": m}
        except Exception:
            continue

    if best["model"] is None:
        m = SARIMAX(
            y, order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            simple_differencing=False,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, method="lbfgs", maxiter=60)
        return m, (1, 1, 1), (0, 0, 0, 0)

    return best["model"], best["order"], best["seasonal_order"]


def forecast_to_date(model, series: pd.Series, target_date: pd.Timestamp):
    """
    Prognoza do wybranej daty w skali POZIOMÓW (po exp()).
    Kroki liczone w częstotliwości serii (W-FRI lub B).
    """
    inferred = pd.infer_freq(series.index) or ""
    if inferred.startswith("W"):
        freq = "W-FRI"
        start = (series.index.max() + pd.offsets.Week(weekday=4))
        if start <= series.index.max():
            start += pd.offsets.Week(weekday=4)
        steps = len(pd.date_range(start=start, end=target_date, freq=freq))
        idx = pd.date_range(start=start, periods=max(steps, 1), freq=freq)
    else:
        freq = "B"
        start = series.index.max() + pd.offsets.BDay(1)
        steps = len(pd.bdate_range(start=start, end=target_date))
        idx = pd.bdate_range(start=start, periods=max(steps, 1))

    # jeżeli data docelowa nie leży w przyszłych okresach – zwróć ostatnią wartość
    if steps <= 0:
        val = float(series.iloc[-1])
        t_idx = pd.DatetimeIndex([pd.to_datetime(target_date)])
        return pd.Series([val], index=t_idx), pd.Series([val], index=t_idx), pd.Series([val], index=t_idx)

    # Prognoza: 'summary_frame()' → mean, mean_ci_lower, mean_ci_upper w SKALI ENDOGENU (tu: log-poziomy)
    fc = model.get_forecast(steps=int(steps))
    sf = fc.summary_frame(alpha=0.2)  # 80% CI
    mean_log = sf["mean"]
    lower_log = sf["mean_ci_lower"]
    upper_log = sf["mean_ci_upper"]

    # Zamiana z logów na poziomy cen
    mean = np.exp(mean_log.to_numpy())
    lower = np.exp(lower_log.to_numpy())
    upper = np.exp(upper_log.to_numpy())

    return pd.Series(mean, index=idx), pd.Series(lower, index=idx), pd.Series(upper, index=idx)

#Kod wstawiony przez zespół który tworzył projekt. jest to logika aplikacji
# ============ LOGIKA ============
if btn:
    try:
        if (target_date - history_start).days > 365 * 8:
            st.info("ℹ️ Długa historia — może wydłużyć trening i zwiększyć niepewność prognozy.")
        if history_start >= target_date:
            st.warning("Data początkowa historii nie może być późniejsza niż data prognozy. Koryguję o 2 lata wstecz.")
            history_start = target_date - relativedelta(years=2)

        with st.spinner(f"Pobieram {symbol} i trenuję model SARIMA..."):
            s_raw = fetch_series(symbol, history_start)
            if s_raw.empty:
                st.error(f"Brak danych dla {symbol}. Spróbuj innego zakresu dat.")
                st.stop()

            # wersja szeregu do modelu (jedna ścieżka danych)
            if use_weekly:
                s = s_raw.resample("W-FRI").last().ffill()
            else:
                s = s_raw.asfreq("B").ffill()

            # trenowanie SARIMA (szybkie)
            model, order, seasonal_order = select_sarima_fast(s, fast=fast_mode)

        tdate = pd.to_datetime(target_date)
        fc_mean, fc_low, fc_high = forecast_to_date(model, s, tdate)

        # Konwersja jednostki (oz -> g) tylko dla GC=F
        unit_label = "USD/oz"
        if symbol == "GC=F" and unit == "USD za 1 gram":
            factor = 31.1034768
            s = s / factor
            fc_mean = fc_mean / factor
            fc_low = fc_low / factor
            fc_high = fc_high / factor
            unit_label = "USD/g"
        elif symbol == "GLD":
            unit_label = "USD (GLD)"

        # Wykres
        fig = plt.figure(figsize=(7.5, 4.8))
        fig.patch.set_alpha(0)
        ax = plt.gca()
        hist_start = max(s.index.min(), s.index.max() - pd.DateOffset(years=5))
        s_hist = s.loc[hist_start:]
        ax.plot(s_hist.index, s_hist.values, label=f"Historia ({unit_label})", linewidth=2)
        ax.plot(fc_mean.index, fc_mean.values, label=f"Prognoza ({unit_label})", linewidth=2)
        ax.fill_between(fc_mean.index, fc_low.values, fc_high.values, alpha=0.18, label="Niepewność prognozy")
        ax.set_title(f"{symbol}: historia i prognoza — SARIMA{order}+{seasonal_order}")
        ax.set_xlabel("Data")
        ax.set_ylabel(unit_label)
        ax.grid(alpha=.15)
        ax.legend()
        st.pyplot(fig)

        # Wynik
        price_on_target = float(fc_mean.iloc[-1])
        st.subheader("📈 Wynik prognozy")
        st.metric(label=f"Prognoza na {tdate.date()}", value=f"{price_on_target:,.2f} {unit_label}")

        with st.expander("📊 Szczegóły techniczne"):
            st.json({
                "symbol": symbol,
                "jednostka": unit_label,
                "ostatnia_obserwacja": str(s.index.max().date()),
                "wartosc_ostatnia": float(s.iloc[-1]),
                "model_order": {"ARIMA": order, "SEASONAL": seasonal_order},
                "AIC": float(model.aic),
                "dlugosc_prognozy_okresy": int(len(fc_mean)),
                "czestotliwosc": "W-FRI" if use_weekly else "B",
            })

    except Exception as e:
        st.exception(e)
