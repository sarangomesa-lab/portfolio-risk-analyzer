{\rtf1\ansi\ansicpg1252\cocoartf2870
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
import yfinance as yf\
import matplotlib.pyplot as plt\
import seaborn as sns\
from scipy import stats\
import warnings\
warnings.filterwarnings('ignore')\
\
st.set_page_config(page_title="\uc0\u55357 \u56522  Portfolio Risk Analyzer", layout="wide")\
st.title("\uc0\u55357 \u56522  Portfolio Risk Analyzer")\
st.write("**Educational Tool: Understand Your Portfolio Risk in Plain English**")\
\
# ============ SIDEBAR: PORTFOLIO BUILDER ============\
st.sidebar.header("\uc0\u55357 \u56520  Build Your Portfolio")\
\
# Stock input\
stocks_input = st.sidebar.text_input(\
    "Enter stock tickers (comma-separated)",\
    "AAPL,MSFT,JPM,GLD",\
    help="e.g., AAPL, MSFT, TSLA, SPY"\
)\
\
tickers = [t.strip().upper() for t in stocks_input.split(",")]\
\
# Weight sliders\
st.sidebar.write("**Set weights for each stock:**")\
weights_dict = \{\}\
total_weight = 0\
\
for ticker in tickers:\
    weight = st.sidebar.slider(\
        f"\{ticker\} weight (%)",\
        0, 100, 25,\
        key=f"weight_\{ticker\}"\
    )\
    weights_dict[ticker] = weight / 100\
    total_weight += weight\
\
# Check if weights sum to 100\
if abs(total_weight - 100) > 1:\
    st.sidebar.warning(f"\uc0\u9888 \u65039  Weights sum to \{total_weight\}% (should be 100%)")\
else:\
    st.sidebar.success(f"\uc0\u9989  Weights sum to 100%")\
\
# Data period\
period_days = st.sidebar.slider("Historical data (years)", 1, 5, 3) * 252\
\
# ============ LOAD DATA ============\
st.write("---")\
with st.spinner("Loading market data..."):\
    try:\
        # Fetch data\
        end_date = pd.Timestamp.now()\
        start_date = end_date - pd.Timedelta(days=period_days)\
        \
        data = yf.download(\
            tickers, \
            start=start_date, \
            end=end_date, \
            progress=False\
        )\
        \
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']\
        \
        if isinstance(prices, pd.Series):\
            prices = prices.to_frame()\
        \
        # Calculate returns\
        returns = prices.pct_change().dropna()\
        \
        # Normalize weights to tickers available\
        available_tickers = [t for t in tickers if t in prices.columns]\
        weights_array = np.array([weights_dict[t] for t in available_tickers])\
        weights_array = weights_array / weights_array.sum()\
        \
        # Portfolio returns\
        portfolio_returns = (returns[available_tickers] * weights_array).sum(axis=1)\
        \
        st.success("\uc0\u9989  Data loaded successfully!")\
        \
    except Exception as e:\
        st.error(f"\uc0\u10060  Error loading data: \{e\}")\
        st.stop()\
\
# ============ METRICS SECTION ============\
st.write("---")\
st.header("\uc0\u55357 \u56522  Risk Metrics")\
\
col1, col2, col3, col4 = st.columns(4)\
\
# Volatility\
volatility_annual = portfolio_returns.std() * np.sqrt(252)\
col1.metric(\
    "\uc0\u55357 \u56520  Annual Volatility",\
    f"\{volatility_annual*100:.2f\}%",\
    help="How much your portfolio typically swings (yearly)"\
)\
\
# Daily VaR\
confidence = 0.95\
var_daily = portfolio_returns.quantile(1 - confidence)\
col2.metric(\
    "\uc0\u9888 \u65039  Daily VaR (95%)",\
    f"\{var_daily*100:.2f\}%",\
    help="Worst expected loss on 95% of days"\
)\
\
# CVaR\
worst_days = portfolio_returns[portfolio_returns <= var_daily]\
cvar = worst_days.mean()\
col3.metric(\
    "\uc0\u55357 \u56485  CVaR (Tail Risk)",\
    f"\{cvar*100:.2f\}%",\
    help="Average loss on bad 5% of days"\
)\
\
# Max Drawdown\
cumulative = (1 + portfolio_returns).cumprod()\
running_max = cumulative.expanding().max()\
drawdown = (cumulative - running_max) / running_max\
max_dd = drawdown.min()\
col4.metric(\
    "\uc0\u55357 \u56521  Max Drawdown",\
    f"\{max_dd*100:.2f\}%",\
    help="Worst peak-to-trough loss"\
)\
\
# ============ VISUALIZATIONS ============\
st.write("---")\
st.header("\uc0\u55357 \u56520  Visualizations")\
\
tab1, tab2, tab3, tab4 = st.tabs([\
    "\uc0\u55357 \u56505  Volatility",\
    "\uc0\u9888 \u65039  Stress Test",\
    "\uc0\u55357 \u56522  Correlations",\
    "\uc0\u55357 \u56521  Drawdown"\
])\
\
# TAB 1: Volatility\
with tab1:\
    rolling_vol = portfolio_returns.rolling(30).std() * np.sqrt(252)\
    \
    fig, ax = plt.subplots(figsize=(12, 5))\
    ax.plot(rolling_vol.index, rolling_vol*100, linewidth=2, color='orange')\
    ax.fill_between(rolling_vol.index, rolling_vol*100, alpha=0.3, color='orange')\
    ax.set_ylabel('Volatility (%)', fontweight='bold')\
    ax.set_title('30-Day Rolling Volatility: When Does Risk Spike?', fontweight='bold')\
    ax.grid(alpha=0.3)\
    st.pyplot(fig)\
    \
    st.write(f"**Average volatility:** \{rolling_vol.mean()*100:.2f\}%")\
    st.write(f"**Peak volatility:** \{rolling_vol.max()*100:.2f\}% (most stressful period)")\
\
# TAB 2: Stress Test\
with tab2:\
    scenarios = \{\
        'Normal': 1.0,\
        '-5% Correction': 0.95,\
        '-10% Decline': 0.90,\
        '-20% Bear': 0.80,\
        '-30% Crash': 0.70,\
    \}\
    \
    losses = \{k: (v-1)*100 for k, v in scenarios.items()\}\
    \
    fig, ax = plt.subplots(figsize=(12, 5))\
    colors = ['green' if v >= 0 else 'red' for v in losses.values()]\
    ax.barh(list(losses.keys()), list(losses.values()), color=colors, alpha=0.7)\
    ax.set_xlabel('Portfolio Return (%)', fontweight='bold')\
    ax.set_title('\uc0\u55357 \u56485  Stress Test: How Your Portfolio Handles Market Crashes', fontweight='bold')\
    ax.grid(axis='x', alpha=0.3)\
    st.pyplot(fig)\
    \
    st.write("\uc0\u55357 \u56481  **Key Insight:** In extreme crashes, your downside is proportional to market decline")\
\
# TAB 3: Correlations\
with tab3:\
    corr = returns[available_tickers].corr()\
    \
    fig, ax = plt.subplots(figsize=(10, 8))\
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, \
                vmin=-1, vmax=1, square=True, ax=ax, cbar_kws=\{"shrink": 0.8\})\
    ax.set_title('Stock Correlations: Lower = Better Diversification', fontweight='bold')\
    st.pyplot(fig)\
    \
    st.write("\uc0\u55357 \u56524  **Correlations < 0.5** = Good diversification (stocks move differently)")\
\
# TAB 4: Drawdown\
with tab4:\
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))\
    \
    ax1.plot((cumulative-1)*100, linewidth=2, color='blue')\
    ax1.fill_between(cumulative.index, (cumulative-1)*100, alpha=0.3, color='blue')\
    ax1.set_ylabel('Return (%)', fontweight='bold')\
    ax1.set_title('Portfolio Growth Over Time', fontweight='bold')\
    ax1.grid(alpha=0.3)\
    \
    ax2.plot(drawdown.index, drawdown*100, linewidth=2, color='red')\
    ax2.fill_between(drawdown.index, drawdown*100, alpha=0.3, color='red')\
    ax2.set_ylabel('Drawdown (%)', fontweight='bold')\
    ax2.set_xlabel('Date', fontweight='bold')\
    ax2.set_title('Drawdown: Distance from Peak', fontweight='bold')\
    ax2.grid(alpha=0.3)\
    \
    st.pyplot(fig)\
    \
    st.write(f"\uc0\u55357 \u56628  **Maximum Drawdown:** \{max_dd*100:.2f\}% on \{drawdown.idxmin().strftime('%Y-%m-%d')\}")\
\
# ============ EXPLANATION SECTION ============\
st.write("---")\
st.header("\uc0\u55357 \u56481  Plain English Explanations")\
\
with st.expander("What is Volatility?"):\
    st.write(f"""\
    **Volatility = how much your portfolio swings up and down**\
    \
    - Low volatility (5%): Calm, steady, boring returns\
    - Medium volatility (15%): Normal ups and downs\
    - High volatility (30%+): Wild swings, exciting and scary\
    \
    Your portfolio: **\{volatility_annual*100:.2f\}%** volatility = Moderate risk\
    """)\
\
with st.expander("What is VaR?"):\
    st.write(f"""\
    **VaR = "Value at Risk" - Your worst expected loss**\
    \
    Your Daily VaR (95%): **\{var_daily*100:.2f\}%**\
    \
    Translation: "95% of the time, I won't lose more than \{abs(var_daily*100):.2f\}% in one day"\
    \
    On a $100,000 portfolio, that's about $\{abs(var_daily)*100000:,.0f\} loss\
    """)\
\
with st.expander("What is CVaR?"):\
    st.write(f"""\
    **CVaR = "Conditional VaR" - Your average loss on BAD days**\
    \
    Your CVaR (95%): **\{cvar*100:.2f\}%**\
    \
    Translation: "On the worst 5% of days (market crashes), I lose \{abs(cvar*100):.2f\}% on average"\
    \
    **CVaR > VaR** means tail risk is real - bad days are REALLY bad\
    """)\
\
with st.expander("What is Drawdown?"):\
    st.write(f"""\
    **Drawdown = How far your portfolio fell from its peak**\
    \
    Your Max Drawdown:}