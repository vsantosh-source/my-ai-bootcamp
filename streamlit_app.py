import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Social Signal Alpha Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

""

# Select End User
"""
# :material/query_stats: What would you like to do?

"""

""  # Add some space.

# Initialize session state for the selected section
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = "Get Started"

# Function to update state
def set_section(section):
    st.session_state.selected_section = section

# Create 3 equal-width columns
NUM_COLS = 3
col1, col2, col3 = st.columns(NUM_COLS)

# Column 1
with col1:
    # Use a button or a hidden radio to trigger state change
    if st.button("🚀 Retail Investor", use_container_width=True, key="btn1"):
        set_section("Retail Investor")
    st.markdown("<p style='text-align: center; color: gray; margin-top: -10px;'>iscover viral products organically.</p>", unsafe_allow_html=True)

# Column 2
with col2:
    if st.button("📊 Equity Analyst", use_container_width=True, key="btn2"):
        set_section("Equity Analyst")
    st.markdown("<p style='text-align: center; color: gray; margin-top: -10px;'>Perform deep fundamental research.</p>", unsafe_allow_html=True)

# Column 3
with col3:
    if st.button("⚙️ Portfolio Manager", use_container_width=True, key="btn3"):
        set_section("Portfolio Manager")
    st.markdown("<p style='text-align: center; color: gray; margin-top: -10px;'>Filter ideas and spot portfolio risks.</p>", unsafe_allow_html=True)
    
st.divider()

# Dynamic Content Below
st.subheader(f"Content for: {st.session_state.selected_section}")

if st.session_state.selected_section == "Retail Investor":
    """
    # :material/query_stats: Select stocks to compare
  
    """
    ""  # Add some space.

    cols = st.columns([1, 3])
    # Will declare right cell later to avoid showing it when no data.

    STOCKS = [
    "AAPL",
    "ABBV",
    "ACN",
    "ADBE",
    "ADP",
    "AMD",
    "AMGN",
    "AMT",
    "AMZN",
    "APD",
    "AVGO",
    "AXP",
    "BA",
    "BK",
    "BKNG",
    "BMY",
    "BRK.B",
    "BSX",
    "C",
    "CAT",
    "CI",
    "CL",
    "CMCSA",
    "COST",
    "CRM",
    "CSCO",
    "CVX",
    "DE",
    "DHR",
    "DIS",
    "DUK",
    "ELV",
    "EOG",
    "EQR",
    "FDX",
    "GD",
    "GE",
    "GILD",
    "GOOG",
    "GOOGL",
    "HD",
    "HON",
    "HUM",
    "IBM",
    "ICE",
    "INTC",
    "ISRG",
    "JNJ",
    "JPM",
    "KO",
    "LIN",
    "LLY",
    "LMT",
    "LOW",
    "MA",
    "MCD",
    "MDLZ",
    "META",
    "MMC",
    "MO",
    "MRK",
    "MSFT",
    "NEE",
    "NFLX",
    "NKE",
    "NOW",
    "NVDA",
    "ORCL",
    "PEP",
    "PFE",
    "PG",
    "PLD",
    "PM",
    "PSA",
    "REGN",
    "RTX",
    "SBUX",
    "SCHW",
    "SLB",
    "SO",
    "SPGI",
    "T",
    "TJX",
    "TMO",
    "TSLA",
    "TXN",
    "UNH",
    "UNP",
    "UPS",
    "V",
    "VZ",
    "WFC",
    "WM",
    "WMT",
    "XOM",
]

    DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META"]

    def stocks_to_str(stocks):
        return ",".join(stocks)
    if "tickers_input" not in st.session_state:
        st.session_state.tickers_input = st.query_params.get(
            "stocks", stocks_to_str(DEFAULT_STOCKS)
        ).split(",")

    # Callback to update query param when input changes
    def update_query_param():
        if st.session_state.tickers_input:
            st.query_params["stocks"] = stocks_to_str(st.session_state.tickers_input)
        else:
            st.query_params.pop("stocks", None)

    top_left_cell = cols[0].container(
        border=True, height="stretch", vertical_alignment="center"
    )

    with top_left_cell:
        # Selectbox for stock tickers
        tickers = st.multiselect(
            "Stock tickers",
            options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
            default=st.session_state.tickers_input,
            placeholder="Choose stocks to compare. Example: NVDA",
            accept_new_options=True,
        )

    # Time horizon selector
    horizon_map = {
      "1 Months": "1mo",
      "3 Months": "3mo",
      "6 Months": "6mo",
      "1 Year": "1y",
      "5 Years": "5y",
      "10 Years": "10y",
      "20 Years": "20y",
    }

    with top_left_cell:
        # Buttons for picking time horizon
        horizon = st.pills(
            "Time horizon",
            options=list(horizon_map.keys()),
            default="6 Months",
    )

    tickers = [t.upper() for t in tickers]

    # Update query param when text input changes
    if tickers:
        st.query_params["stocks"] = stocks_to_str(tickers)
    else:
        # Clear the param if input is empty
        st.query_params.pop("stocks", None)

    if not tickers:
        top_left_cell.info("Pick some stocks to compare", icon=":material/info:")
        st.stop()


    right_cell = cols[1].container(
        border=True, height="stretch", vertical_alignment="center"
    )


    @st.cache_resource(show_spinner=False, ttl="6h")
    def load_data(tickers, period):
        tickers_obj = yf.Tickers(tickers)
        data = tickers_obj.history(period=period)
        if data is None:
            raise RuntimeError("YFinance returned no data.")
        return data["Close"]

    # Load the data
    try:
        data = load_data(tickers, horizon_map[horizon])
    except yf.exceptions.YFRateLimitError as e:
        st.warning("YFinance is rate-limiting us :(\nTry again later.")
        load_data.clear()  # Remove the bad cache entry.
        st.stop()

    empty_columns = data.columns[data.isna().all()].tolist()

    if empty_columns:
        st.error(f"Error loading data for the tickers: {', '.join(empty_columns)}.")
        st.stop()

    # Normalize prices (start at 1)
    normalized = data.div(data.iloc[0])

    latest_norm_values = {normalized[ticker].iat[-1]: ticker for ticker in tickers}
    max_norm_value = max(latest_norm_values.items())
    min_norm_value = min(latest_norm_values.items())

    bottom_left_cell = cols[0].container(
        border=True, height="stretch", vertical_alignment="center"
    )

    with bottom_left_cell:
        cols = st.columns(2)
        cols[0].metric(
            "Best stock",
            max_norm_value[1],
            delta=f"{round(max_norm_value[0] * 100)}%",
            width="content",
        )
        cols[1].metric(
            "Worst stock",
            min_norm_value[1],
            delta=f"{round(min_norm_value[0] * 100)}%",
            width="content",
        )

    # Plot normalized prices
    with right_cell:
        st.altair_chart(
            alt.Chart(
                normalized.reset_index().melt(
                    id_vars=["Date"], var_name="Stock", value_name="Normalized price"
                )
            )
            .mark_line()
            .encode(
                alt.X("Date:T"),
                alt.Y("Normalized price:Q").scale(zero=False),
                alt.Color("Stock:N"),
            )
            .properties(height=400)
        )

    ""

elif st.session_state.selected_section == "Equity Analyst":
    st.write("Displaying your data analytics dashboard...")
    st.selectbox("Select Category", ["Apparel", "Home Goods", "Technology"]) 

elif st.session_state.selected_section == "Portfolio Manager":
    st.write("Adjust your system configuration...")
    st.checkbox("Enable Advanced Mode")
    st.selectbox("Select Theme", ["Light", "Dark"])   

