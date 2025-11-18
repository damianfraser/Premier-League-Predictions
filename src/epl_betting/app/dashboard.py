import streamlit as st
import pandas as pd

from ..config import RESULTS_DIR


def load_upcoming_fixtures() -> pd.DataFrame:
    """
    Placeholder: later this will load upcoming fixtures enriched
    with model/market/posterior probabilities and edges.
    """
    path = RESULTS_DIR / "upcoming_fixtures.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    st.title("Premier League Value Betting Model")

    page = st.sidebar.radio("Page", ["Upcoming Fixtures", "Match Detail", "Performance"])

    if page == "Upcoming Fixtures":
        st.header("Upcoming Fixtures")
        fixtures = load_upcoming_fixtures()
        if fixtures.empty:
            st.info("No upcoming fixtures data found yet.")
        else:
            st.dataframe(fixtures)

    elif page == "Match Detail":
        st.header("Match Detail")
        st.info("Match detail view not implemented yet.")

    elif page == "Performance":
        st.header("Performance & Backtest")
        st.info("Performance view not implemented yet.")


if __name__ == "__main__":
    main()
