"""DemandCast local review dashboard. Run with: streamlit run dashboard/app.py"""

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parent
if str(DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DIRECTORY))
from dashboard_data import BIN_ORDER, MEASURES, forecast_contract, load_bundle, load_forecasts, series_metrics, weighted_metrics


BLUE, ORANGE, DARK = "#2563eb", "#ea8a24", "#172b4d"


def style(figure, height=350):
    figure.update_layout(template="plotly_white", height=height, margin=dict(l=15, r=15, t=35, b=25),
                         font=dict(family="Arial", size=13, color=DARK),
                         legend=dict(orientation="h", y=1.13, x=0), hovermode="x unified")
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#e8edf4", zeroline=False)
    return figure


def show_chart(figure, key, description):
    st.plotly_chart(style(figure), width="stretch", key=key, alt=description,
                    config={"displaylogo": False, "scrollZoom": False})


def download_table(frame, filename, key):
    st.download_button("Download current view", frame.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv", key=key)


def display_table(frame):
    frame = frame.copy()
    for column in ["coverage", "lower_miss_rate", "upper_miss_rate", "median_outside_rate", "wape"]:
        if column in frame:
            frame[column] *= 100
            frame = frame.rename(columns={column: column+"_pct"})
    return frame


@st.cache_data(show_spinner=False, max_entries=4)
def cached_forecasts(data, contract_json):
    return load_forecasts(data, json.loads(contract_json))


def interval_view(frames, receipts):
    st.header("Interval review")
    cuts = sorted(frames["folds"].cutoff.unique())
    chosen = st.multiselect("Evaluation cutoff", cuts, default=cuts, key="interval_cutoffs")
    if not chosen:
        st.info("Select at least one evaluation window.")
        return
    subset = frames["folds"][frames["folds"].cutoff.isin(chosen)]
    baseline = weighted_metrics(subset[subset.method == "uncalibrated"])
    calibrated = weighted_metrics(subset[subset.method == "temporal_cqr"])
    st.caption(f"Validation replay. {baseline['n_observations']:,} observations per method across {len(chosen)} selected windows.")
    cols = st.columns(4)
    cols[0].metric("Interval coverage", f"{baseline['coverage']:.2%}", border=True,
                   help="Inclusive observed coverage. The nominal 80% target applies to the pooled population, not every group.")
    cols[1].metric("Mean width", f"{baseline['mean_width']:.3f}", border=True, help="Sales units per daily series forecast.")
    cols[2].metric("Interval score", f"{baseline['interval_score_80']:.3f}", border=True,
                   help="Width plus 10 times the distance outside the interval. Lower is better.")
    cols[3].metric("Observations", f"{baseline['n_observations']:,}", border=True)
    if all(abs(baseline[k]-calibrated[k]) < 1e-12 for k in MEASURES):
        st.caption("The rolling calibration and q10-q90 envelope have identical metrics in the selected windows.")
    left, right = st.columns(2)
    with left:
        figure = go.Figure()
        for method, label, color in [("uncalibrated", "q10-q90 envelope", BLUE), ("temporal_cqr", "Temporal calibration", ORANGE)]:
            part = subset[subset.method == method].sort_values("cutoff")
            figure.add_trace(go.Scatter(x=part.cutoff, y=100*part.coverage, mode="lines+markers", name=label,
                                       line=dict(color=color, dash="dash" if method == "temporal_cqr" else "solid")))
        figure.add_hline(y=80, line_dash="dot", line_color="#8492a6", annotation_text="Nominal target 80%")
        figure.update_layout(title="Coverage by forecast window")
        figure.update_yaxes(range=[0, 100], ticksuffix="%")
        show_chart(figure, "coverage_history", "Inclusive interval coverage by validation cutoff for both methods.")
    with right:
        figure = go.Figure(go.Bar(x=["Below lower bound", "Above upper bound"],
                                 y=[100*baseline["lower_miss_rate"], 100*baseline["upper_miss_rate"]], marker_color=BLUE))
        figure.update_layout(title="Where observations leave the interval")
        figure.update_yaxes(range=[0, 100], ticksuffix="%")
        show_chart(figure, "tail_misses", "Observed lower and upper tail miss rates in the selected windows.")
    comparison = pd.DataFrame([{"method": label, **values} for label, values in
                              [("q10-q90 envelope", baseline), ("Temporal calibration", calibrated)]])
    with st.expander("Method comparison and calibration audit"):
        st.dataframe(display_table(comparison), hide_index=True, width="stretch")
        st.dataframe(frames["calibration"][frames["calibration"].evaluation_cutoff.isin(chosen)],
                     hide_index=True, width="stretch")
        st.caption("Only complete earlier forecast windows supply correction scores. No split-conformal coverage guarantee is claimed.")
    download_table(display_table(comparison), "interval_comparison_selected_windows.csv", "download_intervals")


def segment_view(frames):
    st.header("Demand segments")
    available = [value for value in BIN_ORDER if value in set(frames["segments"].predicted_mean_bin)]
    chosen = st.multiselect("Predicted mean sales per day", available, default=available, key="mean_bins")
    if not chosen:
        st.info("Select at least one demand segment.")
        return
    part = frames["segments"][(frames["segments"].method == "uncalibrated") &
                              frames["segments"].predicted_mean_bin.isin(chosen)].copy()
    part["predicted_mean_bin"] = pd.Categorical(part.predicted_mean_bin, BIN_ORDER, ordered=True)
    part = part.sort_values("predicted_mean_bin")
    summary = weighted_metrics(part)
    st.caption(f"Both E03 evaluation windows. {summary['n_observations']:,} observations in the selected segments. Groups use the Poisson forecast available at prediction time.")
    cols = st.columns(3)
    cols[0].metric("Selected coverage", f"{summary['coverage']:.2%}", border=True)
    cols[1].metric("Selected mean width", f"{summary['mean_width']:.3f}", border=True)
    cols[2].metric("Selected interval score", f"{summary['interval_score_80']:.3f}", border=True)
    figure = go.Figure(go.Bar(x=part.predicted_mean_bin.astype(str), y=100*part.coverage,
                             marker_color=BLUE, text=[f"{v:.1%}" for v in part.coverage], textposition="outside"))
    figure.update_layout(title="Coverage by predicted demand")
    figure.update_yaxes(range=[0, 100], ticksuffix="%")
    show_chart(figure, "segment_coverage", "Interval coverage for each selected predicted-mean sales group.")
    st.caption("Interval scores and widths use sales units. Larger-demand groups naturally have larger absolute scales; this table does not establish relative model quality.")
    display = part[["predicted_mean_bin", "n_observations", "coverage", "lower_miss_rate", "upper_miss_rate", "mean_width", "interval_score_80"]]
    st.dataframe(display_table(display), hide_index=True, width="stretch")
    download_table(display_table(display), "demand_segments_selected.csv", "download_segments")


def point_view(frames, receipts):
    st.header("Point models")
    points = frames["points"].sort_values("mae").copy()
    count = int(points.n_observations.iloc[0])
    windows = receipts["e02"]["config"]["n_windows"]
    st.caption(f"E02 validation. {count:,} observations per model across {windows} windows. Interval replay elsewhere uses two windows.")
    st.caption("L1 targets the conditional median and Poisson targets the conditional mean. A lower MAE does not establish an unbiased total-sales estimate.")
    left, right = st.columns(2)
    with left:
        figure = go.Figure(go.Bar(y=points.model_label, x=points.mae, orientation="h", marker_color=BLUE))
        figure.update_layout(title="Mean absolute error")
        figure.update_xaxes(title="Sales units", rangemode="tozero")
        figure.update_yaxes(autorange="reversed")
        show_chart(figure, "point_mae", "Validation mean absolute error of the four frozen point models.")
    with right:
        figure = go.Figure(go.Bar(y=points.model_label, x=points.volume_bias_pct, orientation="h", marker_color=ORANGE))
        figure.add_vline(x=0, line_color="#8492a6")
        figure.update_layout(title="Aggregate predicted-sales bias")
        figure.update_xaxes(ticksuffix="%")
        figure.update_yaxes(autorange="reversed")
        show_chart(figure, "volume_bias", "Percentage difference between total predicted and actual sales for each point model.")
    display = points[["model_label", "mae", "wape", "volume_bias_pct", "n_observations"]]
    st.dataframe(display_table(display), hide_index=True, width="stretch")
    download_table(display_table(display), "point_model_comparison.csv", "download_points")


def series_view(frames, receipts):
    st.header("Series explorer")
    st.caption("E02 validation forecasts and subsequently observed sales. Each chart shows one series and one forecast origin.")
    default_path = ROOT/"outputs/interval-validation/interval_predictions.csv"
    path_value = st.text_input("Forecast CSV path", str(default_path), key="forecast_path")
    uploaded = st.file_uploader("Or open interval_predictions.csv", type=["csv"], key="forecast_upload")
    if uploaded is not None:
        payload = uploaded.getvalue()
    elif Path(path_value).is_file():
        payload = Path(path_value).read_bytes()
    else:
        st.info("The aggregate views are ready. To explore a series, open the interval_predictions.csv generated by E02.")
        return
    contract = forecast_contract(receipts, frames["calibration"])
    try:
        forecasts = cached_forecasts(payload, json.dumps(contract, sort_keys=True))
    except (ValueError, OSError, KeyError) as error:
        st.error(str(error))
        return
    category = st.selectbox("Category", sorted(forecasts.category.unique()), key="series_category")
    ids = sorted(forecasts.loc[forecasts.category == category, "unique_id"].unique())
    unique_id = st.selectbox("Series", ids, key="series_id")
    cutoffs = sorted(forecasts.cutoff.dt.strftime("%Y-%m-%d").unique())
    cutoff = st.selectbox("Forecast cutoff", cutoffs, index=len(cutoffs)-1, key="series_cutoff")
    bounds = st.slider("Forecast horizons", 1, contract["horizon"], (1, contract["horizon"]), key="series_horizons")
    chosen = forecasts[(forecasts.unique_id == unique_id) & (forecasts.cutoff == pd.Timestamp(cutoff)) &
                       forecasts.horizon.between(*bounds)].sort_values("ds")
    metrics = series_metrics(chosen)
    cols = st.columns(4)
    cols[0].metric("Median forecast MAE", f"{metrics['median_mae']:.3f}", border=True)
    cols[1].metric("Mean forecast MAE", f"{metrics['mean_mae']:.3f}", border=True)
    cols[2].metric("Interval coverage", f"{metrics['coverage']:.1%}", border=True)
    cols[3].metric("Observed sales", f"{metrics['actual_total']:,.0f}", border=True)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=chosen.ds, y=chosen.lower, mode="lines", line=dict(width=0), name="Lower bound", showlegend=False))
    figure.add_trace(go.Scatter(x=chosen.ds, y=chosen.upper, mode="lines", line=dict(width=0), fill="tonexty",
                               fillcolor="rgba(37,99,235,0.14)", name="q10-q90 envelope"))
    for column, label, color, dash in [("y", "Observed sales", DARK, "solid"),
                                      ("mean_estimate", "Poisson mean", BLUE, "solid"),
                                      ("median_estimate", "L1 median", ORANGE, "dash")]:
        figure.add_trace(go.Scatter(x=chosen.ds, y=chosen[column], name=label, mode="lines+markers",
                                   line=dict(color=color, dash=dash), marker=dict(size=4)))
    figure.update_layout(title=unique_id)
    figure.update_yaxes(title="Daily unit sales", rangemode="tozero")
    show_chart(figure, "series_forecast", "Observed daily sales, conditional mean and median forecasts, and the q10-q90 interval for the selected series.")
    st.caption("The shaded envelope has nominal 80% coverage. Its displayed coverage is empirical for the selected daily observations; it is not an inventory service level or an interval for summed sales.")
    with st.expander("Forecast rows"):
        st.dataframe(chosen, hide_index=True, width="stretch")
    download_table(chosen.drop(columns="category"), "selected_series_forecast.csv", "download_series")


def main():
    st.set_page_config(page_title="DemandCast", layout="wide")
    st.title("DemandCast")
    project_ready = (ROOT/"outputs/calibration-validation/run.json").is_file() and (ROOT/"outputs/interval-validation/run.json").is_file()
    sources = ["Project outputs", "Bundled validation results"] if project_ready else ["Bundled validation results", "Project outputs"]
    source = st.sidebar.selectbox("Results", sources, key="result_source")
    view = st.sidebar.radio("Review", ["Intervals", "Demand segments", "Point models", "Series explorer"], key="review_view")
    try:
        frames, receipts, provenance = load_bundle(ROOT, "project" if source == "Project outputs" else "bundled")
    except (ValueError, FileNotFoundError, KeyError) as error:
        st.error(f"Cannot open these results: {error}")
        st.info("Choose the bundled validation results or complete E02 and E03 in this project.")
        return
    st.sidebar.caption("Validation demo. Final independent-test performance has not been established.")
    if view == "Intervals":
        interval_view(frames, receipts)
    elif view == "Demand segments":
        segment_view(frames)
    elif view == "Point models":
        point_view(frames, receipts)
    else:
        series_view(frames, receipts)
    with st.expander("Sources and measurement definitions"):
        st.dataframe(pd.DataFrame(provenance), hide_index=True, width="stretch")
        st.markdown("Coverage counts observations inside both inclusive bounds. Rates are pooled with observation-count weights. "
                    "Interval score equals width plus 10 times the distance outside the bounds. WAPE is total absolute error "
                    "divided by total observed sales. Volume bias compares total predicted sales with total observed sales.")
        st.caption("Temporal calibration uses complete earlier windows and horizon bands. Dependent panel/time-series scores do not establish a distribution-free coverage guarantee.")


if __name__ == "__main__":
    main()
