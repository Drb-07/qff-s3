"""
Interactive Real-Time Trotter Simulation Dashboard
Qiskit Fall Fest 2026 - S3 Problem

Run with: streamlit run interactive_app.py

Uses the shared engine in src/sweep.py (real Qiskit Aer noisy simulation), so the numbers
match the notebook and CLI for identical inputs.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))   # project root, so `import src...` works

from src.sweep import run_sweep, plot_sweep, load_default_processors

MAX_QUBITS = min(p['num_qubits'] for p in load_default_processors())   # 5 (Processor A)

# Page config
st.set_page_config(
    page_title="S3 Trotter Simulator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #38BDF8;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 700;
    }
    .stSlider > div > div > div {
        background: #334155;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_sweep(J, h, time, num_qubits, two_q_error, pbc, order):
    """Cached real Qiskit sweep (shared engine in src/sweep.py)."""
    return run_sweep(J=J, h=h, time=time, num_qubits=num_qubits, two_q_error=two_q_error,
                     periodic=pbc, order=order)


PRESETS = {
    "Default TFIM":         dict(J=1.0, h=0.5, T=1.0, N=4, err=1.0, pbc=False, order=1),
    "Critical Point (h=J)": dict(J=1.0, h=1.0, T=1.0, N=4, err=1.0, pbc=False, order=1),
    "Periodic Ring":        dict(J=1.0, h=0.5, T=1.0, N=4, err=1.0, pbc=True,  order=1),
    "2nd Order Strang":     dict(J=1.0, h=0.5, T=1.0, N=4, err=1.0, pbc=False, order=2),
    "Long Time":            dict(J=1.0, h=0.5, T=2.0, N=4, err=1.0, pbc=False, order=1),
    "Strong Field":         dict(J=0.5, h=1.5, T=1.0, N=4, err=1.0, pbc=False, order=1),
}
DEFAULTS = PRESETS["Default TFIM"]


def apply_preset():
    """Selectbox callback: push the preset values INTO the widgets so the sliders actually move."""
    p = PRESETS.get(st.session_state.get("preset"))
    if p:
        for k, v in p.items():
            st.session_state[f"w_{k}"] = v


def main():
    st.markdown('<p class="main-header">S3: Noise in Trotter Simulation</p>', unsafe_allow_html=True)
    st.markdown("**Interactive Real-Time Quantum Simulator | Qiskit Fall Fest 2026**")

    # Initialise widget state once
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(f"w_{k}", v)

    st.sidebar.header("🎯 Judge Presets")
    st.sidebar.selectbox("Quick Load", ["Custom"] + list(PRESETS), key="preset", on_change=apply_preset)

    st.sidebar.header("⚛️ Physics Parameters")
    J = st.sidebar.slider("Coupling J (ZZ)", 0.1, 2.5, step=0.1, key="w_J")
    h = st.sidebar.slider("Transverse Field h (X)", 0.1, 2.5, step=0.1, key="w_h")
    time = st.sidebar.slider("Evolution Time T", 0.2, 3.0, step=0.1, key="w_T")
    num_qubits = st.sidebar.select_slider("Qubits N", options=list(range(3, MAX_QUBITS + 1)), key="w_N")

    st.sidebar.header("🔧 Hardware Parameters")
    err_pct = st.sidebar.slider("2Q Error Rate (%)", 0.1, 4.0, step=0.1, key="w_err")

    st.sidebar.header("📐 Method Options")
    pbc = st.sidebar.checkbox("Periodic Boundary (PBC)", key="w_pbc", help="Closed-ring topology")
    order = st.sidebar.radio("Trotter Order", [1, 2], horizontal=True, key="w_order")
    use_mitigation = st.sidebar.checkbox("Show readout mitigation", value=True)

    current = dict(J=J, h=h, time=time, num_qubits=num_qubits,
                   two_q_error=err_pct / 100.0, pbc=pbc, order=order)

    if st.button("▶ Run Simulation", type="primary"):
        with st.spinner("Running Qiskit Aer simulation..."):
            try:
                st.session_state.result = cached_sweep(**current)
                st.session_state.result_inputs = current
            except Exception as e:                       # show the real error, never fake a 100% result
                st.session_state.result = None
                st.error(f"{type(e).__name__}: {e}")

    res = st.session_state.get("result")
    if res is None:
        st.info("👈 Pick a preset or adjust parameters, then click **Run Simulation**.")
        return

    if st.session_state.get("result_inputs") != current:
        st.warning("Inputs changed since the last run - click **Run Simulation** to update the results below.")

    # Everything below describes the run that was actually executed (res), not the live sliders
    df_A, df_B = res["df_A"], res["df_B"]
    opt_A, opt_B = res["opt_A"], res["opt_B"]
    ana = res["analytical"]["processor_B"]
    p = res["params"]
    opt_n = opt_B["optimal_n"]
    swaps_A = int(df_A.loc[df_A["trotter_steps"] == opt_n, "swap_count"].iloc[0])
    mit_at_opt = float(df_B.loc[df_B["trotter_steps"] == opt_n, "mitigated_error"].iloc[0])

    st.success(f"Exact energy: {res['exact_energy']:.6f}")
    st.markdown("### 📊 Key Metrics")
    c1, c2, c3, c4 = st.columns(4)

    def kpi(col, label, value, sub, color="#38BDF8"):
        col.markdown(f"""
        <div class="kpi-card" style="border-left-color: {color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color: {color};">{value}</div>
            <div style="color: #94A3B8; font-size: 12px;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    kpi(c1, "OPTIMAL STEP (n*)", opt_n, "Processor B sweet spot")
    kpi(c2, "MIN TOTAL ERROR", f"{opt_B['min_total_error']*100:.1f}%", f"Mitigated: {mit_at_opt*100:.1f}%", "#4ADE80")
    kpi(c3, "ANALYTICAL n*", f"{ana['analytical_n_continuous']:.2f}", "Formula: √(C_T/C_N)", "#C084FC")
    kpi(c4, "PROC A OVERHEAD", f"{swaps_A} SWAPs", "Routing penalty at n*", "#F87171")

    st.markdown("### 📈 Trotter Error vs Hardware Noise Tradeoff")
    fig = plot_sweep(res, show_mitigated=use_mitigation)
    st.pyplot(fig)
    plt.close(fig)

    if p["periodic"]:
        st.info("⚠️ **Periodic boundary active:** Closing the boundary ring on tree topologies requires routing SWAP gates. "
                "Processor A routes through its central hub, while Processor B routes across heavy-hex tree branches.")

    st.markdown("### 📋 Parametric Sweep Results")
    st.markdown("**Processor A (5Q T-Shape)**")
    st.dataframe(df_A, use_container_width=True, hide_index=True)
    st.markdown("**Processor B (7Q Heavy-Hex)**")
    st.dataframe(df_B, use_container_width=True, hide_index=True)

    col_csv, col_json = st.columns(2)
    tag = f"J{p['J']}_h{p['h']}_T{p['time']}_N{p['num_qubits']}"
    with col_csv:
        st.download_button("📥 Download CSV (Processor B)", df_B.to_csv(index=False).encode("utf-8"),
                           f"trotter_sweep_{tag}.csv", "text/csv")
    with col_json:
        summary = {
            "parameters": {k: v for k, v in p.items() if k != "steps"},
            "optimal_n_A": int(opt_A["optimal_n"]),
            "optimal_n_B": int(opt_B["optimal_n"]),
            "min_error_B": float(opt_B["min_total_error"]),
            "analytical_n_B": float(ana["analytical_n_continuous"]),
        }
        st.download_button("📥 Download JSON", json.dumps(summary, indent=2),
                           "trotter_summary.json", "application/json")


if __name__ == "__main__":
    main()
