"""
ui/
===
Streamlit interface layer, one module per tab:

    theme.py         : theme presets, polish CSS and header.
    shared.py        : helpers shared across tabs (state, caches).
    sidebar.py       : sidebar (load data, vehicle setup, theme).
    tab_inputs.py    : ✏️  Inputs — manual editor with 2D views.
    tab_analysis.py  : 📊 Analysis — complete setup sheet + sweeps.
    tab_view3d.py    : 🌐 View 3D — interactive visualization.
    tab_synthesis.py : 🎯 Synthesis — global optimization (reverse engineering).
    tab_compare.py   : 🔄 Comparison — two geometries side by side.

Each tab module exposes a `render()` function called by app.py inside the
`with` of its respective `st.tabs`.
"""
