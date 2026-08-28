"""
ui/
===
Streamlit interface layer, one module per tab:

    theme.py         : theme presets, polish CSS and header.
    shared.py        : helpers shared across tabs (state, caches).
    sidebar.py       : sidebar (load data, vehicle setup, theme).
    tab_inputs.py    : ✏️  Inputs — manual editor with 2D views.
    tab_vdcore.py    : 📊 Analysis — setup sheet, sweeps and the Altair column.
                       Every geometry row comes from vdcore's DWSolver. This
                       absorbed the old tab_analysis.py, whose static camber,
                       right-side scrub, mechanical trail and every sweep chart
                       were wrong (they ran the strut-to-midpoint solver).
    tab_view3d.py    : 🌐 View 3D — interactive visualization.
    tab_synthesis.py : 🎯 Synthesis — global optimization (reverse engineering).
    tab_compare.py   : 🔄 Comparison — two geometries side by side.

Each tab module exposes a `render()` function called by app.py inside the
`with` of its respective `st.tabs`.
"""
