"""
ui/
===
Camada de interface Streamlit, um módulo por aba:

    theme.py         : presets de tema, CSS de polimento e header.
    shared.py        : helpers compartilhados entre as abas (estado, caches).
    sidebar.py       : sidebar (carregar dados, setup do veículo, tema).
    tab_inputs.py    : ✏️  Inputs — editor manual com vistas 2D.
    tab_analysis.py  : 📊 Análise — ficha de setup completa + sweeps.
    tab_view3d.py    : 🌐 Vista 3D — visualização interativa.
    tab_synthesis.py : 🎯 Síntese — otimização global (engenharia reversa).
    tab_compare.py   : 🔄 Comparação — duas geometrias lado a lado.

Cada módulo de aba expõe uma função `render()` chamada pelo app.py dentro do
`with` do respectivo `st.tabs`.
"""
