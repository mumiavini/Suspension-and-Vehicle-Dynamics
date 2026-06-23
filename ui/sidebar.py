"""
ui/sidebar.py
=============
Sidebar do app: carregamento de hardpoints (upload / demo / template),
limpeza da sessão, setup do veículo e seletor de tema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
import streamlit as st

from analysis.io_hardpoints import (
    read_hardpoints,
    generate_template_dataframe,
    HardpointValidationError,
    VALID_CORNERS,
)
from ui.shared import load_demo_into_session
from ui.theme import THEMES


def _save_uploaded_to_tmp(uploaded_file) -> Path:
    """
    Salva um arquivo upload do Streamlit em diretório temporário e retorna o path.

    Usa `tempfile.gettempdir()` para portabilidade entre SOs:
        - Linux/macOS : /tmp
        - Windows     : C:\\Users\\<user>\\AppData\\Local\\Temp
    """
    import tempfile
    suffix = Path(uploaded_file.name).suffix
    tmp_path = Path(tempfile.gettempdir()) / f"_fsae_upload{suffix}"
    tmp_path.write_bytes(uploaded_file.read())
    return tmp_path


def render_sidebar() -> None:
    with st.sidebar:
        # ─── STATUS DA SESSÃO (sempre visível no topo) ───────────────────────
        if "hardpoints_df" in st.session_state:
            st.success(f"Em uso: **{st.session_state.get('hardpoints_source', '?')}**",
                       icon="📊")
        else:
            st.warning("Nenhum hardpoint carregado", icon="⚠️")

        st.markdown("### 1️⃣ Carregar dados")

        # ─── ETAPA 1: UPLOAD (apenas armazena, NÃO aplica ainda) ─────────────
        uploaded = st.file_uploader(
            "Arquivo de hardpoints",
            type=["xlsx", "csv", "json"],
            help="Colunas: corner, point, x_mm, y_mm, z_mm",
        )

        # Faz parse do arquivo só pra preview/validação, mas não aplica ainda
        pending_df: Optional[pl.DataFrame] = None
        pending_error: Optional[str] = None

        if uploaded is not None:
            try:
                tmp = _save_uploaded_to_tmp(uploaded)
                pending_df = read_hardpoints(tmp)
            except HardpointValidationError as exc:
                pending_error = f"Validação: {exc}"
            except Exception as exc:
                pending_error = str(exc)

        # ─── ETAPA 2: PREVIEW + BOTÃO APLICAR ────────────────────────────────
        if pending_error is not None:
            st.error(f"❌ {pending_error}")
        elif pending_df is not None:
            # Mostra um mini-preview do que foi carregado
            n_rows  = pending_df.height
            corners = sorted(pending_df["corner"].unique().to_list())
            st.success(f"✅ '{uploaded.name}' — {n_rows} pontos · corners: {', '.join(corners)}")

            # Botão explícito que aplica o arquivo (recalcula tudo)
            if st.button("🔄 **Aplicar arquivo**", type="primary", width="stretch",
                          help="Carrega esse arquivo no app e recalcula todos os KPIs e gráficos"):
                st.session_state["hardpoints_df"]     = pending_df
                st.session_state["hardpoints_source"] = uploaded.name
                st.rerun()   # força re-render imediato com o novo arquivo

        # ─── DEMO + TEMPLATE (atalhos) ───────────────────────────────────────
        st.caption("Ou use um atalho:")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("📋 Demo", width="stretch",
                          help="Carrega geometria FSAE realista de exemplo"):
                load_demo_into_session()
                st.rerun()

        with col_b:
            template_df = generate_template_dataframe()
            st.download_button("⬇️ Template", data=template_df.write_csv().encode(),
                                file_name="hardpoints_template.csv", mime="text/csv",
                                width="stretch",
                                help="Baixa o template em CSV para edição manual")

        if "hardpoints_df" in st.session_state:
            if st.button("🗑️ Limpar sessão", width="stretch",
                          help="Remove o arquivo atual da sessão"):
                for key in ["hardpoints_df", "hardpoints_source", "last_optimization",
                            "manual_hardpoints", "manual_synced_source"]:
                    st.session_state.pop(key, None)
                # Limpa também as chaves dos data_editors
                for cid in VALID_CORNERS:
                    st.session_state.pop(f"editor_{cid}", None)
                st.rerun()

        st.divider()
        st.markdown("### 2️⃣ Setup do veículo")
        st.session_state.setdefault("vehicle_setup", {
            "brake_bias": 0.60,
            "c_factor_mm": 100.0,
            "steering_wheel_lock_deg": 270.0,
        })
        vs = st.session_state["vehicle_setup"]
        vs["brake_bias"] = st.slider("Brake bias front", 0.0, 1.0, vs["brake_bias"],
                                       step=0.05, help="Fração na frente")
        with st.expander("🔧 Direção"):
            vs["c_factor_mm"] = st.number_input("c-factor (mm/rev)",
                                                  value=vs["c_factor_mm"], step=1.0,
                                                  help="2π × raio do pinhão")
            vs["steering_wheel_lock_deg"] = st.number_input("Lock total volante (°)",
                                                              value=vs["steering_wheel_lock_deg"],
                                                              step=10.0)

        st.divider()

        # ─── TEMA ────────────────────────────────────────────────────────────
        st.selectbox("🎨 Tema do app", list(THEMES), key="ui_theme",
                     help="Vale para esta sessão; o padrão de boot vem de "
                          ".streamlit/config.toml")
        # A config alterada só chega ao navegador no PRÓXIMO rerun, então
        # forçamos um quando a escolha muda (padrão set_option + rerun).
        if st.session_state["_theme_applied"] != st.session_state["ui_theme"]:
            st.session_state["_theme_applied"] = st.session_state["ui_theme"]
            st.rerun()
