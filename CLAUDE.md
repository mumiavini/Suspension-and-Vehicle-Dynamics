# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projeto

Motor de geometria de suspensão FSAE do time PUCPR Racing (carro FSAE26): analisa e otimiza hardpoints de suspensão duplo-A. Todo o projeto (código, comentários, UI, README) é em **pt-BR** — comunique-se com o usuário em português.

## Comandos

```powershell
# Rodar o app (venv do projeto em .venv\)
& .venv\Scripts\streamlit.exe run app.py

# Instalar dependências
& .venv\Scripts\python.exe -m pip install -r requirements.txt

# Sanity check rápido do motor (sem Streamlit)
& .venv\Scripts\python.exe -c "from geometry import Point3D; print(Point3D(1,2,3))"
```

Não há suíte de testes nem linter configurados. Para validar mudanças:

- **Smoke test**: `streamlit.testing.v1.AppTest.from_file("app.py", default_timeout=60)` — rodar com sessão vazia e com demo (`generate_template_dataframe()`), checar `at.exception`.
- **Verificação visual**: subir o app em porta alternativa (`--server.headless true --server.port 85xx`) e usar **Selenium + Edge headless** (`--headless=new`). O `msedge --screenshot` one-shot NÃO renderiza Streamlit. A sidebar é `section[data-testid='stSidebar']` (não `div`).
- No console Windows, defina `$env:PYTHONIOENCODING='utf-8'` antes de rodar scripts Python que imprimem emoji/acentos (o app usa emoji em labels).

## Arquitetura

Três camadas, com dependência só de cima para baixo:

1. **`geometry/`** — motor matemático puro (numpy/scipy, zero Streamlit).
   - `primitives.py`: `Point3D`/`Vector3D`/`Point2D` + interseções.
   - `model_3d.py`: `ControlArm`, `KingpinGeometry`, `SuspensionCorner`, `Vehicle` — KPIs estáticos (caster, KPI, scrub, trail, RC height…).
   - `solver_3d.py`: `KinematicSolver3D` — trata a manga como corpo rígido e resolve a posição para um estado `(heave, roll, rack)` via interseção de 3 esferas + `least_squares` (Levenberg-Marquardt). É o coração do cálculo dinâmico.
   - `solver_2d.py`: mecanismo de 4 barras na vista frontal Y-Z (usado para Roll Center).
2. **`analysis/`** — usa `geometry/`:
   - `io_hardpoints.py`: leitura/validação/escrita (csv/xlsx/json) e construção de `SuspensionCorner`/`Vehicle` a partir de DataFrames **polars**. Define `VALID_CORNERS` (FL/FR/RL/RR), `REQUIRED_POINTS_PER_CORNER` (10 pontos por corner) e `HardpointValidationError` — toda validação de entrada passa por aqui.
   - `sweeps.py`: `SweepRunner` (varreduras de heave/roll/steer sobre o solver 3D) → arrays numpy; KPIs dinâmicos (camber gain, bump steer, migração de RC) e plots Plotly.
   - `optimizer.py`: síntese via `scipy.optimize.differential_evolution` — `DesignTargets` (targets estáticos + dinâmicos), `HardpointBounds` (keep-out), `validate_against_targets()`.
   - `kpis.py`: KPIs de veículo completo (Ackermann, steer ratio, anti-dive, RC @ 1g, `build_full_report()`).
   - `viz3d.py`: visualização 3D Plotly (corner, veículo, animação).
3. **`app.py` + `ui/`** — camada Streamlit. `app.py` é só orquestração (config da página, tema, header, sidebar, `st.tabs`); cada uma das 5 abas (Inputs / Análise / Vista 3D / Síntese / Comparação) vive em `ui/tab_*.py` expondo `render()`. Apoio: `ui/theme.py` (presets `THEMES`, CSS, header), `ui/sidebar.py` e `ui/shared.py` (empty-state, builders seguros, cache de sweeps via `_geometry_signature()`).

### Fluxo de dados no app

Arquivo de hardpoints → DataFrame polars validado → `st.session_state["hardpoints_df"]` (+ `"hardpoints_source"`) é a **fonte única de verdade**; as abas constroem corners a partir dele a cada rerun. Sweeps são cacheados com `@st.cache_data` chaveado por `_geometry_signature()` em `ui/shared.py` (tupla hashable de todos os hardpoints) — se adicionar um hardpoint novo ao modelo, inclua-o na assinatura. A aba Síntese (`ui/tab_synthesis.py`) usa `st.fragment` e publica `last_optimization`, consumido pela aba Comparação. O editor manual (`ui/tab_inputs.py`) mantém estado próprio (`manual_hardpoints`) sincronizado por `manual_synced_source`.

### Convenções de domínio

- Eixos **SAE J670**: origem no centro do eixo dianteiro ao nível do solo; X+ = frente, Y+ = esquerda, Z+ = cima. Unidades: **mm** e **graus** (nunca polegadas/radianos).
- Escopo é **cinemática pura** (não calcula wheel rate, motion ratio, frequência, damping — ver README §2 e §15 antes de prometer um KPI novo).

## Convenções de UI (manter consistência)

- Largura: `width="stretch"` — **nunca** `use_container_width` (deprecado).
- Métricas com `border=True`; `st.segmented_control`/`st.pills` retornam `None` quando desmarcados — sempre trate o fallback.
- Empty-state padrão via `render_empty_state()` de `ui/shared.py` (botão de demo inline); badges de status no header.
- **Temas**: `.streamlit/config.toml` define só o boot; os presets selecionáveis estão no dict `THEMES` em `ui/theme.py`, aplicados via `st._config.set_option("theme.*")` + rerun forçado (config é global ao processo). Se mudar o `config.toml`, atualize `_DEFAULT_THEME` para o preset equivalente.

## Documentação

O `README.md` é a documentação do usuário final (conceitos físicos, formato de arquivo, tutorial, glossário) — atualize-o quando mudar comportamento visível. Atenção: o tour de abas do README (§8) está parcialmente defasado em relação à ordem real das abas em `app.py`.
