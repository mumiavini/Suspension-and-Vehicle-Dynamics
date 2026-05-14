# 🏎️ FSAE Suspension Geometry Engine

Software em Python para **projetar, analisar e otimizar** a geometria de suspensão de um carro de Fórmula SAE.

Funciona em dois modos:

- **Análise** — você tem um carro (ou montagem CAD) e quer entender como ele se comporta
- **Síntese** — você define metas de comportamento e o software encontra os hardpoints que as atendem

Pode ser usado via **interface gráfica Streamlit** (recomendado) ou via **scripts Python** (para automação).

---

## 📑 Índice

1. [Para quem é](#1-para-quem-é)
2. [O que ele calcula](#2-o-que-ele-calcula)
3. [Conceitos físicos](#3-conceitos-físicos)
4. [Convenções e unidades](#4-convenções-e-unidades)
5. [Instalação](#5-instalação)
6. [Primeiros passos (5 min)](#6-primeiros-passos-5-min)
7. [Tutorial 1: Analisar geometria existente](#7-tutorial-1-analisar-geometria-existente)
8. [Tutorial 2: Síntese a partir de targets](#8-tutorial-2-síntese-a-partir-de-targets)
9. [Tutorial 3: Comparar duas geometrias](#9-tutorial-3-comparar-duas-geometrias)
10. [Workflow SolidWorks ↔ Software](#10-workflow-solidworks--software)
11. [Formato do arquivo de hardpoints](#11-formato-do-arquivo-de-hardpoints)
12. [Estrutura do projeto](#12-estrutura-do-projeto)
13. [Uso via Python (scripts)](#13-uso-via-python-scripts)
14. [KPIs implementados](#14-kpis-implementados)
15. [Troubleshooting](#15-troubleshooting)
16. [Limitações conhecidas](#16-limitações-conhecidas)
17. [Glossário](#17-glossário)

---

## 1. Para quem é

Foi pensado para:

- **Engenheiro de suspensão de FSAE** que precisa iterar geometrias rápido
- **Estudante** querendo entender o efeito de cada hardpoint
- **Time de Fórmula SAE** que quer documentar/versionar as escolhas
- **Quem usa SolidWorks** e quer um motor de cálculo desacoplado do CAD

O software **NÃO** substitui:
- Análise estrutural (precisa de FEA)
- Simulação de dinâmica veicular completa (use OptimumLap, CarMaker)
- Validação experimental (k-rig, testes em pista)

---

## 2. O que ele calcula

### Parâmetros estáticos (sem movimento)

- **Caster**, **KPI** (Kingpin Inclination)
- **Scrub Radius**, **Mechanical Trail**
- **Roll Center Height**
- **Wheelbase**, **Track Width**
- **Ackermann (%)**
- **Steer Arm Length**, **Steer Ratio**, **C-factor**
- **Anti-dive / Anti-squat (%)**

### Parâmetros dinâmicos (durante movimento)

- **Camber gain** (°/mm de heave)
- **Ride Camber** (°/m)
- **Roll Camber** (°/° de roll)
- **Bump Steer** (Δtoe/mm)
- **Migração do Roll Center** (Y e Z durante movimento)
- **Roll Center sob 1g lateral**
- Sweeps completos em **heave**, **roll** e **steer**

### Síntese (engenharia reversa)

- Otimização global (`scipy.differential_evolution`)
- Targets estáticos + dinâmicos combinados
- Bounding boxes para keep-out zones
- Validação automática contra metas
- Export para CSV/Excel/JSON

### Parâmetros que NÃO calcula (exigem inputs externos)

- Wheel rate, Roll rate, Sprung mass natural frequency (precisam de rigidez de mola)
- Motion Ratio (precisa de pushrod/rocker)
- Damping (precisa de dados do amortecedor)
- Camber estático construtivo (depende de como a manga foi fabricada — é input do usuário)

---

## 3. Conceitos físicos

### 3.1 Hardpoints

**Hardpoints** são os pontos de pivô/fixação da suspensão. São o "DNA geométrico" do carro — defini-los é, na prática, definir como o carro vai se comportar.

```
┌─────── CHASSI ───────┐
│                       │
│  UCA_IN_FRONT  ●─────────●  UCA_OUT (na manga)
│  UCA_IN_REAR   ●──────╱
│                       │ ╲
│  LCA_IN_FRONT  ●──────╲    ● (manga de eixo)
│  LCA_IN_REAR   ●─────────●  LCA_OUT
│                       │
│  TIE_ROD_IN    ●─────────●  TIE_ROD_OUT
└───────────────────────┘
                            ●  WHEEL_CENTER (centro de roda)
                            │
                            ●  CONTACT_PATCH (contato pneu-solo)
```

Cada **corner** (FL, FR, RL, RR) tem **10 hardpoints**:
- 3 do **UCA** (braço superior): 2 inboard + 1 outboard
- 3 do **LCA** (braço inferior): 2 inboard + 1 outboard
- 2 do **tie-rod**: 1 inboard + 1 outboard
- **WHEEL_CENTER** e **CONTACT_PATCH**

### 3.2 Parâmetros estáticos típicos FSAE

| Parâmetro | Valor típico | O que afeta |
|---|---|---|
| **Caster** | 3° a 7° | Auto-centragem do volante |
| **KPI** | 5° a 10° | Variação de camber em steer; scrub |
| **Camber** | −1° a −3° | Grip em curva |
| **Scrub Radius** | −10 a +30 mm | Esforço no volante, estabilidade |
| **Trail Mecânico** | 5 a 25 mm | Sensibilidade de direção |
| **RC Height** | 20 a 80 mm | Rolagem do chassi, transferência de carga |

### 3.3 Parâmetros dinâmicos importantes

| Parâmetro | Meta típica |
|---|---|
| **Camber Gain** | −0.015 a −0.025 °/mm |
| **Bump Steer** | < 0.005 °/mm em módulo |
| **RC ΔY** | < 30 mm de migração lateral |
| **Roll Camber** | −0.5 a −1.5 °/° |
| **Anti-dive** | 0 a 30% |
| **Ackermann** | 30% a 100% |

### 3.4 Por que esses parâmetros são acoplados

⚠️ Todos esses parâmetros estão **acoplados geometricamente**. Você não consegue mexer em um sem afetar os outros:

- Aumentar caster → tende a aumentar trail mecânico
- Aumentar KPI → reduz scrub radius
- Subir o LCA inboard → afeta camber gain E roll center juntos

Por isso a **otimização global** (aba Síntese) é mais eficaz que ajustes manuais tentativa-e-erro.

---

## 4. Convenções e unidades

### 4.1 Sistema de eixos (SAE J670)

```
                 Z (cima, positivo)
                 ▲
                 │
                 │
                 ●─────────► Y (esquerda do veículo, positivo)
                /
               /
              ▼
              X (frente do veículo, positivo)
```

- **Origem:** centro do eixo dianteiro, no nível do solo
- **X+** = frente · **Y+** = esquerda · **Z+** = cima

> ⚠️ Se sua origem do CAD não for SAE, **transforme** as coordenadas antes de inserir.

### 4.2 Sinais

| Parâmetro | + significa |
|---|---|
| **Camber** | Topo da roda para FORA (negativo = corrida) |
| **Caster** | Topo do pino ATRÁS da base |
| **KPI** | Topo do pino PARA DENTRO do veículo |
| **Scrub** | Intercepto do pino no solo PARA DENTRO do contato |
| **Heave** | Bump (roda sobe relativa ao chassi) |
| **Roll** | Chassi rola para a DIREITA |
| **Rack** | Rack desloca para a ESQUERDA |

### 4.3 Unidades

- **Comprimentos:** sempre em **mm**
- **Ângulos:** sempre em **graus (°)**
- **Camber gain:** **°/mm**
- **Ride camber:** **°/m**

Nada de polegadas ou radianos. Converta antes (1 in = 25.4 mm).

---

## 5. Instalação

### 5.1 Pré-requisitos

- Python 3.10 ou superior ([download](https://www.python.org/downloads/))
- ~500 MB de espaço (numpy, scipy, plotly, streamlit)
- Windows, macOS ou Linux

### 5.2 Passo-a-passo

**1. Baixe os arquivos** (pasta `fsae_suspension_clean/`)

**2. Abra terminal na pasta:**
- Windows: `Shift + clique direito` → "Abrir terminal aqui"
- macOS: arraste pasta para o Terminal

**3. (Recomendado) Crie ambiente virtual:**
```bash
python -m venv venv
```
Ative:
- Windows: `venv\Scripts\activate`
- macOS/Linux: `source venv/bin/activate`

**4. Instale dependências:**
```bash
pip install numpy scipy plotly polars streamlit openpyxl fastexcel
```

Ou usando `requirements.txt`:
```
numpy>=1.24
scipy>=1.10
plotly>=5.0
polars>=0.20
openpyxl>=3.0
fastexcel>=0.10
streamlit>=1.30
```
```bash
pip install -r requirements.txt
```

**5. Teste:**
```bash
python -c "from geometry import Point3D; print(Point3D(1,2,3))"
```
Se imprimir o ponto, está OK.

### 5.3 Rodando

**Interface gráfica (recomendado):**
```bash
streamlit run app.py
```
Abre no navegador em `http://localhost:8501`.

---

## 6. Primeiros passos (5 min)

```
1. No terminal:                streamlit run app.py
2. Sidebar → "📋 Demo"
3. Aba "📊 Análise"
4. Veja KPIs do corner FL nos cards
5. Em "Tipo de sweep" → "Heave"
6. Gráficos aparecem automaticamente
```

Pronto, você está rodando análise cinemática completa.

---

## 7. Tutorial 1: Analisar geometria existente

**Cenário:** você tem um carro montado no SolidWorks e quer saber seus parâmetros.

### Passo 1 — Extrair hardpoints do SolidWorks

Para cada um dos 10 hardpoints de cada corner:

1. Clique no ponto/sketch
2. "Propriedades de Massa" ou "Medir" → leia X, Y, Z
3. Anote em planilha

> 💡 Se sua origem do CAD não for SAE, transforme antes:
> - Subtraia X da origem desejada
> - Inverta Y se necessário
> - Z = 0 no solo, Z+ para cima

### Passo 2 — Preencher template

1. Sidebar → **"⬇️ Template"** baixa `hardpoints_template.csv`
2. Abra no Excel
3. Substitua os valores
4. Salve como `meu_carro.csv`

### Passo 3 — Carregar

1. Sidebar → upload do `meu_carro.csv`
2. Se erro, corrija conforme a mensagem
3. ✅ confirmação aparece quando OK

### Passo 4 — Análise estática

Aba **📊 Análise**:

1. Escolha o corner (FL para começar)
2. Confira os 6 KPIs nos cards:
 - **Caster** 3–7°
 - **KPI** 5–10°
 - **Camber** −1 a −3°
 - **Scrub** −10 a +30 mm
 - **Trail** 5–25 mm
 - **RC Height** 20–80 mm

Se algum estiver MUITO fora, **revise os hardpoints**.

### Passo 5 — Análise dinâmica

1. "Tipo de sweep" → **"Heave"**
2. Min: −25 mm, Max: +25 mm, Step: 1 mm
3. Aguarde ~2s. Aparecem:
 - **Camber Gain** (°/mm)
 - **Bump Steer** (°/mm)
 - **RC ΔY, ΔZ**

E 3 gráficos:
- Camber vs Heave (curva idealmente linear, inclinação negativa)
- Δ Toe vs Heave (próxima de zero)
- Trajetória do RC (idealmente um ponto)

### Passo 6 — Roll e Steer

Repita com **"Roll"** ou **"Steer"**:
- **Roll**: camber × roll, ratio típico −0.8 a −1.2 °/°
- **Steer**: como Caster e KPI variam (efeito "caster gain")

### Interpretação rápida

| Resultado | Significado |
|---|---|
| Camber gain > 0.03 °/mm | Pode perder grip em pista desnivelada |
| Camber gain POSITIVO | ⚠️ Geometria invertida — revise |
| Bump steer > 0.01 °/mm | Tie-rod mal posicionado |
| RC migra > 50 mm | Comportamento instável |
| Caster < 2° | Volante "morto" |
| Scrub > 30 mm | Esforço alto no volante |

---

## 8. Tutorial 2: Síntese a partir de targets

**Cenário:** você quer projetar o carro do ano que vem e definir onde colocar os hardpoints para atingir metas específicas.

### Passo 1 — Geometria seed

A otimização precisa de **ponto de partida**:
- Carro do ano passado, ou
- Template demo, ou
- Geometria escolhida manualmente

Carregue na sidebar.

### Passo 2 — Aba 🎯 Síntese

Escolha o **corner-seed** (FL é comum).

### Passo 3 — Definir targets

**Estáticos (esquerda):** marque checkbox para ativar.

| Target | Sugestão |
|---|---|
| Caster | 4° a 5° |
| KPI | 6° a 8° |
| Camber estático | −1.5° a −2.5° |
| Scrub Radius | 15 a 25 mm |
| Trail Mecânico | 15 a 25 mm |

**Dinâmicos (direita):** sempre ativos.

| Target | Sugestão |
|---|---|
| Camber Gain | −0.015 a −0.025 °/mm |
| Bump Steer máx | 0.005 °/mm |
| RC Height | 40 a 60 mm |
| RC ΔY máx | 25 mm |

**Faixa do heave:** −25 a +25 mm, step 5 mm (passo maior = mais rápido).

### Passo 4 — Pesos (opcional, avançado)

Expanda **"⚙️ Pesos da função objetivo"**. Aumente o peso do target que está sendo violado.

| Peso padrão | Quando aumentar |
|---|---|
| `w_camber_gain = 1.0` | Camber gain longe do alvo |
| `w_bump_steer = 10.0` | Já alto — bump steer é crítico |
| `w_static_camber = 5.0` | Camber estático fácil de atingir |
| `w_caster = 1.0` | Você prioriza caster |

### Passo 5 — Bounding boxes (opcional)

Expanda **"📦 Bounding Boxes"**. Margens em torno do seed:

| Hardpoint | Margem típica |
|---|---|
| UCA out / LCA out | ±50 mm |
| TR inboard / outboard | ±25 mm |

Comece largo (±50–100), depois aperta.

### Passo 6 — Solver

Expanda **"🔧 Configuração do solver"**:

| Parâmetro | Sugestão |
|---|---|
| População | 12 |
| Iterações | 40 (teste) ou 100+ (final) |
| Random seed | 42 |
| Paralelismo | Todos os cores |

### Passo 7 — Rodar

Botão **"🚀 Rodar Otimização"**. Tempo:
- 40 iter × 12 pop = ~6000 avaliações → 30s–2min
- 100 iter = 3–5 min

### Passo 8 — Interpretar

Tabela comparativa:
```
Parâmetro              Target    Seed     Otimizado  Erro Seed  Erro Opt  OK Seed  OK Opt
Caster (°)             +4.0000   +8.8807  +3.9637    +4.8807    -0.0363   ❌       ✅
KPI (°)                +7.0000   +4.4672  +6.8205    -2.5328    -0.1795   ❌       ✅
...
```

- ❌ Seed = targets que não atendia
- ✅ Otimizado = targets atendidos
- Continuou ❌ = trade-off; aumente o peso ou afrouxe outro target

### Passo 9 — Baixar CSV

Botão **"⬇️ Baixar hardpoints otimizados (CSV)"**.

### Passo 10 — Aplicar no SolidWorks

1. Abra o CSV
2. Para cada hardpoint, edite o sketch
3. Substitua X, Y, Z
4. Atualize referências da montagem

### Iteração

Geralmente roda várias vezes:
```
1ª rodada: bounds amplos, todos targets ligados → vê viabilidade
2ª rodada: bounds apertados, refina pesos → resultado final
3ª rodada (opcional): valida no CAD, ajusta para interferências físicas
```

---

## 9. Tutorial 3: Comparar duas geometrias

Aba **"🔄 Comparação"**:

1. **Geometria A:** "Última geometria SEED (Aba 2)"
2. **Geometria B:** "Última geometria OTIMIZADA (Aba 2)"
3. Faixa heave: padrão −25 a +25 mm

Você verá:

**Tabela KPIs estáticos** com Δ (B−A):
```
Parâmetro              A         B         Δ
Caster (°)            +8.881   +3.964    -4.917
KPI (°)               +4.467   +6.821    +2.354
...
```

**Métricas dinâmicas** lado a lado com deltas.

**3 gráficos sobrepostos:**
- Camber vs Heave (A azul, B vermelho)
- Δ Toe vs Heave
- Trajetória do RC

Outros usos:
- FL vs RL (frontal vs traseiro)
- Duas iterações de design
- Carro atual vs ano passado

---

## 10. Workflow SolidWorks ↔ Software

```
                  ┌─────────────────────────────────┐
                  │     PROJETO DE SUSPENSÃO        │
                  └─────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────┐
   │  1. CONCEITUAL                                       │
   │     - Bitola, wheelbase, altura de CG                │
   │     - Tipo de pneu (raio, contact patch)             │
   │     - Define WHEEL_CENTER e CONTACT_PATCH            │
   └─────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────┐
   │  2. SÍNTESE (este software, aba 🎯)                  │
   │     - Define targets                                 │
   │     - Roda otimização                                │
   │     - Baixa CSV                                      │
   └─────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────┐
   │  3. MODELAGEM CAD (SolidWorks)                       │
   │     - Cria sketches com X,Y,Z do CSV                 │
   │     - Modela braços, manga, mancal                   │
   │     - Verifica interferências físicas                │
   └─────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────┐
   │  4. ANÁLISE FINAL (este software, aba 📊)            │
   │     - Carrega hardpoints definitivos                 │
   │     - Sweeps completos                               │
   │     - Confere KPIs vs targets                        │
   └─────────────────────────────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────┐
   │  5. FABRICAÇÃO + VALIDAÇÃO                           │
   │     - Construção dos braços                          │
   │     - Setup com k-rig                                │
   │     - Validação em pista                             │
   └─────────────────────────────────────────────────────┘
```

---

## 11. Formato do arquivo de hardpoints

### 11.1 Estrutura

5 colunas, 40 linhas (4 corners × 10 pontos):

| Coluna | Tipo | Descrição |
|---|---|---|
| `corner` | texto | "FL", "FR", "RL", "RR" |
| `point` | texto | nome do ponto (lista abaixo) |
| `x_mm` | número | coordenada X em mm |
| `y_mm` | número | coordenada Y em mm |
| `z_mm` | número | coordenada Z em mm |

### 11.2 Os 10 pontos por corner

| Nome | O que é |
|---|---|
| `UCA_IN_FRONT` | Inboard frontal do braço superior |
| `UCA_IN_REAR` | Inboard traseiro do braço superior |
| `UCA_OUT` | Outboard do braço superior (= UBJ) |
| `LCA_IN_FRONT` | Inboard frontal do braço inferior |
| `LCA_IN_REAR` | Inboard traseiro do braço inferior |
| `LCA_OUT` | Outboard do braço inferior (= LBJ) |
| `TIE_ROD_IN` | Ponto do tie-rod no rack |
| `TIE_ROD_OUT` | Ponto do tie-rod na manga |
| `WHEEL_CENTER` | Centro de roda |
| `CONTACT_PATCH` | Contato pneu-solo (sempre Z=0) |

### 11.3 Formatos

- `.csv` (recomendado)
- `.xlsx` (precisa de `openpyxl` ou `fastexcel`)
- `.json`

### 11.4 Exemplo CSV (corner FL)

```csv
corner,point,x_mm,y_mm,z_mm
FL,UCA_IN_FRONT,60,150,295
FL,UCA_IN_REAR,-70,150,295
FL,UCA_OUT,-5,590,280
FL,LCA_IN_FRONT,90,130,162
FL,LCA_IN_REAR,-70,130,162
FL,LCA_OUT,15,600,152
FL,TIE_ROD_IN,-50,180,200
FL,TIE_ROD_OUT,-60,580,195
FL,WHEEL_CENTER,5,610,220
FL,CONTACT_PATCH,5,610,0
```

### 11.5 Erros comuns de validação

| Mensagem | Causa | Correção |
|---|---|---|
| `Corners inválidos: ['fl']` | Minúsculo | Use FL maiúsculo |
| `Pontos desconhecidos: ['UCA_INBOARD']` | Nome errado | Use `UCA_IN_FRONT` |
| `Corner 'FL' faltando pontos: ['TIE_ROD_OUT']` | Linha faltante | Adicione |
| `Coluna 'x_mm' contém nulos` | Célula vazia | Preencha |
| `Coluna 'x_mm' deve ser numérica` | Texto | Use ponto decimal, não vírgula |

---

## 12. Estrutura do projeto

```
fsae_suspension_clean/
│
├── geometry/                       # Motor matemático puro (numpy/scipy)
│   ├── __init__.py
│   ├── primitives.py               # Point3D, Vector3D, Point2D
│   ├── solver_2d.py                # Mecanismo 4 barras (vista frontal Y-Z)
│   ├── model_3d.py                 # ControlArm, SuspensionCorner, Vehicle
│   └── solver_3d.py                # Solver 3D (3 esferas + LM)
│
├── analysis/                       # Análise dinâmica + I/O + KPIs
│   ├── __init__.py
│   ├── sweeps.py                   # Sweeps heave/roll/steer + plots Plotly
│   ├── optimizer.py                # Otimização (differential_evolution)
│   ├── io_hardpoints.py            # Leitura/escrita xlsx/csv/json
│   └── kpis.py                     # KPIs avançados (Ackermann, etc.)
│
├── app.py                          # 🌐 INTERFACE GRÁFICA STREAMLIT
├── README.md                       # Este arquivo
└── requirements.txt                # Dependências
```

### O que cada módulo faz

**`geometry/primitives.py`** — Tipos base (`Point3D`, `Vector3D`, `Point2D`) e funções de interseção (círculo-círculo, reta-reta). Pura matemática.

**`geometry/solver_2d.py`** — Resolve a suspensão como mecanismo de 4 barras na vista frontal (plano Y-Z). Usado para o cálculo do Roll Center.

**`geometry/model_3d.py`** — Classes OOP: `ControlArm`, `KingpinGeometry`, `SuspensionCorner`, `Vehicle`. Calcula parâmetros **estáticos** (caster, KPI, scrub, trail, RC).

**`geometry/solver_3d.py`** — Solver cinemático 3D. Trata a manga como corpo rígido. Resolve a posição em (heave, roll, rack) via interseção de 3 esferas + `least_squares` (Levenberg-Marquardt).

**`analysis/sweeps.py`** — Roda varreduras de heave/roll/steer chamando o solver 3D repetidamente. Calcula camber gain, bump steer, migração do RC. Gera plots Plotly.

**`analysis/optimizer.py`** — Otimização global (`differential_evolution`). Aceita targets estáticos e dinâmicos. Inclui `validate_against_targets()` que gera relatório linha-por-linha.

**`analysis/io_hardpoints.py`** — Leitura, validação e escrita de arquivos de hardpoints. Construção de `SuspensionCorner` e `Vehicle` a partir de DataFrame.

**`analysis/kpis.py`** — KPIs avançados (Ackermann, Steer Ratio, Ride/Roll Camber, RC sob 1g, Anti-dive). Função `build_full_report()` gera relatório completo no formato de ficha de setup.

**`app.py`** — Interface Streamlit com abas: Análise, Síntese, Comparação.

---

## 13. Uso via Python (scripts)

Para automação, batch, integração com outras ferramentas.

### 13.1 Carregar e analisar

```python
from analysis.io_hardpoints import read_hardpoints, build_corner_from_dataframe
from geometry import KinematicSolver3D
from analysis.sweeps import SweepRunner, camber_gain_per_mm, bump_steer_per_mm

# Carrega e valida
df = read_hardpoints("meu_carro.xlsx")

# Constrói corner FL
corner, tie_rod = build_corner_from_dataframe(df, "FL")

# KPIs estáticos
print(f"Caster: {corner.static_caster_deg():+.3f}°")
print(f"KPI:    {corner.static_kpi_deg():+.3f}°")
print(f"Camber: {corner.static_camber_deg():+.3f}°")

# Sweep dinâmico
solver = KinematicSolver3D(corner, tie_rod)
runner = SweepRunner(solver=solver)
sweep  = runner.heave_sweep(-25.0, 25.0, 1.0)

print(f"Camber gain: {camber_gain_per_mm(sweep):+.5f} °/mm")
print(f"Bump steer:  {bump_steer_per_mm(sweep):+.5f} °/mm")
```

### 13.2 Otimização

```python
from analysis.optimizer import (
    SuspensionOptimizer, DesignTargets, validate_against_targets,
)

targets = DesignTargets(
    caster_target_deg=4.5,
    kpi_target_deg=7.0,
    static_camber_target_deg=-1.5,
    camber_gain_target_deg_per_mm=-0.020,
    rc_height_target_mm=50.0,
    heave_step_mm=5.0,
)

opt = SuspensionOptimizer(
    seed_corner=corner,
    seed_tie_rod=tie_rod,
    targets=targets,
    population_size=15,
    max_iterations=60,
    workers=-1,
)
result = opt.run()
print(result.summary())

# Valida
report = validate_against_targets(
    result.optimal_corner, result.optimal_tie_rod, targets,
)
print(report.summary())
```

### 13.3 Relatório completo (formato ficha de setup)

```python
from analysis.io_hardpoints import build_vehicle_from_dataframe
from analysis.kpis import build_full_report

vehicle, tie_rods = build_vehicle_from_dataframe(df)

report = build_full_report(
    vehicle, tie_rods,
    cg_height_mm=280.0,
    brake_bias_pct=60.0,
    drive_type="RWD",
    roll_stiffness_deg_per_g=1.5,
)

print(f"Wheelbase: {report.wheelbase_mm:.1f} mm")
print(f"Track F:   {report.track_front_mm:.1f} mm")
print(f"\nFront: {report.front}")
print(f"\nRear:  {report.rear}")
```

### 13.4 Exportar

```python
from analysis.io_hardpoints import dataframe_from_corner, save_dataframe

df_out = dataframe_from_corner(result.optimal_corner, result.optimal_tie_rod)
save_dataframe(df_out, "geometria_otimizada.xlsx")
```

---

## 14. KPIs implementados

### 14.1 Por corner (`SuspensionCorner`)

| Método | Retorna | Unidade |
|---|---|---|
| `static_caster_deg()` | Caster | ° |
| `static_kpi_deg()` | Kingpin Inclination | ° |
| `static_camber_deg()` | Camber estático (input) | ° |
| `static_scrub_radius_mm()` | Scrub Radius | mm |
| `static_mechanical_trail_mm()` | Trail mecânico | mm |
| `static_kingpin_offset_mm()` | Offset do pino na altura WC | mm |
| `roll_center_height_mm()` | RC Height | mm |
| `steer_arm_length_mm(tro)` | Comprimento do steering arm | mm |
| `anti_dive_percent(...)` | Anti-dive simplificado | % |
| `anti_squat_percent(...)` | Anti-squat simplificado | % |

### 14.2 Avançados (`analysis/kpis.py`)

| Função | Retorna |
|---|---|
| `wheelbase_mm(front, rear)` | Wheelbase |
| `track_width_mm(left, right)` | Bitola |
| `ride_camber_deg_per_m(corner, tr)` | Ride Camber (°/m) |
| `roll_camber_deg_per_deg(corner, tr)` | Roll Camber (°/°) |
| `static_toe_deg(corner, tr)` | Toe estático |
| `static_sum_toe_deg(L, R, ...)` | Sum Toe |
| `ackermann_geometry(...)` | Dict com Ackermann %, steer arms |
| `steer_ratio_and_cfactor(...)` | Dict com rack/wheel° e wheel°/rack |
| `steer_ratio_from_pinion(...)` | Steer Ratio (x:1) |
| `roll_center_at_1g_lat(...)` | RC sob 1g lateral |
| `anti_dive_percent(...)` | Anti-dive |
| `anti_squat_percent(...)` | Anti-squat |
| `build_full_report(...)` | `FullKPIReport` completo |

### 14.3 Dinâmicos (de sweeps)

| Função | Calcula |
|---|---|
| `camber_gain_per_mm(sweep)` | Inclinação de camber vs heave |
| `bump_steer_per_mm(sweep)` | Inclinação de toe vs heave |
| `rc_migration_range(sweep)` | (ΔY, ΔZ) do RC durante sweep |

---

## 15. Troubleshooting

### 15.1 Streamlit

| Problema | Solução |
|---|---|
| `command not found: streamlit` | Ative venv e `pip install streamlit` |
| Página em branco | Outra app na porta. `streamlit run app.py --server.port 8502` |
| Erro import polars | `pip install polars openpyxl fastexcel` |

### 15.2 Upload de arquivo

| Mensagem | Solução |
|---|---|
| `Corners inválidos` | Use FL/FR/RL/RR maiúsculo |
| `Coluna x_mm contém nulos` | Preencha as 40 linhas |
| `ModuleNotFoundError: openpyxl` | `pip install openpyxl` |
| Could not read xlsx | Use CSV, ou `pip install fastexcel` |

### 15.3 Valores absurdos nos KPIs

| Resultado | Causa | Verificação |
|---|---|---|
| Caster = 0° | Outboards com mesmo X | Diferença X em UCA_OUT vs LCA_OUT |
| KPI = 0° | Outboards com mesmo Y | Diferença Y em UCA_OUT vs LCA_OUT |
| Camber/KPI = ±70° (absurdo) | Manga estreita (Z UBJ ≈ Z LBJ) | Distância vertical UCA_OUT vs LCA_OUT deve ser 80–180 mm |
| RC Height < 0 | RC abaixo do solo | UCA inboard mais baixo que outboard? Z trocados? |
| Scrub > 100 mm | WC em Y errado | Verifique WHEEL_CENTER.y |

### 15.4 Diagnóstico: a manga

A manga (UBJ-LBJ) deve ter:
- **Altura vertical (Z)**: 80–180 mm (FSAE típico)
- **Distância total**: 100–200 mm

```python
manga = corner.upper_arm.outboard.distance_to(corner.lower_arm.outboard)
altura_z = abs(corner.upper_arm.outboard.z - corner.lower_arm.outboard.z)
print(f"Manga: {manga:.1f} mm, altura Z: {altura_z:.1f} mm")
```

Se manga < 60 mm ou altura Z < 50 mm → revise hardpoints.

### 15.5 Origem do CAD diferente do SAE

O sinal mais óbvio:
- Sua **`WHEEL_CENTER.z`** deveria ser **raio do pneu** (positivo, ~220–260 mm)
- Sua **`CONTACT_PATCH.z`** deveria ser **0**

Se essas duas estão trocadas ou negativas, seu CAD usa convenção diferente. Conversão típica:

```python
# Se seu Z aponta para baixo, com origem no centro de roda:
Z_sae = TIRE_RADIUS - Z_user

# Se seu Y está negativo para o lado esquerdo:
Y_sae = -Y_user
```

### 15.6 Otimização não converge

| Sintoma | Solução |
|---|---|
| Custo > 100 após muitas iterações | Targets impossíveis simultaneamente; afrouxe um |
| Hardpoints sem variação | Bounds apertados; aumente margens |
| Resultado pior que seed | Aumente iterações (≥ 100) |
| Demora > 10 min | `heave_step_mm=5` ou reduza pop |
| "Mecanismo fora de alcance" | Bounds geram geometria impossível; aperte |

### 15.7 Sweep com descontinuidades

Saltos no gráfico camber × heave:
- Solver encontrou "outra solução" do mecanismo
- Reduza `step_mm` (mais pontos)
- Evite comprimentos quase-iguais (configuração singular)

---

## 16. Limitações conhecidas

### 16.1 O que FAZ ✅

- Cinemática 3D completa em (heave, roll, steer)
- 6+ parâmetros estáticos
- 10+ parâmetros dinâmicos
- Otimização global com targets estáticos + dinâmicos
- Bounding boxes para keep-out zones
- Validação contra targets
- Export CSV/Excel/JSON

### 16.2 O que NÃO FAZ ❌

- **Dinâmica vertical** (massa-mola-amortecedor)
- **Compliance dos braços** (rigidez não-infinita, bushings)
- **Pushrod/pullrod/rocker** → não calcula motion ratio
- **Visualização 3D dos hardpoints** (só gráficos 2D)
- **Otimização das 4 pontas juntas** (só uma por vez)
- **Detecção de interferências físicas** (responsabilidade do CAD)
- **Tire load** (transferência de carga, grip residual)

### 16.3 Aproximações

- **Roll axis na origem** — rotação aplicada em torno de X passando por (0,0,0). Para roll < 3°, erro < 1 mm
- **Toe absoluto não interpretável** — software reporta **Δ toe** relativo ao estado estático
- **Manga rígida** — sem compliance
- **Camber estático = input** — não inferido dos hardpoints (depende da construção da manga)
- **Anti-dive simplificado** — assume freio outboard, sem considerar transferência de carga

---

## 17. Glossário

| Termo | Significado |
|---|---|
| **A-arm / Wishbone** | Braço em formato de "A", padrão FSAE |
| **Anti-dive / Anti-squat** | Geometrias na vista lateral que reduzem mergulho na frenagem / agachamento na aceleração |
| **Ball joint** | Junta esférica (rótula) que liga braço à manga |
| **Bounding box** | Caixa 3D que define onde um hardpoint pode ficar |
| **Bump** | Roda subindo relativa ao chassi (heave positivo) |
| **Bump steer** | Variação INVOLUNTÁRIA do toe com heave |
| **Camber** | Inclinação da roda em relação à vertical |
| **Camber Gain** | Taxa de variação do camber com heave (°/mm) |
| **Caster** | Inclinação do pino mestre na vista lateral |
| **Compliance** | Deformação elástica (bushings, braços) |
| **Contact patch (CP)** | Área de contato pneu-solo |
| **Differential evolution** | Algoritmo evolutivo global do otimizador |
| **DOF** | Degree of Freedom (grau de liberdade) |
| **FSAE** | Formula SAE — competição estudantil |
| **Hardpoint** | Ponto de pivô/fixação da suspensão |
| **Heave** | Deslocamento vertical relativo chassi-roda |
| **Inboard / Outboard** | Lado do chassi / lado da roda |
| **Instant Center (IC)** | Centro instantâneo de rotação da manga (vista frontal) |
| **Jounce** | Sinônimo de bump (compressão) |
| **KPI** | Kingpin Inclination (inclinação do pino na vista frontal) |
| **LBJ / UBJ** | Lower / Upper Ball Joint |
| **LCA / UCA** | Lower / Upper Control Arm |
| **least_squares** | Método numérico para sistemas não-lineares |
| **Levenberg-Marquardt (LM)** | Algoritmo de least_squares |
| **Mechanical Trail** | Distância longitudinal entre intercepto do pino no solo e CP |
| **Motion Ratio** | Razão entre deslocamento da roda e da mola |
| **Pickup point** | Sinônimo de hardpoint |
| **Pushrod / Pullrod** | Barra que conecta manga ao rocker |
| **Rack** | Cremalheira de direção |
| **Rebound** | Roda descendo relativa ao chassi (heave negativo) |
| **Rocker / Bell-crank** | Alavanca pushrod → mola |
| **Roll** | Rotação do chassi em X |
| **Roll Axis** | Linha unindo RC dianteiro e RC traseiro |
| **Roll Center (RC)** | Pivô instantâneo de rolagem na vista frontal |
| **Scrub Radius** | Distância lateral entre intercepto do pino no solo e CP |
| **Seed** | Geometria inicial da otimização |
| **Steer** | Esterçamento (rotação em torno do pino mestre) |
| **Sweep** | Varredura paramétrica |
| **SVIC** | Side View Instant Center (vista lateral) |
| **Tie-rod** | Barra de direção (rack → manga) |
| **Toe** | Convergência/divergência da roda |
| **TRO / TRI** | Tie Rod Outboard / Inboard |
| **Upright** | Manga de eixo |
| **Wheel Center (WC)** | Centro de roda |

---

## 📞 Sobre

Este software foi desenvolvido como projeto educacional para times de Fórmula SAE. Não é produto comercial.

A estrutura modular foi pensada para extensão:
- Pushrod/pullrod → estenda `SuspensionCorner` e solver 3D
- Visualização 3D → use `plotly.graph_objects.Scatter3d`
- Análise vista lateral (anti-dive completo) → crie `solver_xz.py`
- Integração SolidWorks → use a API COM (Windows)

**Última atualização:** 2026
