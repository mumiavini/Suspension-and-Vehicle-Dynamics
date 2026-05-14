# 🏎️ FSAE Suspension Geometry Engine

Software em Python para **projetar, analisar e otimizar** a geometria de suspensão de um carro de Fórmula SAE.

Funciona em dois modos:

- **Análise** — você tem um carro (ou montagem CAD) e quer entender como ele se comporta
- **Síntese** — você define metas de comportamento e o software encontra os hardpoints que as atendem

Interface via **Streamlit** (recomendado) ou **scripts Python** (para automação).

---

## 📑 Índice

1. [Para quem é](#1-para-quem-é)
2. [O que o software FAZ e o que NÃO faz](#2-o-que-o-software-faz-e-o-que-não-faz)
3. [Por que peso/motor/mola NÃO são necessários para a maioria dos KPIs](#3-por-que-pesomotormola-não-são-necessários-para-a-maioria-dos-kpis)
4. [Conceitos físicos](#4-conceitos-físicos)
5. [Convenções e unidades](#5-convenções-e-unidades)
6. [Instalação](#6-instalação)
7. [Primeiros passos (5 min)](#7-primeiros-passos-5-min)
8. [Tour das abas do app](#8-tour-das-abas-do-app)
9. [Tutorial completo: do CAD ao otimizado](#9-tutorial-completo-do-cad-ao-otimizado)
10. [Formato do arquivo de hardpoints](#10-formato-do-arquivo-de-hardpoints)
11. [Estrutura do projeto](#11-estrutura-do-projeto)
12. [Uso via Python (scripts)](#12-uso-via-python-scripts)
13. [Lista completa de KPIs](#13-lista-completa-de-kpis)
14. [Troubleshooting](#14-troubleshooting)
15. [Limitações](#15-limitações)
16. [Glossário](#16-glossário)

---

## 1. Para quem é

Foi pensado para:

- **Engenheiro de suspensão de FSAE** que precisa iterar geometrias rápido
- **Estudante** querendo entender o efeito de cada hardpoint
- **Time de Fórmula SAE** que quer documentar/versionar as escolhas
- **Quem usa SolidWorks** e quer um motor de cálculo desacoplado do CAD

**NÃO substitui:**
- Análise estrutural (precisa de FEA)
- Simulação de dinâmica veicular completa (use OptimumLap, CarMaker)
- Validação experimental (k-rig, testes em pista)

---

## 2. O que o software FAZ e o que NÃO faz

### ✅ FAZ — Cinemática e geometria

| Categoria | KPIs calculados |
|---|---|
| **Estáticos** | Caster, KPI, Camber, Scrub Radius, Mechanical Trail, RC Height, Kingpin Offset @ WC |
| **Dimensões** | Wheelbase, Track Width F/R |
| **Direção** | Steer Arm Length, Ackermann %, Rack/grau, Steer Ratio (com input de c-factor) |
| **Dinâmicos** | Camber Gain (°/mm), Ride Camber (°/m), Roll Camber (°/°), Bump Steer (°/mm) |
| **Roll Center** | Migração ΔY/ΔZ durante sweeps, RC @ 1g lateral (aproximação) |
| **Vista lateral** | Anti-dive %, Anti-squat % (versão simplificada) |
| **Estado** | Static Sum Toe |
| **Síntese** | Otimização global com targets estáticos + dinâmicos, bounding boxes, validação |

### ❌ NÃO FAZ (atualmente) — Dinâmica e estrutura

| Categoria | Por que falta |
|---|---|
| **Wheel Rate** (N/mm) | Precisa de **rigidez de mola** + motion ratio |
| **Roll Rate** (Nm/°) | Wheel rate + ARB + bitola |
| **Sprung Mass Frequency** (Hz) | Wheel rate + **massa suspensa** |
| **Motion Ratio** | Precisa modelar **pushrod/pullrod/rocker** |
| **Jounce/Rebound Damping** | Curva F×v do **amortecedor** |
| **FEA dos braços** | Outro tipo de software (Ansys, etc.) |
| **Simulação de lap time** | Outro tipo de software (OptimumLap) |

---

## 3. Por que peso/motor/mola NÃO são necessários para a maioria dos KPIs

Esta é uma dúvida comum, e a resposta é importante:

### 🟢 Cinemática pura — só geometria

A maior parte dos KPIs depende **apenas das posições dos hardpoints** e de **como eles se movem**:

```
Caster, KPI, Camber, Scrub, Trail, Roll Center
Camber Gain, Bump Steer, Ride/Roll Camber
Ackermann %, Steer Arm Length
```

Esses parâmetros são **invariantes em relação à massa**. Um carro de 200 kg e outro de 300 kg com a mesma geometria de hardpoints terão o mesmo Caster, mesmo Camber, mesmo Ackermann.

A massa só importa para:
- Calcular **frequência natural** (precisa de wheel rate × massa)
- Calcular **cargas absolutas** (para FEA)
- Calcular **transferência de carga**

Nenhum desses três o software calcula.

### 🟡 Aproximações que usam parâmetros externos (com defaults razoáveis)

**Anti-dive / Anti-squat:**
- Fórmula: `tan(θ_SVIC) × wheelbase/cg_height × brake_bias × 100`
- Precisa de **altura do CG** e **brake bias** → você informa no setup do veículo na sidebar
- **NÃO precisa do peso absoluto** — só da posição relativa do CG

**Roll Center @ 1g lateral:**
- Fórmula: aplica roll equivalente a 1g e mede onde o RC fica
- Precisa de **roll stiffness** (graus por g) → input do usuário, default 1.5 °/g
- Esse valor depende de mola+ARB+bitola, mas como aproximação aceita-se o típico de FSAE

### 🔴 KPIs que requerem dados externos (futuro)

Se você quiser **wheel rate, roll rate, motion ratio, frequência natural, damping** — preciso adicionar:

1. Modelo de **pushrod/rocker** (motion ratio)
2. **Rigidez de mola** como input
3. **Massa suspensa** como input
4. **Curvas do amortecedor** como input

Isso multiplicaria o tamanho do projeto. Por isso o escopo atual está **só na cinemática** — o que já cobre cerca de 70% dos KPIs da ficha de setup típica de FSAE.

### Resumo da tabela

| Você precisa para... | O software calcula HOJE? |
|---|---|
| Mexer em hardpoints e ver Caster/KPI/Camber/etc. | ✅ Sim |
| Otimizar geometria para targets | ✅ Sim |
| Ver Ackermann, Steer Ratio | ✅ Sim |
| Anti-dive/squat (precisa CG e brake bias) | ✅ Sim, com inputs |
| RC @ 1g (precisa roll stiffness) | ✅ Sim, aproximado |
| Frequência natural, wheel rate, damping | ❌ Não |
| FEA, análise de estresse | ❌ Não |
| Simulação de lap time | ❌ Não |

---

## 4. Conceitos físicos

### 4.1 Hardpoints

**Hardpoints** são os pontos de pivô/fixação da suspensão. Definir os hardpoints é definir como o carro se comporta.

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

Cada **corner** (FL, FR, RL, RR) tem **10 hardpoints**.

### 4.2 Valores típicos FSAE

**Estáticos:**

| Parâmetro | Valor típico | O que afeta |
|---|---|---|
| **Caster** | 3° a 7° | Auto-centragem do volante |
| **KPI** | 5° a 10° | Variação de camber em steer |
| **Camber** | −1° a −3° | Grip em curva |
| **Scrub Radius** | −10 a +30 mm | Esforço no volante |
| **Trail Mecânico** | 5 a 25 mm | Sensibilidade de direção |
| **RC Height** | 20 a 80 mm | Rolagem do chassi |

**Dinâmicos:**

| Parâmetro | Meta típica |
|---|---|
| **Camber Gain** | −0.015 a −0.025 °/mm |
| **Bump Steer** | < 0.005 °/mm em módulo |
| **RC ΔY** | < 30 mm de migração lateral |
| **Roll Camber** | −0.5 a −1.5 °/° |
| **Anti-dive** | 0 a 30% |
| **Ackermann** | 30% a 100% |

### 4.3 Acoplamento dos parâmetros

⚠️ Todos esses parâmetros são **acoplados geometricamente** — não dá pra mexer em um sem afetar os outros:

- Aumentar Caster → tende a aumentar Trail
- Aumentar KPI → reduz Scrub Radius
- Subir o LCA inboard → afeta Camber Gain E RC ao mesmo tempo

Por isso a **otimização global** (aba Síntese) é mais eficaz que tentativa-e-erro.

---

## 5. Convenções e unidades

### 5.1 Sistema de eixos (SAE J670)

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
> Sinais comuns de problema:
> - `WHEEL_CENTER.z` ≈ 0 → seu Z não é "altura do solo" (deveria ser o raio do pneu)
> - `LCA_IN.z > UCA_IN.z` → seu Z aponta pra baixo

### 5.2 Sinais

| Parâmetro | + significa |
|---|---|
| **Camber** | Topo da roda PARA FORA (negativo = corrida) |
| **Caster** | Topo do pino ATRÁS da base |
| **KPI** | Topo do pino PARA DENTRO do veículo |
| **Scrub** | Pino cruza o solo PARA DENTRO do contato |
| **Heave** | Bump (roda sobe relativa ao chassi) |
| **Roll** | Chassi rola para a DIREITA |
| **Rack** | Rack desloca para a ESQUERDA |

### 5.3 Unidades

- Comprimentos: **mm**
- Ângulos: **graus (°)**
- Camber gain: **°/mm**
- Ride camber: **°/m**

Nada de polegadas ou radianos. Converta antes (1 in = 25.4 mm).

---

## 6. Instalação

### 6.1 Pré-requisitos

- Python 3.10+ ([download](https://www.python.org/downloads/))
- ~500 MB livres

### 6.2 Passo-a-passo

**1. Baixe os arquivos** (pasta `fsae_suspension_clean/`)

**2. Terminal na pasta do projeto**

**3. Ambiente virtual (recomendado):**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

**4. Instale dependências:**
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.24
scipy>=1.10
plotly>=5.0
polars>=0.20
openpyxl>=3.0
fastexcel>=0.10
streamlit>=1.30
```

**5. Teste:**
```bash
python -c "from geometry import Point3D; print(Point3D(1,2,3))"
```

### 6.3 Rodando o app

```bash
streamlit run app.py
```
Abre no navegador em `http://localhost:8501`.

---

## 7. Primeiros passos (5 min)

```
1. Terminal:                    streamlit run app.py
2. Sidebar → "📋 Demo"
3. Aba "📊 Análise"
4. Veja KPIs do FL aparecerem
5. "Tipo de sweep" → "Heave"
6. Gráficos aparecem automaticamente
```

---

## 8. Tour das abas do app

O app tem **5 áreas**:

### 📂 Sidebar — Hardpoints e Setup

```
┌─ Hardpoints ─────────────────┐
│ 1. Carregue arquivo          │  ← upload de .csv/.xlsx/.json
│ [arquivo.csv mostrado aqui]  │
│ ✅ '...' — 40 pontos        │  ← preview
│ [🔄 Aplicar arquivo]         │  ← clica aqui pra aplicar
│                              │
│ Ou use:                      │
│ [📋 Demo] [⬇️ Template]      │
│                              │
│ ─────────────                │
│ 📊 Em uso: arquivo.csv       │
│ [🗑️ Limpar]                 │
│                              │
│ ─────────────                │
│ ⚙️ Setup do veículo          │
│ Brake bias front:    0.60    │  ← afeta anti-dive
│ ▼ Direção                    │
│   c-factor:          100 mm  │  ← afeta steer ratio
│   Volante lock:      270°    │
└──────────────────────────────┘
```

**Fluxo recomendado:**

1. **Carregue arquivo** OU clica em **Demo** OU edita manualmente (aba 5)
2. Veja o preview, clica em **🔄 Aplicar arquivo**
3. Ajuste **Setup do veículo** se quiser anti-dive/steer ratio corretos
4. Vá para as abas

### 📊 Aba 1: Análise

Mostra os KPIs e gráficos do corner selecionado.

- **6 cards no topo:** Caster, KPI, Camber, Scrub, Trail, RC Height
- **Seleção de sweep:** Heave / Roll / Steer com configuração inline
- **KPIs dinâmicos:** Camber Gain, Bump Steer, RC ΔY/ΔZ
- **Gráficos Plotly:** Camber vs Heave, Δ Toe vs Heave, trajetória do RC
- **Expander "Dados completos":** tabela com todos os pontos do sweep

### 🎯 Aba 2: Síntese / Otimização

Engenharia reversa — você define metas e o software encontra os hardpoints.

**Estrutura:**

```
┌─ Definição dos Targets ─────────────────────┐
│ ESTÁTICOS                   DINÂMICOS       │
│ ☑ Caster        4°          Camber Gain     │
│ ☑ KPI           7°          Bump Steer max  │
│ ☑ Camber       -1.5°        RC Height       │
│ ☐ Scrub                     RC ΔY max       │
│ ☐ Trail                                     │
└─────────────────────────────────────────────┘

▼ ⚙️ Pesos da função objetivo (avançado)
▼ 📦 Bounding Boxes / Keep-out zones
▼ 🔧 Configuração do solver evolutivo

[🚀 Rodar Otimização]

────────────────────────────────────────
Tabela: Target × Seed × Otimizado
Tabela: hardpoints otimizados
[⬇️ Baixar CSV]
```

### 🔄 Aba 3: Comparação

Compara duas geometrias (A vs B) lado a lado.

- Fontes: arquivo / seed / última otimização
- Tabela com KPIs estáticos e Δ
- Gráficos sobrepostos

### ✏️ Aba 4: Editor Manual

Edição manual dos hardpoints com visualização 2D ao vivo.

```
┌─────────────────────┬─────────────────────┐
│ Tabela editável     │ 3 vistas 2D:        │
│ (10 pts por corner) │  - YZ (frontal)     │
│                     │  - XZ (lateral)     │
│                     │  - XY (topo)        │
│                     │ Atualiza em tempo   │
│                     │ real ao digitar     │
└─────────────────────┴─────────────────────┘
[✅ Aplicar como hardpoints]  [⬇️ Baixar CSV]
[🪞 Espelhar FL→FR]
[📋 Carregar template neste corner]
[🔁 Recarregar do arquivo]
```

### 📋 Aba 5: KPIs Completos

Relatório completo no formato da ficha de setup do carro.

```
DIMENSÕES        | Wheelbase, Track F, Track R
DIANTEIRO        | Camber L/R, Sum Toe, Caster L/R, KPI L/R,
                 | Scrub L/R, Trail L/R
                 | RC static / @ 1g (Y, Z)
                 | Ride Camber, Roll Camber, Anti-dive
                 | Ackermann, Steer Arm, Steer Ratio
TRASEIRO         | (mesma estrutura, sem caster/ackermann)
NÃO CALCULADO    | Lista do que precisaria de mola/amortecedor
```

---

## 9. Tutorial completo: do CAD ao otimizado

### Cenário 1: tenho um carro pronto, quero analisar

**Passo 1 — Extrair hardpoints do SolidWorks**

Para cada um dos 10 hardpoints de cada corner:
1. Clique no ponto/sketch
2. "Propriedades de Massa" ou "Medir" → leia X, Y, Z
3. Anote em planilha

> 💡 **Verificação de origem:** antes de gastar tempo, confira que:
> - `WHEEL_CENTER.z ≈ raio do pneu` (positivo, 220-260 mm típico)
> - `CONTACT_PATCH.z = 0`
> - `UCA_IN.z > LCA_IN.z` (UCA está mais alto)
> Se essas não baterem, sua origem do CAD não é SAE — transforme.

**Passo 2 — Preencher template**

1. Sidebar → **"⬇️ Template"** baixa `hardpoints_template.csv`
2. Abra no Excel, substitua os valores
3. Salve como `meu_carro.csv`

**Passo 3 — Carregar e analisar**

1. Sidebar → upload do `meu_carro.csv`
2. Confira o preview
3. Clica em **"🔄 Aplicar arquivo"**
4. Vai para **Aba 1 (Análise)**
5. Veja os 6 KPIs nos cards — se algum estiver MUITO fora, revise

**Passo 4 — Sweep dinâmico**

1. "Tipo de sweep" → **"Heave"**
2. Min: −25 mm, Max: +25 mm, Step: 1 mm
3. Aguarde ~2s. Aparecem:
 - **Camber Gain** (°/mm) — meta −0.015 a −0.025
 - **Bump Steer** (°/mm) — meta < 0.005
 - **RC ΔY, ΔZ** — meta ΔY < 30 mm

**Passo 5 — Ver tudo na ficha de setup**

Aba **"📋 KPIs Completos"** mostra todos os parâmetros do veículo numa tabela.

### Cenário 2: vou projetar carro novo, quero descobrir hardpoints ideais

**Passo 1 — Geometria seed**

Carregue um ponto de partida (Demo, carro do ano passado, etc.).

**Passo 2 — Aba 🎯 Síntese**

1. Escolha **corner-seed** (FL)
2. **Marque os checkboxes** dos targets que importam:
 - ☑ Caster = 4.5°
 - ☑ KPI = 7°
 - ☑ Camber estático = −1.5°
3. Ajuste targets dinâmicos:
 - Camber Gain = −0.020 °/mm
 - RC Height = 50 mm

**Passo 3 — (Opcional) Bounding boxes**

Expanda **"📦 Bounding Boxes"**. Margens típicas:
- UCA out / LCA out: ±50 mm
- TR in/out: ±25 mm

Comece largo (±50-100), depois aperta.

**Passo 4 — Rodar**

Clica em **"🚀 Rodar Otimização"**. Tempo:
- 40 iter × 12 pop = ~6000 avaliações → 30s-2min
- 100 iter = 3-5 min

**Passo 5 — Interpretar**

Aparece tabela:
```
Parâmetro          Target   Seed    Otimizado  OK Seed  OK Opt
Caster (°)         +4.50    +8.88   +4.49      ❌       ✅
KPI (°)            +7.00    +4.47   +6.82      ❌       ✅
Camber (°)         -1.50    +0.00   -1.51      ❌       ✅
...
```

- ❌ Seed = não atendia
- ✅ Opt = atendido após otimização
- Se algum continua ❌ → trade-off; aumente o peso ou afrouxe

**Passo 6 — Baixar CSV e aplicar no CAD**

1. Botão **"⬇️ Baixar hardpoints otimizados"**
2. Abra o CSV
3. No SolidWorks, edite cada sketch com os novos X, Y, Z

**Passo 7 — Validar**

Aba **🔄 Comparação**:
- A = "Última geometria SEED"
- B = "Última geometria OTIMIZADA"
- Vê gráficos sobrepostos confirmando o ganho

### Cenário 3: quero brincar com hardpoints manualmente

**Aba ✏️ Editor Manual:**

1. Escolha um corner para editar
2. Edite os valores X, Y, Z direto na tabela
3. As 3 vistas 2D atualizam ao vivo
4. Botão "🪞 Espelhar Esquerdo → Direito" se quiser simetria
5. Clica em **"✅ Aplicar como hardpoints carregados"** quando terminar
6. Vá para Aba 1 e veja os KPIs resultantes

---

## 10. Formato do arquivo de hardpoints

### 10.1 Estrutura

5 colunas, 40 linhas (4 corners × 10 pontos):

| Coluna | Tipo | Descrição |
|---|---|---|
| `corner` | texto | "FL", "FR", "RL", "RR" |
| `point` | texto | nome do ponto |
| `x_mm` | número | coordenada X em mm |
| `y_mm` | número | coordenada Y em mm |
| `z_mm` | número | coordenada Z em mm |

### 10.2 Os 10 pontos por corner

| Nome | O que é |
|---|---|
| `UCA_IN_FRONT` | Inboard frontal do braço superior |
| `UCA_IN_REAR` | Inboard traseiro do braço superior |
| `UCA_OUT` | Outboard do braço superior (= UBJ) |
| `LCA_IN_FRONT` | Inboard frontal do braço inferior |
| `LCA_IN_REAR` | Inboard traseiro do braço inferior |
| `LCA_OUT` | Outboard do braço inferior (= LBJ) |
| `TIE_ROD_IN` | Inboard do tie-rod (no rack) |
| `TIE_ROD_OUT` | Outboard do tie-rod (na manga) |
| `WHEEL_CENTER` | Centro de roda |
| `CONTACT_PATCH` | Contato pneu-solo (sempre Z=0) |

### 10.3 Exemplo CSV

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
... (repetir para FR, RL, RR)
```

### 10.4 Erros de validação

| Mensagem | Causa | Correção |
|---|---|---|
| `Corners inválidos: ['fl']` | Minúsculo | Use FL maiúsculo |
| `Pontos desconhecidos: ['UCA_INBOARD']` | Nome errado | Use `UCA_IN_FRONT` |
| `Corner 'FL' faltando pontos: ['...']` | Linha faltante | Adicione |
| `Coluna 'x_mm' contém nulos` | Célula vazia | Preencha |
| `Coluna 'x_mm' deve ser numérica` | Texto | Use ponto decimal, não vírgula |

---

## 11. Estrutura do projeto

```
fsae_suspension_clean/
│
├── geometry/                       # Motor matemático puro
│   ├── __init__.py
│   ├── primitives.py               # Point3D, Vector3D, Point2D, interseções
│   ├── solver_2d.py                # Mecanismo 4 barras (vista frontal Y-Z)
│   ├── model_3d.py                 # ControlArm, KingpinGeometry,
│   │                                 SuspensionCorner, Vehicle
│   └── solver_3d.py                # Solver 3D cinemático (3 esferas + LM)
│
├── analysis/                       # Análise + I/O + KPIs + Otimização
│   ├── __init__.py
│   ├── sweeps.py                   # SweepRunner + plots Plotly
│   ├── optimizer.py                # DesignTargets, SuspensionOptimizer
│   ├── io_hardpoints.py            # read/write csv/xlsx/json
│   └── kpis.py                     # Ackermann, Steer Ratio, Ride/Roll Camber,
│                                     RC@1g, Anti-dive, build_full_report
│
├── app.py                          # 🌐 Streamlit (5 áreas)
├── README.md                       # Este arquivo
└── requirements.txt
```

### O que cada módulo faz

**`geometry/primitives.py`** — Tipos base (`Point3D`, `Vector3D`, `Point2D`) e funções de interseção (círculo-círculo, reta-reta). Pura matemática.

**`geometry/solver_2d.py`** — Resolve a suspensão como mecanismo de 4 barras na vista frontal (plano Y-Z). Usado para Roll Center.

**`geometry/model_3d.py`** — Classes OOP: `ControlArm`, `KingpinGeometry`, `SuspensionCorner`, `Vehicle`. Calcula KPIs estáticos.

**`geometry/solver_3d.py`** — Solver cinemático 3D. Trata a manga como corpo rígido. Resolve a posição em (heave, roll, rack) via interseção de 3 esferas + `least_squares` (Levenberg-Marquardt).

**`analysis/sweeps.py`** — Roda varreduras (`SweepRunner`). Calcula camber gain, bump steer, migração do RC. Gera plots Plotly.

**`analysis/optimizer.py`** — Otimização global (`differential_evolution`). `DesignTargets` com targets estáticos + dinâmicos, `HardpointBounds` para keep-out zones, `validate_against_targets()` para relatório.

**`analysis/io_hardpoints.py`** — Leitura, validação, escrita. Constrói `SuspensionCorner` e `Vehicle` a partir de DataFrames.

**`analysis/kpis.py`** — KPIs avançados (Ackermann, Steer Ratio, Ride/Roll Camber, RC@1g, Anti-dive). `build_full_report()` gera relatório completo.

**`app.py`** — Streamlit com 5 áreas: Sidebar + 4 abas + Editor Manual.

---

## 12. Uso via Python (scripts)

### 12.1 Carregar e analisar

```python
from analysis.io_hardpoints import read_hardpoints, build_corner_from_dataframe
from geometry import KinematicSolver3D
from analysis.sweeps import SweepRunner, camber_gain_per_mm, bump_steer_per_mm

df = read_hardpoints("meu_carro.xlsx")
corner, tie_rod = build_corner_from_dataframe(df, "FL")

print(f"Caster: {corner.static_caster_deg():+.3f}°")
print(f"KPI:    {corner.static_kpi_deg():+.3f}°")

solver = KinematicSolver3D(corner, tie_rod)
runner = SweepRunner(solver=solver)
sweep  = runner.heave_sweep(-25.0, 25.0, 1.0)

print(f"Camber gain: {camber_gain_per_mm(sweep):+.5f} °/mm")
print(f"Bump steer:  {bump_steer_per_mm(sweep):+.5f} °/mm")
```

### 12.2 Otimização

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
    seed_corner=corner, seed_tie_rod=tie_rod, targets=targets,
    population_size=15, max_iterations=60, workers=-1,
)
result = opt.run()
print(result.summary())

report = validate_against_targets(
    result.optimal_corner, result.optimal_tie_rod, targets,
)
print(report.summary())
```

### 12.3 Relatório completo

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
print(f"Front:     {report.front}")
```

### 12.4 Exportar

```python
from analysis.io_hardpoints import dataframe_from_corner, save_dataframe

df_out = dataframe_from_corner(result.optimal_corner, result.optimal_tie_rod)
save_dataframe(df_out, "geometria_otimizada.xlsx")
```

---

## 13. Lista completa de KPIs

### 13.1 Por corner (`SuspensionCorner`)

| Método | Retorna | Unidade |
|---|---|---|
| `static_caster_deg()` | Caster | ° |
| `static_kpi_deg()` | Kingpin Inclination | ° |
| `static_camber_deg()` | Camber estático (input construtivo) | ° |
| `static_scrub_radius_mm()` | Scrub Radius | mm |
| `static_mechanical_trail_mm()` | Trail mecânico | mm |
| `static_kingpin_offset_mm()` | Offset do pino na altura WC | mm |
| `roll_center_height_mm()` | RC Height estático | mm |
| `steer_arm_length_mm(tro)` | Steering arm length | mm |
| `anti_dive_percent(...)` | Anti-dive simplificado | % |
| `anti_squat_percent(...)` | Anti-squat simplificado | % |

### 13.2 Avançados (`analysis/kpis.py`)

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

### 13.3 Dinâmicos (de sweeps)

| Função | Calcula |
|---|---|
| `camber_gain_per_mm(sweep)` | Inclinação de camber vs heave |
| `bump_steer_per_mm(sweep)` | Inclinação de toe vs heave |
| `rc_migration_range(sweep)` | (ΔY, ΔZ) do RC durante sweep |

---

## 14. Troubleshooting

### 14.1 Streamlit

| Problema | Solução |
|---|---|
| `command not found: streamlit` | Ative venv e `pip install streamlit` |
| Página em branco | `streamlit run app.py --server.port 8502` |
| Erro import polars | `pip install polars openpyxl fastexcel` |
| `[Errno 2] No such file or directory: '/tmp/...'` | Bug do Windows — atualize para versão mais recente do app.py |

### 14.2 Upload de arquivo

| Mensagem | Solução |
|---|---|
| `Corners inválidos` | Use FL/FR/RL/RR maiúsculo |
| `Coluna x_mm contém nulos` | Preencha as 40 linhas |
| `ModuleNotFoundError: openpyxl` | `pip install openpyxl` |

### 14.3 Valores absurdos nos KPIs

| Resultado | Causa | Verificação |
|---|---|---|
| Caster = 0° | Outboards com mesmo X | Diferença X UCA_OUT vs LCA_OUT |
| KPI = 0° | Outboards com mesmo Y | Diferença Y UCA_OUT vs LCA_OUT |
| Camber/KPI = ±70° | Manga estreita (Z UBJ ≈ Z LBJ) | Distância vertical 80-180 mm |
| RC Height < 0 | RC abaixo do solo | UCA inboard mais baixo que outboard? Z trocado? |
| Scrub > 100 mm | WC em Y errado | Verifique WHEEL_CENTER.y |

### 14.4 Diagnóstico da manga

A manga (UBJ-LBJ) deve ter:
- **Altura vertical (Z)**: 80-180 mm
- **Distância total**: 100-200 mm

```python
manga = corner.upper_arm.outboard.distance_to(corner.lower_arm.outboard)
altura_z = abs(corner.upper_arm.outboard.z - corner.lower_arm.outboard.z)
print(f"Manga: {manga:.1f} mm, altura Z: {altura_z:.1f} mm")
```

Se manga < 60 mm ou altura Z < 50 mm → revise hardpoints.

### 14.5 Origem do CAD diferente do SAE

Sinais óbvios:
- `WHEEL_CENTER.z` deveria ser **raio do pneu** (~220-260 mm, positivo)
- `CONTACT_PATCH.z` deveria ser **0**

Conversão típica:
```python
# Se Z do CAD aponta para baixo com origem no centro de roda:
Z_sae = TIRE_RADIUS - Z_user

# Se Y é negativo para o lado esquerdo:
Y_sae = -Y_user
```

### 14.6 Otimização

| Sintoma | Solução |
|---|---|
| Custo > 100 após muitas iterações | Targets impossíveis simultaneamente; afrouxe um |
| Hardpoints sem variação | Bounds apertados; aumente |
| Resultado pior que seed | Aumente iterações (≥100) |
| Demora > 10 min | `heave_step_mm=5` ou reduza pop |
| "Mecanismo fora de alcance" | Bounds geram geometria impossível |

### 14.7 Editor manual perde dados

Se ao clicar em "🔄 Aplicar arquivo" o editor manual descarta suas edições:
- É o comportamento correto: aplicar arquivo **sobrescreve** o estado do editor
- Para manter suas edições: clique em "✅ Aplicar como hardpoints" no editor ANTES de carregar outro arquivo

---

## 15. Limitações

### 15.1 O que FAZ ✅

- Cinemática 3D em (heave, roll, steer)
- 6+ KPIs estáticos
- 10+ KPIs dinâmicos
- Otimização global com targets misturados
- Bounding boxes
- Validação
- Export CSV/Excel/JSON

### 15.2 O que NÃO FAZ ❌

- **Dinâmica vertical** (massa-mola-amortecedor)
- **Compliance dos braços**
- **Pushrod/pullrod/rocker** (sem motion ratio)
- **Visualização 3D dos hardpoints** (só 2D)
- **Otimização das 4 pontas juntas**
- **Detecção de interferências físicas**
- **Tire load** (transferência de carga)
- **Wheel rate, roll rate, frequência natural, damping**

### 15.3 Aproximações importantes

- **Roll axis na origem** — rotação em torno de X passando por (0,0,0). Para roll < 3°, erro < 1 mm
- **Toe absoluto não-interpretável** — reporta **Δ toe** relativo ao estático
- **Manga rígida** — sem compliance
- **Camber estático = input** — não inferido dos hardpoints
- **Anti-dive simplificado** — assume freio outboard
- **RC @ 1g aproximado** — usa roll stiffness fixo (default 1.5 °/g)

---

## 16. Glossário

| Termo | Significado |
|---|---|
| **A-arm / Wishbone** | Braço em formato de "A", padrão FSAE |
| **Anti-dive / Anti-squat** | Geometrias na vista lateral que reduzem mergulho/agachamento |
| **Ball joint** | Junta esférica (rótula) braço↔manga |
| **Bounding box** | Caixa 3D para keep-out zones |
| **Bump** | Roda subindo relativa ao chassi (heave +) |
| **Bump steer** | Variação INVOLUNTÁRIA do toe com heave |
| **Camber** | Inclinação da roda vs vertical |
| **Camber Gain** | d(camber)/d(heave) em °/mm |
| **Caster** | Inclinação do pino mestre na vista lateral |
| **Compliance** | Deformação elástica (bushings, braços) |
| **Contact patch (CP)** | Área de contato pneu-solo |
| **Differential evolution** | Algoritmo evolutivo do otimizador |
| **DOF** | Degree of Freedom |
| **FSAE** | Formula SAE — competição estudantil |
| **Hardpoint** | Ponto de pivô/fixação |
| **Heave** | Deslocamento vertical chassi-roda |
| **Inboard / Outboard** | Lado do chassi / lado da roda |
| **Instant Center (IC)** | Centro instantâneo de rotação da manga |
| **Jounce** | Sinônimo de bump |
| **KPI** | Kingpin Inclination |
| **LBJ / UBJ** | Lower / Upper Ball Joint |
| **LCA / UCA** | Lower / Upper Control Arm |
| **Levenberg-Marquardt (LM)** | Algoritmo de least_squares |
| **Mechanical Trail** | Distância long. pino-solo até CP |
| **Motion Ratio** | Razão deslocamento roda / mola |
| **Pickup point** | Sinônimo de hardpoint |
| **Pushrod / Pullrod** | Barra manga → rocker |
| **Rack** | Cremalheira de direção |
| **Rebound** | Roda descendo (heave −) |
| **Rocker / Bell-crank** | Alavanca pushrod → mola |
| **Roll** | Rotação do chassi em X |
| **Roll Axis** | Linha unindo RC F e RC R |
| **Roll Center (RC)** | Pivô instantâneo de rolagem (vista frontal) |
| **Scrub Radius** | Distância lat. pino-solo até CP |
| **Seed** | Geometria inicial da otimização |
| **Steer** | Esterçamento |
| **Sweep** | Varredura paramétrica |
| **SVIC** | Side View Instant Center |
| **Tie-rod** | Barra de direção rack→manga |
| **Toe** | Convergência/divergência |
| **TRO / TRI** | Tie Rod Outboard / Inboard |
| **Upright** | Manga de eixo |
| **Wheel Center (WC)** | Centro de roda |

---

## 📞 Sobre

A estrutura modular permite extensão:
- Pushrod/pullrod → estenda `SuspensionCorner` e solver 3D
- Visualização 3D → use `plotly.graph_objects.Scatter3d`
- Vista lateral completa (anti-dive) → crie `solver_xz.py`
- Wheel rate / frequência natural → módulo novo + inputs de mola
- Integração SolidWorks → use a API COM (Windows)

**Autor** Vinicius Andrade Trento
**Versão:** 2026
