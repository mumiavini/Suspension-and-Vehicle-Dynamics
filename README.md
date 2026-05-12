# FSAE Suspension Geometry Engine

Motor de cálculo em Python para **projetar, dimensionar e simular** a geometria de suspensão de um carro de Fórmula SAE. Foco no backend matemático — sem GUI obrigatória, com saída opcional via Streamlit.

---

## Índice

1. [O que esse software faz](#1-o-que-esse-software-faz)
2. [Estrutura do projeto](#2-estrutura-do-projeto)
3. [Conceitos físicos — leitura obrigatória](#3-conceitos-físicos--leitura-obrigatória)
4. [Convenções](#4-convenções)
5. [Como cada módulo funciona](#5-como-cada-módulo-funciona)
6. [Decisões de arquitetura](#6-decisões-de-arquitetura)
7. [Como usar — receitas práticas](#7-como-usar--receitas-práticas)
8. [Instalação](#8-instalação)
9. [Limitações conhecidas](#9-limitações-conhecidas)
10. [Próximos passos](#10-próximos-passos)

---

## 1. O que esse software faz

A partir das **coordenadas dos hardpoints** da suspensão (pontos de ancoragem dos braços, manga, tie-rod, centro de roda, contato do pneu), o software calcula:

**Estaticamente:**
- Caster, KPI (Kingpin Inclination), Camber estático
- Scrub Radius, Mechanical Trail
- Roll Center e Roll Axis do veículo

**Dinamicamente (durante o movimento):**
- Cinemática da manga de eixo em **heave** (bump/rebound), **roll** (rolagem) e **steer** (esterçamento)
- Camber gain (°/mm de heave)
- Bump steer (°/mm)
- Migração do Roll Center

**Para projeto (síntese):**
- Otimização global dos hardpoints para atingir metas de design (camber gain alvo, bump steer mínimo, RC numa altura específica) respeitando keep-out zones do chassi.

---

## 2. Estrutura do projeto

```
fsae_suspension/
│
├── geometry/                       # Motor matemático puro
│   ├── __init__.py                 # Re-exporta as classes principais
│   ├── primitives.py               # Point3D, Vector3D, Point2D, interseções
│   ├── solver_2d.py                # Mecanismo de 4 barras (vista frontal)
│   ├── model_3d.py                 # ControlArm, SuspensionCorner, Vehicle
│   └── solver_3d.py                # Solver 3D (interseção de 3 esferas)
│
├── analysis/                       # Análise dinâmica e otimização
│   ├── __init__.py
│   ├── sweeps.py                   # Heave/Roll/Steer sweeps + plots Plotly
│   ├── optimizer.py                # differential_evolution
│   └── io_hardpoints.py            # Leitura xlsx/csv/json com polars
│
├── app.py                          # Streamlit (interface gráfica opcional)
├── main.py                         # Demo dos Degraus 1-2 (estática 2D + 3D)
├── demo_advanced.py                # Demo dos Degraus 3-5 (solver 3D, sweeps, opt)
└── README.md                       # Este arquivo
```

**Hierarquia de dependências (de baixo para cima):**

```
primitives.py  ← base, depende só de numpy
    ↑
    ├── solver_2d.py
    └── model_3d.py  ← usa primitives
            ↑
            └── solver_3d.py  ← usa model_3d + scipy
                    ↑
                    └── analysis/sweeps.py
                            ↑
                            ├── analysis/optimizer.py
                            └── analysis/io_hardpoints.py (polars)
                                    ↑
                                    └── app.py (streamlit)
```

Cada módulo só importa de módulos "abaixo" — sem ciclos.

---

## 3. Conceitos físicos — leitura obrigatória

### 3.1 Sistema double wishbone

A suspensão modelada é a clássica de braços A (**double wishbone**), padrão em FSAE:

```
       UCA  ●═══════● UBJ          UCA = Upper Control Arm (braço superior)
                    │              UBJ = Upper Ball Joint
                    │ manga        LBJ = Lower Ball Joint
                    │              LCA = Lower Control Arm
       LCA  ●═══════● LBJ          TR  = Tie Rod (terminal de direção)
                    │
                    ● contato pneu-solo
```

Cada braço A tem **dois** pontos inboard (no chassi) e **um** outboard (na manga). Os dois pontos inboard definem o **eixo de rotação** do braço — quando a suspensão se move, o braço pivota em torno dessa linha.

### 3.2 Vista frontal como mecanismo de 4 barras

Olhando o carro de frente (plano Y-Z), a suspensão vira um **mecanismo de quatro barras planar**:

```
   elo fixo (chassi):   UCA_in ↔ LCA_in
   elo 1 (UCA):         UCA_in ↔ UCA_out
   elo 2 (LCA):         LCA_in ↔ LCA_out
   acoplador (manga):   UCA_out ↔ LCA_out
```

Os três comprimentos rígidos a serem preservados a todo instante:
- `|UCA_in - UCA_out| = L_UCA`
- `|LCA_in - LCA_out| = L_LCA`
- `|UCA_out - LCA_out| = L_manga`

**Como o solver 2D resolve isso:** dado um deslocamento de heave (chassi sobe → inboards sobem em Z), procura novas posições para UCA_out e LCA_out que satisfaçam os três comprimentos. Cada outboard é encontrado como **interseção de dois círculos** (um de cada constraint). Itera até convergir.

### 3.3 Vista 3D — manga como corpo rígido

Em 3D, a manga é um **corpo rígido** com 3 ball joints (UBJ, LBJ, TRO). Cada um deve estar a distância fixa do seu ponto inboard correspondente — isso são **três esferas no espaço**:

```
   (X - x_UBJ_in)² + (Y - y_UBJ_in)² + (Z - z_UBJ_in)² = L_UCA²
   (X - x_LBJ_in)² + (Y - y_LBJ_in)² + (Z - z_LBJ_in)² = L_LCA²
   (X - x_TRO_in)² + (Y - y_TRO_in)² + (Z - z_TRO_in)² = L_TR²
```

Além disso, as três distâncias INTERNAS da manga (UBJ-LBJ, UBJ-TRO, LBJ-TRO) também são fixas (manga rígida).

São **9 incógnitas** (3 coords × 3 pontos) e **6 constraints**. Restam 3 DOF, que correspondem fisicamente aos 3 graus de liberdade do chassi (heave, roll, rack). Para fechar o sistema, adicionamos **regularização suave** (ancora a solução próximo do estado anterior) — o solver resolve por mínimos quadrados não-lineares (Levenberg-Marquardt).

### 3.4 Roll Center

O **Roll Center (RC)** é o ponto em torno do qual o chassi gira instantaneamente quando rola. Calculado pelo método do Centro Instantâneo:

1. **IC** (Centro Instantâneo da manga) = interseção das prolongações das retas dos braços
2. Linha do IC até o **contact patch** do pneu
3. **RC** = onde essa linha cruza o plano de simetria (Y=0)

```
                     UCA_in ●─────────● UCA_out
                            │              \
                            │               ● IC (longe)
                            │              /│
                     LCA_in ●─────────● LCA_out
                                       │
                                       ● contact patch
                            ● RC (no plano de simetria)
```

### 3.5 Parâmetros do pino mestre

O **pino mestre** (steering axis) é a linha de UBJ até LBJ. A roda esterça em torno dele. Suas inclinações geram:

| Parâmetro | Plano | Definição | Típico FSAE |
|---|---|---|---|
| **Caster** | X-Z (lateral) | Inclinação para trás do topo do pino | 3° a 7° |
| **KPI** | Y-Z (frontal) | Inclinação para dentro do topo do pino | 5° a 10° |
| **Scrub Radius** | — | Dist. lateral entre intercepto do pino no solo e contact patch | -10 a +30 mm |
| **Mechanical Trail** | — | Dist. longitudinal entre intercepto e contact patch | 5 a 25 mm |

---

## 4. Convenções

### 4.1 Eixos (padrão SAE J670)

```
       Z (cima)
       ▲
       │
       │
       └────────► Y (esquerda do veículo)
      /
     /
    ▼
    X (frente)
```

- **X positivo** = frente do veículo
- **Y positivo** = lado esquerdo
- **Z positivo** = para cima

### 4.2 Sinais dos parâmetros

| Parâmetro | Convenção |
|---|---|
| **Camber** | Negativo = topo para dentro (configuração de corrida) |
| **Caster** | Positivo = topo do pino atrás da base |
| **KPI** | Positivo = topo do pino para dentro |
| **Scrub** | Positivo = intercepto do pino para dentro do contato |
| **Heave** | Positivo = bump (roda sobe relativo ao chassi) |
| **Roll** | Positivo = chassi rola para a direita |
| **Rack** | Positivo = rack desloca para a esquerda |

### 4.3 Unidades

- **Comprimentos:** sempre em **milímetros (mm)**
- **Ângulos:** sempre em **graus (°)**

Nada de mistura — funções que recebem ou retornam ângulos sempre são `_deg` no nome.

---

## 5. Como cada módulo funciona

### 5.1 `geometry/primitives.py`

Tipos básicos. Sem física, só matemática.

**Classes:**
- `Point3D(x, y, z)` — ponto no espaço
- `Point2D(u, v)` — ponto no plano (usado em vistas projetadas)
- `Vector3D(x, y, z)` — vetor com `dot`, `cross`, `normalize`, `angle_to_deg`

**Funções:**
- `circle_circle_intersection(c1, r1, c2, r2)` — interseção de dois círculos no plano. Levanta ValueError se não há solução. Base do solver 2D.
- `line_intersection_2d(p1, p2, p3, p4)` — interseção de duas retas. Base do cálculo de Roll Center.

**Detalhe importante:** todos os Point/Vector têm `.to_array()` que retorna numpy array. Use isso quando precisar fazer álgebra linear pesada — é mais rápido que operações em objetos Python.

### 5.2 `geometry/solver_2d.py`

Resolve o mecanismo de 4 barras na vista frontal Y-Z.

**Classe principal: `SuspensionGeometry2D`**

Recebe os 4 pontos inboard/outboard + WC + CP (todos como `Point2D`, onde u=Y, v=Z).

**Método mais importante: `solve_heave(heave_mm)`**

Retorna um `KinematicState2D` com:
- Novas posições da manga
- Cambagem
- Roll Center

**Algoritmo:**
1. Move os inboards verticalmente (heave)
2. Itera o mecanismo de 4 barras (interseção de 2 círculos, com tracking de continuidade)
3. Reconstrói o centro de roda na nova orientação da manga (manga é rígida)
4. Calcula cambagem (ângulo da manga vs vertical)
5. Calcula Roll Center (método do IC)

**Por que iteração de ponto fixo?** Porque o sistema tem 3 constraints (L_uca, L_lca, L_manga) acoplados — resolver as duas interseções de círculos sucessivamente converge em 2-5 iterações.

### 5.3 `geometry/model_3d.py`

Classes OOP para a geometria estática 3D.

```
ControlArm        ← braço (2 inboards + 1 outboard)
KingpinGeometry   ← eixo do pino mestre + caster/KPI/scrub/trail
SuspensionCorner  ← uma ponta (UCA + LCA + WC + CP)
Vehicle           ← carro completo (4 corners + dimensões gerais)
```

**Por que separar `KingpinGeometry`?** O cálculo de Caster, KPI, Scrub e Trail só depende dos 4 pontos: UBJ, LBJ, WC, CP. Não precisa saber dos braços completos. Separar evita duplicação e torna a API mais limpa.

**Acessores rápidos no `SuspensionCorner`:**
```python
corner.static_caster_deg()
corner.static_kpi_deg()
corner.static_camber_deg()
corner.static_scrub_radius_mm()
corner.static_mechanical_trail_mm()
corner.roll_center_height_mm()        # usa o solver 2D internamente
```

### 5.4 `geometry/solver_3d.py`

Solver cinemático 3D — o cérebro do projeto.

**Classe principal: `KinematicSolver3D`**

```python
solver = KinematicSolver3D(corner, tie_rod)
state = solver.solve(heave_mm=10.0, roll_deg=0.5, rack_mm=0.0)
```

**O que faz internamente:**

1. **Pré-computa invariantes:**
   - 3 comprimentos de elos (L_uca, L_lca, L_tr)
   - 3 distâncias internas da manga (UBJ-LBJ, UBJ-TRO, LBJ-TRO)
   - Offsets locais do WC e CP no referencial da manga (manga rígida)
   - Toe estático de referência (datum para reportar Δtoe)

2. **Em cada chamada de `solve`:**
   - Move os inboards segundo (heave + roll)
   - Adiciona deslocamento do rack ao tie-rod inboard
   - Chama `scipy.optimize.least_squares` com método Levenberg-Marquardt
   - Reconstrói WC e CP usando o referencial local da manga
   - Calcula camber, caster, KPI, toe (relativo ao estático)

3. **Continuidade física:**
   - Mantém cache do último estado (`_last_state`)
   - Usa esse cache como **seed** para a próxima chamada
   - Garante que o mecanismo siga uma trajetória contínua sem "pular" para a outra solução do quadrilátero
   - `solver.reset_seed()` zera o cache antes de novo sweep

**Por que least_squares e não fsolve/root?** Porque o sistema é **sub-determinado** (9 incógnitas, 6 constraints físicos). Para tornar bem-condicionado, adicionamos 9 termos de regularização suave que ancoram a solução próximo do seed. O least_squares lida bem com sistemas retangulares, fsolve não.

**Por que Levenberg-Marquardt?** Bom equilíbrio entre Gauss-Newton (rápido perto da solução) e Gradient Descent (robusto longe). Para sistemas com Jacobiano bem-condicionado, é a primeira escolha.

### 5.5 `analysis/sweeps.py`

Varreduras paramétricas (sweeps) — o que se faz com o solver.

**Classe principal: `SweepRunner`**

```python
runner = SweepRunner(solver=my_solver)

# Heave: -25mm a +25mm em passos de 1mm
heave_data = runner.heave_sweep(-25.0, 25.0, 1.0)

# Roll: -3° a +3°
roll_data = runner.roll_sweep(-3.0, 3.0, 0.2)

# Steer: -30mm a +30mm de rack
steer_data = runner.steer_sweep(-30.0, 30.0, 1.0)
```

Retorna **numpy structured array** com colunas nomeadas:
```python
sweep["heave_mm"]    # vetor de inputs
sweep["camber_deg"]  # vetor de cambagens correspondentes
sweep["rc_z_mm"]     # vetor de alturas do RC
# etc...
```

**Por que structured array e não DataFrame?** Performance — o overhead do pandas/polars para arrays pequenos (≤200 pontos) é alto. Numpy estruturado serializa bem e converte para DataFrame trivialmente quando necessário.

**Métricas derivadas:**
```python
camber_gain_per_mm(sweep)      # regressão linear °/mm
bump_steer_per_mm(sweep)       # regressão linear de toe vs heave
rc_migration_range(sweep)      # (ΔY, ΔZ) de migração do RC
```

**Plots Plotly (import lazy):**
```python
plot_camber_vs_heave(sweep)
plot_bump_steer(sweep)
plot_rc_migration(sweep)
plot_caster_kpi_vs_steer(sweep)
```

### 5.6 `analysis/optimizer.py`

Motor de síntese — projeta a suspensão para atingir metas.

**Fluxo:**

```python
# 1. Defina as metas e pesos
targets = DesignTargets(
    camber_gain_target_deg_per_mm=-0.020,
    bump_steer_max_abs_deg_per_mm=0.005,
    rc_height_target_mm=45.0,
    rc_y_migration_max_mm=20.0,
    w_camber_gain=1.0,
    w_bump_steer=10.0,
    w_rc_height=0.01,
)

# 2. Defina os bounds dos hardpoints (keep-out zones)
opt = SuspensionOptimizer(
    seed_corner=corner,           # geometria inicial
    seed_tie_rod=tie_rod,
    targets=targets,
    bounds_uca_outboard=HardpointBounds(-30, 30, 560, 620, 260, 300),
    # ... outros bounds
    population_size=15,
    max_iterations=60,
)

# 3. Rode
result = opt.run()
print(result.summary())
new_corner = result.optimal_corner
new_tie_rod = result.optimal_tie_rod
```

**Função objetivo:**
```
cost = w_cg  * (camber_gain - target)²
     + w_bs  * Σ(Δtoe)²
     + w_rch * (rc_height - target)²
     + w_rcm * max(0, ΔY_rc - max_allowed)²
```

Se a geometria gera uma configuração que quebra o solver (braços muito curtos, mecanismo travado), retorna `penalty_non_converged = 1e6` — o algoritmo evolutivo aprende a evitar essas regiões.

**Por que differential_evolution?**
- **Global**: não fica preso em mínimos locais (o espaço de hardpoints é cheio deles)
- **Não precisa de gradiente**: funcionaria mesmo com a função objetivo descontínua
- **Trivial paralelizar**: aumentar `workers=-1` usa todos os cores

**Custo computacional:** ~1500-3000 avaliações da função objetivo para 12 dimensões. Cada avaliação roda um sweep curto (11 pontos). Total: alguns minutos no laptop padrão.

### 5.7 `analysis/io_hardpoints.py`

Leitura de hardpoints de arquivos Excel/CSV/JSON via **polars**.

**Formato esperado:**

| corner | point          | x_mm | y_mm | z_mm |
|--------|----------------|------|------|------|
| FL     | UCA_IN_FRONT   |  60  |  150 |  295 |
| FL     | UCA_IN_REAR    | -70  |  150 |  295 |
| FL     | UCA_OUT        |  -5  |  590 |  280 |
| FL     | LCA_IN_FRONT   |  90  |  130 |  162 |
| ...    | ...            |  ... |  ... |  ... |

Cada corner deve ter exatamente 10 pontos:
`UCA_IN_FRONT`, `UCA_IN_REAR`, `UCA_OUT`, `LCA_IN_FRONT`, `LCA_IN_REAR`, `LCA_OUT`, `TIE_ROD_IN`, `TIE_ROD_OUT`, `WHEEL_CENTER`, `CONTACT_PATCH`.

**Funções:**
```python
df = read_hardpoints("car.xlsx")                    # lê + valida
corner, tie_rod = build_corner_from_dataframe(df, "FL")
vehicle, tie_rods = build_vehicle_from_dataframe(df)
save_template("template.xlsx")                       # gera arquivo modelo
```

**Por que polars e não pandas?** Mais rápido, type-safe, API consistente, e a validação de schema é trivial. Para arquivos pequenos a diferença é irrelevante, mas o código fica mais limpo.

### 5.8 `app.py` (Streamlit)

Interface gráfica simples para engenheiros sem Python.

**Como rodar:**
```bash
streamlit run app.py
```

**Fluxo do usuário:**
1. Upload do arquivo de hardpoints (ou click em "Usar template demo")
2. Seleciona o corner (FL/FR/RL/RR) e o tipo de sweep
3. Configura amplitudes e passos
4. Vê tabela de KPIs e gráficos Plotly interativos no navegador

---

## 6. Decisões de arquitetura

### 6.1 Por que dataclasses?

Quase tudo é `@dataclass`:
- Boilerplate zero para `__init__`, `__repr__`, `__eq__`
- Type hints visíveis na assinatura
- Imutabilidade opcional via `frozen=True` (não usamos por enquanto)

Alternativa seria Pydantic, mas adicionaria dependência por benefício marginal.

### 6.2 Type hints rigorosos

Toda função tem assinatura tipada. Permite:
- IDE auto-complete confiável
- `mypy --strict` passa
- Documentação implícita

### 6.3 Separação geometry/analysis

`geometry/` é matemática pura: só depende de `numpy` e `scipy`. Pode ser usado em qualquer contexto (testes, scripts, outras GUIs).

`analysis/` adiciona ferramentas de alto nível: sweeps, otimização, I/O, plots. Aqui aparecem as dependências "pesadas" (polars, plotly).

Isso permite que o `geometry/` rode em ambientes minimalistas (CI, embarcado, AWS Lambda).

### 6.4 Import lazy para plotly

```python
def plot_camber_vs_heave(sweep):
    import plotly.graph_objects as go   # lazy
    ...
```

Quem só quer computar não precisa do plotly instalado. Reduz tempo de import inicial em ~500ms.

### 6.5 Tracking de continuidade no solver

O ponto mais sutil do código:

Cada interseção de círculos tem **2 soluções**. Para o mecanismo de 4 barras, escolher a errada significa que a manga "pula" para a configuração oposta (UCA passando por baixo, fisicamente impossível mas matematicamente válida).

A solução: sempre escolher a solução **mais próxima da posição anterior** (`_closest_intersection`). Para o primeiro ponto, usar a posição estática como referência. Para os seguintes, usar o resultado anterior.

Isso é o que faz os sweeps funcionarem suavemente. Se tirar essa lógica, em algum ponto do sweep a manga "vira" e os gráficos ficam descontínuos.

### 6.6 Regularização no solver 3D

O sistema 3D tem mais variáveis que constraints (9 vs 6). Adicionar regularização suave (peso 1e-4) ancora a solução próximo do seed sem prejudicar a precisão dos constraints físicos. Sem isso, o solver chuta soluções aleatórias dentro da subvariedade de soluções válidas.

### 6.7 Toe como delta

O **toe absoluto** depende de uma escolha arbitrária de quem é o "vetor para frente da roda" — alterar o tie-rod outboard muda o valor de toe absoluto sem mudar nada fisicamente.

Por isso, o solver pré-calcula o **toe estático** no `__init__` e sempre reporta `toe_deg = toe_atual - toe_estático`. Esse delta é o que importa fisicamente: **bump steer** (mudança de toe com heave) e **steer angle** (mudança com rack).

---

## 7. Como usar — receitas práticas

### 7.1 Calcular parâmetros estáticos de uma ponta

```python
from geometry import Point3D, ControlArm, SuspensionCorner

uca = ControlArm(
    inboard_front=Point3D( 60.0, 150.0, 295.0),
    inboard_rear =Point3D(-70.0, 150.0, 295.0),
    outboard     =Point3D( -5.0, 590.0, 280.0),
    name="UCA_FL",
)
lca = ControlArm(
    inboard_front=Point3D( 90.0, 130.0, 162.0),
    inboard_rear =Point3D(-70.0, 130.0, 162.0),
    outboard     =Point3D( 15.0, 600.0, 152.0),
    name="LCA_FL",
)
corner = SuspensionCorner(
    upper_arm=uca, lower_arm=lca,
    wheel_center =Point3D(5.0, 610.0, 220.0),
    contact_patch=Point3D(5.0, 610.0,   0.0),
    corner_id="FL",
)

print(corner.summary())
# ═══ SuspensionCorner [FL] ═══
#   Caster   : +8.881°
#   KPI      : +4.467°
#   ...
```

### 7.2 Rodar uma análise dinâmica de heave

```python
from geometry import KinematicSolver3D, TieRod
from analysis.sweeps import SweepRunner, camber_gain_per_mm, plot_camber_vs_heave

tie_rod = TieRod(
    inboard =Point3D(-50.0, 180.0, 200.0),
    outboard=Point3D(-60.0, 580.0, 195.0),
)

solver = KinematicSolver3D(corner, tie_rod)
runner = SweepRunner(solver=solver)

sweep = runner.heave_sweep(heave_min_mm=-25.0, heave_max_mm=25.0, step_mm=1.0)

print(f"Camber gain: {camber_gain_per_mm(sweep):+.5f} °/mm")

# Renderiza no navegador (precisa de plotly)
fig = plot_camber_vs_heave(sweep)
fig.show()
```

### 7.3 Otimizar uma geometria para atingir metas

```python
from analysis.optimizer import SuspensionOptimizer, DesignTargets, HardpointBounds

targets = DesignTargets(
    camber_gain_target_deg_per_mm=-0.020,
    bump_steer_max_abs_deg_per_mm=0.005,
    rc_height_target_mm=45.0,
)

opt = SuspensionOptimizer(
    seed_corner=corner,
    seed_tie_rod=tie_rod,
    targets=targets,
    # Sem bounds = ±50mm em torno dos hardpoints atuais
    population_size=15,
    max_iterations=40,
    workers=-1,                  # usa todos os cores
)

result = opt.run()
print(result.summary())

# Valida o resultado
new_solver = KinematicSolver3D(result.optimal_corner, result.optimal_tie_rod)
new_sweep = SweepRunner(solver=new_solver).heave_sweep(-25, 25, 1)
print(f"Camber gain após otimização: {camber_gain_per_mm(new_sweep):+.5f} °/mm")
```

### 7.4 Carregar do Excel

```python
from analysis.io_hardpoints import read_hardpoints, build_vehicle_from_dataframe

df = read_hardpoints("meu_carro.xlsx")   # valida automaticamente
vehicle, tie_rods = build_vehicle_from_dataframe(df)

print(vehicle.summary())
```

### 7.5 Gerar um template para começar do zero

```python
from analysis.io_hardpoints import save_template
save_template("template.xlsx")
# Abra no Excel, edite os valores, depois leia de volta
```

### 7.6 Rodar a interface gráfica

```bash
streamlit run app.py
```
Abre no navegador. Upload do arquivo → resultados interativos.

---

## 8. Instalação

### Dependências mínimas (motor de cálculo)

```bash
pip install numpy scipy
```

Com isso já dá pra usar todo o `geometry/` e o otimizador.

### Dependências completas

```bash
pip install numpy scipy plotly polars openpyxl streamlit
```

`openpyxl` é o backend que polars usa para Excel. Sem ele, só CSV/JSON funcionam.

### Estrutura de pastas para Streamlit

Para `streamlit run app.py` funcionar a partir da raiz, mantenha o layout:

```
fsae_suspension/
├── app.py              ← ponto de entrada Streamlit
├── geometry/
└── analysis/
```

E rode sempre a partir da pasta raiz. Os imports são relativos a essa raiz (`from geometry import ...`).

### requirements.txt sugerido

```
numpy>=1.24
scipy>=1.10
plotly>=5.0
polars>=0.20
openpyxl>=3.0
streamlit>=1.30
```

---

## 9. Limitações conhecidas

### 9.1 Sem dinâmica vertical (massa-mola-amortecedor)

O motor é **cinemático puro**. Não calcula:
- Movimento real sob acelerações verticais
- Resposta a perturbações do solo
- Frequências naturais

Para isso, seria necessário adicionar um modelo de quarter-car ou full-car com integração temporal.

### 9.2 Sem flexibilidade dos braços

Todos os elos são tratados como rígidos. Compliance (flexão dos braços, bushings, fatigue) não está modelada. Para FSAE, isso costuma ser aceitável; para carros de produção, faltam termos importantes.

### 9.3 Roll axis passa pela origem (Y=0, Z=0)

A rotação de roll é aplicada em torno do eixo X passando pela origem. Para roll axis real (passando pelo RC, que tem altura não-nula), bastaria transladar antes/depois da rotação. Para roll < 3° (típico FSAE) a diferença é < 1mm — desprezível.

### 9.4 Toe absoluto não é interpretável

Como discutido, o valor absoluto de toe retornado pelo solver é função da escolha arbitrária de orientação do tie-rod. Use sempre `delta_toe = state.toe_deg` (que é relativo ao estático) ou pegue a diferença entre dois estados.

### 9.5 Otimizador não respeita simetria FL/FR

O `SuspensionOptimizer` atua em **uma ponta**. Para otimizar o veículo simetricamente (FL e FR juntos), seria necessário um wrapper que aplica a simetria automaticamente. Por enquanto, otimize um lado e espelhe Y manualmente.

### 9.6 Sem visualização 3D da geometria

Os plots são todos 2D (gráficos cartesianos). Visualização 3D da suspensão em movimento seria útil para debug — fica como próximo passo (Plotly tem `Scatter3d`, mas não está implementado).

---

## 10. Próximos passos

Em ordem de prioridade:

1. **Visualização 3D Plotly** dos hardpoints e do movimento da manga
2. **Anti-dive / anti-squat** — análise da vista lateral (plano X-Z)
3. **Pushrod / pullrod / rocker** — adicionar a cinemática de mola e bell-crank para calcular motion ratio
4. **Validação cruzada** — comparar resultados com OptimumKinematics ou Lotus Suspension Analyzer em uma geometria conhecida
5. **Testes unitários** — `pytest` cobrindo casos de teste com soluções analíticas (geometrias planares simples)
6. **Camber gain analítico vs numérico** — para geometrias 2D existe fórmula fechada (via FVSA); útil para validar
7. **Sweeps combinados** — UI para varreduras 2D (heave + roll simultâneos) com heatmaps
8. **Wrapper de simetria** para o otimizador (FL+FR juntos)
9. **Export de relatório** — PDF com resumo dos KPIs + gráficos
10. **Suporte a unidades** via `pint` — para times que trabalham em polegadas

---

## Apêndice: glossário

| Termo | Significado |
|---|---|
| **Heave** | Deslocamento vertical relativo entre chassi e roda |
| **Bump / Rebound** | Heave positivo / negativo |
| **Roll** | Rotação do chassi em torno do eixo longitudinal (X) |
| **Steer** | Rotação da roda em torno do pino mestre |
| **Rack** | Deslocamento lateral do sistema de direção |
| **UCA / LCA** | Upper / Lower Control Arm (braço superior / inferior) |
| **UBJ / LBJ** | Upper / Lower Ball Joint (ponta do braço na manga) |
| **TR / TRO** | Tie Rod / Tie Rod Outboard |
| **WC** | Wheel Center (centro de roda) |
| **CP** | Contact Patch (contato pneu-solo) |
| **RC** | Roll Center (centro de rolagem) |
| **IC** | Instant Center (centro instantâneo de rotação da manga) |
| **KPI** | Kingpin Inclination (inclinação do pino mestre na vista frontal) |
| **Camber gain** | Taxa de variação da cambagem com o heave (°/mm) |
| **Bump steer** | Variação involuntária do toe com o heave |
| **Motion ratio** | Razão entre deslocamento da roda e deslocamento da mola |

---

**Autor:** Vinicius Andrade trento
**Última atualização:** Maio 2026