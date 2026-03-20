![CI](https://github.com/emiliomunozai/rl_games/actions/workflows/ci.yml/badge.svg?branch=main)

A hands-on repo for understanding how Reinforcement Learning works.
Train, inspect, and visualise RL agents on [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) (or any other Gymnasium environment).

## LunarLander-v3 environment

The default environment is [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/).
The lander starts at the top of the screen and the goal is to land it softly on the landing pad (between the two flags) using thrust and rotation.

### State (observation) — 8 continuous values

| Index | Variable | Description | Range |
|:---:|---|---|---|
| 0 | x | Horizontal position | -2.5 to 2.5 |
| 1 | y | Vertical position | -2.5 to 2.5 |
| 2 | vx | Horizontal velocity | -10 to 10 |
| 3 | vy | Vertical velocity | -10 to 10 |
| 4 | angle | Angle of the lander (radians) | -6.28 to 6.28 |
| 5 | angular velocity | Rotation speed | -10 to 10 |
| 6 | left leg contact | 1 if left leg touches ground, 0 otherwise | 0 or 1 |
| 7 | right leg contact | 1 if right leg touches ground, 0 otherwise | 0 or 1 |

### Actions — 4 discrete

| Value | Action |
|:---:|---|
| 0 | Do nothing |
| 1 | Fire left orientation engine (rotate right) |
| 2 | Fire main engine (thrust up) |
| 3 | Fire right orientation engine (rotate left) |

### Rewards

| Event | Reward |
|---|---|
| Moving towards the landing pad | positive, proportional to distance reduction |
| Moving away from the landing pad | negative |
| Crash | **-100** |
| Successful landing (come to rest) | **+100** |
| Each leg ground contact | **+10** |
| Firing main engine (per frame) | **-0.3** |
| Firing side engine (per frame) | **-0.03** |

An episode is considered **solved** at **+200** points average over 100 episodes.
The episode ends when the lander crashes, lands, or after **1000 time steps** (truncation).

> Run `rlgames inspect` to see live state/action/reward values from the environment.

## Quick concepts — Q-Learning methods

### The core idea

An RL agent interacts with an **environment** in discrete time steps.
At each step it observes a **state** $s$, picks an **action** $a$, receives a **reward** $r$, and transitions to a new state $s'$.
The goal is to learn a **policy** $\pi(s) \to a$ that maximises the total (discounted) reward over time.

### Q-values and the Bellman equation

A **Q-value** $Q(s, a)$ estimates the expected cumulative reward of taking action $a$ in state $s$ and then following the optimal policy.
The optimal Q-values satisfy the **Bellman optimality equation**:

$$
Q^*(s, a) = r + \gamma \max_{a'} Q^*(s', a')
$$

where $\gamma \in [0, 1]$ is the **discount factor** (how much the agent cares about future vs. immediate rewards).

### Tabular Q-Learning

When the state and action spaces are small (or can be discretized), we store Q-values in a table and update them after every transition:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \bigr]
$$

- $\alpha$ (learning rate) — how fast we update.
- **$\varepsilon$-greedy** exploration — with probability $\varepsilon$ pick a random action, otherwise pick $\arg\max_a Q(s, a)$. $\varepsilon$ decays over time so the agent gradually shifts from exploring to exploiting.

> See `src/rl_games/agents/qlearning.py` for a complete tabular implementation.

### Deep Q-Network (DQN)

When the state space is continuous (like the 8-dimensional LunarLander observation), a table no longer works.
**DQN** replaces the table with a neural network $Q_\theta(s, a)$ and introduces two key tricks:

| Trick | Why |
|---|---|
| **Experience replay** | Store transitions in a buffer, sample random mini-batches — breaks correlation between consecutive samples and reuses data. |
| **Target network** | Keep a frozen copy of the Q-network and update it periodically — stabilises the moving Bellman target. |

Training step (one gradient update):

1. Sample a mini-batch $\{(s, a, r, s', \text{done})\}$ from the replay buffer.
2. Compute targets: $y = r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a') \cdot (1 - \text{done})$.
3. Minimise MSE between $Q_\theta(s, a)$ and $y$.

> See `src/rl_games/agents/dqn.py` for a from-scratch PyTorch implementation where every component (network, replay buffer, training loop) is visible and editable.

### Exploration vs. Exploitation

This is the fundamental trade-off in RL.
**Explore** (random actions) to discover new, potentially better strategies.
**Exploit** (greedy actions) to collect the highest reward based on current knowledge.
The $\varepsilon$-greedy schedule balances both: start with high $\varepsilon$ (mostly exploring) and anneal towards low $\varepsilon$ (mostly exploiting).

## Agents

| Agent | Algorithm | State representation | File |
|---|---|---|---|
| `qlearning` | Tabular Q-Learning | Discretized (8 bins per dim) | `agents/qlearning.py` |
| `dqn` | DQN from scratch (PyTorch) | Raw continuous | `agents/dqn.py` |

## Setup

```bash
uv sync
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

## CLI usage

```bash
rlgames <command> [agent] [options]
```

### Version

```bash
rlgames version
```

### List agents and their save status

```bash
rlgames list
```

### Inspect an environment

Show state/action spaces and sample a few random transitions to see what the agent observes.

```bash
rlgames inspect                          # LunarLander-v3 (default)
rlgames inspect --steps 10              # more sample transitions
```

### Initialize a new untrained agent

```bash
rlgames init qlearning
rlgames init dqn
```

### Train an agent

Creates a save if none exists, resumes from an existing save otherwise.

```bash
rlgames train qlearning --episodes 20000
rlgames train dqn       --episodes 500
```

### Load a save and display info

```bash
rlgames load qlearning
rlgames load dqn --eval
```

### Simulate episodes (text output)

Run a trained agent and see every action, reward, and outcome in the terminal.

```bash
rlgames sim qlearning --episodes 3              # full episodes
rlgames sim dqn       --episodes 2 --verbose    # full episodes with state vectors
rlgames sim dqn       --episodes 5 --steps 10   # only first 10 steps per episode
```

### Render episodes (graphical window)

```bash
rlgames render qlearning --episodes 3
rlgames render dqn       --episodes 3
```

### Delete a saved agent

```bash
rlgames delete qlearning
rlgames delete dqn
```

## Project structure

```
src/rl_games/
├── cli.py                  # CLI entry point
└── agents/
    ├── qlearning.py        # Tabular Q-Learning agent
    └── dqn.py              # DQN agent from scratch (PyTorch)
```

Saves are written to `saves/` in the working directory.


------------------------------------------------------------------------------------------
## DESARROLLO DEL TALLER
## Diagramas del entendimiento de los agentes 

## <font color="green">  Agente Qlearning <code>

<details>
  <summary>📸 Haz clic para ver el diagrama que explica el proceso del agente</summary>
  <br>
  <a href="Diagramas\Diagrama QLearning.jpg">
  <img src="Diagramas\Diagrama QLearning.jpg"style="max-width: 100%; border-radius: 10px;">
  </a>
</details>
 > El proceso de Qlearning se enfoca en tomar decisiones e ir mirando que recompensa obtiene y por medio de la ecuación de Bellman va guardando los datos de Q futuro, teniendo presente que tan buena es la acción en el estado actual y a futuro. Con este aprendizaje toma la siguiente decisión y repite el proceso optimizando cada vez el mejor valor futuro o Max Q. Comienza explorando mucho y después va usando más el aprendizaje por lo que va disminuyendo el valor Epsilon. 
<br>
<br>

## <font color="lightgreen">  Agente DQN <code>

<details>
  <summary>📸 Haz clic para ver el diagrama que explica el proceso del agente</summary>
  <br>
  <a href="Diagramas\Diagrama DQN.jpg">
  <img src="Diagramas\Diagrama DQN.jpg"style="max-width: 100%; border-radius: 10px;">
  </a>
</details>
> Se comienza realizando la definición de la red neuronal, incluyendo los parámetros de buffer y el agente. Después el agente va tomando decisiones, a partir de un batch del buffer. E igualmente va utilizando la ecuación de Bellman para seguir calculando los mejores valores futuros. Con la diferencia que ahora no solo optimiza la ecuación sino que adicional va mejorando el entrenamiento de la red lo que termina dando mejores resultados significativamente. Pero también a un costo computacional mucho mayor. 
<br>
<br>

## Mejores modelos desarrollados 

## <font color="lightblue">  Agente DQN <code>

<details>
  <summary>📸 Haz clic para ver el los mejores resultados del Agente Qlearning</summary>
  <br>
  <a href="Resultados\Mejores Qlearning.png">
  <img src="Resultados\Mejores Qlearning.png"style="max-width: 100%; border-radius: 10px;">
  </a>
</details>

> Aunque el modelo no logra obtener resultados positivos en las medias, hay pocos casos donde la media logra ser positiva llegando a un valor máximo de 21.95 después de 12000 episodios. 
<br>
<br>

<details>
  <summary>📸 Haz clic para ver el los mejores resultados simulados del Agente Qlearning</summary>
  <br>
  <a href="Resultados\Mejores simulados Qlearning.png">
  <img src="Resultados\Mejores simulados Qlearning.png"style="max-width: 100%; border-radius: 10px;">
  </a>
</details>
> Sin embargo, simulando resultados una vez se tiene el modelo entrenado, se logran obtener buenos resultados llegando a valores altos y asta un valor de superación de la prueba de 207, lo cual muestra que, aunque con mayor dificultad el modelo es capaz de aprender lo suficiente para superar el reto 
<br>
<br>

## <font color="bluelig">  Agente DQN <code>
<details>
  <summary>📸 Haz clic para ver el los mejores resultados simulados del Agente DQN</summary>
  <br>
  <a href="Resultados\Mejores resultados DQN simulados.png">
  <img src="Resultados\Mejores resultados DQN simulados.png"style="max-width: 100%; border-radius: 10px;">
  </a>
</details>

> Por otro lado, el agente DQN fue capaz de aprender de manera excelente como aterrizar logrando en 3000 episodios excelentes resultados. Y entre esos al momento de simular los datos se logran obtener valores de más de 300 de recompensas, específicamente el valor máximo logra 309.21
<br>
<br>

Los dos modelos son capaces de aprender y aterrizar la nave, aunque uno de los modelos llega en menos episodios y con mucha mejor eficiencia, mostrando la gran capacidad de aprendizaje de las redes neuronales 
