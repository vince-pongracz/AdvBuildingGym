# User Intervention During Evaluation: Interactive EV Control

## Problem

During evaluation (`run_eval_ray.py`), the EV charger connection schedule is driven by a static CSV via `EVState`. For interactive testing, a user needs to connect and disconnect different EVs on the fly — with arbitrary specs — while the RL agent continues to control the charging rate.

## Solution: Gymnasium Wrapper + Command Queue

A `gym.Wrapper` around `AdvBuildingGym` checks a thread-safe `queue.Queue` at the start of each `step()`. A background daemon thread reads user commands from stdin (or a socket) and pushes them onto the queue. The wrapper pops commands, finds the `LinearEVCharger` in `self.env.infras`, and calls `set_ev_connected()` directly — bypassing the `EVState` CSV mechanism entirely.

The RL agent's actions flow through the wrapper unchanged. Only EV connection state is user-controlled.

```
                              ┌────────────────────────┐
┌──────────────┐    Queue     │ EVInteractiveWrapper   │     ┌─────────────────────┐
│ Input thread │─────────────>│ (gym.Wrapper)          │────>│  AdvBuildingGym     │
│ (CLI/socket) │  EV commands │  step() checks queue   │     │  └─ LinearEVCharger │
└──────────────┘              │  calls set_ev_connected│     └─────────────────────┘
                              └────────────────────────┘
```

## Components

### 1. `EVInteractiveWrapper(gym.Wrapper)`

- **`__init__`**: Takes the inner env + a `queue.Queue`. Locates the `LinearEVCharger` instance from `self.env.infras` by checking `hasattr(infr, 'set_ev_connected')`.
- **`step(action)`**: Before delegating to `super().step(action)`, drains the queue. For each command:
  - **Connect**: Build an `EvSpec` from the command parameters, call `charger.set_ev_connected(True, ev_spec)`.
  - **Disconnect**: Call `charger.set_ev_connected(False)`.
- **`reset(**kwargs)`**: Optionally drain the queue on reset to avoid stale commands carrying over between episodes.

### 2. Input Thread

A daemon thread (`threading.Thread(daemon=True)`) that runs a read loop:

```
[Step 42 | 03:30 | SoC: 0.65 | EV: connected]
> disconnect
```

Parses command formats:
- `connect <max_cap_kWh> <max_charging_kW> <charger_eff> <discharge_eff> <v2g:true/false> <start_soc> <target_soc>`
- `disconnect`
- `break` — pause the eval loop after the current step completes
- `resume` — continue the eval loop

EV commands push a dict onto the queue: `{"connected": True, "ev_spec": EvSpec(...)}` or `{"connected": False}`.
Flow-control commands (`break`/`resume`) operate on a shared `threading.Event` directly — see section 5.

Thread safety: `queue.Queue` is thread-safe by design. The wrapper is the only consumer (in the main thread's `step()` call), the input thread is the only producer.

### 3. Eval Script Integration

In `run_eval_ray.py`, wrap the env before entering the eval loop:

```python
import queue, threading

cmd_queue = queue.Queue()
run_gate = threading.Event()
run_gate.set()  # Start in running state

env = AdvBuildingGym(...)
env = EVInteractiveWrapper(env, cmd_queue)

# Start input thread (receives both EV commands and break/resume)
input_thread = threading.Thread(
    target=cli_input_loop,
    args=(cmd_queue, run_gate),
    daemon=True,
)
input_thread.start()

# Eval loop with break/resume support
while not done:
    run_gate.wait()  # Blocks here while paused; passes through instantly when running
    action = algo.compute_single_action(obs, explore=False)
    obs, reward, terminated, truncated, info = env.step(action)
```

### 4. Removing EVState for Interactive Mode

When using the wrapper, `EVState` should be removed from the statesources list to avoid conflicts (both trying to control the charger). Either:
- Don't include `EVState` in the config when running interactive eval.
- Or have the wrapper set a flag that makes `EVState.update_state()` skip event processing.

### 5. Break/Resume (Eval Loop Pause)

Break/resume uses a `threading.Event` (`run_gate`) as a gate for the eval loop — separate from the command queue, because flow control must take effect **immediately** rather than being deferred to the next `step()` call.

**Mechanism**:
- `run_gate = threading.Event()`, initialized with `run_gate.set()` (running).
- The eval loop calls `run_gate.wait()` at the top of each iteration. When the gate is set (running), `wait()` returns instantly. When cleared (paused), `wait()` blocks until the gate is set again.
- The input thread handles:
  - `break` → `run_gate.clear()` — closes the gate; the loop blocks after the current step finishes.
  - `resume` → `run_gate.set()` — opens the gate; the loop continues from where it paused.

**Input thread dispatch logic**:
```python
def cli_input_loop(cmd_queue: queue.Queue, run_gate: threading.Event):
    while True:
        line = input("> ").strip().lower()
        if line == "break":
            run_gate.clear()
            print("PAUSED — type 'resume' to continue")
        elif line == "resume":
            run_gate.set()
            print("RESUMED")
        elif line == "disconnect":
            cmd_queue.put({"connected": False})
        elif line.startswith("connect"):
            # parse args, build EvSpec, push to queue
            ...
        else:
            print(f"Unknown command: {line}")
```

**Why not use the queue for break/resume?**
The queue is drained inside `EVInteractiveWrapper.step()`, which runs *during* the step. If the loop is what we want to pause, the control signal must be checked *before* entering `step()`. A `threading.Event` is the standard Python primitive for this — zero polling, zero latency, and `wait()` is the idiomatic blocking call.

**User experience while paused**:
- The status line shows `[PAUSED]` and the current env state.
- The user can still issue `connect`/`disconnect` commands while paused — they queue up and are applied on the next `step()` after `resume`.
- This lets the user inspect state, compose commands, then resume — useful for debugging agent behaviour at specific moments.

## Control Split

| Component        | Controlled by |
|------------------|---------------|
| EV connection    | User (via wrapper) |
| EV specs (EvSpec)| User (via connect command) |
| Charging rate    | RL agent (policy action) |
| Other infras     | RL agent (policy actions) |

## Timing

Commands are applied at the **start of the next `step()` call**, before the charger's `exec_action` runs. This means:
- User issues `connect` command at any point.
- At the next environment step, the charger sees `ev_connected=True` with the new spec.
- The RL agent's action for that step already accounts for the new connection (since it observes `ev_connected` from the previous step — 1-step delay, 5 minutes of simulation time).

## HPC / SLURM Considerations

### The problem: no stdin in batch jobs

SLURM batch jobs (`sbatch`) connect stdin to `/dev/null`. The `input()` call in the input thread raises `EOFError` immediately, killing the entire interactive mechanism. There is no TTY attached — `sys.stdin.isatty()` returns `False`.

The wrapper, queue, and `threading.Event` are all SLURM-agnostic. **Only the input source needs to be swappable.**

### Input source options on HPC

| Approach | Works with `sbatch`? | Notes |
|----------|---------------------|-------|
| **Named pipe (FIFO)** | Yes | Recommended. User writes commands from login node. Requires shared filesystem (standard on HAICORE). |
| **TCP socket** | Yes | Input thread listens on a port. User connects via SSH tunnel + `nc`. Works across nodes. |
| **Watched file** | Yes | Input thread polls a command file. User appends lines. Crude but zero-dependency. |
| **`salloc` + `srun --pty`** | Yes (interactive job) | stdin works normally. Ties up a terminal for the job's duration. |
| **Unix domain socket** | Partially | Only if user can SSH to the same compute node. |

### Recommended for HAICORE: Named pipe (FIFO)

Lowest-friction option. Login node and compute node share the same parallel filesystem (`/home`, project dirs), so a FIFO created by the job is accessible from the login node.

**SLURM script setup** (`slurm_eval_ray.sh`):
```bash
FIFO_PATH="eval_commands_${SLURM_JOB_ID}.fifo"
mkfifo "$FIFO_PATH"
trap "rm -f $FIFO_PATH" EXIT

echo "=== Interactive FIFO created: $FIFO_PATH ==="
echo "Send commands from login node with:"
echo "  echo 'connect 40 6.6 0.92 0.90 false 0.25 0.90' > $FIFO_PATH"
echo "  echo 'break' > $FIFO_PATH"

python run_eval_ray.py --interactive --fifo "$FIFO_PATH" ...
```

**Input thread reading from FIFO**:
```python
def fifo_input_loop(fifo_path: str, cmd_queue: queue.Queue, run_gate: threading.Event):
    """Read commands from a named pipe. Re-opens after each writer disconnects."""
    while True:
        with open(fifo_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                dispatch_command(line, cmd_queue, run_gate)
        # Writer closed the pipe — loop back and wait for next writer
```

The `with open` blocks until a writer opens the pipe, reads all lines, then re-opens when the writer disconnects. This allows multiple commands over time without keeping a persistent connection.

**User on login node** sends commands at any time:
```bash
# Connect an EV
echo "connect 40 6.6 0.92 0.90 false 0.25 0.90" > eval_commands_1586793.fifo

# Pause the loop
echo "break" > eval_commands_1586793.fifo

# Send commands while paused (they queue up)
echo "disconnect" > eval_commands_1586793.fifo

# Resume
echo "resume" > eval_commands_1586793.fifo
```

### Auto-detection of input source

The eval script should choose the input source automatically:

```python
import sys

if args.fifo:
    # HPC / SLURM mode — read from named pipe
    input_fn = fifo_input_loop
    input_args = (args.fifo, cmd_queue, run_gate)
elif sys.stdin.isatty():
    # Local dev — interactive terminal
    input_fn = cli_input_loop
    input_args = (cmd_queue, run_gate)
else:
    # Non-interactive, no FIFO — skip input thread entirely
    input_fn = None

if input_fn is not None:
    input_thread = threading.Thread(target=input_fn, args=input_args, daemon=True)
    input_thread.start()
```

### `salloc` as a simpler alternative

For short interactive sessions (10-30 min), `salloc` avoids FIFO complexity entirely:
```bash
salloc --partition=normal --cpus-per-task=2 --gres=gpu:full:1 --time=00:30:00
srun --pty python run_eval_ray.py --interactive --episodes 10
```
stdin works normally here. Downside: the user must keep the SSH session alive for the entire job duration.

### Summary by scenario

| Scenario | Input source | CLI arg | Changes needed |
|----------|-------------|---------|----------------|
| Local dev / `salloc --pty` | stdin (`input()`) | `--interactive` | None — current design works |
| `sbatch` on HPC | Named pipe (FIFO) | `--interactive --fifo PATH` | Add FIFO reader, SLURM script creates pipe |
| Remote / multi-user | TCP socket | `--interactive --port PORT` | Replace input thread with socket listener |
| Non-interactive batch | None | *(no flag)* | No input thread started; static EVState CSV used |

## Future Extensions

- **Web dashboard**: Streamlit/Gradio frontend that pushes commands to the same queue via a socket, with real-time SoC/reward charts.
- **Predefined EV presets**: Named profiles (e.g., `connect tesla3`, `connect leaf`) that expand to full EvSpec parameters.
- **Scenario replay**: Record interactive sessions as timestamped command logs, replay them for reproducible experiments.
