# Ultimate Tic-Tac-Toe AI

An evolving Ultimate Tic-Tac-Toe AI powered by AlphaZero-style self-play, a
policy-value network, and Monte Carlo Tree Search (MCTS). The model was trained
from random initialization and is already a strong opponent, but development is
ongoing as we identify weaknesses and release improved versions.

## Quick start

Python 3.10 or later is required. A virtual environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python web_server.py
```

Open `http://127.0.0.1:8765` in a browser. The repository includes the compact
Sophon v1.1 inference model, so no training checkpoint download is needed to
start playing.

## Game rules

- The first move may be played in any empty cell.
- The local cell chosen in one small board determines the small board in which
  the opponent must play next.
- A small board closes as soon as it is won by three in a row or filled as a
  draw. No further moves may be played there.
- If the required small board is already closed, the next player may move in
  any open small board.
- This project uses the score-only ending: the game ends after all nine small
  boards close, and the player who owns more small boards wins. An equal score
  is a draw. A three-in-a-row pattern on the large board does not end the game.

## Project status

1. **Rules engine and tests — complete:** state representation, coordinates,
   legal moves, small-board closure, final scoring, and neural input encoding.
2. **Baselines and environment validation — complete:** random and heuristic
   opponents, stress games, and invariant checks.
3. **Policy-value network — complete:** a `(6, 9, 9)` input tensor, 81 policy
   logits, and one scalar value estimate.
4. **Monte Carlo Tree Search — complete:** legal-action masking, PUCT, value
   backup, and root exploration noise.
5. **Self-play pipeline — complete:** `(state, MCTS policy, result)` examples,
   D4 symmetry augmentation, and a fixed-capacity replay buffer.
6. **Training loop — complete:** self-play, optimization, checkpointing, and
   recovery from full training checkpoints.
7. **Evaluation and tuning — complete:** alternating-side matches against
   random and heuristic baselines, heuristic curriculum training, and pure
   self-play fine-tuning.
8. **Human play and continued improvement — v1.1 released:** command-line and
   browser play, game-record auditing, deeper self-play, paired-opening model
   leagues, D4 symmetry training and inference, and guarded champion promotion.

## Current model: Tic-Tac-Toe Sophon v1.1

The compact champion model is stored at `checkpoints/best.pth`; the named
release is at `checkpoints/releases/Tic-Tac-Toe_Sophon_v1.1.pth`. The v1.0
release remains available separately.

Starting from the full v1.0 training checkpoint, v1.1 added 200 self-play games
at 96 MCTS simulations per move, 10,575 training positions, and 1,000 optimizer
updates. Training combined existing D4 data augmentation with policy and value
consistency losses over all eight rotations/reflections. At inference time the
eight transformed positions are evaluated in one batch, their policies are
mapped back to the original board, and their outputs are averaged. This makes
the released predictions D4-consistent to numerical precision. It uses eight
network evaluations per position, so play may be slower than v1.0 at the same
search budget; batching means wall-clock cost is not necessarily eightfold.

In the formal paired-opening league against v1.0, each model used 96 MCTS
simulations per move. Across 200 openings with sides swapped (400 games), v1.1
scored 227 wins, 37 losses, and 136 draws: a 73.75% score rate. The paired
bootstrap 95% interval was 70.63%–76.88%. The candidate passed the 55% score
threshold and 50% lower-bound threshold. The raw network's mean policy
symmetry discrepancy improved from 0.22356 to 0.19899; its mean value
discrepancy improved from 0.19381 to 0.16019 on the fixed audit positions.
See `checkpoints/releases/v1.1_release_report.json` for the full result and
audit protocol. This is a model-vs-model result, not a human win rate.

### Earlier v1.0 release

Sophon v1.0 added 400 pure self-play games at 96 MCTS simulations per move,
producing 21,946 new training positions. Across the full project, training used
760 pure self-play games and 400 heuristic-curriculum games. The network always
started from random parameters and did not reuse weights from the classic
Tic-Tac-Toe project.

The released model was evaluated for 40 games per opponent, alternating sides
and using 96 MCTS simulations per move:

| Opponent | Win-Loss-Draw | Score rate | Average small-board margin |
|---|---:|---:|---:|
| Random baseline | 40-0-0 | 100% | +3.13 |
| Strong heuristic baseline | 22-5-13 | 71.25% | +0.55 |

The formal league against the frozen previous champion used 100 random legal
openings, with the players swapping sides for every opening (200 games total).
Sophon v1.0 scored 98 wins, 35 losses, and 67 draws for a 65.75% score rate.
The paired bootstrap 95% interval was 60.5%–70.75%, passing both the 55% score
threshold and the 50% confidence-interval lower-bound threshold. See
`checkpoints/releases/v1.0_release_report.json` for that release's report.

Human evaluation is kept separate from model-league evaluation. Human game
records are local runtime data and are not published in this repository. Run
`python clean_records.py` to generate a current audit. The human benchmark
requires at least 20 complete, unassisted games and an AI non-loss rate of at
least 50%. Assisted games do not count, and a result is not declared conclusive
while the sample remains below the minimum.

## Play against the AI

### Browser interface

```powershell
python web_server.py
```

On Windows, `start_web.bat` starts the service if necessary, opens the browser,
and prints a LAN address that phones on the same Wi-Fi network can use. Running
it repeatedly does not create duplicate servers. Closing the browser does not
stop the server.

The computer and phone must be able to reach each other on the same private
IPv4 network. The LAN server accepts `10.*`, `172.16.*`–`172.31.*`, and
`192.168.*` addresses while rejecting public hosts, cross-origin requests, and
non-local clients. Guest or campus Wi-Fi may isolate devices; a phone hotspot
is often a simple alternative.

For a Windows network marked Public, the following command may be run once in
an **Administrator PowerShell**. It allows TCP 8765 only from the local subnet:

```powershell
netsh advfirewall firewall add rule name="Tic-Tac-Toe Sophon 8765" dir=in action=allow protocol=TCP localport=8765 remoteip=LocalSubnet profile=public
```

The browser UI is responsive and requires no web framework. HTML, CSS, and
JavaScript are served locally, while PyTorch inference runs in the Python
service. Opening `web/index.html` directly will not load the model.

- **Match:** play the champion from `checkpoints/best.pth` as X or O.
- **Guided play:** evaluate every legal move and highlight the top three from
  the human player's perspective. The panel shows coordinates, estimated score
  after the move, and the change from the current position.
- The probability bar displays human and AI values that sum to 100%. The model
  estimates approximately `P(win) - P(loss)`, which the UI maps with
  `(v + 1) / 2`. It is therefore an expected score with draws worth half a
  point, not a calibrated pure win probability.
- Every accepted move is saved atomically to
  `records/human_games/web_<game-id>.json`. Active snapshots use the `aborted`
  status with reason `web_autosave_incomplete`; they are resumable snapshots,
  not finished results.
- Refreshing the page resumes the current game while the service is running.
  Restarting the service starts a new game without deleting old records.

Search budgets and the port can be changed explicitly:

```powershell
python web_server.py --port 8765 --simulations 128 --analysis-simulations 24
```

`--simulations` controls AI moves and current-position evaluation.
`--analysis-simulations` is the budget for **each** legal candidate in guided
play. The defaults are 128 and 24 respectively. Restart the server after
replacing the champion model.

### Command-line interface

Moves may be entered as global `row column` coordinates from 1 to 9, such as
`3 7`, or as action numbers from 0 to 80. Closed and occupied cells are rejected.

```bash
# Play first. Without --side, the program asks interactively.
python play.py --side first --simulations 128

# Play second.
python play.py --side second --simulations 128

# Non-interactive demonstrations.
python play.py --demo heuristic --simulations 128
python play.py --demo self --simulations 128
```

Larger search budgets usually produce stronger but slower play. The CLI default
of 48 simulations is suitable for casual games.

## Human game records

Human games are saved under `records/human_games/` by default. Completed games
and games aborted with `q` both leave JSON records; demonstration games do not.
Each record contains:

- model path and training iteration, search parameters, and random seed;
- actor, action number, global coordinates, small board, and local cell for
  every move;
- the forced board before each move and score/board status afterward;
- final state, outcome, timestamps, and a SHA-256 integrity digest.

```bash
# Save records to a custom directory.
python play.py --side first --records-dir records/my_games

# Disable recording temporarily.
python play.py --side first --no-record
```

Use `game_record.load_record(path)` to load a record and
`game_record.replay_record(path)` to validate and replay it strictly from an
empty board. Records are never added to training automatically.

Audit and clean local records with:

```bash
# Audit completed matches, assisted games, incomplete games, and invalid files.
python clean_records.py

# Move incomplete or invalid records to a recoverable quarantine; nothing is deleted.
python clean_records.py --quarantine records/quarantine
```

Assisted records remain available for qualitative analysis but are excluded
from the human benchmark because the player saw model suggestions.

## Public quick tunnels

`launch_public.py` can expose a rate-limited, loopback-only game server through
a temporary HTTPS URL. New-game and MCTS requests have queue and rate limits.

The default localtunnel path expects a Mihomo mixed/SOCKS5 proxy on
`127.0.0.1:7897`:

```powershell
python launch_public.py
```

Use `--proxy-host` and `--proxy-port` to change the proxy. Quick-tunnel URLs
change when the machine or tunnel restarts and are intended for temporary play
sessions. A stable deployment should use an authenticated named tunnel and a
domain you control.

If the current network permits Cloudflare Tunnel edge connections:

```powershell
python launch_public.py --provider cloudflare
```

`cloudflared` is not distributed with this repository. See `tools/README.md`
for the official download and placement instructions.

## Training and evaluation

Common commands, run from the repository root:

```bash
python -B -m unittest discover -s tests -v
python validate_environment.py --games 5000
python train.py
python evaluate.py checkpoints/best.pth --games 40 --simulations 96
```

Full checkpoints contain the optimizer and replay buffer and can resume
training. The compact `best.pth` file is intended only for inference and as the
current league champion. Full training checkpoints are excluded from the public
repository; a fresh clone can first run `python train.py` to generate one. The
following reproduces the v1.1 continuation only if the local full v1.0
checkpoint is available:

```bash
python train.py --resume checkpoints/candidate_v1/trained.pth \
  --iterations 55 --games-per-iter 20 --simulations 96 --train-steps 100 \
  --resume-learning-rate 0.0002 --symmetry-fraction 0.25 \
  --symmetry-policy-weight 0.2 --symmetry-value-weight 0.2 \
  --inference-symmetry d4 --model-version v1.1 \
  --checkpoint-dir checkpoints/candidate_v11

python symmetry_audit.py checkpoints/releases/Tic-Tac-Toe_Sophon_v1.0.pth \
  --raw --report checkpoints/candidate_v11/v1_baseline_symmetry.json
python symmetry_audit.py checkpoints/candidate_v11/trained.pth \
  --raw --report checkpoints/candidate_v11/v11_raw_symmetry.json
python symmetry_audit.py checkpoints/candidate_v11/trained.pth \
  --report checkpoints/candidate_v11/v11_ensemble_symmetry.json
python league.py checkpoints/candidate_v11/trained.pth \
  checkpoints/releases/Tic-Tac-Toe_Sophon_v1.0.pth \
  --pairs 200 --opening-plies 4 --simulations 96 \
  --report checkpoints/candidate_v11/league_vs_v1.json --no-promote
python release_v11.py --dry-run
```

For every random legal opening, the league swaps model sides and uses identical
search budgets without exploration noise. Promotion requires a candidate score
rate of at least 55% and a paired-bootstrap 95% interval lower bound of at least
50%. For the v1.1 release, `release_v11.py` additionally requires 200 opening
pairs, the symmetry audit improvements, and D4 inference consistency. Its
default inputs are local generated reports and checkpoints; do not promote a
freshly trained model using the published v1.1 report from a different run.

The rules engine depends only on NumPy; neural inference and training use
PyTorch. Earlier environment validation completed 5,000 random stress games
without a rules error. The heuristic baseline scored 499 wins, 0 losses, and
1 draw in 500 alternating-side games against the random baseline.

## Repository layout

| Path | Contents | Public repository |
|---|---|---|
| Root `*.py` | Rules, model, MCTS, training, evaluation, and entry points | Included |
| `tests/` | Automated test suite | Included |
| `web/` | Framework-free browser client | Included |
| `checkpoints/releases/` | Compact releases and formal evaluation report | Included |
| `checkpoints/best.pth` | Default compact champion | Included |
| Other `checkpoints/` files | Full training state and intermediate runs | Local only |
| `records/` | Human games, audits, and quarantine data | Local only |
| `logs/` | Server and tunnel logs | Generated locally |
| `tools/` | Optional third-party tunnel binaries | Documentation only |

`.gitignore` excludes private records, logs, full training artifacts, caches,
and downloaded third-party binaries. These files may remain on the development
machine without being included by `git add .`.

## GitHub publication

Before publishing, review ignored files with `git status --short --ignored`.
Commit the source, compact release model, and public report; keep full training
checkpoints and private game records local.

```bash
git add .
git commit -m "Release Sophon v1.1"
git push -u origin main
```

No open-source license is selected yet. Add a `LICENSE` file before publication
if you intend to grant reuse, modification, or redistribution rights.
