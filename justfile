# Setup the repo .venv via uv
setup:
    uv sync

# Run the test suite
test:
    uv run pytest

# Run static analysis and automatically fix issues where possible
check:
    uvx ruff check . --fix

# Format code according to project style
format:
    uvx ruff format .

# Run formatting and linting (CI-style target)
clean: format check

# --- The figures ------------------------------------------------------
#
# There are three results these studies produce, and one script each.
# No recipe carries an experiment parameter: the axes a study
# sweeps -- charts, pilot budgets, lambda grid, rate, dataset, encoder
# pairs -- all live in `config/hydra/`, where they can be read, diffed
# and overridden by name. A `just --set` variable can do none of those,
# and split the description of a run across two files.
#
# Everything below is a Hydra override, forwarded verbatim -- but
# `just` interpolates the arguments into the recipe *unquoted*, so bash
# re-splits any override containing a space and Hydra sees half of one.
# Write them without spaces, which its grammar accepts everywhere:
#
#   just lambda-sweep charts=[{preprocess:pca}] ranks=[0.8,0.6,0.2]   # yes
#   just lambda-sweep 'charts=[{preprocess: pca}]'                    # no
#
# Call the script directly when an override really needs a space.
#
#   just lambda-sweep data=semasia_mnist
#   just lambda-sweep pilots.counts=[500,1000] symbol_divisor=2
#   just pilot-sweep seeds=[0,1,2] wandb.mode=disabled
#
# Both recipes retry rather than `set -e`. These runs segfault
# intermittently on this machine (roughly half of full-grid runs, in
# varying native frames; cause not yet identified), and both scripts
# write each unit of work -- a (chart, budget) cell, a seed -- to disk as
# it finishes and skip what is already there on the next invocation. So a
# crash costs the unit in flight and nothing more, and re-running is
# cheap by construction. Pass `resume=false` to force a refit.

# Figure (i): RKA against Procrustes over the RKHS regularisation, one
# figure per (chart, pilot budget, metric).
#
#   just lambda-sweep
#   just lambda-sweep lam.min=1e-6 lam.max=1e-1 lam.per_decade=8
#   just lambda-sweep plot_only=true      # redraw, fit nothing
[doc('Figure (i): RKA vs Procrustes over the RKHS regularisation')]
lambda-sweep *ARGS:
    #!/usr/bin/env bash
    set -uo pipefail
    for attempt in 1 2 3 4 5 6; do
        uv run scripts/lambda_sweep.py {{ARGS}} && exit 0
        code=$?
        # Only a signal is worth retrying. The retries exist for the
        # intermittent segfault (bash reports 128+signo); a bad override
        # or a raised exception exits 1 and will do so six times over,
        # burying the one line that says what is wrong.
        if [ "$code" -lt 128 ]; then
            echo "!! exited $code -- not a crash, so not retried."
            exit $code
        fi
        echo "!! attempt $attempt died on signal $((code - 128)) -- the"
        echo "   cells that finished are on disk; retrying, resume=true"
        echo "   skips them."
    done
    echo "!! gave up after 6 attempts; the last exit code was $code."
    exit $code

# Figure (ii): RKA, Procrustes and the MLP baselines against the pilot
# budget, with RKA's lambda read back per budget from figure (i).
# Run `just lambda-sweep` first, on the same axes.
#
#   just pilot-sweep
#   just pilot-sweep plot_only=true drop=[direct_mlp,residual_mlp]
#   just pilot-sweep drop=[direct_mlp:stratified,residual_mlp:stratified]
[doc('Figure (ii): RKA, Procrustes and the MLP baselines over the pilot budget')]
pilot-sweep *ARGS:
    #!/usr/bin/env bash
    set -uo pipefail
    for attempt in 1 2 3 4 5 6; do
        uv run scripts/pilot_sweep.py {{ARGS}} && exit 0
        code=$?
        # See `lambda-sweep`: retry the crash, not the usage error.
        if [ "$code" -lt 128 ]; then
            echo "!! exited $code -- not a crash, so not retried."
            exit $code
        fi
        echo "!! attempt $attempt died on signal $((code - 128)) -- the"
        echo "   seeds that finished are on disk; retrying, resume=true"
        echo "   skips them."
    done
    echo "!! gave up after 6 attempts; the last exit code was $code."
    exit $code

# Figure (iii): RKA and Procrustes read back from figure (i), CCA, SVCCA
# and Proto-PFE fitted on the same pilots, all against the compression dimension
# at one fixed pilot budget. Run `just lambda-sweep` first with a `ranks`
# axis and `pilots.n_pilots` among its budgets; the run refits Procrustes
# to check it is reading the numbers it thinks it is.
#
#   just dimension-sweep 'ranks=[16,32,64,128]' decoder.kind=linear
#   just dimension-sweep pilots.n_pilots=4096 'ranks=[16,32,64,128]'
#   just dimension-sweep plot_only=true 'ranks=[16,32,64,128]'
[doc('Figure (iii): RKA, Procrustes, CCA, SVCCA and Proto-PFE over the compression dimension')]
dimension-sweep *ARGS:
    #!/usr/bin/env bash
    set -uo pipefail
    for attempt in 1 2 3 4 5 6; do
        uv run scripts/dimension_sweep.py {{ARGS}} && exit 0
        code=$?
        # See `lambda-sweep`: retry the crash, not the usage error.
        if [ "$code" -lt 128 ]; then
            echo "!! exited $code -- not a crash, so not retried."
            exit $code
        fi
        echo "!! attempt $attempt died on signal $((code - 128)) -- the"
        echo "   fits that finished are on disk; retrying, resume=true"
        echo "   skips them."
    done
    echo "!! gave up after 6 attempts; the last exit code was $code."
    exit $code

# Figures (ii) and (iii) averaged over encoder pairs instead of seeds.
# Fits nothing: it pools what `pilot-sweep` and `dimension-sweep` wrote
# for every pair listed in `config/hydra/pair_average.yaml`, and for a
# pair that is missing it stops and prints the commands that produce it.
#
#   just pair-average
#   just pair-average 'pilot.drop=[direct_mlp:round_robin]'
[doc('Figures (ii) and (iii) averaged over encoder pairs')]
pair-average *ARGS:
    uv run scripts/pair_average.py {{ARGS}}

# Figures (i) and (ii), in the order they depend on each other. Figure
# (iii) is left out: it needs a `ranks` axis the default axes do not set.
# The overrides go
# to both, which is what you want for the axes they share (`data=`,
# `symbol_divisor=`, `charts=`, `pilots.counts=`) and not for the ones
# only one of them has -- pass those to the recipe that owns them.
#
#   just figures
#   just figures data=semasia_mnist
[doc('Both figures, in dependency order')]
figures *ARGS: (lambda-sweep ARGS) (pilot-sweep ARGS)
