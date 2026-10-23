"""Small helper shared by the experiment scripts (added for the public release).

It lets you change any configuration constant defined at the top of a script
from the command line, without editing the file, and saves every figure:

    python src/bbvi_synthetic.py --STRUCTURED True --FRAME_STEP 3 --seed 0
    python src/bbvi_synthetic.py --help-config        # list the constants you can set

* `--NAME VALUE` overrides the module-level constant NAME. The value is parsed
  as Python/JSON (True, 3, 0.0873, [0.25, 0.5, 1.0], "path/to/file.mp4").
* `--seed N` fixes numpy and torch random seeds (the original runs used none).
* `--output-dir DIR` is where figures and run_config.json go (default: outputs/).
* `--show` also opens the figures in a window; otherwise they are only saved.

The estimation code itself is not touched by this module.
"""
import ast
import json
import os
import sys

_FIG_COUNTER = [0]


def _parse_value(text, default):
    if isinstance(default, str):
        return text
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def configure(g, output_key="OUTPUT_DIR"):
    """Apply command-line overrides to the module globals `g` and set up figure saving."""
    import matplotlib
    import matplotlib.pyplot as plt

    argv = sys.argv[1:]
    settable = {k: v for k, v in g.items()
                if not k.startswith("_") and k not in ("device",)
                and (isinstance(v, (bool, int, float, str, list, tuple, type(None)))
                     or (hasattr(v, "ndim") and getattr(v, "ndim", 0) == 1))}

    if "--help-config" in argv:
        print("Configurable constants (use --NAME VALUE):")
        for k, v in settable.items():
            print(f"  --{k:<22} {v!r}")
        sys.exit(0)

    show, seed, out_dir = False, None, g.get(output_key, "outputs")
    changed = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--show":
            show = True; i += 1; continue
        if tok in ("--seed", "--output-dir"):
            if i + 1 >= len(argv):
                sys.exit(f"{tok} needs a value")
            if tok == "--seed":
                seed = int(argv[i + 1])
            else:
                out_dir = argv[i + 1]
            i += 2; continue
        if tok.startswith("--"):
            name = tok[2:]
            if name not in settable:
                sys.exit(f"Unknown option {tok}. Run with --help-config to list the options.")
            if i + 1 >= len(argv):
                sys.exit(f"{tok} needs a value")
            g[name] = changed[name] = _parse_value(argv[i + 1], settable[name])
            i += 2; continue
        sys.exit(f"Unexpected argument {tok!r}")

    # keep simple derived constants consistent
    import numpy as np
    for k in changed:                       # keep numpy arrays as numpy arrays
        if hasattr(settable[k], "ndim"):
            g[k] = np.asarray(g[k], dtype=settable[k].dtype)
    if "radii" in changed and "num_radii" in g:
        g["num_radii"] = len(g["radii"])
    if "pyramid_scales" in changed and "num_levels" in g:
        g["num_levels"] = len(g["pyramid_scales"])

    if seed is not None:
        import numpy as np
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass

    os.makedirs(out_dir, exist_ok=True)
    g[output_key] = out_dir

    if not show:
        plt.switch_backend("Agg")
    script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    # name figures after the experiment switches that were changed, e.g.
    # bbvi_synthetic_STRUCTURED-True_FRAME_STEP-3_fig01.png, so runs do not overwrite each other
    tag = "_".join(f"{k}-{changed[k]}" for k in changed
                   if isinstance(changed[k], (bool, int)) and k not in ("OUTER_EM_ITERS", "E_UPDATES", "M_UPDATES", "S_mc"))
    if tag:
        script = f"{script}_{tag}"

    def _save_and_maybe_show(*args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            _FIG_COUNTER[0] += 1
            stem = os.path.join(out_dir, f"{script}_fig{_FIG_COUNTER[0]:02d}")
            fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
            fig.savefig(stem + ".pdf", bbox_inches="tight")
            print(f"[figure] saved {stem}.png/.pdf")
        if show:
            _orig_show(*args, **kwargs)
        plt.close("all")

    _orig_show = plt.show
    plt.show = _save_and_maybe_show

    record = {k: (g[k].tolist() if hasattr(g[k], "tolist") else g[k])
              for k in settable if k in g}
    record.update(seed=seed, overridden=sorted(changed), argv=sys.argv)
    try:
        import torch
        record.update(torch=torch.__version__, cuda=torch.cuda.is_available())
    except ImportError:
        pass
    with open(os.path.join(out_dir, f"{script}_run_config.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)
    if changed:
        print("[config] overridden:", {k: g[k] for k in changed})
