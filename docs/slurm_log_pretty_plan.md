Analyse the todo: Slurm log names (.err and .out) -- appending the trial name to the end, after the jobID, how complicated is this?

Complexity: low, but it needs a wrapper — you can't do it purely inside the sbatch script.

The blocker is that #SBATCH --output=… directives are parsed at submission time, before $@ is available inside the script. SLURM filename patterns only support %j, %x (job-name), %u, etc. — no way to inject a CLI arg at that stage. --job-name has the same limitation (the directive is static).

Two viable approaches:

1. Wrapper script (recommended, ~15 lines) — slurm_scripts/submit_train.sh that:

parses --trial path/to/trial.yaml from $@,
derives a slug (basename "${trial%.yaml}"),
calls sbatch --output=slurm_logs/train/slurm-train-ray-%j-<slug>.out --error=…%j-<slug>.err --job-name=ray-train-<slug> slurm_scripts/slurm_train_ray.sh "$@".
CLI sbatch flags override the #SBATCH directives in the script, so the existing script needs no changes. Same pattern works for slurm_eval_ray.sh. Total work: one new tiny wrapper per entry point (train + eval = 2 files), plus a one-line README update.

2. In-script symlink (uglier, ~5 lines) — keep the current jobid-only filenames, and after parsing $@ create slurm-train-ray-<jobid>-<slug>.{out,err} symlinks pointing at the real files. Works because SLURM keeps the underlying fd open by inode. Downside: the "real" file is still jobid-only, so ls shows both.

Go with (1). Estimated effort: 20-30 min including testing on a small trial.