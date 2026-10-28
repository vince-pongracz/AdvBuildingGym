

snapshots/20260714_161830_gen_battery_lin_so_only_17kW_sac_cp_pp
snapshots/20260714_235241_gen_battery_lin_so_only_15kW_sac_cp_pp
snapshots/20260714_162614_gen_battery_lin_so_all_sac_v2_pp
snapshots/20260714_165943_gen_battery_lin_capacity_sampled_sac_pp

snapshots/20260714_235444_gen_battery_lin_so_only_17kW_ppo_cp_pp
snapshots/20260714_235342_gen_battery_lin_so_only_15kW_ppo_cp_pp
snapshots/20260714_235542_gen_battery_lin_so_all_ppo_v2_cp_pp
snapshots/20260714_235647_gen_battery_lin_capacity_sampled_ppo_cp_pp

snapshots/20260714_225444_gen_battery_lin_so_only_17kW_dreamer_pp
snapshots/20260714_225342_gen_battery_lin_so_only_15kW_dreamer_pp
snapshots/20260714_225542_gen_battery_lin_so_all_dreamer_v2_pp
snapshots/20260714_225647_gen_battery_lin_capacity_sampled_dreamer_pp

-- eval started ^^
--- training started ^^


---
Training script:

for s in 46 71 97 141; do
  python -m tools.snapshot.submit_snapshot \
    --snapshot snapshots/20260714_225647_gen_battery_lin_capacity_sampled_dreamer_pp \
    --kind train --common-seed $s --sbatch="--time=16:00:00"
done

---
Eval script:

SNAPS=(
  snapshots/20260714_225444_gen_battery_lin_so_only_17kW_dreamer_pp snapshots/20260714_225342_gen_battery_lin_so_only_15kW_dreamer_pp snapshots/20260714_225542_gen_battery_lin_so_all_dreamer_v2_pp snapshots/20260714_225647_gen_battery_lin_capacity_sampled_dreamer_pp
)

for SNAP in "${SNAPS[@]}"; do
  for cs in 46 71 97 141; do
    CKPT=$(ls -d $SNAP/runs/train_*_cseed${cs}/models/*/ray/*/*/*/checkpoint_* 2>/dev/null | sort | tail -n 1)
    if [ -z "$CKPT" ]; then echo "$SNAP cseed${cs}: no checkpoint found, skipping" >&2; continue; fi
    CKPT=$(realpath "$CKPT")
    echo "$SNAP cseed${cs} -> $CKPT"
    python -m tools.snapshot.submit_snapshot --snapshot $SNAP --kind eval \
        --seed 42 --common-seed $cs --checkpoint "$CKPT" -- --episodes 10 --plot-all
  done
done