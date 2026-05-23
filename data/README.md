# data/

Place your input files here before running the script.

## Required Files

| File | Description |
|------|-------------|
| `fasciculation_simulation_accel_decel_test_new4.mp4` | Synthetic video from `Fasciculation-Simulation` repo |
| `ring_initial_rotation_values.csv` | Ground-truth rotation angles from `Fasciculation-Simulation` repo |

## CSV Format

```
Frame, Rectangle_0, Rectangle_1, ..., Rectangle_9
0,     0.0,         0.0,         ..., 0.0
1,     <angle_rad>, ...
```

Both files are outputs of the **[Fasciculation-Simulation](https://github.com/ShzdiTrnum/Fasciculation-Simulation)** repository.

> **Note:** Data files are excluded from version control via `.gitignore`.
