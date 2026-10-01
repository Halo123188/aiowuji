# Wuji Hand 2 MuJoCo model

Unmodified copies of Wuji Technology's official Wuji Hand 2 (Beta 2) MuJoCo
models, left and right, with the meshes they reference:

- Source: <https://github.com/wuji-technology/wuji-description>,
  `hand2/hand2_beta2/body/{mjcf,meshes}` at commit
  `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95` (2026-09-11).
- License: MIT, Copyright (c) 2025 Wuji Technology (see `LICENSE`).

Beta 2 is Beta 1 plus a tactile sensor pad on each fingertip, so its fingertip
collision geometry is the more conservative of the two for collision checks.

The model's joints are the hardware's, in aiowuji's order (per finger:
S1 flexion, S2 spread, S3, S4), and their ranges equal the documented joint
limits. Positive S2 moves every finger toward the pinky side; hand poses
measured on a right hand are free of interpenetration only under this
convention. aiowuji uses the collision meshes (geom group 2) for kinematic
clearance checks in `aiowuji.model`.
