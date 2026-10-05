# Franka Panda parser fixture

This is a **parser fixture**, not a runnable MuJoCo model. It holds the MuJoCo
Menagerie Franka Panda from <https://github.com/google-deepmind/mujoco_menagerie>,
directory `franka_emika_panda/`, at commit `feadf76d42f8a2162426f7d226a3b539556b3bf5`. The license is Apache 2.0; see
`LICENSE`.

- `scene.xml`, `panda.xml`, `README.md`, and `LICENSE` are copied byte-for-byte.
- The 67 files under `assets/` are empty placeholders with the upstream
  names. The loader only resolves and hashes meshes, and the originals total about 33 MB,
  which stays out of this repository. MuJoCo cannot load the model until they are
  replaced.

`provenance.json` records the commit and the SHA-256 and size of every copied file and
of every original mesh. `tests/test_loaders.py` enforces it: a copied file that drifts,
or a placeholder that goes missing, gets renamed, or acquires content, fails the suite.

To make a runnable copy outside this repository, replace each placeholder from a
Menagerie checkout:

```sh
for f in assets/*; do git -C /path/to/mujoco_menagerie show feadf76d42f8a2162426f7d226a3b539556b3bf5:franka_emika_panda/$f > $f; done
```

Each file then matches its `upstream_sha256` in `provenance.json`.
