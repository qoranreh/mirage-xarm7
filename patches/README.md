# Local compatibility patches

The upstream projects are Git submodules, so local fixes inside them are kept
as explicit patch files instead of leaving an untracked dirty submodule.

After initializing the submodules, apply the single-GPU checkpoint fix from
the repository root:

```bash
git submodule update --init --recursive
git -C robomimic apply ../patches/robomimic-cpu-checkpoint-load.patch
```

The patch loads checkpoint tensors on CPU first. Robomimic then moves the
policy to the selected runtime device. This avoids the upstream hard-coded
`cuda:4` device, which fails on a one-GPU machine.
