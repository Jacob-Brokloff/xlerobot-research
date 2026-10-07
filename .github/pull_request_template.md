## Review

- [ ] No secrets, tokens, or credentials are in the diff
- [ ] `train:hf` or `train:homelab` is applied only when this merge should route a training job
- [ ] At most one of those two labels is applied
- [ ] Auto-merge stays off

Training still does not start unless the matching repository variable is `true` (`XLEROBOT_DISPATCH_HF` or `XLEROBOT_DISPATCH_HOMELAB`). A manual kick is `.github/workflows/train-hf.yml`.
