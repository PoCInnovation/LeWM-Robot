"""Le predictor utilise-t-il les actions ? Loss sur la val avec vraies actions / mélangées / nulles."""
import sys, torch
sys.path.insert(0, sys.argv[1])
torch.set_num_threads(6)
from src.checkpoint import load_world_model
from src.config import load_config
from src.data import split_train_val
from src.latents import LatentPairs, load_encoded
ckpt, data_path = sys.argv[2], sys.argv[3]
cfg = load_config(sys.argv[1] + "/configs/rtx4090.yaml")
pred, fusion, meta = load_world_model(ckpt, "cpu")
pred.eval(); fusion.eval()
from src.latents import frameskip_pairs
data = frameskip_pairs(load_encoded(data_path), meta["frameskip"])
_, val = split_train_val(data, seed=cfg["seed"], val_fraction=0.2)
aug = data.get("is_augmented")
if aug is not None:
    val = val[~torch.as_tensor(aug).flatten().bool()[val]]
lat = LatentPairs(data, "cpu", "cpu", verbose=False)
lat.normalize_actions(meta["action_stats"])
g = torch.Generator().manual_seed(0)
idx = val[torch.randperm(len(val), generator=g)[:1024]]
res = {k: 0.0 for k in ["identité", "vraies actions", "actions mélangées", "action nulle (moyenne)", "actions x2"]}
with torch.no_grad():
    for b in idx.split(128):
        zw, zg, zw1, zg1, a = lat.fetch(b)
        z, z1 = fusion(zw.float(), zg.float()), fusion(zw1.float(), zg1.float())
        a = a.float()
        mse = lambda p: float(((p - z1) ** 2).mean()) * len(b)
        res["identité"] += mse(z)
        res["vraies actions"] += mse(pred(z, a))
        res["actions mélangées"] += mse(pred(z, a[torch.randperm(len(b), generator=g)]))
        res["action nulle (moyenne)"] += mse(pred(z, torch.zeros_like(a)))
        res["actions x2"] += mse(pred(z, a * 2))
true = res["vraies actions"]
for k, v in res.items():
    print("%-24s mse %.6f   (x%.3f vs vraies actions)" % (k, v / len(idx), v / true))
