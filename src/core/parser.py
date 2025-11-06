import argparse

def build_parser(desc: str = ""):
    p = argparse.ArgumentParser(description=desc or "Run")
    # general
    p.add_argument("--mode", choices=["train", "explain"], default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")  # "cuda", "cpu", or "auto"
    p.add_argument("--out_dir", type=str, default="results")

    # data
    p.add_argument("--dataset", choices=["indian_pines", "pavia", "salinas", "synthetic"],
                   default="synthetic")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--mat_x", type=str, default="")     # optional: .mat path to cube (HxWxB)
    p.add_argument("--mat_y", type=str, default="")     # optional: .mat path to labels (HxW)
    p.add_argument("--mat_x_key", type=str, default="X")
    p.add_argument("--mat_y_key", type=str, default="y")
    p.add_argument("--knn", type=int, default=8)

    # model
    p.add_argument("--hid", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--norm_mode", type=str, default="PN-SI")
    p.add_argument("--norm_scale", type=float, default=1.0)

    # train
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=10)

    # explain
    p.add_argument("--explainer", choices=["ig", "saliency", "gradcam"], default="ig")
    p.add_argument("--topk", type=int, default=10)  # top-k bands per class for summary CSV
    return p
