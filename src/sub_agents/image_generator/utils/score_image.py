from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Dict, Any

from PIL import Image


def load_openclip(model_name: str, pretrained: str, device: str):
    import torch
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return torch, model, preprocess, tokenizer


def load_aesthetic_head(torch, model, device: str, head_path: str):
    if not os.path.exists(head_path):
        raise FileNotFoundError(
            f"Aesthetic head weights not found: {head_path}. "
            f"Set AESTHETIC_HEAD_PATH or place weights at models/aesthetic_head.pth"
        )

    # infer dim
    with torch.no_grad():
        dummy = torch.zeros((1, 3, 224, 224), device=device)
        img_feat = model.encode_image(dummy)
        dim = img_feat.shape[-1]

    head = torch.nn.Linear(dim, 1).to(device)
    head.eval()

    sd = torch.load(head_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    cleaned = {}
    for k, v in sd.items():
        ck = k[len("module.") :] if k.startswith("module.") else k
        cleaned[ck] = v

    head.load_state_dict(cleaned, strict=False)
    return head


def score_image(
    *,
    image_path: str,
    prompt: str,
    openclip_model: str,
    openclip_pretrained: str,
    aesthetic_head_path: str,
    device: str,
) -> Dict[str, Any]:
    torch, model, preprocess, tokenizer = load_openclip(openclip_model, openclip_pretrained, device)
    aes_head = load_aesthetic_head(torch, model, device, aesthetic_head_path)

    # load image
    with open(image_path, "rb") as f:
        b = f.read()
    im = Image.open(io.BytesIO(b)).convert("RGB")

    image_in = preprocess(im).unsqueeze(0).to(device)
    text_tokens = tokenizer([prompt]).to(device)

    with torch.no_grad():
        img_feat = model.encode_image(image_in)
        txt_feat = model.encode_text(text_tokens)

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        clip_sim = (img_feat * txt_feat).sum(dim=-1).item()
        aes_score = aes_head(img_feat).squeeze().item()

    return {
        "ok": True,
        "clip_score": float(clip_sim),
        "aesthetic_score": float(aes_score),
        "openclip_model": openclip_model,
        "openclip_pretrained": openclip_pretrained,
        "aesthetic_head_path": aesthetic_head_path,
        "device": device,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out_json", required=True)

    ap.add_argument("--openclip_model", default=os.getenv("OPENCLIP_MODEL", "ViT-B-32"))
    ap.add_argument("--openclip_pretrained", default=os.getenv("OPENCLIP_PRETRAINED", "openai"))
    ap.add_argument("--aesthetic_head_path", default=os.getenv("AESTHETIC_HEAD_PATH", "models/aesthetic_head.pth"))
    ap.add_argument("--device", default=os.getenv("OPENCLIP_DEVICE", "cpu"))

    args = ap.parse_args()

    try:
        result = score_image(
            image_path=args.image_path,
            prompt=args.prompt,
            openclip_model=args.openclip_model,
            openclip_pretrained=args.openclip_pretrained,
            aesthetic_head_path=args.aesthetic_head_path,
            device=args.device,
        )
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # exit code non-zero if failed (useful for subprocess)
    sys.exit(0 if result.get("ok") else 2)


if __name__ == "__main__":
    main()