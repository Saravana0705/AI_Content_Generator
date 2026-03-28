from __future__ import annotations
from typing import Any, Dict, List

STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "photorealistic": {
        "positive_en": "photorealistic, natural lighting, high detail, sharp focus",
        "positive_de": "fotorealistisch, natürliche Beleuchtung, hohe Details, scharfer Fokus",
        "negative_en": ["blurry", "low resolution", "text", "watermark", "logo"],
        "negative_de": ["unscharf", "niedrige Auflösung", "Text", "Wasserzeichen", "Logo"],
    },
    "anime": {
        "positive_en": "anime style, clean linework, vibrant colors, studio quality",
        "positive_de": "Anime-Stil, klare Linienführung, lebendige Farben, Studioqualität",
        "negative_en": ["text", "watermark", "logo"],
        "negative_de": ["Text", "Wasserzeichen", "Logo"],
    },
    "3d": {
        "positive_en": "3D render, soft global illumination, high poly, cinematic lighting",
        "positive_de": "3D-Render, weiche globale Beleuchtung, hohe Polygonzahl, cinematische Beleuchtung",
        "negative_en": ["text", "watermark", "logo"],
        "negative_de": ["Text", "Wasserzeichen", "Logo"],
    },
    "illustration": {
        "positive_en": "digital illustration, crisp shapes, pleasing composition",
        "positive_de": "digitale Illustration, klare Formen, ansprechende Komposition",
        "negative_en": ["text", "watermark", "logo"],
        "negative_de": ["Text", "Wasserzeichen", "Logo"],
    },
}


class StyleRetriever:
    def retrieve(self, *, style: str) -> Dict[str, Any]:
        key = (style or "photorealistic").strip().lower()
        return STYLE_PRESETS.get(key, STYLE_PRESETS["photorealistic"])

    def build_final_prompt(
        self,
        *,
        intent: Dict[str, Any],
        style_payload: Dict[str, Any],
        language: str = "en",
    ) -> str:
        lang = (language or intent.get("source_language") or "en").strip().lower()
        base = (intent.get("raw_prompt") or "").strip()

        if lang == "de":
            pos = (style_payload.get("positive_de") or "").strip()
            negatives = set(style_payload.get("negative_de", []))
        else:
            pos = (style_payload.get("positive_en") or "").strip()
            negatives = set(style_payload.get("negative_en", []))

        # merge explicit exclusions from analyzer intent
        for ex in intent.get("exclude", []):
            if not ex:
                continue
            ex = str(ex).strip()
            if not ex:
                continue

            if lang == "de":
                # map canonical exclusions to German text
                neg_map_de = {
                    "text": "Text",
                    "watermark": "Wasserzeichen",
                    "logo": "Logo",
                    "signature": "Signatur",
                    "captions": "Untertitel",
                }
                negatives.add(neg_map_de.get(ex.lower(), ex))
            else:
                negatives.add(ex.lower())

        spatial = intent.get("spatial_relations", []) or []

        # optional: map canonical spatial tags to German display text
        if lang == "de":
            spatial_map_de = {
                "foreground": "im Vordergrund",
                "background": "im Hintergrund",
                "beside": "nebeneinander / daneben",
                "next to": "neben",
                "holding": "in der Hand / hält",
                "in front of": "vor",
                "behind": "hinter",
                "on": "auf",
                "under": "unter",
                "between": "zwischen",
            }
            spatial_display = [spatial_map_de.get(str(s).lower(), str(s)) for s in spatial]
        else:
            spatial_display = [str(s) for s in spatial]

        parts: List[str] = [base]

        if pos:
            parts.append(pos)

        if spatial_display:
            if lang == "de":
                parts.append("Räumliche Beziehungen beachten: " + ", ".join(spatial_display) + ".")
            else:
                parts.append("Spatial constraints: " + ", ".join(spatial_display) + ".")

        if negatives:
            if lang == "de":
                parts.append("Nicht enthalten: " + ", ".join(sorted(negatives)) + ".")
            else:
                parts.append("Negative: " + ", ".join(sorted(negatives)) + ".")

        return "\n".join(p for p in parts if p).strip()