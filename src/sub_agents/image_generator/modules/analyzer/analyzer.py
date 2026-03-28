from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

DEFAULT_SIZE = "1024x1024"

NEGATIVE_HINTS_EN = {
    "no text": "text",
    "no watermark": "watermark",
    "no logo": "logo",
    "no signature": "signature",
    "no captions": "captions",
}

NEGATIVE_HINTS_DE = {
    "ohne text": "text",
    "kein text": "text",
    "ohne wasserzeichen": "watermark",
    "kein wasserzeichen": "watermark",
    "ohne logo": "logo",
    "kein logo": "logo",
    "ohne signatur": "signature",
    "keine signatur": "signature",
    "ohne untertitel": "captions",
}

STYLE_KEYWORDS_EN = {
    "photorealistic": ["photo", "realistic", "dslr", "cinematic"],
    "anime": ["anime", "manga"],
    "3d": ["3d", "render", "octane", "blender"],
    "illustration": ["illustration", "storybook", "vector"],
}

STYLE_KEYWORDS_DE = {
    "photorealistic": ["foto", "realistisch", "fotorealistisch", "kinematisch"],
    "anime": ["anime", "manga"],
    "3d": ["3d", "render", "octane", "blender"],
    "illustration": ["illustration", "illustriert", "vektor"],
}

def _infer_style_en(prompt: str) -> str:
    p = prompt.lower()
    for style, keys in STYLE_KEYWORDS_EN.items():
        if any(k in p for k in keys):
            return style
    return "photorealistic"

def _infer_style_de(prompt: str) -> str:
    p = prompt.lower()
    for style, keys in STYLE_KEYWORDS_DE.items():
        if any(k in p for k in keys):
            return style
    return "photorealistic"

def _detect_negatives(prompt: str) -> List[str]:
    p = prompt.lower()
    out = []
    for n in NEGATIVE_HINTS:
        if n in p:
            out.append(n)
    return out

class ImageAnalyzer:
    def analyze(
        self,
        *,
        prompt: str,
        style: str = "",
        size: str = "",
        language: str = "en",
    ) -> Dict[str, Any]:
        raw = (prompt or "").strip()
        if not raw:
            return {"error": "Empty prompt", "size": size or DEFAULT_SIZE}

        lang = (language or "en").strip().lower()

        if lang == "de":
            final_style = (style or "").strip() or _infer_style_de(raw)
            subjects = self._extract_subjects_de(raw)
            counts = self._extract_counts_de(raw)
            spatial = self._extract_spatial_de(raw)
            negatives = self._detect_negatives_de(raw)
            enhanced_prompt = ""
        else:
            final_style = (style or "").strip() or _infer_style_en(raw)
            subjects = self._extract_subjects_en(raw)
            counts = self._extract_counts_en(raw)
            spatial = self._extract_spatial_en(raw)
            negatives = self._detect_negatives_en(raw)
            enhanced_prompt = ""

        final_size = (size or "").strip() or DEFAULT_SIZE

        intent: Dict[str, Any] = {
            "raw_prompt": raw,
            "source_language": lang,
            "style": final_style,
            "size": final_size,
            "subjects": subjects,
            "counts": counts,
            "spatial_relations": spatial,
            "negative_constraints": negatives,
            "must_have": subjects,
            "exclude": negatives,
            "enhanced_prompt": enhanced_prompt,
        }
        return intent

    def _extract_subjects_en(self, prompt: str) -> List[str]:
        # Simple heuristic: nouns/objects after "of"/"with"/"showing"
        # Keep it basic at first; you can replace with an LLM parser later if needed.
        p = prompt.lower()
        candidates = []
        for m in re.findall(r"(?:of|with|showing)\s+([a-z0-9\s\-]+)", p):
            chunk = m.strip()
            if chunk and len(chunk) <= 60:
                candidates.append(chunk)
        return candidates[:6]

    def _extract_counts_en(self, prompt: str) -> Dict[str, int]:
        # “two dogs”, “3 apples”
        counts: Dict[str, int] = {}
        p = prompt.lower()
        for num, obj in re.findall(r"\b(\d+|one|two|three|four|five)\s+([a-z][a-z\s\-']+)", p):
            n = {"one":1,"two":2,"three":3,"four":4,"five":5}.get(num, None)
            n = int(num) if n is None and num.isdigit() else n
            if n:
                counts[obj.strip()] = n
        return counts

    def _extract_spatial_en(self, prompt: str) -> List[str]:
        p = prompt.lower()
        relations = []
        for rel in ["foreground", "background", "beside", "next to", "holding", "in front of", "behind"]:
            if rel in p:
                relations.append(rel)
        return relations
    
    def _detect_negatives_en(self, prompt: str) -> List[str]:
        p = prompt.lower()
        out = []
        for phrase, canonical in NEGATIVE_HINTS_EN.items():
            if phrase in p:
                out.append(canonical)
        return list(dict.fromkeys(out))
    
    def _detect_negatives_de(self, prompt: str) -> List[str]:
        p = prompt.lower()
        out = []
        for phrase, canonical in NEGATIVE_HINTS_DE.items():
            if phrase in p:
                out.append(canonical)
        return list(dict.fromkeys(out))
    
    def _extract_counts_de(self, prompt: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        p = prompt.lower()

        number_map = {
            "ein": 1, "eine": 1, "einen": 1, "einem": 1, "einer": 1,
            "zwei": 2,
            "drei": 3,
            "vier": 4,
            "fünf": 5, "funf": 5,
        }

        pattern = r"\b(\d+|ein|eine|einen|einem|einer|zwei|drei|vier|fünf|funf)\s+([a-zäöüß][a-zäöüß\s\-]+)"
        for num, obj in re.findall(pattern, p):
            n = number_map.get(num)
            if n is None and num.isdigit():
                n = int(num)
            if n:
                counts[obj.strip()] = n
        return counts
    
    def _extract_spatial_de(self, prompt: str) -> List[str]:
        p = prompt.lower()
        rel_map = {
            "im vordergrund": "foreground",
            "im hintergrund": "background",
            "neben": "beside",
            "vor": "in front of",
            "hinter": "behind",
            "auf": "on",
            "unter": "under",
            "zwischen": "between",
            "in der hand": "holding",
            "hält": "holding",
        }
        out = []
        for phrase, canonical in rel_map.items():
            if phrase in p:
                out.append(canonical)
        return list(dict.fromkeys(out))
    
    def _extract_subjects_de(self, prompt: str) -> List[str]:
        p = prompt.lower()

        # Remove known negative clauses first
        p = re.sub(r"(ohne|kein|keine)\s+[a-zäöüß\s\-]+", "", p)

        candidates = []

        # Example patterns:
        # "zwei Hunde", "ein rotes Haus", "eine weiße Katze"
        for m in re.findall(r"\b(?:\d+|ein|eine|einen|einem|einer|zwei|drei|vier|fünf|funf)\s+([a-zäöüß][a-zäöüß\s\-]{1,40})", p):
            chunk = m.strip(" ,.")
            if chunk and len(chunk) <= 60:
                candidates.append(chunk)

        # Fallback: after common relation prepositions
        for m in re.findall(r"(?:neben|vor|hinter|auf|unter|zwischen)\s+([a-zäöüß][a-zäöüß\s\-]{1,40})", p):
            chunk = m.strip(" ,.")
            if chunk and len(chunk) <= 60:
                candidates.append(chunk)

        # dedupe
        return list(dict.fromkeys(candidates))[:6]
    
    def _enhance_prompt_de(
        self,
        *,
        raw: str,
        style: str,
        subjects: List[str],
        counts: Dict[str, int],
        spatial: List[str],
        negatives: List[str],
    ) -> str:
        style_text_map = {
            "photorealistic": "fotorealistisches Bild",
            "anime": "Anime-Stil",
            "3d": "3D-Render",
            "illustration": "Illustration",
        }

        parts = [raw.strip()]

        style_text = style_text_map.get(style)
        if style_text:
            parts.append(f"Stil: {style_text}")

        if subjects:
            parts.append("Wichtige Motive: " + ", ".join(subjects))

        if counts:
            count_text = ", ".join(f"{k}: {v}" for k, v in counts.items())
            parts.append("Anzahl genau beachten: " + count_text)

        if spatial:
            parts.append("Räumliche Beziehungen beachten: " + ", ".join(spatial))

        if negatives:
            parts.append("Nicht enthalten: " + ", ".join(negatives))

        parts.append("Klare Komposition, hohe Details, visuell konsistent.")

        return ". ".join(parts)

