"""Automatic query enhancer: Vietnamese-to-English translation and semantic concept
expansion for CLIP/BGE-VL/MetaCLIP visual embeddings.

Visual models are pre-trained predominantly on English descriptions. Translating
and augmenting Vietnamese queries with visual synonyms significantly improves
Recall@100 on hard Known-Item Search (KIS) and TRAKE queries without degrading
lexical OCR/ASR matching.
"""
from __future__ import annotations

import re
import unicodedata
import urllib.parse
import urllib.request
import json
from functools import lru_cache
from typing import Dict, List, Tuple

# Comprehensive dictionary mapping common Vietnamese visual concepts to English
_CONCEPT_MAP: Dict[str, str] = {
    # Actions
    "phat bieu": "giving a speech speaking at microphone",
    "thuyet trinh": "presentation giving a talk",
    "phong van": "interview speaking to reporter",
    "trao giai": "award ceremony giving trophy medal",
    "nhan giai": "receiving award holding trophy",
    "bieu dien": "performing stage show",
    "hat": "singing on stage with microphone",
    "mua": "dancing performance",
    "mua lan": "lion dance traditional performance",
    "mua rong": "dragon dance performance",
    "chu lan": "lion dance costume performer",
    "nau an": "cooking food in kitchen preparation",
    "so che": "food preparation cutting ingredients",
    "thai": "slicing chopping food",
    "cat": "cutting with knife",
    "xao": "stir frying in pan wok",
    "chien": "deep frying food in oil",
    "nuong": "grilling barbecue cooking",
    "luoc": "boiling water cooking pot",
    "do dau": "pouring oil into pan",
    "dat len": "placing objects onto surface",
    "cam len": "holding up objects",
    "doi thoai": "talking face to face conversation",
    "deo tap de": "wearing an apron",
    "bat lua": "lighting stove fire burner",
    "nem nem": "seasoning tasting food",
    "an": "eating food with chopsticks fork",
    "uong": "drinking beverage glass cup",
    "chay": "running sprinting jogging",
    "di bo": "walking on street sidewalk",
    "di dao": "strolling walking outdoors",
    "boi": "swimming in pool water",
    "lan": "diving underwater",
    "nhay": "jumping in the air",
    "giam nhay": "takeoff jump takeoff athlete",
    "tiep dat": "landing on ground mat",
    "cham dat": "touching ground landing",
    "dung day": "standing up rising",
    "nga": "falling to the ground",
    "bat tay": "shaking hands greeting",
    "om": "hugging embracing",
    "vo tay": "clapping hands applauding",
    "cuoi": "smiling laughing happy face",
    "khoc": "crying tears emotional",
    "lai xe": "driving vehicle car",
    "tu lai": "self-driving autonomous driverless car",
    "khong nguoi lai": "autonomous driverless vehicle",
    "vo lang": "steering wheel inside car",
    "re phai": "turning right",
    "re trai": "turning left",
    "di xe may": "riding motorcycle motorbike",
    "dap xe": "riding bicycle cycling",
    "sua chua": "repairing fixing tools",
    "xay dung": "construction building site worker",
    "ve tranh": "painting drawing artwork canvas",
    "doc sach": "reading book newspaper",
    "su dung dien thoai": "using mobile smartphone screen",
    "dung may tinh": "typing on laptop computer",
    "keo co": "playing tug of war pulling rope competition",
    "vay quanh": "surrounding isolated central person",

    # Objects / Items
    "micro": "microphone",
    "mic": "microphone",
    "cup": "trophy award cup",
    "huy chuong": "medal gold medal",
    "bang khen": "certificate diploma award",
    "hoa": "flowers bouquet",
    "hoa tuoi": "fresh flower bouquet",
    "qua bong": "sports ball soccer basketball",
    "bong da": "football soccer ball",
    "bong ro": "basketball",
    "bong chuyen": "volleyball",
    "vot": "racket tennis badminton",
    "dao": "kitchen knife",
    "thot": "cutting board",
    # ``cháo`` (porridge) and ``chảo`` (pan) collapse to the same accent-folded
    # token. Compound porridge phrases are matched first, while bare ``chao``
    # is guarded by an accent-sensitive pattern below.
    "to chao": "bowl of porridge rice porridge congee",
    "mon chao": "porridge rice porridge congee",
    "chao": "cooking pan wok",
    "noi": "cooking pot",
    "bep": "kitchen stove burner",
    "bep gas": "gas stove flame",
    "bat": "bowl",
    "dia": "plate dish",
    "dia trang": "white plate dish",
    "lo hoa rieng tia": "vase of purple galangal ginger flowers",
    "hoa rieng tia": "purple galangal ginger flowers",
    "4 con": "four small animals food ingredients",
    "2 con": "two small animals food ingredients",
    "dua": "chopsticks",
    "thuc an": "food dish meal",
    "mon an": "cooked food cuisine dish",
    "chen nho mau den": "small black bowl",
    "mau cam": "orange",
    "mau den": "black",
    "topping": "topping garnish",
    "nhung soi nho": "thin shredded strands",
    "soi nho": "thin shredded strands",
    "trang tri": "garnished decorated finished dish",
    "nguyen lieu nhieu mau sac": "colorful raw cooking ingredients vegetables mushrooms edible flowers food preparation",
    "nguyen lieu": "raw cooking ingredients vegetables mushrooms food preparation",
    "hai san": "seafood ingredients prawns shrimp fish shellfish",
    "day thung": "thick rope",
    "nhan vat 3d": "3D figures",
    "nguoi 3d": "3D human figures",
    "hai nhan vat hoat hinh nam": "two cartoon men",
    "nhan vat hoat hinh nam": "cartoon men",
    "nhan vat": "figures people",
    "hoat hinh": "cartoon characters",
    "bieu do": "chart graph with axes and plotted curves",
    "do thi": "chart graph with axes and plotted curves",
    "cot danh so": "numbered performance poles",
    "rong xoan oc": "spiral dragon model",
    "dan nhung con so": "numbered labels stickers",
    "dan so": "numbered labels stickers",
    "mo hinh con rong": "dragon model sculpture",
    "hinh xoan oc": "spiral coil shape",
    "rau": "vegetables green leaves",
    "hanh": "green onions scallions herbs",
    "thit": "meat beef pork",
    "ca": "fish seafood",
    "nam": "mushrooms",
    "dau hu": "tofu white bean curd",
    "cu nang": "water chestnuts",
    "trung": "eggs",
    "banh mi": "bread sandwich baguette",
    "dien thoai": "smartphone mobile phone",
    "may tinh": "laptop computer monitor",
    "may anh": "camera photographer",
    "may quay": "video camera recording",
    # Keep the generic object identity in compound-device queries. Bare
    # ``can`` is intentionally omitted because accent folding also maps the
    # common Vietnamese word ``cần`` to it.
    "can dien tu": "weighing scale digital electronic scale",
    "cai can": "weighing scale",
    "chiec can": "weighing scale",
    "can nang": "weighing body weight on scale",
    "co": "flag banner",
    "co do sao vang": "vietnam red flag yellow star",
    "khau trang": "face mask medical mask",
    "non": "hat cap",
    "non la": "conical leaf hat vietnamese hat",
    "mu bao hiem": "helmet safety helmet",
    "kinh": "glasses spectacles",
    "kinh ram": "sunglasses",

    # Colors & Clothing
    "ao do": "red shirt red dress clothing",
    "ao trang": "white shirt white clothing",
    "ao den": "black shirt black clothing",
    "ao vang": "yellow shirt yellow clothing",
    "ao xanh": "blue shirt green shirt",
    "ao dai": "vietnamese traditional ao dai dress",
    "ao so mi": "collared button-up shirt",
    "ao thun": "t-shirt casual shirt",
    "ao khoac": "jacket coat outerwear",
    "vest": "suit jacket formal blazer",
    "vay": "dress skirt female clothing",
    "quan jean": "blue jeans denim pants",
    "quan tay": "formal trousers dress pants",
    "dong phuc": "uniform matching clothing",
    "tap de": "apron cooking clothing",
    "mau trang": "white",
    "mau do": "red",
    "mau xanh": "blue green colored",
    "nhieu mau sac": "colorful multicolored",

    # People / Roles
    "nguoi dan ong": "man male person",
    "nguoi phu nu": "woman female person",
    "co gai": "young woman girl",
    "chang trai": "young man boy",
    "tre em": "child kid children",
    "em be": "baby infant toddler",
    "nguoi gia": "elderly senior old person",
    "dien gia": "speaker keynote lecturer",
    "ca si": "singer musical artist",
    "nhac si": "musician playing instrument",
    "dien vien": "actor actress performing",
    "mc": "host presenter emcee",
    "phong vien": "reporter journalist news correspondent",
    "van dong vien": "athlete sports player",
    "dau bep": "chef cook kitchen apron",
    "canh sat": "police officer in uniform",
    "bac si": "doctor medical coat physician",
    "y ta": "nurse medical uniform",
    "hoc sinh": "student school uniform",
    "sinh vien": "college university student",
    "khan gia": "audience crowd spectators watching",
    "giam khao": "judges panel jury",
    "canh sat giao thong": "traffic police officer",

    # Vehicles & Transportation
    "xe cuu hoa": "fire truck emergency fire engine",
    "xe cuu thuong": "ambulance emergency vehicle",
    "xe canh sat": "police car police vehicle",
    "xe o to": "car automobile vehicle",
    "xe hoi": "car sedan automobile",
    "xe mau trang": "white car automobile",
    "xe may": "motorcycle motorbike scooter",
    "xe dap": "bicycle bike cyclist",
    "xe bus": "public bus transit bus",
    "xe tai": "truck cargo lorry",
    "may bay": "airplane aircraft flying",
    "truc thang": "helicopter rotor aircraft",
    "thuyen": "boat watercraft sailing",
    "tau hoa": "train railway locomotive",

    # Locations / Settings
    "ngoai troi": "outdoors outside open air nature",
    "trong nha": "indoors inside interior room",
    "san khau": "stage spotlight indoor stage",
    "hoi truong": "auditorium conference hall hall",
    "phong hop": "meeting room conference table",
    "duong pho": "city street road traffic urban",
    "cong vien": "park green trees grass lawn",
    "bai bien": "beach ocean coast seashore sand",
    "song": "river water riverbank",
    "ho": "lake water pond",
    "nha bep": "kitchen cooking area counter",
    "sieu thi": "supermarket grocery store aisle",
    "cho": "outdoor traditional market vendor",
    "benh vien": "hospital clinic medical ward",
    "truong hoc": "school classroom campus",
    "san van dong": "sports stadium arena field",
    "nha ga": "train station airport terminal",
    "cay xanh": "green trees foliage plants nature",
    "troi mua": "rain raining wet weather",
    "troi nang": "sunny bright daylight",
    "ban dem": "night nighttime dark lights",
    "hoang hon": "sunset golden hour evening",
    "binh minh": "sunrise morning light dawn",
    "bien hieu do": "red sign red shop sign",
    "chu han": "Chinese characters Chinese writing",
    "can canh": "close-up view",
    "tu tren xuong": "overhead top-down view",
    "toan canh": "wide shot full view",
    "bai giang": "lecture slide",
    "do man": "salinity axis value",
}

# Accent folding is useful for spelling variation, but a few short Vietnamese
# words collapse into unrelated concepts (``nam``/``nấm``, ``cuối``/``cười``,
# ``tĩnh``/``tính``). Require the intended accented spelling for those keys so
# query expansion never injects an object that the user did not describe.
_ACCENT_SENSITIVE_PATTERNS: Dict[str, str] = {
    "an": r"ăn",
    "bat": r"bát",
    "chao": r"chảo",
    "to chao": r"tô\s+cháo",
    "mon chao": r"món\s+cháo",
    "boi": r"bơi",
    "co": r"cờ",
    "cuoi": r"cười",
    "cho": r"chợ",
    "dao": r"dao",
    "dua": r"đũa",
    "hat": r"hát",
    "hanh": r"hành",
    "hoa": r"hoa",
    "ho": r"hồ",
    "lan": r"lặn",
    "may tinh": r"máy\s+tính",
    "mua": r"múa",
    "nam": r"nấm",
    "nga": r"ngã",
    "song": r"sông",
    "vay": r"váy",
}

# Conventional visual symbols can identify the *spoken* lecture topic even
# when that topic is not written in the user's description.  Keep this map
# deliberately small and auditable: values are Vietnamese retrieval terms for
# OCR/ASR, not expected video IDs, titles, or frame numbers.
_SEMANTIC_TEXT_LOCATOR_MAP: Dict[str, str] = {
    "keo co": "cạnh tranh ganh đua đấu tranh",
}


def semantic_text_locator_query(text: str) -> str | None:
    """Infer a conservative text-index query from conventional slide symbols.

    Activation requires a lecture/slide context so a real sports scene is not
    reinterpreted as an abstract lesson topic.
    """
    folded = strip_vietnamese_accents(text)
    if not re.search(r"\b(?:bai giang|giang bai|slide|lecture|lesson)\b", folded):
        return None
    terms = [
        locator for symbol, locator in _SEMANTIC_TEXT_LOCATOR_MAP.items()
        if re.search(rf"\b{re.escape(symbol)}\b", folded)
    ]
    if not terms:
        return None
    return " ".join(dict.fromkeys(terms))


def strip_vietnamese_accents(text: str) -> str:
    """Normalize and remove Vietnamese diacritics for robust matching."""
    folded = "".join(
        c for c in unicodedata.normalize("NFD", text.casefold())
        if unicodedata.category(c) != "Mn"
    ).replace("đ", "d").replace("Đ", "d")
    return re.sub(r"\s+", " ", folded).strip()


@lru_cache(maxsize=512)
def _try_online_translation(text: str) -> str | None:
    """Attempt fast free Google Translate API call with aggressive timeout.
    Returns None if offline or failed, falling back to concept expansion.
    """
    if not text.strip():
        return None
    try:
        url = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl=vi&tl=en&dt=t&q={urllib.parse.quote(text)}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=1.8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, list) and data and isinstance(data[0], list):
                translated_parts = [part[0] for part in data[0] if part and isinstance(part[0], str)]
                translated = " ".join(translated_parts).strip()
                if translated and translated.casefold() != text.casefold():
                    return translated
    except Exception:
        pass
    return None


def enhance_visual_query(
    vietnamese_query: str, *, allow_online_translation: bool = True,
) -> str:
    """Generate an optimal English visual search query for CLIP/BGE-VL/MetaCLIP models.
    
    1. Tries fast online translation if available.
    2. Maps domain-specific concepts (actions, objects, clothing, scenes).
    3. Combines translated terms into a rich visual prompt.
    """
    cleaned = vietnamese_query.strip()
    if not cleaned:
        return ""

    # Check if query is already predominantly English
    ascii_ratio = sum(1 for c in cleaned if ord(c) < 128) / max(1, len(cleaned))
    if ascii_ratio > 0.95 and not any(k in cleaned.casefold() for k in ("nguoi", "mot", "dang", "trong", "ngoai", "tren", "duoi")):
        return cleaned

    folded = strip_vietnamese_accents(cleaned)
    original_folded_case = unicodedata.normalize("NFC", cleaned.casefold())
    matched_english_terms: List[str] = []
    matched_positions: List[Tuple[int, str]] = []

    # Match concept phrases (longest keys first for compound match)
    sorted_concepts = sorted(_CONCEPT_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    covered_spans: List[Tuple[int, int]] = []

    for vi_key, en_val in sorted_concepts:
        sensitive = _ACCENT_SENSITIVE_PATTERNS.get(vi_key)
        pattern = r"\b" + (sensitive if sensitive is not None else re.escape(vi_key)) + r"\b"
        haystack = original_folded_case if sensitive is not None else folded
        for match in re.finditer(pattern, haystack):
            start, end = match.span()
            if not any(cs <= start and end <= ce for cs, ce in covered_spans):
                covered_spans.append((start, end))
                matched_positions.append((start, en_val))

    # Longest-pattern matching above resolves compound overlaps; source order
    # then makes the resulting prompt read like the user's scene instead of
    # arbitrary dictionary-key length. This also keeps early distinguishing
    # subjects/colors inside encoder token limits.
    for _, en_val in sorted(matched_positions, key=lambda item: item[0]):
        if en_val not in matched_english_terms:
            matched_english_terms.append(en_val)

    # Try fast online translation to get full grammatical sentence
    online_translation = _try_online_translation(cleaned) if allow_online_translation else None

    if online_translation:
        # Combine online translation with extra concept keywords
        extra_terms = [t for t in matched_english_terms if t.casefold() not in online_translation.casefold()]
        if extra_terms:
            return f"{online_translation}, {', '.join(extra_terms[:4])}\nOriginal description: {cleaned}"
        return f"{online_translation}\nOriginal description: {cleaned}"

    if matched_english_terms:
        # Keywords are incomplete hints, never a replacement for actions,
        # entities, counts, negation or temporal constraints in the source.
        return cleaned + "\nVisual keywords: " + ", ".join(matched_english_terms)

    return cleaned
