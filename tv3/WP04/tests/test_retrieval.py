from wp04.normalization import normalize_text
from wp04.contracts import ObjectDetection, SearchCandidate
from wp04.retrieval import ElasticTextIndex, LocalTextIndex, object_match


def test_local_index_finds_diacritic_free_and_fuzzy_query():
    index = LocalTextIndex.from_documents([("a", "Bánh mì"), ("b", "bánh ngọt")])
    assert index.search("banh mi", 1)[0].document_id == "a"
    assert index.search("banh mii", 1)[0].document_id == "a"


def test_normalization_folds_vietnamese_d_and_local_bm25_ranks_match_first():
    index = LocalTextIndex.from_documents([
        ("a", "Đường phố bánh mì"),
        ("b", "đường ngọt"),
    ])
    assert normalize_text("Đường").without_diacritics == "duong"
    assert index.search("duong banh mi", 1)[0].document_id == "a"


def test_local_index_does_not_return_fuzzy_only_non_matches():
    index = LocalTextIndex.from_documents([
        ("fire", "cháy rừng lan nhanh"),
        ("noise", "bốn xe máy đều bị hư hỏng"),
    ])
    assert [hit.document_id for hit in index.search("cháy rừng", 10)] == ["fire"]


def test_elastic_failure_degrades_to_local_index():
    class OfflineClient:
        def search(self, query: str, limit: int):
            raise ConnectionError("offline")

    local = LocalTextIndex.from_documents([("a", "bánh mì")])
    assert ElasticTextIndex(OfflineClient(), local).search("banh mi", 1)[0].document_id == "a"


def test_object_miss_does_not_remove_candidate_and_match_only_suggests_boost():
    candidate = SearchCandidate("q", "v", 42, 1400, "ocr", 1, "tv1")
    detection = ObjectDetection("tv1", "v", 42, 1400, "person", (0.1, 0.2, 0.3, 0.4), 0.8, "rf-detr", "v1", "object:v:42:0")
    assert object_match(candidate, [], {"person"}).suggested_boost == 0.0
    match = object_match(candidate, [detection], {"person"})
    assert match.evidence_refs == ("object:v:42:0",)
    assert match.suggested_boost > 0.0
