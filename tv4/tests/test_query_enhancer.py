from tv4.query_enhancer import enhance_visual_query, semantic_text_locator_query
from tv4.external_assist import ExternalQueryAssist, is_temporal_visual_query, split_visual_moments


def test_electronic_scale_expansion_preserves_generic_object_identity():
    query = enhance_visual_query(
        "Người phụ nữ đứng cạnh cân điện tử",
        allow_online_translation=False,
    )

    assert "weighing scale" in query
    assert "electronic scale" in query


def test_offline_expansion_does_not_call_online_translator(monkeypatch):
    def fail_if_called(text):
        raise AssertionError("online translator must not run")

    monkeypatch.setattr("tv4.query_enhancer._try_online_translation", fail_if_called)

    result = enhance_visual_query(
        "múa lân", allow_online_translation=False,
    )
    assert result.startswith("múa lân")
    assert "lion dance" in result


def test_porridge_and_pan_remain_distinct_after_accent_folding():
    porridge = enhance_visual_query(
        "Một tô cháo đặt cạnh chén nhỏ màu đen chứa topping màu cam dạng những sợi nhỏ",
        allow_online_translation=False,
    )
    pan = enhance_visual_query("Đổ dầu vào chảo", allow_online_translation=False)

    assert "porridge" in porridge
    assert "small black bowl" in porridge
    assert "orange" in porridge
    assert "thin shredded strands" in porridge
    assert "cooking pan wok" not in porridge
    assert "cooking pan wok" in pan


def test_visual_assist_rejects_fluent_translation_that_loses_food_and_color(monkeypatch):
    assist = ExternalQueryAssist(
        enabled=True, base_url="http://unused", model="test", timeout=0.1,
    )
    monkeypatch.setattr(
        assist,
        "_chat_json",
        lambda *args, **kwargs: {
            "query": "A completed bowl of noodle soup beside a camouflaged topping."
        },
    )

    variants = assist.visual_queries(
        "Một tô cháo đã hoàn chỉnh, cạnh chén nhỏ màu đen có topping màu cam."
    )

    assert "porridge" in variants[0]
    assert "orange" in variants[0]
    assert "black" in variants[0]
    assert "noodle soup" not in variants[0]


def test_unknown_counted_food_sequence_keeps_apron_counts_and_does_not_infer_market():
    query = enhance_visual_query(
        "Cô gái đeo tạp dề màu trắng cạnh lọ hoa riềng tía, đặt 4 con X lên "
        "đĩa trắng rồi cầm lên 2 con X để đối thoại.",
        allow_online_translation=False,
    )

    assert "wearing an apron" in query
    assert "vase of purple galangal ginger flowers" in query
    assert "four small animals food ingredients" in query
    assert "white plate" in query
    assert "holding up objects" in query
    assert "two small animals food ingredients" in query
    assert "talking face to face" in query
    assert "outdoor traditional market" not in query


def test_self_driving_query_has_audited_offline_visual_fallback(monkeypatch):
    assist = ExternalQueryAssist(
        enabled=True, base_url="http://unused", model="test", timeout=0.1,
    )
    monkeypatch.setattr(
        assist,
        "_chat_json",
        lambda *args, **kwargs: {"query": "a car driving scene with visible elements"},
    )

    query = assist.visual_query(
        "Xe ô tô tự lái có vô lăng xoay, sau đó xe màu trắng rẽ trái dưới biển hiệu đỏ chữ Hán"
    )

    assert "self-driving" in query
    assert "steering wheel" in query
    assert "white car" in query
    assert "turning left" in query
    assert "red sign" in query
    assert "Chinese" in query


def test_faithful_english_translation_is_retained(monkeypatch):
    assist = ExternalQueryAssist(
        enabled=True, base_url="http://unused", model="test", timeout=0.1,
    )
    translated = (
        "Inside a self-driving car, the steering wheel turns right; then a white car "
        "turns left beneath a red sign with six Chinese characters."
    )
    monkeypatch.setattr(
        assist, "_chat_json", lambda *args, **kwargs: {"query": translated},
    )

    assert assist.visual_query("Một xe ô tô tự lái rẽ phải") == translated


def test_visual_moment_split_handles_semicolons_transitions_and_bullets():
    moments = split_visual_moments(
        "Cận cảnh nguyên liệu đầu tiên; chuyển sang nguyên liệu thứ hai; "
        "cuối cùng là toàn cảnh món ăn.\n- Hai nhân vật đang kéo co."
    )

    assert moments == (
        "Cận cảnh nguyên liệu đầu tiên",
        "nguyên liệu thứ hai",
        "toàn cảnh món ăn",
        "Hai nhân vật đang kéo co",
    )


def test_temporal_query_detection_requires_order_cue_and_multiple_moments():
    assert is_temporal_visual_query(
        "First the camera shows raw seafood; then it shows colorful vegetables; "
        "finally it shows a wide view of all ingredients."
    )
    assert not is_temporal_visual_query(
        "On the table are seafood; colorful vegetables; and a white bowl."
    )


def test_temporal_visual_queries_use_one_assistant_call_and_stable_query_roles(monkeypatch):
    assist = ExternalQueryAssist(enabled=True, base_url="http://unused", model="test")
    calls = []

    def fake_chat(*args, **kwargs):
        calls.append((args, kwargs))
        return {"query": "First seafood, then colorful vegetables, finally all ingredients."}

    monkeypatch.setattr(assist, "_chat_json", fake_chat)
    variants = assist.visual_queries(
        "First the camera shows raw seafood; then it shows colorful vegetables; "
        "finally it shows all cooking ingredients."
    )

    assert len(calls) == 1
    assert len(variants) >= 4
    assert variants[0]
    assert variants[1]  # video-level locator; temporal moments start at index 2


def test_vqa_visual_queries_use_deterministic_anchor_without_translation(monkeypatch):
    assist = ExternalQueryAssist(enabled=True, base_url="http://unused", model="unused")

    def unexpected_chat(*args, **kwargs):
        raise AssertionError("QA retrieval must not depend on optional translation")

    monkeypatch.setattr(assist, "_chat_json", unexpected_chat)
    variants = assist.vqa_visual_queries(
        "Một cô gái đeo tạp dề màu trắng rồi cầm nguyên liệu lên",
        "Đó là gì?",
    )

    assert variants
    assert "wearing an apron" in variants[0]


def test_temporal_vqa_uses_compact_opening_moment_as_primary_anchor():
    assist = ExternalQueryAssist(enabled=False, base_url="", model="")
    variants = assist.vqa_visual_queries(
        "Một cô gái đeo tạp dề trắng cạnh lọ hoa. Sau đó cô đặt lên đĩa các nguyên liệu. Sau đó cô cầm chúng lên.",
        "Nguyên liệu là gì?",
    )

    assert "wearing an apron" in variants[0]
    assert "placing objects" not in variants[0]
    assert any("placing objects" in query for query in variants[1:])


def test_visual_queries_have_deterministic_multi_moment_fallback(monkeypatch):
    assist = ExternalQueryAssist(enabled=False, base_url="", model="")
    query = (
        "Nhóm nhân vật màu trắng vây quanh một nhân vật màu đỏ.\n"
        "- Hai nhân vật hoạt hình nam đang kéo co bằng dây thừng."
    )

    variants = assist.visual_queries(query)

    assert 2 <= len(variants) <= 5
    assert variants[0]
    assert any("figures people" in variant for variant in variants)
    assert any("tug of war" in variant for variant in variants)
    assert any(
        "surrounding isolated central person" in variant
        and "tug of war" not in variant
        for variant in variants
    )
    assert any(
        "tug of war" in variant
        and "surrounding isolated central person" not in variant
        for variant in variants
    )
    assert any("two cartoon men" in variant for variant in variants)


def test_visual_moment_split_drops_intro_and_header_but_keeps_bullets():
    moments = split_visual_moments(
        "Đây là một đoạn trong bài giảng.\nTrên slide bao gồm:\n"
        "- Nhóm nhân vật trắng vây quanh nhân vật đỏ.\n"
        "- Hai nhân vật hoạt hình đang kéo co."
    )

    assert moments == (
        "Nhóm nhân vật trắng vây quanh nhân vật đỏ",
        "Hai nhân vật hoạt hình đang kéo co",
    )


def test_lecture_symbol_can_supply_conservative_transcript_locator():
    query = (
        "Đây là một đoạn trong bài giảng. Trên slide có hai nhân vật "
        "hoạt hình đang thi đấu kéo co."
    )

    assert semantic_text_locator_query(query) == "cạnh tranh ganh đua đấu tranh"
    assert semantic_text_locator_query("Hai đội đang thi đấu kéo co ngoài trời") is None
