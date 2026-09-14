from dataclasses import replace
from types import SimpleNamespace

import pytest

from tv4.contracts import EvidencePack, SearchCandidate
from tv4.kis_pipeline import _rank_multimoment_kis
from tv4.qa_query_plan import build_qa_query_plan
from tv4.qa_reranker import select_evidence_candidates
from tv4.qa_verifier import verify_answer
from tv4.query_enhancer import enhance_visual_query
from tv4.trake_event_parser import parse_event
from tv4.trake_pipeline import run_trake_queries, visual_query_for_trake
from tv4.trake_verifier import verify_sequence
from tv4.visual_verifier import OllamaVisualVerifier, VisualVerification
from tv4.wp11_vqa import VqaResult, answer_query


def candidate(video="V1", frame=10, rank=1):
    return SearchCandidate(query_id="regression", video_id=video, frame_id=frame,
                           timestamp_ms=frame * 100, source="visual", rank=rank,
                           score=1 / rank, preprocess_run_id="run-test")


def test_visual_subprocess_roundtrips_vietnamese_under_legacy_windows_encoding(monkeypatch, tmp_path):
    import json
    import sys
    from tv4.clients.tv2_visual_client import TV2VisualClient
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    client = TV2VisualClient(python_executable=sys.executable, wp03_cwd=tmp_path, artifact_root=tmp_path)
    monkeypatch.setattr(client, "_server_command", lambda: [
        sys.executable, '-c',
        'import sys,json; value=json.loads(sys.stdin.readline()); print(json.dumps(value,ensure_ascii=False),flush=True)',
    ])
    process = client._ensure_process()
    payload = {"query": "Người thợ cầm đèn, trộn nước cam, chọn chiếc thứ 7."}
    try:
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
        process.stdin.flush()
        assert json.loads(process.stdout.readline()) == payload
        process.wait(timeout=5)
    finally:
        client.close()


def pack(c, text=""):
    return EvidencePack(query_id=c.query_id, video_id=c.video_id, frame_id=c.frame_id,
                        timestamp_ms=c.timestamp_ms, keyframe_path=None, asr_texts=(text,) if text else ())


def test_unseen_event_keeps_count_negation_and_completion_offline():
    event = "Người thợ hoàn tất đặt chiếc đèn thứ 7 xuống bàn, không bật đèn."
    expanded = visual_query_for_trake(event, context="Một xưởng làm đồ thủ công.")
    assert event in expanded
    assert "Một xưởng làm đồ thủ công." in expanded


def test_translation_keeps_original_when_translation_omits_constraint(monkeypatch):
    monkeypatch.setattr("tv4.query_enhancer._try_online_translation", lambda _: "A person puts down a lamp")
    source = "Người thợ đặt chiếc đèn thứ 7 xuống bàn, không bật đèn."
    result = enhance_visual_query(source)
    assert "A person puts down a lamp" in result and source in result


def test_kis_keeps_another_window_and_bounds_duplicates():
    originals = [candidate("V1", 10), candidate("V1", 90, 2), candidate("V1", 91, 3)]
    result = _rank_multimoment_kis({"visual": originals}, weights=None, top_k=100)
    assert [c.frame_id for c in result] == [10, 90]
    many = [candidate(f"V{i}", i * 20, i + 1) for i in range(100)]
    duplicates = [replace(c, frame_id=c.frame_id + 500, timestamp_ms=c.timestamp_ms + 50_000) for c in many]
    bounded = _rank_multimoment_kis({"visual": many + duplicates}, weights=None, top_k=100)
    assert len(bounded) == 100
    assert len({c.video_id for c in bounded}) >= 75


def test_vietnamese_homographs_do_not_become_actions():
    fruit = parse_event(0, "Trộn nước cam với vỏ cam trên chiếc bàn cao nhất.")
    assert fruit.state_transition is None
    assert "trộn" in fruit.actions
    hold = parse_event(1, "Người thợ nhặt chiếc đèn và cầm lên.")
    assert hold.state_transition == "holding"
    finish = parse_event(2, "Hoàn tất đặt chiếc đèn thứ 7 xuống bàn.")
    assert finish.ordinal == 7 and "completion" in finish.temporal_cues
    assert finish.state_transition == "released"


def test_unknown_action_needs_observation_even_with_ordered_frames():
    result = verify_sequence(("Unrecognized event description",), (candidate(),))
    assert result.micro_clip_support < 0.5 and result.clip_manual_required
    checked = verify_sequence(("Unrecognized event description",), (candidate(),),
                              visual_verifications=(VisualVerification(available=True, score=0.9),))
    assert checked.micro_clip_support == 0.9


@pytest.mark.parametrize("question,answer,text", [
    ("Tên trường tiểu học nào?", "CHANNEL", "CHANNEL"),
    ("Giáo viên dạy chính ở trường tiểu học nào?", "trường", "Giáo viên dạy tại trường tiểu học."),
    ("Có bao nhiêu màu trên đĩa?", "CHANNEL", "CHANNEL"),
    ("Có bao nhiêu học sinh?", "1", "Lớp có 14 học sinh."),
])
def test_qa_rejects_unrelated_or_wrong_type_answer(question, answer, text):
    result = verify_answer(question, answer, pack(candidate(), text), producer_supported=True)
    assert not result.supported


def test_visible_lettering_cannot_be_verified_from_speech_alone():
    question = 'Dòng chữ neon là gì?'
    spoken = pack(candidate(), 'Dòng chữ neon trên biển là OPEN.')
    assert not verify_answer(question, 'OPEN', spoken, producer_supported=True).supported
    visible = replace(spoken, asr_texts=(), ocr_texts=('Dòng chữ neon OPEN',))
    assert verify_answer(question, 'OPEN', visible, producer_supported=True).supported


def test_quantity_question_does_not_count_the_word_qua_trinh():
    question = 'Trong quá trình nêm gia vị, lượng nước tương được sử dụng là bao nhiêu?'
    assert not build_qa_query_plan('', question).requires_visual_read


@pytest.mark.parametrize('text,expected', [
    ('Nước tương: 2M', True),
    ('NƯỚCTƯƠNG.2M', True),
    ('Dầu ăn: 2M', False),
    ('Nước tương: 1M; dầu ăn: 2M', False),
    ('Nước tương: 1M, dầu ăn: 2M', False),
])
def test_quantity_belongs_to_requested_ingredient(text, expected):
    question = 'Lượng nước tương được sử dụng là bao nhiêu?'
    evidence = replace(pack(candidate()), ocr_texts=(text,))
    assert verify_answer(question, '2M', evidence).supported is expected


@pytest.mark.parametrize('scene,expected', [
    (None, False),
    (VisualVerification(available=False, score=None, reason='timeout'), False),
    (VisualVerification(available=True, score=0.1, contradiction=True), False),
    (VisualVerification(available=True, score=0.5), False),
    (VisualVerification(available=True, score=0.9), True),
])
def test_same_answer_in_another_scene_does_not_verify(scene, expected):
    evidence = replace(pack(candidate()), ocr_texts=('Nước tương: 2M',))
    result = verify_answer('Lượng nước tương được sử dụng là bao nhiêu?', '2M', evidence,
                           producer_supported=True, description='Preparing an orange fillet with berries.',
                           scene_verification=scene)
    assert result.supported is expected and result.scene_supported is expected


def test_scene_verifier_is_answer_blind_and_limits_images(monkeypatch, tmp_path):
    from tv4.visual_verifier import VisualFrame
    verifier = OllamaVisualVerifier(enabled=True, model='test', max_images=2)
    calls = []
    def chat(prompt, frames):
        calls.append((prompt, frames))
        return {'score':0.2, 'contradiction':True, 'rationale':'Different subject',
                'checks':[{'criterion':0, 'matched':False, 'frame_ids':[]}]}
    monkeypatch.setattr(verifier, '_chat', chat)
    frames = tuple(VisualFrame('V1', i, i*100, tmp_path/f'{i}.jpg') for i in range(7))
    result = verifier.verify_scene('An orange fillet with berries.', frames)
    assert result.contradiction and result.frame_ids == (0, 1)
    assert 'An orange fillet with berries.' in calls[0][0]
    assert 'Proposed answer:' not in calls[0][0] and len(calls[0][1]) == 2


@pytest.mark.parametrize("description,question,operation", [
    ("Một cửa tiệm.", "Dòng chữ neon là gì?", "read_visible_text"),
    ("Các bánh trên đĩa.", "Trên đĩa có mấy màu bột khác nhau?", "count_visible"),
])
def test_explicit_visual_questions_enter_image_reader(description, question, operation):
    plan = build_qa_query_plan(description, question)
    assert plan.requires_visual_read and plan.operation == operation


def test_spoken_grade_question_does_not_trigger_object_counting():
    assert not build_qa_query_plan("Phỏng vấn học sinh.", "Em sắp lên lớp mấy?").requires_visual_read


@pytest.mark.parametrize("question,answer", [("Dòng chữ neon là gì?", "OPEN"), ("Có mấy màu bột trên đĩa?", "4")])
def test_visual_reader_executes_new_operations_with_bounded_frames(monkeypatch, tmp_path, question, answer):
    from tv4.visual_verifier import VisualFrame
    verifier = OllamaVisualVerifier(enabled=True, model="test", max_images=2)
    calls = []
    def chat(prompt, frames, **kwargs):
        calls.append((prompt, frames))
        return {"answer": answer, "observed_values": [], "score": 0.9}
    monkeypatch.setattr(verifier, "_chat", chat)
    frames = tuple(VisualFrame("V1", i, i*100, tmp_path/f'{i}.jpg') for i in range(8))
    result = verifier.answer_qa(question, build_qa_query_plan("", question), frames)
    assert result.available and result.answer == answer
    assert len(calls) == 1 and len(calls[0][1]) == 2
    assert question in calls[0][0]


def test_trake_translation_is_one_call_and_rejects_wrong_event_count(monkeypatch):
    from tv4.external_assist import ExternalQueryAssist
    assist = ExternalQueryAssist(enabled=False, base_url="", model="", timeout=1)
    calls = []
    def translate(*args):
        calls.append(args)
        return {"queries": ["A workshop"]}
    monkeypatch.setattr(assist, "_chat_json", translate)
    result = assist.trake_translations("Xưởng chế tác", ("Cầm chiếc đèn đầu tiên", "Đặt đèn xuống bàn"))
    assert result == (None, None, None) and len(calls) == 1


def test_empty_answer_at_maximum_window_does_not_repeat_same_evidence(monkeypatch):
    c = candidate()
    initial = replace(pack(c, "hello"), provenance={"window_ms": 30_000})
    def unexpected(*a, **k):
        raise AssertionError("Already inspected the maximum window")
    monkeypatch.setattr("tv4.wp11_vqa.build_evidence_pack", unexpected)
    engine = SimpleNamespace(answer=lambda *a: "", verify=lambda *a: False)
    result = answer_query(c, "Tên trường tiểu học nào?", None, None, engine, evidence=initial)
    assert result.answer == "" and result.retry_count == 0


def test_qa_empty_answer_gets_one_bounded_local_retry(monkeypatch):
    c = candidate()
    initial = replace(pack(c, "Chào các bạn"), provenance={"window_ms": 3_000})
    calls = []
    def rebuild(*args, **kwargs):
        calls.append(kwargs)
        return pack(c, "Trường tiểu học Mộc Sơn")
    monkeypatch.setattr("tv4.wp11_vqa.build_evidence_pack", rebuild)
    engine = SimpleNamespace(answer=lambda q, e: "Mộc Sơn" if "Mộc Sơn" in e.asr_texts[0] else "",
                             verify=lambda *a: True)
    result = answer_query(c, "Tên trường tiểu học nào?", None, None, engine, evidence=initial)
    assert result.answer == "Mộc Sơn" and result.retry_count == 1
    assert len(calls) == 1 and calls[0]["window_ms"] <= 30_000
    assert calls[0]["search_branches"] is False


def test_evidence_pool_reaches_other_videos_despite_repeated_frames():
    rows = [candidate("V1", frame=i * 20, rank=i + 1) for i in range(40)]
    rows += [candidate("V2", rank=41), candidate("V3", rank=42)]
    selected = select_evidence_candidates(rows, 30)
    assert len(selected) == 30 and {c.video_id for c in selected} == {"V1", "V2", "V3"}


def test_trake_missing_event_gets_scoped_search_and_diagnostics(monkeypatch):
    monkeypatch.setenv("TV4_EXTERNAL_ASSIST_ENABLED", "0")
    first, second = candidate(frame=10), candidate(frame=30)
    class Visual:
        def __init__(self): self.calls = []
        def search_batch(self, requests, **kwargs):
            self.calls.append(kwargs)
            if not kwargs:
                return [[first], [first], []]
            assert kwargs['candidate_frames'] == [{"video_id": "V1", "frame_id": 10}, {"video_id": "V1", "frame_id": 30}]
            return [[first], [second]]
    visual = Visual()
    services = SimpleNamespace(
        visual=visual, tv3=SimpleNamespace(search_all=lambda *a: {}),
        tv1=SimpleNamespace(frames=lambda v: [
            {"video_id": v, "frame_id": f, "preprocess_run_id": "run-test"} for f in [10, 30]
        ]), preprocess_run_id="run-test", refine=None, original_media_root=None,
        derivative_asset_root=None,
    )
    diagnostics = {}
    results = run_trake_queries("Workshop", services, events=["pick up a lamp", "put down a lamp"],
                                diagnostics=diagnostics,
                                visual_verifier=OllamaVisualVerifier(enabled=False, model=""))
    assert len(visual.calls) == 2 and diagnostics["supplement_attempted"]
    assert diagnostics["video_coverage_before"][0]["missing_event_indexes"] == [1]
    assert results and list(results[0].frame_ids) == [10, 30]


@pytest.mark.parametrize('visual_question', [False, True])
@pytest.mark.parametrize('scene_matches', [False, True])
def test_qa_api_tries_deeper_videos_and_returns_supported_answer(monkeypatch, visual_question, scene_matches):
    import tv4.api as api
    from tv4.external_assist import ExternalQueryAssist
    rows = [candidate(f"V{i}", rank=i) for i in range(1, 46)]
    services = SimpleNamespace(tv1=None, tv3=None, refine=None, dense_ocr=None,
                               derivative_asset_root=None, allowed_image_extensions=('.jpg',),
                               preprocess_run_id='run-test')
    monkeypatch.setattr(api, "FIXTURE_MODE", False)
    monkeypatch.setattr(api, "_get_services", lambda: services)
    monkeypatch.setattr(api, "run_kis_query", lambda *a, **k: rows)
    monkeypatch.setattr(api.OllamaAnswerEngine, "from_environment", lambda: SimpleNamespace(visual_retrieval_query=lambda t: t))
    monkeypatch.setattr(ExternalQueryAssist, "from_environment", lambda: SimpleNamespace(vqa_visual_queries=lambda *a: (), branches=lambda *a: ()))
    verifier = OllamaVisualVerifier(enabled=True, model='test', max_qa_candidates=3)
    image_calls = []
    def read_images(question, plan, frames):
        image_calls.append((question, frames))
        return SimpleNamespace(available=False, answer='')
    monkeypatch.setattr(verifier, 'answer_qa', read_images)
    monkeypatch.setattr(verifier, 'verify_qa', lambda *a: verifier.unavailable('test'))
    monkeypatch.setattr(verifier, 'verify_scene', lambda description, frames:
                        VisualVerification(available=True, score=0.9 if scene_matches and frames == ('V40',) else 0.1))
    monkeypatch.setattr(api.OllamaVisualVerifier, "from_environment", lambda: verifier)
    monkeypatch.setattr(api, 'canonical_qa_frames', lambda evidence, *a, **k: (evidence.video_id,))
    evidence_text = 'Dòng chữ neon Mộc Sơn' if visual_question else 'Trường tiểu học Mộc Sơn'
    monkeypatch.setattr(api, "build_evidence_pack", lambda c, *a, **k: replace(
        pack(c, evidence_text if c.video_id == "V40" else "hello"), query_text="Trường học",
        ocr_texts=(evidence_text,) if visual_question and c.video_id == 'V40' else (),
    ))
    monkeypatch.setattr(api, "direct_answer_query", lambda *a: None)
    def answer(c, question, *args, evidence, **kwargs):
        answer = "Mộc Sơn" if c.video_id == "V40" else ""
        return VqaResult(candidate=c, evidence=evidence, answer=answer, proposal=answer,
                         verified=bool(answer), manual_fallback=not bool(answer), approved=False,
                         verifier_status="verified" if answer else "abstained", retry_count=0,
                         manual_required=not bool(answer), status="proposal" if answer else "manual_required", degraded_reasons=())
    monkeypatch.setattr(api, "answer_query", answer)
    question = 'Dòng chữ neon là gì?' if visual_question else 'Tên trường tiểu học nào?'
    response = api.vqa_answer(api.VqaRequest(query_text="Trường học", question=question))
    assert response["diagnostics"]["deeper_retrieval_attempted"]
    assert len(response["results"]) <= 5
    if not scene_matches:
        assert all(r['answer'] == '' and not r['verified'] for r in response['results'])
        assert any(r['proposal'] == 'Mộc Sơn' for r in response['results'])
        return
    assert response["results"][0]["answer"] == "Mộc Sơn"
    assert response["results"][0]["video_id"] == "V40"
    assert {'V1', 'V2'} <= {r['video_id'] for r in response['results']}
    if visual_question:
        assert [frames[0] for _, frames in image_calls] == ['V1', 'V2', 'V3']
        assert all('Trường học' in prompt and question in prompt for prompt, _ in image_calls)


def test_named_quantity_evidence_outranks_generic_cooking_text():
    from tv4.qa_reranker import score_answerability
    question = 'Lượng dầu ô liu được sử dụng là bao nhiêu?'
    c = candidate()
    generic = replace(pack(c, 'Cho thêm gia vị rồi trộn đều'), ocr_texts=('Muối 1M',))
    specific = replace(pack(c, 'Cho thêm gia vị rồi trộn đều'), ocr_texts=('DẦUÔLIU3M',))
    assert score_answerability(c, specific, question).total > score_answerability(c, generic, question).total


@pytest.mark.parametrize('texts,expected', [
    (['DẦUÔLIU3M'], '3M'),
    (['Dầu ô liu: 3ml', 'DẦUÔLIU3ml'], '3ml'),
    (['Dầu ô liu: 3M', 'Dầu ô liu: 4M'], ''),
    (['Muối 3M'], ''),
])
def test_quantity_extraction_preserves_units_and_rejects_conflicts(texts, expected):
    from tv4.qa_reranker import extract_quantity_ocr_answer
    assert extract_quantity_ocr_answer('Lượng dầu ô liu được sử dụng là bao nhiêu?', texts) == expected


def test_generic_high_scene_score_cannot_override_missing_description_clause(monkeypatch, tmp_path):
    from tv4.visual_verifier import VisualFrame
    verifier = OllamaVisualVerifier(enabled=True, model='test')
    monkeypatch.setattr(verifier, '_chat', lambda *a: {
        'score':0.95, 'contradiction':False,
        'checks':[{'criterion':0,'matched':True,'frame_ids':[1]},
                  {'criterion':1,'matched':False,'frame_ids':[]}],
    })
    result = verifier.verify_scene('An orange fillet. Berries removed from a branch.',
                                    (VisualFrame('V1',1,100,tmp_path/'1.jpg'),))
    assert result.available and result.score < 0.5
