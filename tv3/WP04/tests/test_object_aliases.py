from wp04.object_enrichment import build_alias_map, canonical_label


def test_electronic_scale_alias_preserves_generic_scale_label():
    aliases = build_alias_map()

    assert canonical_label("cân điện tử", aliases)[0] == "scale"
    assert canonical_label("electronic scale", aliases)[0] == "scale"
