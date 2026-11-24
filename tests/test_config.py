from helmet_detection.config import REPO_ROOT, load_config, resolve


def test_default_config_loads_and_has_thesis_values():
    cfg = load_config()
    assert cfg["inference"]["helmet_conf"] == 0.329
    assert cfg["association"] == {"expand_x": 1.3, "expand_up": 2.0, "expand_down": 0.1}
    assert cfg["classes"] == {"with_helmet": 0, "without_helmet": 1}


def test_resolve_relative_and_absolute(tmp_path):
    assert resolve("weights/x.pt") == REPO_ROOT / "weights" / "x.pt"
    assert resolve(tmp_path) == tmp_path


def test_expansion_factors_from_config():
    from helmet_detection.association import ExpansionFactors

    assert ExpansionFactors.from_config(load_config()["association"]) == ExpansionFactors(1.3, 2.0, 0.1)
