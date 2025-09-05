import pytest

from sundew.config_presets import list_presets, get_preset
from sundew.config import SundewConfig

def test_list_presets_nonempty():
    names = list_presets()
    assert isinstance(names, list) and names, "No presets registered"

@pytest.mark.parametrize("name", list_presets())
def test_every_preset_instantiates(name):
    cfg = get_preset(name)
    assert isinstance(cfg, SundewConfig)
    # Basic sanity checks that also execute lines inside builders
    assert 0.0 <= cfg.min_threshold <= cfg.max_threshold <= 1.0
    assert 0.0 <= cfg.activation_threshold <= 1.0
    assert 0.0 <= cfg.target_activation_rate <= 1.0

def test_get_preset_overrides_and_errors():
    cfg = get_preset("tuned_v2", overrides=dict(target_activation_rate=0.30, gate_temperature=0.15))
    assert cfg.target_activation_rate == pytest.approx(0.30)
    assert cfg.gate_temperature == pytest.approx(0.15)

    with pytest.raises(KeyError):
        get_preset("does_not_exist")

    with pytest.raises(AttributeError):
        get_preset("tuned_v2", overrides={"not_a_field": 123})
