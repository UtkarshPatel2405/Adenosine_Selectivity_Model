from src.mlflow_tracking import _flatten_dict


class TestFlattenDict:
    def test_simple_dict(self):
        result = _flatten_dict({"a": 1.0, "b": 2.0})
        assert result == {"a": 1.0, "b": 2.0}

    def test_nested_dict(self):
        result = _flatten_dict({"outer": {"inner": 1.0, "other": 2.0}})
        assert result == {"outer_inner": 1.0, "outer_other": 2.0}

    def test_mixed_types(self):
        result = _flatten_dict({"a": 1.0, "b": "string", "c": {"d": 3.0}})
        assert "a" in result
        assert "b" not in result
        assert "c_d" in result
