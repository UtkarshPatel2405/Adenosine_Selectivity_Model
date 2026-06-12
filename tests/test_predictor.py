import pytest


class TestPredict:
    def test_invalid_smiles_raises(self):
        from src.predictor import predict
        with pytest.raises(ValueError, match="Invalid SMILES"):
            predict("INVALID")

    def test_returns_dict(self):
        from src.predictor import predict
        result = predict("CCO")
        assert isinstance(result, dict)
        assert "smiles" in result
        assert "descriptors" in result
        assert "predictions" in result

    def test_descriptors_content(self):
        from src.predictor import predict
        result = predict("CCO")
        desc = result["descriptors"]
        assert "MW" in desc
        assert "LogP" in desc
        assert "HBD" in desc
        assert "HBA" in desc
