import json
import unittest
import snapped
from models.pydantic_models import (
    MemoryBudget,
    DeviceSpec,
    TensorShape,
    QuantizationConfig,
    NormConfig,
    AttentionConfig,
    _BaseOptimiser,
    AdamWConfig,
    SGDConfig,
    AdaFactorConfig,
    LionConfig,
    OptimizerConfig,
)


class TestPydantic(unittest.TestCase):

    def _test(self, model: type) -> tuple[snapped.SnappedModel, snapped.SnappedModel]:
        source = snapped.snap(MemoryBudget)
        target = snapped.snap(snapped.unsnap(source))
        return (source, target)

    def test_write(self):
        source = snapped.snap(MemoryBudget)
        target = snapped.snap(snapped.unsnap(source))

        with open("model.source.json", "w") as f:
            json.dump(source.to_dict(), f, indent=2)
        with open("model.target.json", "w") as f:
            json.dump(target.to_dict(), f, indent=2)

        self.assertEqual(source.schema, target.schema)

    def test_memory_budget(self):
        source, target = self._test(MemoryBudget)
        self.assertEqual(source.schema, target.schema)

    def test_device_spec(self):
        source, target = self._test(DeviceSpec)
        self.assertEqual(source.schema, target.schema)

    def test_tensor_shape(self):
        source, target = self._test(TensorShape)
        self.assertEqual(source.schema, target.schema)

    def test_quant_config(self):
        source, target = self._test(QuantizationConfig)
        self.assertEqual(source.schema, target.schema)

    def test_norm_config(self):
        source, target = self._test(NormConfig)
        self.assertEqual(source.schema, target.schema)

    def test_attention_config(self):
        source, target = self._test(AttentionConfig)
        self.assertEqual(source.schema, target.schema)

    def test_optimisers(self):
        for opt in [
            _BaseOptimiser,
            AdamWConfig,
            SGDConfig,
            AdaFactorConfig,
            LionConfig,
        ]:

            source, target = self._test(opt)
            self.assertEqual(source.schema, target.schema)


if __name__ == "__main__":
    unittest.main()
