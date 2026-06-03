from types import SimpleNamespace

import train


def test_peft_save_disables_embedding_layer_autosave(monkeypatch, tmp_path):
    calls = {}

    class FakePeftModel:
        def save_pretrained(self, output_dir, **kwargs):
            calls["output_dir"] = output_dir
            calls["kwargs"] = kwargs

    class FakeAccelerator:
        def unwrap_model(self, model):
            return model

    class FakeProcessingClass:
        def save_pretrained(self, output_dir):
            calls["processing_output_dir"] = output_dir

    monkeypatch.setattr(train, "PeftModel", FakePeftModel)

    trainer = object.__new__(train.NoEmbeddingSaveTrainer)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), save_safetensors=True)
    trainer.model = FakePeftModel()
    trainer.accelerator = FakeAccelerator()
    trainer.processing_class = FakeProcessingClass()

    trainer._save(str(tmp_path))

    assert calls["kwargs"]["save_embedding_layers"] is False
    assert calls["kwargs"]["safe_serialization"] is True
    assert calls["processing_output_dir"] == str(tmp_path)
    assert (tmp_path / "training_args.bin").exists()
