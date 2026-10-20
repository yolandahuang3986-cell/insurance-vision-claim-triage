from pathlib import Path
import hashlib
import json
from zipfile import ZipFile
import pytest

from scripts.build_submission_zip import build_archive


def test_submission_archive_excludes_local_data(tmp_path: Path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "private.jpg").write_bytes(b"not for submission")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    archive = build_archive(tmp_path, "01", tmp_path / "out")
    with ZipFile(archive) as zipped:
        names = set(zipped.namelist())
    assert "src/main.py" in names
    assert all(not name.startswith("data/") for name in names)


def test_submission_includes_corrective_code_but_not_runtime_artifacts(tmp_path: Path):
    (tmp_path / "experiments" / "results").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "experiments" / "unified_eval.py").write_text("pass\n")
    (tmp_path / "scripts" / "build_submission_zip.py").write_text("pass\n")
    (tmp_path / "experiments" / "model.pt").write_bytes(b"weights")
    (tmp_path / "experiments" / "results" / "prediction.json").write_text("[]")
    archive = build_archive(tmp_path, "01", tmp_path / "out")
    with ZipFile(archive) as zipped:
        names = set(zipped.namelist())
    assert "experiments/unified_eval.py" in names
    assert "scripts/build_submission_zip.py" in names
    assert "experiments/model.pt" not in names
    assert "experiments/results/prediction.json" not in names


def test_manifest_hashes_match_actual_zip(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.py").write_text("pass\n")
    archive = build_archive(tmp_path, "15", tmp_path / "out")
    manifest = json.loads((tmp_path / "out/SUBMISSION_MANIFEST.json").read_text())
    assert manifest["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with ZipFile(archive) as z:
        assert set(z.namelist()) == set(manifest["file_sha256"])
        for name in z.namelist():
            assert hashlib.sha256(z.read(name)).hexdigest() == manifest["file_sha256"][name]


@pytest.mark.parametrize("host_path", ["drive.google.com/file/d/FAKE/view", "colab.research.google.com/drive/FAKE", "example.sharepoint.com/file/FAKE"])
def test_cloud_substitute_links_rejected(tmp_path, host_path):
    (tmp_path / "README.md").write_text("Code is at " + "https://" + host_path)
    with pytest.raises(ValueError, match="Cloud-hosted"):
        build_archive(tmp_path, "15", tmp_path / "out")


def test_any_external_url_rejected(tmp_path):
    (tmp_path / "README.md").write_text("Documentation: " + "https://" + "docs.python.org/3/")
    with pytest.raises(ValueError, match="External link"):
        build_archive(tmp_path, "15", tmp_path / "out")


def test_link_free_manifest_flags(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.py").write_text("pass\n")
    build_archive(tmp_path, "15", tmp_path / "out")
    manifest = json.loads((tmp_path / "out/SUBMISSION_MANIFEST.json").read_text())
    assert manifest["contains_external_links"] is False
    assert manifest["contains_cloud_file_links"] is False


def test_misnamed_dataset_json_rejected(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/notes.json").write_text(json.dumps({"images": [], "annotations": []}))
    with pytest.raises(ValueError, match="Dataset"):
        build_archive(tmp_path, "15", tmp_path / "out")


def test_symlink_images_secrets_and_weights_excluded(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/main.py").write_text("pass")
    for name in ["photo.jpg", "model.PTH", ".env"]:
        (tmp_path / "src" / name).write_text("secret")
    (tmp_path / "src/leak.py").symlink_to(tmp_path / "src/main.py")
    archive = build_archive(tmp_path, "15", tmp_path / "out")
    with ZipFile(archive) as z:
        assert z.namelist() == ["src/main.py"]


def test_group_id_cannot_escape_output_directory(tmp_path):
    with pytest.raises(ValueError, match="Group ID"):
        build_archive(tmp_path, "../15", tmp_path / "out")


def test_baseline_source_included_without_manifest_or_notebook(tmp_path):
    code=tmp_path/'CarDD project/cardd_code/training'
    code.mkdir(parents=True)
    (code/'train_yolo.py').write_text('pass')
    (tmp_path/'CarDD project/01_train.ipynb').write_text('{}')
    manifests=tmp_path/'CarDD project/manifests';manifests.mkdir()
    (manifests/'cardd_manifest.json').write_text('{}')
    archive=build_archive(tmp_path,'15',tmp_path/'out')
    with ZipFile(archive) as z:
        assert z.namelist()==['CarDD project/cardd_code/training/train_yolo.py']
