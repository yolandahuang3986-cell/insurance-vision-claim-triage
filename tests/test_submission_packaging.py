from pathlib import Path
from zipfile import ZipFile

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
