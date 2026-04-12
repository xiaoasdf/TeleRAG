import io
import zipfile

from src.standards.ingest import extract_source_to_index_ready, stage_standard_sources


def test_stage_standard_sources_extracts_zip_and_skips_unchanged(tmp_path):
    standards_root = tmp_path / "raw" / "standards"
    archive_path = standards_root / "3gpp" / "rel18" / "38_series" / "38101-1-ic0.zip"
    archive_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("38101-1-ic0.txt", "Beamforming improves directional transmission.")

    index_ready_root = tmp_path / "raw" / "index_ready" / "standards"
    state_path = tmp_path / "raw" / "index_ready" / "standards_ingest_state.json"

    first = stage_standard_sources(standards_root, index_ready_root, state_path)
    extracted_path = index_ready_root / "3gpp" / "rel18" / "38_series" / "38101-1-ic0.txt"
    assert extracted_path.exists()
    assert first["new_or_updated_sources"] == 1
    assert first["skipped_unchanged_sources"] == 0

    second = stage_standard_sources(standards_root, index_ready_root, state_path)
    assert second["new_or_updated_sources"] == 0
    assert second["skipped_unchanged_sources"] == 1
    assert second["extracted_documents"] == 1


def test_extract_source_to_index_ready_converts_docx_to_txt(tmp_path):
    standards_root = tmp_path / "raw" / "standards"
    source_path = standards_root / "itu" / "itu-r" / "sample.docx"
    source_path.parent.mkdir(parents=True)

    docx_bytes = io.BytesIO()
    with zipfile.ZipFile(docx_bytes, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
                "<w:body><w:p><w:r><w:t>Beamforming in wireless systems</w:t></w:r></w:p></w:body>"
                "</w:document>"
            ),
        )
    source_path.write_bytes(docx_bytes.getvalue())

    extracted = extract_source_to_index_ready(
        source_path,
        standards_root=standards_root,
        index_ready_root=tmp_path / "raw" / "index_ready" / "standards",
    )

    assert len(extracted) == 1
    assert extracted[0].suffix == ".txt"
    assert "Beamforming in wireless systems" in extracted[0].read_text(encoding="utf-8")
