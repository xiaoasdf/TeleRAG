import json

from src.standards.downloader import build_download_jobs, download_jobs


def test_build_download_jobs_expands_directory_targets():
    html = """
    <html>
      <body>
        <a href="38101-1-ic0.zip">38101-1-ic0.zip</a>
        <a href="38104-ic0.zip">38104-ic0.zip</a>
        <a href="subdir/">subdir/</a>
      </body>
    </html>
    """
    targets = [
        {
            "name": "3gpp_rel18_38_series",
            "source_org": "3GPP",
            "kind": "directory",
            "url": "https://www.3gpp.org/ftp/specs/2025-12/Rel-18/38_series",
            "destination_dir": "standards/3gpp/rel18/38_series",
            "file_suffixes": [".zip"],
            "release": "Rel-18",
            "series": "38_series",
        }
    ]

    jobs = build_download_jobs(targets, fetch_text_func=lambda _: html)

    assert [job.filename for job in jobs] == ["38101-1-ic0.zip", "38104-ic0.zip"]
    assert all(job.destination_dir == "standards/3gpp/rel18/38_series" for job in jobs)


def test_download_jobs_writes_manifest_and_skips_existing(tmp_path):
    jobs = build_download_jobs(
        [
            {
                "name": "itu_r_p1239",
                "source_org": "ITU-R",
                "kind": "file",
                "url": "https://example.com/itu-r-p1239.pdf",
                "destination_dir": "standards/itu/itu-r",
                "filename": "ITU-R_P.1239-0.pdf",
            }
        ]
    )
    manifest_path = tmp_path / "manifest.json"

    manifest = download_jobs(
        jobs,
        output_root=tmp_path,
        manifest_path=manifest_path,
        fetch_bytes_func=lambda _: b"pdf-bytes",
    )

    downloaded_file = tmp_path / "standards" / "itu" / "itu-r" / "ITU-R_P.1239-0.pdf"
    assert downloaded_file.exists()
    assert downloaded_file.read_bytes() == b"pdf-bytes"
    assert manifest["downloaded_bytes"] == len(b"pdf-bytes")

    second_manifest = download_jobs(
        jobs,
        output_root=tmp_path,
        manifest_path=manifest_path,
        fetch_bytes_func=lambda _: b"new-pdf-bytes",
    )
    assert second_manifest["records"][0]["status"] == "skipped_existing"

    manifest_on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_on_disk["records"][0]["filename"] == "ITU-R_P.1239-0.pdf"
