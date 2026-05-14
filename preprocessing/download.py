"""File download and archive extraction utilities."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from preprocessing.utils import fetch_with_retry

logger = logging.getLogger(__name__)

DOWNLOAD_CHUNK_SIZE: int = 1024 * 1024


def read_links_file(links_file: Path) -> list[str]:
    """Read non-empty links from a text file."""
    if not links_file.exists():
        raise FileNotFoundError(f"Links file not found: {links_file}")

    links: list[str] = []
    for line in links_file.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned:
            links.append(cleaned)
    return links


def filename_from_url(url: str) -> str:
    """Extract filename from URL path."""
    parsed = urlparse(url)
    filename = Path(parsed.path).name
    if not filename:
        raise ValueError(f"Could not infer filename from URL: {url}")
    return filename


def download_file(url: str, destination: Path, overwrite: bool) -> bool:
    """Download a single file; returns True if written, False if skipped/failed."""
    if destination.exists() and not overwrite:
        logger.info("Skip existing file: %s", destination)
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = fetch_with_retry(url, timeout=120, stream=True)
        with response:
            with destination.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        output_file.write(chunk)
    except requests.exceptions.RequestException as exc:
        logger.warning("Download failed for %s after retries: %s — skipping file", url, exc)
        # Clean up partial download
        if destination.exists():
            destination.unlink()
        return False

    logger.info("Downloaded: %s", destination)
    return True


def download_links(links_file: Path, output_dir: Path, overwrite: bool) -> list[Path]:
    """Download all links listed in links_file into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    links = read_links_file(links_file)

    downloaded_files: list[Path] = []
    for url in links:
        filename = filename_from_url(url)
        destination = output_dir / filename
        wrote_file = download_file(url, destination, overwrite=overwrite)
        if wrote_file:
            downloaded_files.append(destination)

    return downloaded_files


def extract_zip_files(directory: Path, overwrite: bool) -> list[Path]:
    """Extract zip files in directory; returns list of extracted members."""
    extracted_files: list[Path] = []

    for zip_path in sorted(directory.glob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                members = [Path(member) for member in zip_file.namelist() if not member.endswith("/")]

                if not members:
                    logger.info("Skip empty archive: %s", zip_path)
                    continue

                if not overwrite:
                    targets = [directory / member for member in members]
                    if all(target.exists() for target in targets):
                        logger.info("Skip already extracted archive: %s", zip_path)
                        continue

                zip_file.extractall(directory)
                logger.info("Extracted archive: %s", zip_path)
                extracted_files.extend(directory / member for member in members)
        except zipfile.BadZipFile:
            logger.warning("Skip invalid zip file: %s", zip_path)

    return extracted_files
