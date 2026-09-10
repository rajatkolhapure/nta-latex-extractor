"""
Resilient question image downloader with retry logic and concurrency.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import requests

from nta_extractor.config import (
    DEFAULT_HEADERS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_FACTOR,
)


def download_single_image(
    image_url: str,
    output_dir: str,
    session: Optional[requests.Session] = None,
    max_retries: int = MAX_RETRIES,
    timeout: int = REQUEST_TIMEOUT,
) -> Tuple[str, bool]:
    """
    Downloads a single question JPEG image to output_dir with exponential backoff.

    Returns:
        Tuple of (local_file_path, success_flag)
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = image_url.split("/")[-1]
    # Remove any query params from filename if present
    if "?" in filename:
        filename = filename.split("?")[0]

    local_path = target_dir / filename
    rel_path = f"./{target_dir.as_posix()}/{filename}".replace("//", "/")

    # Skip if already downloaded and has content
    if local_path.exists() and local_path.stat().st_size > 0:
        return rel_path, True

    requester = session or requests.Session()
    headers = dict(DEFAULT_HEADERS)
    headers["Referer"] = "https://nta.ac.in/Quiz/Home/Paper"

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requester.get(image_url, headers=headers, timeout=timeout, stream=True)
            if resp.status_code == 200:
                temp_path = local_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                # Atomic rename
                if temp_path.exists():
                    temp_path.replace(local_path)
                return rel_path, True
            else:
                last_error = f"HTTP {resp.status_code}"
        except Exception as e:
            last_error = str(e)

        # Exponential backoff before retry
        if attempt < max_retries:
            sleep_time = RETRY_BACKOFF_FACTOR ** attempt
            time.sleep(sleep_time)

    # Failed after retries
    return rel_path, False


def download_images_batch(
    image_urls: List[str],
    output_dir: str,
    concurrency: int = 5,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[Tuple[str, str, bool]]:
    """
    Downloads a batch of image URLs concurrently.

    Returns:
        List of tuples: (image_url, local_path, success)
    """
    results: List[Tuple[str, str, bool]] = []
    total = len(image_urls)
    completed = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {
            executor.submit(download_single_image, url, output_dir): url
            for url in image_urls
        }
        for future in as_completed(future_map):
            url = future_map[future]
            try:
                local_path, ok = future.result()
                results.append((url, local_path, ok))
            except Exception:
                results.append((url, "", False))
            completed += 1
            if on_progress:
                on_progress(completed, total)

    return results
