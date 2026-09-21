import os
import sys
import csv
import time
import shutil
import argparse
import threading
import concurrent.futures
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT_DATA = os.environ.get("PATH_ROOT_DATA", str(BASE_DIR))
DEFAULT_TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "15d2ea6d0dc1d476efbca3eba2b9bbfb")

thread_local = threading.local()
csv_lock = threading.Lock()


def get_session():
    """Returns a thread-local requests.Session with connection pooling and headers."""
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=2
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        thread_local.session = session
    return thread_local.session


def get_poster_from_tmdb_api(session, tmdb_id, api_key, timeout=10):
    """Tier 1: Retrieves poster URL using the official TMDB API."""
    api_url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}"
    try:
        for attempt in range(2):
            r = session.get(api_url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.status_code == 200:
                data = r.json()
                poster_path = data.get("poster_path")
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    movie_url = f"https://www.themoviedb.org/movie/{int(tmdb_id)}"
                    return poster_url, movie_url
            break
    except Exception:
        pass
    return None, None


def get_poster_from_tmdb_web(session, tmdb_id, timeout=10):
    """Tier 2: Retrieves poster URL from TMDB webpage without requiring an API key."""
    page_url = f"https://www.themoviedb.org/movie/{int(tmdb_id)}"
    try:
        for attempt in range(2):
            r = session.get(page_url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                meta_img = soup.find("meta", property="og:image")
                if meta_img and meta_img.get("content"):
                    content = meta_img["content"]
                    if content.startswith("http") and "media.themoviedb.org" in content:
                        return content, page_url

                img = soup.find("img", class_="poster")
                if img and img.get("src"):
                    src = img["src"]
                    if src.startswith("/"):
                        src = f"https://media.themoviedb.org{src}"
                    return src, page_url
            break
    except Exception:
        pass
    return None, None


def get_poster_from_tmdb_find_imdb(session, imdb_id, api_key, timeout=10):
    """Tier 3: Retrieves poster from TMDB using the imdbId via /find endpoint."""
    tt_code = f"tt{str(int(imdb_id)).zfill(7)}"
    api_url = f"https://api.themoviedb.org/3/find/{tt_code}?api_key={api_key}&external_source=imdb_id"
    try:
        for attempt in range(2):
            r = session.get(api_url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.status_code == 200:
                results = r.json().get("movie_results", [])
                if results and results[0].get("poster_path"):
                    poster_url = f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
                    movie_url = f"https://www.themoviedb.org/movie/{results[0]['id']}"
                    return poster_url, movie_url
            break
    except Exception:
        pass
    return None, None


def get_poster_from_imdb_cdn(session, imdb_id, timeout=10):
    """Tier 4: Retrieves poster from IMDb official CDN suggestion API (bypasses WAF)."""
    tt_code = f"tt{str(int(imdb_id)).zfill(7)}"
    first_char = tt_code[2]
    api_url = f"https://v3.sg.media-imdb.com/suggestion/titles/{first_char}/{tt_code}.json"
    try:
        r = session.get(api_url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            for entry in data.get("d", []):
                if entry.get("id") == tt_code and entry.get("i"):
                    img_url = entry["i"].get("imageUrl")
                    if img_url:
                        return img_url, f"https://www.imdb.com/title/{tt_code}/"
    except Exception:
        pass
    return None, None


def create_placeholder_image(output_path, title, movie_id):
    """Tier 5: Generates a clean placeholder poster image when no poster is found online."""
    try:
        width, height = 300, 450
        img = Image.new("RGB", (width, height), color=(30, 34, 42))
        draw = ImageDraw.Draw(img)

        # Draw border
        draw.rectangle([10, 10, width - 10, height - 10], outline=(70, 78, 96), width=2)

        # Format title into multiple lines
        text_lines = [
            f"Movie ID: {movie_id}",
            "",
            title[:38],
            title[38:76] if len(title) > 38 else "",
            title[76:114] if len(title) > 76 else "",
            "",
            "(Poster Not Available)"
        ]
        text_lines = [l for l in text_lines if l != ""]

        y_offset = 140
        for line in text_lines:
            draw.text((25, y_offset), line, fill=(200, 205, 215))
            y_offset += 24

        img.save(output_path, "JPEG", quality=85)
        return True
    except Exception:
        return False


def download_image(session, image_url, output_path, timeout=15):
    """Downloads an image and validates its integrity."""
    try:
        r = session.get(image_url, timeout=timeout, stream=True)
        if r.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return True
            if os.path.exists(output_path):
                os.remove(output_path)
    except Exception:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
    return False


def append_csv_safe(file_path, row):
    """Thread-safe CSV append."""
    with csv_lock:
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


def process_movie_row(row, path_image_output, file_movie_url, file_poster_url, file_missing, api_key=None, create_placeholders=False):
    """Processes a single movie through a 5-tier fallback cascade."""
    movie_id = int(row["movieId"])
    tmdb_id = row.get("tmdbId")
    imdb_id = row.get("imdbId")
    title = str(row.get("title", f"Movie {movie_id}"))
    img_filename = Path(path_image_output) / f"{movie_id}.jpg"

    if img_filename.exists() and img_filename.stat().st_size > 1000:
        return movie_id, "exists"

    session = get_session()
    poster_url, movie_page_url = None, None

    # Tier 1: TMDB API by tmdb_id
    if api_key and pd.notna(tmdb_id):
        poster_url, movie_page_url = get_poster_from_tmdb_api(session, tmdb_id, api_key)

    # Tier 2: TMDB Web scraping by tmdb_id
    if not poster_url and pd.notna(tmdb_id):
        poster_url, movie_page_url = get_poster_from_tmdb_web(session, tmdb_id)

    # Tier 3: TMDB API find by imdb_id
    if not poster_url and api_key and pd.notna(imdb_id):
        poster_url, movie_page_url = get_poster_from_tmdb_find_imdb(session, imdb_id, api_key)

    # Tier 4: IMDb official CDN suggestion API (bypasses AWS WAF)
    if not poster_url and pd.notna(imdb_id):
        poster_url, movie_page_url = get_poster_from_imdb_cdn(session, imdb_id)

    # Download image if found
    if poster_url and movie_page_url:
        if download_image(session, poster_url, img_filename):
            append_csv_safe(file_movie_url, [movie_id, movie_page_url])
            append_csv_safe(file_poster_url, [movie_id, poster_url])
            return movie_id, "downloaded"

    # Tier 5: Placeholder generator if requested
    if create_placeholders:
        if create_placeholder_image(img_filename, title, movie_id):
            append_csv_safe(file_movie_url, [movie_id, "placeholder"])
            append_csv_safe(file_poster_url, [movie_id, "placeholder"])
            return movie_id, "placeholder"

    # Truly missing
    append_csv_safe(file_missing, [movie_id])
    return movie_id, "missing"


def load_processed_ids(file_poster_url, file_missing=None, legacy_poster_url=None):
    """Loads previously processed movie IDs into a set for O(1) lookup."""
    processed = set()

    for path in [file_poster_url, legacy_poster_url]:
        if path and Path(path).exists():
            try:
                df = pd.read_csv(path, names=["id", "url"], header=None, engine="python")
                valid_ids = pd.to_numeric(df["id"], errors="coerce").dropna().astype(int)
                processed.update(valid_ids)
            except Exception:
                pass

    if file_missing and Path(file_missing).exists():
        try:
            df = pd.read_csv(file_missing, names=["id"], header=None, engine="python")
            valid_ids = pd.to_numeric(df["id"], errors="coerce").dropna().astype(int)
            processed.update(valid_ids)
        except Exception:
            pass

    return processed


def main():
    parser = argparse.ArgumentParser(description="Download MovieLens poster images safely and concurrently.")
    parser.add_argument("--root-data", default=DEFAULT_ROOT_DATA, help="Path to data root directory")
    parser.add_argument("--dataset", default="ml-25m", help="MovieLens dataset size (e.g. ml-25m, ml-20m)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of movies to process (for testing)")
    parser.add_argument("--api-key", default=DEFAULT_TMDB_API_KEY, help="TMDB API key")
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="Ignore previously saved missing file and retry downloading all missing movies"
    )
    parser.add_argument(
        "--create-placeholders",
        action="store_true",
        help="Generate default placeholder images for movies with no poster online"
    )
    args = parser.parse_args()

    root_data = Path(args.root_data).resolve()
    dataset = args.dataset

    path_image_output = root_data / "image_orgin" / "movielens"
    path_image_output.mkdir(parents=True, exist_ok=True)

    alias_image_origin = root_data / "image_origin"
    if not alias_image_origin.exists() and not alias_image_origin.is_symlink():
        try:
            alias_image_origin.symlink_to(root_data / "image_orgin", target_is_directory=True)
        except (OSError, NotImplementedError):
            pass

    file_movie_data = root_data / "movielens" / dataset / "movies.csv"
    file_links_data = root_data / "movielens" / dataset / "links.csv"

    file_movie_url_output = root_data / "movielens" / f"movie_url_{dataset}.csv"
    file_movie_url_poster_output = root_data / "movielens" / f"movie_poster_url_{dataset}.csv"
    file_id_movie_missing_output = root_data / "movielens" / f"movie_id_poster_missing_{dataset}.csv"
    legacy_poster_url = root_data / "movielens" / f"movie_poster_url_{dataset}g.csv"

    for file_path in [file_movie_url_output, file_movie_url_poster_output, file_id_movie_missing_output]:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.touch()

    if not file_movie_data.exists():
        print(f"Error: Movies file not found at: {file_movie_data}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading movie metadata from {file_movie_data}...")
    df_movies = pd.read_csv(file_movie_data)

    if file_links_data.exists():
        print(f"Loading link mappings from {file_links_data}...")
        df_links = pd.read_csv(file_links_data)
        df_all = pd.merge(df_movies, df_links, on="movieId", how="left")
    else:
        print(f"Notice: {file_links_data} not found. Operating with movie title only.")
        df_all = df_movies
        if "tmdbId" not in df_all.columns:
            df_all["tmdbId"] = None
        if "imdbId" not in df_all.columns:
            df_all["imdbId"] = None

    if args.retry_missing:
        print("Option --retry-missing enabled: Retrying previously missing movies...")
        backup_missing = root_data / "movielens" / f"movie_id_poster_missing_{dataset}.backup.csv"
        if file_id_movie_missing_output.exists() and not backup_missing.exists():
            shutil.copy(file_id_movie_missing_output, backup_missing)
            print(f"Backed up old missing file to: {backup_missing}")
        open(file_id_movie_missing_output, "w").close()
        processed_ids = load_processed_ids(
            file_movie_url_poster_output,
            file_missing=None,
            legacy_poster_url=legacy_poster_url
        )
    else:
        processed_ids = load_processed_ids(
            file_movie_url_poster_output,
            file_missing=file_id_movie_missing_output,
            legacy_poster_url=legacy_poster_url
        )

    print(f"Found {len(processed_ids)} previously completed movies.")

    to_process = []
    for _, row in df_all.iterrows():
        mid = int(row["movieId"])
        if mid not in processed_ids:
            img_file = path_image_output / f"{mid}.jpg"
            if img_file.exists() and img_file.stat().st_size > 1000:
                processed_ids.add(mid)
                continue
            to_process.append(row)

    if args.limit:
        to_process = to_process[:args.limit]

    total_tasks = len(to_process)
    print(f"Movies to process: {total_tasks} (using {args.workers} worker threads)")

    if total_tasks == 0:
        print("All movies have already been processed!")
        return

    downloaded_count = 0
    missing_count = 0
    exists_count = 0
    placeholder_count = 0

    pbar = tqdm(total=total_tasks, desc="Downloading posters", unit="movie") if tqdm else None

    # Bounded concurrent submission to prevent memory overload
    max_in_flight = args.workers * 4
    in_flight = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        row_iter = iter(to_process)

        # Prefill initial tasks
        for _ in range(min(max_in_flight, total_tasks)):
            try:
                row = next(row_iter)
                future = executor.submit(
                    process_movie_row,
                    row,
                    path_image_output,
                    file_movie_url_output,
                    file_movie_url_poster_output,
                    file_id_movie_missing_output,
                    args.api_key,
                    args.create_placeholders
                )
                in_flight.add(future)
            except StopIteration:
                break

        # Process sliding window
        while in_flight:
            done, in_flight = concurrent.futures.wait(
                in_flight,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                try:
                    _, status = future.result()
                    if status == "downloaded":
                        downloaded_count += 1
                    elif status == "placeholder":
                        placeholder_count += 1
                    elif status == "missing":
                        missing_count += 1
                    elif status == "exists":
                        exists_count += 1
                except Exception:
                    missing_count += 1

                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        "downloaded": downloaded_count,
                        "placeholder": placeholder_count,
                        "missing": missing_count,
                        "exists": exists_count
                    })
                else:
                    completed = downloaded_count + placeholder_count + missing_count + exists_count
                    if completed % 100 == 0 or completed == total_tasks:
                        print(f"Progress: {completed}/{total_tasks} | Downloaded: {downloaded_count} | Missing: {missing_count}")

                try:
                    next_row = next(row_iter)
                    next_future = executor.submit(
                        process_movie_row,
                        next_row,
                        path_image_output,
                        file_movie_url_output,
                        file_movie_url_poster_output,
                        file_id_movie_missing_output,
                        args.api_key,
                        args.create_placeholders
                    )
                    in_flight.add(next_future)
                except StopIteration:
                    pass

    if pbar:
        pbar.close()

    print("\nFinished!")
    print(f"- Total processed: {total_tasks}")
    print(f"- Successfully downloaded: {downloaded_count}")
    if placeholder_count > 0:
        print(f"- Placeholders generated: {placeholder_count}")
    print(f"- Missing / Failed: {missing_count}")
    print(f"- Already existing: {exists_count}")
    print(f"- Image directory: {path_image_output}")


if __name__ == "__main__":
    main()
