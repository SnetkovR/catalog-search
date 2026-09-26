"""Build first, then: python3 tests/docker_smoke.py catalog-search:ci.

The host needs only Python's standard library and Docker. The injected encoder is
mounted into a separate test container; the production image has no test mode.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path


def serve_fixture():
    import numpy as np
    import torch
    import uvicorn
    from PIL import Image

    from catalog_search.api import create_app
    from catalog_search.config import Settings

    assert torch.version.cuda is None, "Linux image must use CPU-only PyTorch"

    class ColorEncoder:
        dimension = 3
        signature = "docker-smoke-colors-v1"

        def encode(self, images):
            return np.stack(
                [np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) for image in images]
            )

    root = Path("/tmp/smoke-catalog")
    root.mkdir()
    Image.new("RGB", (20, 40), "red").save(root / "red.png")
    Image.new("RGB", (20, 40), "blue").save(root / "blue.png")
    settings = Settings(
        catalog=root,
        storage=Path("/tmp/smoke-index"),
        index_interval=0.1,
        index_settle_seconds=0,
        threads=1,
    )
    uvicorn.run(create_app(settings, encoder=ColorEncoder()), host="0.0.0.0", port=8000)


def docker(*args):
    return subprocess.check_output(
        [os.environ.get("DOCKER_BIN", "docker"), *args], text=True, timeout=60
    ).strip()


def request(url, *, data=None, headers=None):
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(url, data=data, headers=headers or {}), timeout=5
        )
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, response.read()


def wait_for(check, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if check():
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise AssertionError("Timed out waiting for Docker service")


@contextmanager
def container(image, fixture=False):
    name = f"catalog-smoke-{uuid.uuid4().hex[:12]}"
    args = [
        "run",
        "--detach",
        "--name",
        name,
        "--publish",
        "127.0.0.1::8000",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--health-interval=1s",
        "--health-start-period=0s",
    ]
    if fixture:
        args += [
            "--entrypoint",
            "python",
            "--mount",
            f"type=bind,source={Path(__file__).resolve()},target=/smoke.py,readonly",
        ]
    try:
        docker(*args, image, *(["/smoke.py", "--serve"] if fixture else []))
        port = docker("port", name, "8000/tcp").rsplit(":", 1)[1]
        url = f"http://127.0.0.1:{port}"
        wait_for(lambda: request(url + "/health/live")[0] == 200)
        wait_for(
            lambda: docker("inspect", "--format", "{{.State.Health.Status}}", name) == "healthy"
        )
        yield name, url
    finally:
        logs = Path("var/ci/docker")
        logs.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [os.environ.get("DOCKER_BIN", "docker"), "logs", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        (logs / ("fixture.log" if fixture else "production.log")).write_text(
            result.stdout + result.stderr
        )
        docker("rm", "--force", name)


def check_image(image):
    # Exercise the real ENTRYPOINT/CMD with an empty cache and no network model downloads.
    with container(image) as (_, url):
        assert request(url + "/")[0] == 200
        assert b"refreshStatus" in request(url + "/static/app.js")[1]
        assert request(url + "/static/styles.css")[0] == 200
        assert request(url + "/health/ready")[0] == 503
        wait_for(lambda: json.loads(request(url + "/api/status")[1])["phase"] == "model_error")
    # Exercise the installed package with a deterministic encoder, not host source imports.
    with container(image, fixture=True) as (name, url):
        wait_for(lambda: request(url + "/health/ready")[0] == 200)
        items = json.loads(request(url + "/api/catalog")[1])["items"]
        assert len(items) == 2
        assert request(url + items[0]["thumbnail_url"])[0] == 200
        code, body = request(url + f"/api/search/catalog/{items[0]['id']}", data=b"")
        assert code == 200 and len(json.loads(body)["results"]) == 1
        # Use a published thumbnail as a query to check real multipart parsing and inference.
        thumbnail = request(url + items[0]["thumbnail_url"])[1]
        upload = (
            b'--smoke\r\nContent-Disposition: form-data; name="file"; filename="q.jpg"\r\n'
            b"Content-Type: image/jpeg\r\n\r\n" + thumbnail + b"\r\n--smoke--\r\n"
        )
        code, body = request(
            url + "/api/search",
            data=upload,
            headers={"Content-Type": "multipart/form-data; boundary=smoke"},
        )
        assert code == 200 and json.loads(body)["results"][0]["path"] == items[0]["path"]
        docker(
            "exec",
            name,
            "python",
            "-c",
            "from pathlib import Path; Path('/tmp/smoke-catalog/blue.png').unlink()",
        )
        wait_for(lambda: json.loads(request(url + "/api/catalog")[1])["total"] == 1)
    print(
        "Docker smoke passed: production startup, health, static assets, "
        "CPU search, background indexing"
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve_fixture()
    else:
        check_image(sys.argv[1] if len(sys.argv) > 1 else "catalog-search:ci")
