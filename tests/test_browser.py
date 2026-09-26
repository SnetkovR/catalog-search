"""Opt-in headless UI checks: RUN_BROWSER_TESTS=1 pytest tests/test_browser.py."""

import os
import socket
import threading
import time

import pytest
import uvicorn
from PIL import Image
from playwright.sync_api import expect, sync_playwright
from test_catalog import ColorEncoder

from catalog_search.api import create_app
from catalog_search.catalog import index_catalog
from catalog_search.config import Settings

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_BROWSER_TESTS") != "1", reason="Set RUN_BROWSER_TESTS=1 to launch Chromium"
)


@pytest.fixture
def live_app(tmp_path):
    root = tmp_path / "catalog"
    root.mkdir()
    Image.new("RGB", (100, 200), "red").save(root / "red.png")
    Image.new("RGB", (100, 200), "blue").save(root / "blue.png")
    settings = Settings(catalog=root, storage=tmp_path / "var", threads=1)
    encoder = ColorEncoder()
    index_catalog(settings, encoder)
    server = uvicorn.Server(
        uvicorn.Config(create_app(settings, encoder=encoder), log_level="error")
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, "Test server failed to start"
    try:
        yield f"http://127.0.0.1:{port}", root, tmp_path
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


@pytest.mark.parametrize(
    "viewport", [{"width": 1365, "height": 1000}, {"width": 390, "height": 844}]
)
def test_upload_crop_neighbors_and_responsive_layout(live_app, viewport):
    url, root, tmp_path = live_app
    query = Image.new("RGB", (100, 200), "red")
    query.paste("blue", (0, 100, 100, 200))
    query.save(tmp_path / "query.png")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=viewport)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        expect(page.locator(".card")).to_have_count(2)
        expect(page.locator("#search")).to_be_disabled()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

        page.locator("#file").set_input_files(str(tmp_path / "query.png"))
        expect(page.locator("#search")).to_be_enabled()
        page.locator("summary").click()
        page.locator("#crop-top").fill("50")
        page.locator("#apply-crop").click()
        page.locator("#search").click()
        expect(page.locator("#results-title")).to_have_text("Похожие фотографии")
        expect(page.locator(".card-name").first).to_have_text("blue.png")
        expect(page.locator("#clear-crop")).to_be_visible()

        page.locator("#show-catalog").click()
        page.locator(".card button").first.click()
        expect(page.locator("#results-title")).to_have_text("Похожие фотографии")
        expect(page.locator(".card")).to_have_count(1)
        expect(page.locator("#exclude")).to_be_disabled()
        expect(page.locator("#crop-details")).to_be_hidden()

        # Reloading the same file still triggers selection after the input was reset.
        page.locator("#file").set_input_files(str(root / "red.png"))
        expect(page.locator("#exclude")).to_be_enabled()
        page.locator("#exclude").uncheck()
        page.locator("#search").click()
        expect(page.locator(".card-name").first).to_have_text("red.png")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert not errors
        browser.close()
