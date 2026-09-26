"use strict";
const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const context = canvas.getContext("2d");
let selectedFile = null, selectedCatalog = null, picture = null, crop = null;
let pointerStart = null, busy = false, ready = false, imageVersion = 0;
let catalogItems = [], catalogTotal = 0, objectURL = null;
let catalogGeneration, showingResults = false, refreshingStatus = false;

function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}
function setBusy(value) {
  busy = value;
  $("search").disabled = busy || !ready || (!selectedFile && !selectedCatalog);
  $("search").firstElementChild.textContent = busy ? "Ищем похожие…" : "Найти похожие";
  document.querySelector(".results-panel").setAttribute("aria-busy", String(busy));
  ["change", "show-catalog", "load-more", "top-k", "exclude", "clear-crop", "apply-crop", "file"].forEach(id => $(id).disabled = busy);
  $("exclude").disabled = busy || Boolean(selectedCatalog);
  document.querySelectorAll(".card button").forEach(button => button.disabled = busy || !ready);
}
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Проверьте файл и параметры поиска.");
  return data;
}
function card(item, rank) {
  const article = document.createElement("article");
  article.className = "card";
  const photo = document.createElement("div"); photo.className = "card-photo";
  const img = document.createElement("img");
  img.src = item.thumbnail_url; img.alt = item.path; img.loading = "lazy";
  photo.append(img);
  if (rank) { const badge = document.createElement("span"); badge.className = "rank"; badge.textContent = rank; photo.append(badge); }
  const content = document.createElement("div"); content.className = "card-content";
  const name = document.createElement("p"); name.className = "card-name"; name.textContent = item.path; name.title = item.path;
  const bottom = document.createElement("div"); bottom.className = "card-bottom";
  if (item.score !== undefined) {
    const score = document.createElement("span"); score.className = "score";
    score.textContent = `Сходство ${item.score.toFixed(3)}`;
    score.title = "Косинусное сходство, не вероятность совпадения"; bottom.append(score);
  }
  const button = document.createElement("button"); button.type = "button"; button.className = "text-button";
  button.textContent = "Найти похожие ↗"; button.disabled = !ready;
  button.addEventListener("click", () => searchCatalog(item));
  bottom.append(button); content.append(name, bottom); article.append(photo, content);
  return article;
}
function render(items, results = false) {
  $("grid").replaceChildren(...items.map((item, i) => card(item, results ? i + 1 : null)));
  $("empty").hidden = items.length > 0;
  $("empty-text").textContent = results ? "Других фотографий не найдено. Попробуйте отключить исключение точных копий." : "Добавьте фотографии в папку каталога. При включённом автообновлении они появятся здесь после обработки.";
}
function showCatalog() {
  showingResults = false;
  $("results-label").textContent = "ВАША КОЛЛЕКЦИЯ";
  $("results-title").textContent = "Фотографии каталога";
  $("show-catalog").hidden = true;
  $("load-more").hidden = catalogItems.length >= catalogTotal;
  render(catalogItems);
  message(ready ? "Выберите фотографию в каталоге или загрузите свою." : "Каталог пока не готов к поиску. Состояние обновления показано выше.");
}
async function loadCatalog(append = false, display = true) {
  const offset = append ? catalogItems.length : 0;
  const data = await api(`/api/catalog?limit=24&offset=${offset}`);
  catalogItems = append ? [...catalogItems, ...data.items] : data.items;
  catalogGeneration = data.generation;
  catalogTotal = data.total; $("count").textContent = data.total;
  if (display && !showingResults) showCatalog();
}
function indexingStatus(job) {
  const element = $("indexing-status");
  let text;
  if (!job || !job.enabled) text = "Автообновление выключено. Каталог обновляется вручную.";
  else if (job.state === "running") text = "Проверяем каталог и обрабатываем новые фотографии…";
  else if (job.state === "error") text = `Не удалось обновить каталог: ${job.error}. Повторим попытку автоматически.`;
  else if (job.state === "waiting") text = job.error ? "Каталог обновляется другим процессом. Проверим снова автоматически." : "Ожидаем проверки каталога…";
  else if (job.report?.errors.length) text = `Не удалось обработать файлов: ${job.report.errors.length}. Остальные фотографии доступны.`;
  else if (job.report?.deferred) text = `Ожидаем завершения записи файлов: ${job.report.deferred}.`;
  else text = `Каталог обновляется автоматически. Интервал проверки: ${job.interval_seconds} с.`;
  element.textContent = text;
  element.classList.toggle("error", job?.state === "error" || Boolean(job?.report?.errors.length));
  element.title = (job?.report?.errors || []).slice(0, 5).map(item => `${item.path}: ${item.error}`).join("\n");
}
async function refreshStatus() {
  if (refreshingStatus) return;
  refreshingStatus = true;
  try {
    const status = await api("/api/status");
    ready = status.ready;
    indexingStatus(status.indexing);
    $("count").textContent = status.count;
    if (!busy && status.generation !== catalogGeneration) {
      // Refresh the collection without replacing an in-progress query or its results.
      const data = await api("/api/catalog?limit=24");
      catalogItems = data.items; catalogTotal = data.total; catalogGeneration = data.generation;
      if (!busy && !showingResults) showCatalog();
    }
    setBusy(busy);
  } catch {
    $("indexing-status").textContent = "Нет связи с сервером. Повторим проверку автоматически.";
    $("indexing-status").classList.add("error");
  } finally { refreshingStatus = false; }
}
function draw() {
  if (!picture) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(picture, 0, 0, canvas.width, canvas.height);
  if (crop) {
    const [l, t, r, b] = crop;
    context.fillStyle = "rgba(12,28,15,.48)";
    context.beginPath(); context.rect(0, 0, canvas.width, canvas.height);
    context.rect(l * canvas.width, t * canvas.height, (r-l) * canvas.width, (b-t) * canvas.height);
    context.fill("evenodd"); context.strokeStyle = "#c6f17e"; context.lineWidth = 2;
    context.strokeRect(l * canvas.width, t * canvas.height, (r-l) * canvas.width, (b-t) * canvas.height);
  }
}
function syncCrop() {
  const values = crop || [0, 0, 1, 1];
  ["left", "top", "right", "bottom"].forEach((side, i) => $("crop-" + side).value = Math.round(values[i] * 100));
  $("clear-crop").hidden = !crop;
  draw();
}
async function preview(src, filename, fromCatalog = false) {
  const version = ++imageVersion;
  const image = new Image();
  image.src = src;
  await image.decode();
  if (version !== imageVersion) return false;
  picture = image; crop = null;
  const scale = Math.min(600 / image.naturalWidth, 640 / image.naturalHeight, 1);
  canvas.width = Math.round(image.naturalWidth * scale);
  canvas.height = Math.round(image.naturalHeight * scale);
  canvas.classList.toggle("catalog-preview", fromCatalog);
  $("filename").textContent = filename;
  $("preview").hidden = false; $("dropzone").hidden = true;
  $("crop-help").textContent = fromCatalog ? "Ищем по фотографии из каталога. Сам снимок и его точные копии исключены." : "Выделите предмет на фотографии, чтобы искать по выбранной области.";
  $("crop-details").hidden = fromCatalog;
  syncCrop(); return true;
}
async function selectFile(file) {
  if (!file || busy) return;
  if (file.size > 20 * 1024 * 1024) { message("Файл больше 20 МБ.", true); return; }
  if (!/\.(jpe?g|png|webp)$/i.test(file.name)) { message("Выберите JPEG, PNG или WebP.", true); return; }
  const url = URL.createObjectURL(file);
  try {
    if (!await preview(url, file.name)) { URL.revokeObjectURL(url); return; }
    if (objectURL) URL.revokeObjectURL(objectURL);
    objectURL = url; selectedFile = file; selectedCatalog = null;
    setBusy(false); message("Фотография готова. Можно выделить предмет или искать по всему кадру.");
  } catch { URL.revokeObjectURL(url); message("Не удалось открыть фотографию.", true); }
}
function showResults(data) {
  showingResults = true;
  $("results-label").textContent = "РЕЗУЛЬТАТЫ ПОИСКА";
  $("results-title").textContent = "Похожие фотографии";
  $("show-catalog").hidden = false; $("load-more").hidden = true;
  render(data.results, true);
  const plural = new Intl.PluralRules("ru").select(data.results.length);
  const noun = {one: "результат", few: "результата", many: "результатов", other: "результата"}[plural];
  message(`${data.results.length} ${noun} · ${(data.elapsed_ms / 1000).toFixed(2)} с · Сходство не является вероятностью совпадения.`);
}
async function search() {
  if (busy || !ready || (!selectedFile && !selectedCatalog)) return;
  setBusy(true); message("Ищем похожие фотографии…");
  try {
    let data;
    if (selectedCatalog) {
      data = await api(`/api/search/catalog/${selectedCatalog.id}?top_k=${$("top-k").value}`, {method: "POST"});
    } else {
      const form = new FormData(); form.append("file", selectedFile);
      form.append("top_k", $("top-k").value); form.append("exclude_identical", $("exclude").checked);
      if (crop) form.append("crop", JSON.stringify(crop));
      data = await api("/api/search", {method: "POST", body: form});
    }
    showResults(data);
  } catch (error) { message(error.message || "Не удалось выполнить поиск.", true); }
  finally { setBusy(false); }
}
async function searchCatalog(item) {
  if (busy || !ready) return;
  setBusy(true);
  try {
    await preview(item.thumbnail_url, item.path, true);
    selectedCatalog = item; selectedFile = null;
    $("exclude").checked = true;
  } catch { message("Не удалось открыть фотографию каталога.", true); setBusy(false); return; }
  setBusy(false); await search();
}
function point(event) {
  const rect = canvas.getBoundingClientRect();
  return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
}
canvas.addEventListener("pointerdown", event => {
  if (!selectedFile || busy) return;
  event.preventDefault(); pointerStart = point(event); canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", event => {
  if (!pointerStart) return;
  const p = point(event);
  crop = [Math.min(p[0], pointerStart[0]), Math.min(p[1], pointerStart[1]), Math.max(p[0], pointerStart[0]), Math.max(p[1], pointerStart[1])];
  syncCrop();
});
function finishCrop() {
  if (!pointerStart) return;
  pointerStart = null;
  if (crop && (crop[2] - crop[0] < .01 || crop[3] - crop[1] < .01)) crop = null;
  syncCrop();
}
canvas.addEventListener("pointerup", finishCrop);
canvas.addEventListener("pointercancel", finishCrop);
$("clear-crop").addEventListener("click", () => { crop = null; syncCrop(); });
$("apply-crop").addEventListener("click", () => {
  const values = ["left", "top", "right", "bottom"].map(side => Number($("crop-" + side).value) / 100);
  if (!values.every(v => Number.isFinite(v) && v >= 0 && v <= 1) || values[0] >= values[2] || values[1] >= values[3]) { message("Проверьте координаты области: от 0 до 100%, с положительной шириной и высотой.", true); return; }
  crop = values; syncCrop(); message("Область выбрана. Нажмите «Найти похожие».");
});
$("file").addEventListener("change", event => { selectFile(event.target.files[0]); event.target.value = ""; });
$("change").addEventListener("click", () => $("file").click());
$("search").addEventListener("click", search);
$("show-catalog").addEventListener("click", showCatalog);
$("load-more").addEventListener("click", async () => { setBusy(true); try { await loadCatalog(true); } catch (e) { message(e.message, true); } finally { setBusy(false); } });
const panel = document.querySelector(".query-panel");
panel.addEventListener("dragover", event => { event.preventDefault(); $("dropzone").classList.add("dragging"); });
panel.addEventListener("dragleave", () => $("dropzone").classList.remove("dragging"));
panel.addEventListener("drop", event => { event.preventDefault(); $("dropzone").classList.remove("dragging"); selectFile(event.dataTransfer.files[0]); });
refreshStatus();
setInterval(refreshStatus, 3000);
