(() => {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const cleanToken = (value) => (value || "").replace(/[^A-Za-z0-9._-]/g, "").slice(0, 64);
  const partner = cleanToken(params.get("partner"));
  const campaign = cleanToken(params.get("campaign"));
  const requested = (params.get("companies") || "")
    .split(",")
    .map((value) => value.trim())
    .filter((value) => /^\d{4}$/.test(value));

  const eventPayload = (event, companyId = null) => ({
    source: "bonus-media-widget",
    schema_version: 1,
    event,
    partner: partner || null,
    campaign: campaign || null,
    company_id: companyId,
  });

  const emit = (event, companyId = null) => {
    window.parent.postMessage(eventPayload(event, companyId), "*");
  };

  const fmtMonths = (item) => {
    if (item.status === "unavailable" || item.months.central == null) return "利用不可";
    if (item.status === "verified") return `${item.months.central}か月（確認済み）`;
    return `${item.months.minimum ?? "?"}–${item.months.maximum ?? "?"}か月（推定）`;
  };

  const fmtAmount = (item) => {
    if (item.amount.status !== "available" || item.amount.central_yen == null) return "利用不可";
    return `${Number(item.amount.central_yen).toLocaleString("ja-JP")}円`;
  };

  const statusLabel = (status) => ({ verified: "確認済み", estimated: "モデル推定", unavailable: "利用不可" }[status] || "利用不可");

  const sourceList = (item) => {
    if (!item.sources.length) return '<span class="muted">公開可能な一次資料リンクなし</span>';
    return item.sources
      .map((source) => {
        const label = source.title || source.type || "出典";
        const href = source.url.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
        const text = label.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        return `<a class="source" href="${href}" target="_blank" rel="noopener noreferrer" data-source-company="${item.company_id}">${text}</a>`;
      })
      .join(" ");
  };

  const render = (payload) => {
    if (payload.schema_version !== "bonus.media-widget.v1" || !Array.isArray(payload.records)) {
      throw new Error("unsupported media widget contract");
    }
    const byId = new Map(payload.records.map((item) => [item.company_id, item]));
    const ids = requested.length ? [...new Set(requested)] : payload.records.slice(0, 2).map((item) => item.company_id);
    if (ids.length < 2 || ids.length > 5) throw new Error("companiesには2〜5社の証券コードを指定してください");
    const selected = ids.map((id) => byId.get(id));
    const missing = ids.filter((id, index) => !selected[index]);
    if (missing.length) throw new Error(`未登録の会社ID: ${missing.join(", ")}`);

    document.getElementById("as-of").textContent = `更新日 ${payload.as_of || "不明"} / contract ${payload.schema_version}`;
    document.getElementById("cards").innerHTML = selected.map((item) => `
      <article class="card">
        <div class="card-head">
          <h2>${item.company_name_ja.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")}</h2>
          <span class="badge badge-${item.status}">${statusLabel(item.status)}</span>
        </div>
        <dl>
          <dt>賞与月数</dt><dd>${fmtMonths(item)}</dd>
          <dt>金額</dt><dd>${fmtAmount(item)}</dd>
          <dt>信頼度</dt><dd>${item.confidence.level || "未設定"}${item.confidence.score == null ? "" : ` (${item.confidence.score})`}</dd>
          <dt>基準日</dt><dd>${item.as_of || "不明"}</dd>
        </dl>
        <div class="sources"><strong>一次資料</strong><br>${sourceList(item)}</div>
      </article>`).join("");

    document.querySelectorAll("[data-source-company]").forEach((link) => {
      link.addEventListener("click", () => emit("source_opened", link.dataset.sourceCompany || null));
    });
    emit("embed_loaded");
  };

  document.getElementById("full-link").addEventListener("click", () => emit("full_comparison_opened"));
  document.getElementById("inquiry-link").addEventListener("click", () => emit("business_inquiry_started"));

  fetch("../../data/media-widget-v1.json", { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`data fetch failed: ${response.status}`);
      return response.json();
    })
    .then(render)
    .catch((error) => {
      const box = document.getElementById("error");
      box.hidden = false;
      box.textContent = `表示できません: ${error.message}`;
      document.getElementById("as-of").textContent = "データ契約を確認できませんでした";
    });
})();
