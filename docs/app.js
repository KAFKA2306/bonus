const state = { data: null, status: 'all', query: '' };

const labels = {
  confirmed: '確認済み',
  partially_confirmed: '一部確認',
  unknown: '一次情報未確認',
  estimated: '仮説推定',
  hybrid: 'ハイブリッド',
  performance_linked: '業績連動',
  base_salary_linked: '基本給連動',
  discretionary: '総合判断',
  low: '低',
  medium: '中',
  high: '高'
};

const accents = [
  'var(--blue)',
  'var(--lav)',
  'var(--mint)',
  'var(--rose)',
  'var(--apricot)'
];

function escapeHtml(value = '') {
  return String(value).replace(
    /[&<>'"]/g,
    char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]
  );
}

function normalize(value = '') {
  return String(value).normalize('NFKC').toLocaleLowerCase('ja');
}

function annualMonths(bonus = {}) {
  const annual = bonus.annual_months;
  if (!annual) return '一次情報では未確認';
  if (annual.kind === 'minimum') return `${annual.value}か月以上`;
  if (annual.kind === 'maximum') return `${annual.value}か月以下`;
  if (annual.kind === 'range') return `${annual.minimum}–${annual.maximum}か月`;
  return `${annual.value}か月`;
}

function hypothesisRange(hypothesis = {}) {
  const estimate = hypothesis.estimate;
  if (!estimate) return '推定なし';
  return `${estimate.minimum}–${estimate.maximum}か月`;
}

function recordSummary(record) {
  const bonus = record.bonus || {};
  return (
    bonus.pool_basis ||
    bonus.allocation_logic ||
    record.notes?.[0] ||
    '一次情報で制度の詳細を確認できていません。'
  );
}

function hypothesisText(hypothesis = {}) {
  return [
    hypothesis.method,
    hypothesis.classification_hypothesis,
    labels[hypothesis.classification_hypothesis],
    hypothesis.confidence?.level,
    labels[hypothesis.confidence?.level],
    ...(hypothesis.basis || []).flatMap(item => [item.statement, item.reference]),
    ...(hypothesis.assumptions || []),
    ...(hypothesis.falsifiers || [])
  ];
}

function searchText(record) {
  const bonus = record.bonus || {};
  return normalize([
    record.company_name_ja,
    record.stock_code,
    record.employee_scope,
    record.classification,
    labels[record.classification],
    record.evidence_status,
    labels[record.evidence_status],
    bonus.pool_basis,
    bonus.allocation_logic,
    bonus.base_salary_link,
    ...hypothesisText(record.hypothesis),
    ...(record.notes || [])
  ].filter(Boolean).join(' '));
}

function listItems(items = []) {
  return items.map(item => `<li>${escapeHtml(item)}</li>`).join('');
}

function hypothesisPanel(hypothesis) {
  if (!hypothesis) return '';
  const estimate = hypothesis.estimate;
  const confidence = hypothesis.confidence || {};
  const confidenceLabel = labels[confidence.level] || confidence.level || '—';
  const score = Number.isFinite(confidence.score)
    ? `${Math.round(confidence.score * 100)}%`
    : '—';
  const classification = labels[hypothesis.classification_hypothesis] || '未確定';
  const basis = (hypothesis.basis || []).map(item => `
    <li><strong>${item.type === 'verified_fact' ? '確認事実' : '事前分布'}</strong>${escapeHtml(item.statement)}</li>
  `).join('');

  return `<section class="hypothesis" aria-label="仮説推定">
    <div class="hypothesis-head">
      <div>
        <span class="hypothesis-label">HYPOTHESIS</span>
        <strong>${escapeHtml(hypothesisRange(hypothesis))}</strong>
      </div>
      <div class="confidence confidence-${escapeHtml(confidence.level)}">
        確度 ${escapeHtml(confidenceLabel)} · ${escapeHtml(score)}
      </div>
    </div>
    <div class="hypothesis-grid">
      <div><span>中心値</span><b>${escapeHtml(estimate.central)}か月</b></div>
      <div><span>制度仮説</span><b>${escapeHtml(classification)}</b></div>
      <div><span>支給回数仮説</span><b>${escapeHtml(hypothesis.frequency_per_year_hypothesis)}回 / 年</b></div>
    </div>
    <ul class="basis-list">${basis}</ul>
    <details>
      <summary>前提と反証条件</summary>
      <div class="hypothesis-details">
        <div><h4>前提</h4><ul>${listItems(hypothesis.assumptions)}</ul></div>
        <div><h4>反証条件</h4><ul>${listItems(hypothesis.falsifiers)}</ul></div>
      </div>
    </details>
    <p class="estimate-warning">推定値です。確認済み月数の集計には含めません。</p>
  </section>`;
}

function card(record) {
  const accentIndex = Number.parseInt(record.stock_code, 10) % accents.length;
  const accent = accents[Number.isNaN(accentIndex) ? 0 : accentIndex];
  const statusLabel = labels[record.evidence_status] || record.evidence_status;
  const classificationLabel = labels[record.classification] || '一次情報では未確定';
  const frequency = record.bonus?.frequency_per_year;
  const sources = (record.sources || []).map(source => `
    <a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">
      ${escapeHtml(source.title)} <span aria-hidden="true">↗</span>
    </a>
  `).join('');
  const estimateBadge = record.hypothesis
    ? '<span class="status status-estimated">仮説あり</span>'
    : '';

  return `<article class="card" style="--accent:${accent}">
    <div class="card-top">
      <div>
        <div class="company-code">${escapeHtml(record.stock_code)}</div>
        <h3>${escapeHtml(record.company_name_ja)}</h3>
      </div>
      <div class="badges">
        <span class="status status-${escapeHtml(record.evidence_status)}">
          ${escapeHtml(statusLabel)}
        </span>
        ${estimateBadge}
      </div>
    </div>
    <p class="scope">${escapeHtml(record.employee_scope)}</p>
    <div class="fact-layer-label">確認事実</div>
    <div class="facts">
      <div class="fact"><span>制度分類</span><strong>${escapeHtml(classificationLabel)}</strong></div>
      <div class="fact"><span>支給回数</span><strong>${frequency == null ? '一次情報では未確認' : `${frequency}回 / 年`}</strong></div>
      <div class="fact"><span>年換算月数</span><strong>${escapeHtml(annualMonths(record.bonus))}</strong></div>
      <div class="fact"><span>基準日</span><strong>${escapeHtml(record.as_of)}</strong></div>
    </div>
    <p class="summary-text">${escapeHtml(recordSummary(record))}</p>
    <div class="source-list" aria-label="${escapeHtml(record.company_name_ja)}の一次資料">
      ${sources}
    </div>
    ${hypothesisPanel(record.hypothesis)}
  </article>`;
}

function statusMatch(record) {
  if (state.status === 'all') return true;
  if (state.status === 'estimated') return Boolean(record.hypothesis);
  return record.evidence_status === state.status;
}

function render() {
  if (!state.data) return;

  const query = normalize(state.query.trim());
  const records = state.data.records.filter(record => {
    const queryMatch = !query || searchText(record).includes(query);
    return statusMatch(record) && queryMatch;
  });

  document.querySelector('#cards').innerHTML = records.map(card).join('');
  document.querySelector('#empty').hidden = records.length !== 0;
  document.querySelector('#result-count').textContent =
    `${records.length}件 / 全${state.data.records.length}件`;
}

function setMetrics(data) {
  document.querySelector('#as-of').textContent = `基準日 ${data.as_of}`;
  document.querySelector('#metric-records').textContent = data.summary.record_count;
  document.querySelector('#metric-confirmed').textContent =
    data.summary.confirmed_or_partial_count;
  document.querySelector('#metric-hypotheses').textContent =
    data.summary.hypothesis_count;
  document.querySelector('#metric-universe').textContent =
    data.universe.tracked_companies;
}

function selectFilter(selected) {
  document.querySelectorAll('.filter').forEach(button => {
    const active = button === selected;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  state.status = selected.dataset.status;
  render();
}

async function init() {
  try {
    const response = await fetch('./data/bonus.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    setMetrics(state.data);
    render();
  } catch (error) {
    document.querySelector('#cards').innerHTML =
      `<p class="empty">データを読み込めませんでした: ${escapeHtml(error.message)}</p>`;
    document.querySelector('#result-count').textContent = '読み込み失敗';
  }
}

document.querySelector('#search').addEventListener('input', event => {
  state.query = event.target.value;
  render();
});

document.querySelectorAll('.filter').forEach(button => {
  button.addEventListener('click', () => selectFilter(button));
});

init();
