const state = {
  data: null,
  status: 'all',
  query: '',
  sortKey: 'company',
  sortDirection: 'asc'
};

const labels = {
  confirmed: '確認済み',
  partially_confirmed: '一部確認',
  unknown: '未確認',
  hybrid: 'ハイブリッド',
  performance_linked: '業績連動',
  base_salary_linked: '基本給連動',
  discretionary: '総合判断',
  low: '低',
  medium: '中',
  high: '高'
};

const statusOrder = {
  confirmed: 0,
  partially_confirmed: 1,
  unknown: 2
};

function escapeHtml(value = '') {
  return String(value).replace(
    /[&<>'"]/g,
    char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]
  );
}

function normalize(value = '') {
  return String(value).normalize('NFKC').toLocaleLowerCase('ja');
}

function annualMonthsValue(bonus = {}) {
  const annual = bonus.annual_months;
  if (!annual) return null;
  if (annual.kind === 'range') return (Number(annual.minimum) + Number(annual.maximum)) / 2;
  return Number(annual.value);
}

function annualMonthsLabel(bonus = {}) {
  const annual = bonus.annual_months;
  if (!annual) return '—';
  if (annual.kind === 'minimum') return `${annual.value}か月以上`;
  if (annual.kind === 'maximum') return `${annual.value}か月以下`;
  if (annual.kind === 'range') return `${annual.minimum}–${annual.maximum}か月`;
  return `${annual.value}か月`;
}

function hypothesisRange(hypothesis = {}) {
  const estimate = hypothesis.estimate;
  if (!estimate) return '—';
  return `${estimate.minimum}–${estimate.maximum}か月`;
}

function effectiveClassification(record) {
  return record.classification || record.hypothesis?.classification_hypothesis || null;
}

function effectiveFrequency(record) {
  return record.bonus?.frequency_per_year ?? record.hypothesis?.frequency_per_year_hypothesis ?? null;
}

function confidenceScore(record) {
  const score = record.hypothesis?.confidence?.score;
  return Number.isFinite(score) ? Number(score) : null;
}

function basisTypeLabel(type) {
  return ({
    verified_fact: '確認事実',
    legacy_prior: '旧調査',
    sector_prior: '業界事前分布',
    calculation: '計算'
  })[type] || '根拠';
}

function searchText(record) {
  const hypothesis = record.hypothesis || {};
  const bonus = record.bonus || {};
  return normalize([
    record.company_name_ja,
    record.stock_code,
    record.employee_scope,
    record.evidence_status,
    labels[record.evidence_status],
    record.classification,
    labels[record.classification],
    hypothesis.classification_hypothesis,
    labels[hypothesis.classification_hypothesis],
    bonus.pool_basis,
    bonus.allocation_logic,
    bonus.base_salary_link,
    hypothesis.method,
    ...(record.notes || []),
    ...(hypothesis.basis || []).flatMap(item => [item.statement, item.reference]),
    ...(hypothesis.assumptions || []),
    ...(hypothesis.falsifiers || []),
    ...(record.sources || []).flatMap(source => [source.title, source.page_or_section])
  ].filter(Boolean).join(' '));
}

function listItems(items = []) {
  if (!items.length) return '<li>記録なし</li>';
  return items.map(item => `<li>${escapeHtml(item)}</li>`).join('');
}

function sourceLinks(record) {
  const sources = record.sources || [];
  if (!sources.length) return '<p class="muted">一次資料は未整備です。</p>';
  return `<ul class="source-list">${sources.map(source => `
    <li><a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">
      ${escapeHtml(source.title)} <span aria-hidden="true">↗</span>
    </a></li>
  `).join('')}</ul>`;
}

function evidenceSummary(record) {
  const bonus = record.bonus || {};
  return bonus.pool_basis || bonus.allocation_logic || record.notes?.[0] || '一次情報による制度詳細は未確認です。';
}

function detailsCell(record) {
  const hypothesis = record.hypothesis || {};
  const basis = hypothesis.basis || [];
  const basisHtml = basis.length
    ? `<ul class="basis-list">${basis.map(item => `
        <li><span>${escapeHtml(basisTypeLabel(item.type))}</span>${escapeHtml(item.statement)}</li>
      `).join('')}</ul>`
    : '<p class="muted">推定根拠は未登録です。</p>';

  return `<details class="row-details">
    <summary>根拠を見る</summary>
    <div class="detail-panel">
      <section>
        <h3>確認事実</h3>
        <p>${escapeHtml(evidenceSummary(record))}</p>
        <p class="scope-note">対象範囲: ${escapeHtml(record.employee_scope)}</p>
        ${sourceLinks(record)}
      </section>
      <section>
        <h3>推定根拠</h3>
        ${basisHtml}
      </section>
      <section>
        <h3>成立前提</h3>
        <ul>${listItems(hypothesis.assumptions)}</ul>
      </section>
      <section>
        <h3>反証条件</h3>
        <ul>${listItems(hypothesis.falsifiers)}</ul>
      </section>
    </div>
    <p class="estimate-warning">推定値は確認済み月数の集計に含めません。</p>
  </details>`;
}

function tableRow(record) {
  const hypothesis = record.hypothesis || {};
  const estimate = hypothesis.estimate || {};
  const statusLabel = labels[record.evidence_status] || record.evidence_status;
  const classification = effectiveClassification(record);
  const classificationLabel = labels[classification] || '未確定';
  const classificationIsEstimated = !record.classification && Boolean(classification);
  const verifiedFrequency = record.bonus?.frequency_per_year;
  const frequency = effectiveFrequency(record);
  const frequencyIsEstimated = verifiedFrequency == null && frequency != null;
  const confidence = hypothesis.confidence || {};
  const confidenceLabel = labels[confidence.level] || '—';
  const score = confidenceScore(record);

  return `<tr>
    <th scope="row" class="company-cell">
      <strong>${escapeHtml(record.company_name_ja)}</strong>
      <span>${escapeHtml(record.stock_code)}</span>
    </th>
    <td><span class="status status-${escapeHtml(record.evidence_status)}">${escapeHtml(statusLabel)}</span></td>
    <td>
      <span class="cell-main">${escapeHtml(classificationLabel)}</span>
      ${classificationIsEstimated ? '<small>推定</small>' : ''}
    </td>
    <td>
      <span class="cell-main">${escapeHtml(annualMonthsLabel(record.bonus))}</span>
      ${record.bonus?.annual_months ? '<small>一次情報</small>' : '<small>未確認</small>'}
    </td>
    <td class="estimate-cell">
      <strong>${escapeHtml(hypothesisRange(hypothesis))}</strong>
      <small>中心 ${estimate.central == null ? '—' : `${escapeHtml(estimate.central)}か月`}</small>
    </td>
    <td>
      <span class="cell-main">${frequency == null ? '—' : `${escapeHtml(frequency)}回 / 年`}</span>
      ${frequencyIsEstimated ? '<small>推定</small>' : '<small>確認値</small>'}
    </td>
    <td>
      <span class="confidence confidence-${escapeHtml(confidence.level || 'low')}">${escapeHtml(confidenceLabel)}</span>
      <small>${score == null ? '—' : `${Math.round(score * 100)}%`}</small>
    </td>
    <td class="details-cell">${detailsCell(record)}</td>
  </tr>`;
}

function statusMatch(record) {
  return state.status === 'all' || record.evidence_status === state.status;
}

function sortValue(record, key) {
  switch (key) {
    case 'company': return `${record.company_name_ja}-${record.stock_code}`;
    case 'status': return statusOrder[record.evidence_status] ?? 9;
    case 'classification': return labels[effectiveClassification(record)] || '';
    case 'verified': return annualMonthsValue(record.bonus) ?? -1;
    case 'central': return Number(record.hypothesis?.estimate?.central ?? -1);
    case 'frequency': return Number(effectiveFrequency(record) ?? -1);
    case 'confidence': return confidenceScore(record) ?? -1;
    default: return record.stock_code;
  }
}

function sortRecords(records) {
  const direction = state.sortDirection === 'asc' ? 1 : -1;
  return [...records].sort((a, b) => {
    const aValue = sortValue(a, state.sortKey);
    const bValue = sortValue(b, state.sortKey);
    if (typeof aValue === 'string' || typeof bValue === 'string') {
      return String(aValue).localeCompare(String(bValue), 'ja') * direction;
    }
    if (aValue === bValue) return a.stock_code.localeCompare(b.stock_code);
    return (aValue - bValue) * direction;
  });
}

function updateSortHeaders() {
  document.querySelectorAll('th[data-column]').forEach(header => {
    const active = header.dataset.column === state.sortKey;
    header.setAttribute(
      'aria-sort',
      active ? (state.sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'
    );
  });
}

function render() {
  if (!state.data) return;
  const query = normalize(state.query.trim());
  const filtered = state.data.records.filter(record => {
    const queryMatch = !query || searchText(record).includes(query);
    return statusMatch(record) && queryMatch;
  });
  const records = sortRecords(filtered);

  document.querySelector('#table-body').innerHTML = records.map(tableRow).join('');
  document.querySelector('#empty').hidden = records.length !== 0;
  document.querySelector('#result-count').textContent = `${records.length}社 / 全${state.data.records.length}社`;
  updateSortHeaders();
}

function setMetrics(data) {
  document.querySelector('#as-of').textContent = `基準日 ${data.as_of}`;
  document.querySelector('#metric-records').textContent = data.summary.record_count;
  document.querySelector('#metric-verified').textContent = data.summary.verified_record_count;
  document.querySelector('#metric-hypotheses').textContent = data.summary.hypothesis_count;
  document.querySelector('#metric-coverage').textContent = `${Math.round(data.universe.coverage_ratio * 100)}%`;
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

function selectSort(button) {
  const key = button.dataset.sort;
  if (state.sortKey === key) {
    state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
  } else {
    state.sortKey = key;
    state.sortDirection = ['central', 'verified', 'frequency', 'confidence'].includes(key) ? 'desc' : 'asc';
  }
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
    document.querySelector('#table-body').innerHTML = `
      <tr><td colspan="8" class="load-error">データを読み込めませんでした: ${escapeHtml(error.message)}</td></tr>
    `;
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

document.querySelectorAll('.sort-button').forEach(button => {
  button.addEventListener('click', () => selectSort(button));
});

init();
