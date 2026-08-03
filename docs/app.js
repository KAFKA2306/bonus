const state = { data: null, confidence: 'all', query: '', sortKey: 'months', sortDirection: 'desc' };

const tierLabels = {
  primary_company: '会社一次',
  primary_collective: '労使一次',
  official_disclosure: '公式開示',
  official_benchmark: '公的ベンチマーク',
  discovery_only: '探索専用'
};
const classificationLabels = {
  performance_linked: '業績連動',
  base_salary_linked: '基本給連動',
  discretionary: '総合判断',
  hybrid: 'ハイブリッド'
};
const releaseLabels = { first: '第1回・暫定', final: '最終' };
const confidenceLabels = { high: '高', medium: '中', low: '低' };
const upsideLabels = { very_high: '最大', high: '大', medium: '中', low: '小' };
const formulaLabels = {
  explicit: '算式明示',
  not_disclosed: '算式非開示',
  not_applicable: '利益連動式なし',
  unknown: '開示状況不明'
};
const amountMethodLabels = {
  matched_sector_sample: '同一業種標本による参考換算',
  official_company_base_projection: '会社公式の季別モデル額から年間投影',
  not_estimable_from_available_samples: '対応標本がなく算定不可'
};
const estimateStatusLabels = {
  verified_numeric: '一次資料の明示値',
  estimated_with_verified_bound: '一次資料の境界付き推定',
  estimated_with_verified_structure: '制度確認済み推定',
  estimated: 'モデル推定'
};

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'})[char]);
}
function normalize(value = '') { return String(value).normalize('NFKC').toLocaleLowerCase('ja'); }
function list(items = []) { return items.length ? items.map(item => `<li>${escapeHtml(item)}</li>`).join('') : '<li>なし</li>'; }
function number(value) { return Number(value).toLocaleString('ja-JP'); }
function yen(value) { return value == null ? '—' : `¥${number(Math.round(Number(value)))}`; }
function months(value) { return `${Number(value).toFixed(2)}か月`; }
function percent(value) { return value == null ? '—' : `${Math.round(Number(value) * 100)}%`; }
function valueLabel(value, unit) {
  if (unit === 'yen') return yen(value);
  if (unit === 'months') return months(value);
  return `${value}`;
}
function changeLabel(value, unit) {
  const numeric = Number(value);
  const sign = numeric > 0 ? '+' : '';
  if (unit === 'percent') return `${sign}${numeric.toFixed(2)}%`;
  if (unit === 'yen') return `${sign}${yen(numeric)}`;
  if (unit === 'months') return `${sign}${numeric.toFixed(2)}か月`;
  return `${sign}${numeric}`;
}
function sampleLabel(sample = {}) {
  const organizations = sample.organizations == null ? '—' : `${number(sample.organizations)}組織`;
  return sample.workers == null ? organizations : `${organizations} / ${number(sample.workers)}人`;
}
function aggregationLabel(value) {
  return value === 'worker_weighted_average' ? '労働者加重平均' : value === 'company_average' ? '企業平均' : value;
}
function periodLabel(value) {
  return String(value).replace('-summer', ' 夏季').replace('-yearend', ' 年末');
}
function benchmarkRow(item) {
  return `<tr>
    <th scope="row" class="benchmark-title">
      <strong>${escapeHtml(item.title)}</strong>
      <small>${escapeHtml(item.publisher)} / ${escapeHtml(periodLabel(item.period))} / 公表 ${escapeHtml(item.published_at)}</small>
    </th>
    <td><span class="release release-${escapeHtml(item.release_status)}">${escapeHtml(releaseLabels[item.release_status] || item.release_status)}</span><small>${escapeHtml(aggregationLabel(item.aggregation))}</small></td>
    <td class="numeric"><strong>${escapeHtml(valueLabel(item.value, item.unit))}</strong></td>
    <td class="numeric">${escapeHtml(valueLabel(item.previous_value, item.unit))}</td>
    <td class="numeric change-positive">${escapeHtml(changeLabel(item.change_value, item.change_unit))}</td>
    <td class="numeric">${escapeHtml(sampleLabel(item.sample))}</td>
    <td><strong>${escapeHtml(item.scope)}</strong><small>${escapeHtml(item.note)}</small></td>
    <td><a class="evidence-link" href="${escapeHtml(item.source_url)}" target="_blank" rel="noopener noreferrer">一次資料 ↗</a></td>
  </tr>`;
}
function verifiedFacts(record) {
  const facts = [];
  if (record.classification) facts.push(`方式: ${classificationLabels[record.classification] || record.classification}`);
  if (record.bonus?.frequency_per_year != null) facts.push(`支給回数: 年${record.bonus.frequency_per_year}回`);
  const annual = record.bonus?.annual_months;
  if (annual) {
    const label = annual.kind === 'range' ? `${annual.minimum}–${annual.maximum}か月` : `${annual.value}か月${annual.kind === 'minimum' ? '以上' : annual.kind === 'maximum' ? '以下' : ''}`;
    facts.push(`一次資料の年換算月数: ${label}`);
  }
  if (record.bonus?.pool_basis) facts.push(`原資: ${record.bonus.pool_basis}`);
  return facts;
}
function sourceLinks(record) {
  if (!record.sources?.length) return '<p class="muted">個社一次資料は未登録です。</p>';
  return `<ul class="source-list">${record.sources.map(source => `<li><a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.title)} ↗</a><small>${escapeHtml(source.page_or_section || source.type)}</small></li>`).join('')}</ul>`;
}
function sourceRow(source) {
  const link = source.url ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">開く ↗</a>` : '<span class="muted">企業・労組ごとに探索</span>';
  return `<tr>
    <td><strong>${source.priority}</strong></td>
    <th scope="row"><strong>${escapeHtml(source.name_ja)}</strong><small>${escapeHtml(source.use_when)}</small></th>
    <td><span class="tier tier-${escapeHtml(source.tier)}">${escapeHtml(tierLabels[source.tier] || source.tier)}</span></td>
    <td>${escapeHtml(source.verifies.join(' / '))}</td>
    <td>${escapeHtml(source.limitations)}</td>
    <td>${link}</td>
  </tr>`;
}
function referenceLink(reference) {
  const text = escapeHtml(reference);
  if (String(reference).startsWith('https://')) return `<a href="${text}" target="_blank" rel="noopener noreferrer">一次資料 ↗</a>`;
  return `<code>${text}</code>`;
}
function basisList(items = []) {
  return items.map(item => `<li><strong>${escapeHtml(item.statement)}</strong><small>${referenceLink(item.reference)}</small></li>`).join('');
}
function officialObservations(estimate) {
  const items = [];
  const annual = estimate.official_observations?.annual_months;
  const seasonal = estimate.official_observations?.latest_seasonal;
  if (annual) items.push(`<li><strong>${escapeHtml(annual.period)} 年間 ${months(annual.value)}</strong><small>${escapeHtml(annual.note)} ${referenceLink(annual.source_url)}</small></li>`);
  if (seasonal) items.push(`<li><strong>${escapeHtml(periodLabel(seasonal.period))} ${months(seasonal.months)} / ${yen(seasonal.amount_yen)}</strong><small>${escapeHtml(seasonal.note)} ${referenceLink(seasonal.source_url)}</small></li>`);
  return items.length ? `<ul class="basis-list">${items.join('')}</ul>` : '<p class="muted">会社公式の個別月数・モデル額は未登録です。</p>';
}
function estimateDetails(record) {
  const estimate = record.estimate;
  const anchors = estimate.anchors;
  const mechanism = estimate.mechanism;
  const facts = verifiedFacts(record);
  return `<details class="row-details"><summary>根拠と式</summary><div class="detail-panel estimate-panel">
    <section><h3>制度類型とアップサイド</h3><dl class="formula-grid">
      <dt>詳細分類</dt><dd>${escapeHtml(mechanism.label_ja)}</dd>
      <dt>アップサイド</dt><dd>${escapeHtml(upsideLabels[mechanism.upside_profile] || mechanism.upside_profile)} / ${percent(mechanism.upside_score)}</dd>
      <dt>算式開示</dt><dd>${escapeHtml(formulaLabels[mechanism.formula_disclosure] || mechanism.formula_disclosure)}</dd>
      <dt>金額算定</dt><dd>${escapeHtml(amountMethodLabels[estimate.amount_method] || estimate.amount_method)}</dd>
      <dt>金額可用性</dt><dd>${escapeHtml(estimate.amount_status)}</dd>
      <dt>金額標本</dt><dd>${escapeHtml(estimate.amount_conversion?.amount_sample_id || "—")}</dd>
      <dt>月数標本</dt><dd>${escapeHtml(estimate.amount_conversion?.months_sample_id || "—")}</dd>
    </dl>${mechanism.source_url ? `<p><a class="evidence-link" href="${escapeHtml(mechanism.source_url)}" target="_blank" rel="noopener noreferrer">制度の一次資料 ↗</a></p>` : ''}${mechanism.source_note ? `<p class="muted">${escapeHtml(mechanism.source_note)}</p>` : ''}</section>
    <section><h3>会社公式の数値</h3>${officialObservations(estimate)}</section>
    <section><h3>計算入力</h3><dl class="formula-grid">
      <dt>旧個社事前分布</dt><dd>${months(anchors.company_prior_months.minimum)}–${months(anchors.company_prior_months.maximum)}（中心 ${months(anchors.company_prior_months.central)}）</dd>
      <dt>業種実測</dt><dd>${months(anchors.sector_actual_months)} / ${yen(anchors.sector_actual_amount_yen)}</dd>
      <dt>会社重み</dt><dd>${percent(estimate.weights.company_prior)}</dd>
      <dt>業種重み</dt><dd>${percent(estimate.weights.sector_actual)}</dd>
      <dt>業種標本</dt><dd>${escapeHtml(sampleLabel(anchors.sector_sample_months))}</dd>
    </dl>${estimate.override_note ? `<p class="override-note">${escapeHtml(estimate.override_note)}</p>` : ''}<p class="muted">${escapeHtml(estimate.amount_caution)}</p></section>
    <section><h3>一次資料で確認済み</h3>${facts.length ? `<ul>${list(facts)}</ul>` : '<p class="muted">個社の数値・制度は未確認。モデル推定を表示。</p>'}${sourceLinks(record)}</section>
    <section><h3>推定根拠</h3><ul class="basis-list">${basisList(estimate.basis)}</ul></section>
    <section><h3>前提・反証条件</h3><h4>前提</h4><ul>${list(estimate.assumptions)}</ul><h4>反証条件</h4><ul>${list(estimate.falsifiers)}</ul></section>
    <section><h3>残る調査</h3><ul>${list(record.survey.open_questions)}</ul><p>${escapeHtml(record.employee_scope)}</p></section>
  </div></details>`;
}
function companyRow(record) {
  const estimate = record.estimate;
  const m = estimate.months;
  const amount = estimate.amount_yen;
  const amountCell = amount && amount.central != null
    ? `<strong>${yen(amount.central)}</strong><span>${yen(amount.minimum)}–${yen(amount.maximum)}</span><small>${escapeHtml(amountMethodLabels[estimate.amount_method] || '参考換算')}</small>`
    : `<strong>算定不可</strong><span>金額・月数の対応標本なし</span><small>${escapeHtml(amountMethodLabels[estimate.amount_method] || estimate.amount_method)}</small>`;
  const anchor = estimate.anchors;
  const broadMethod = classificationLabels[estimate.classification] || estimate.classification;
  const mechanism = estimate.mechanism;
  return `<tr>
    <th scope="row" class="company-cell"><strong>${escapeHtml(record.company_name_ja)}</strong><span>${escapeHtml(record.stock_code)}</span><small>${escapeHtml(estimateStatusLabels[estimate.status] || estimate.status)}</small></th>
    <td class="estimate-main numeric"><strong>${months(m.central)}</strong><span>${months(m.minimum)}–${months(m.maximum)}</span></td>
    <td class="numeric">${amountCell}</td>
    <td><strong>${escapeHtml(estimate.sector_name_ja)} ${months(anchor.sector_actual_months)}</strong><span>${yen(anchor.sector_actual_amount_yen)}</span><small>${escapeHtml(sampleLabel(anchor.sector_sample_months))}</small></td>
    <td><strong>${escapeHtml(mechanism.label_ja)}</strong><span>${escapeHtml(broadMethod)} / 年${escapeHtml(estimate.frequency_per_year)}回</span><small>アップサイド ${escapeHtml(upsideLabels[mechanism.upside_profile] || mechanism.upside_profile)} ${percent(mechanism.upside_score)}</small></td>
    <td><span class="confidence confidence-${escapeHtml(estimate.confidence.level)}">${escapeHtml(confidenceLabels[estimate.confidence.level])} ${percent(estimate.confidence.score)}</span><small>金額 ${estimate.confidence.amount_score == null ? '算定不可' : percent(estimate.confidence.amount_score)}</small></td>
    <td>${estimateDetails(record)}</td>
  </tr>`;
}
function searchText(record) {
  const estimate = record.estimate;
  return normalize([
    record.company_name_ja, record.stock_code, estimate.sector_name_ja,
    classificationLabels[estimate.classification], estimate.mechanism.label_ja,
    upsideLabels[estimate.mechanism.upside_profile], formulaLabels[estimate.mechanism.formula_disclosure],
    confidenceLabels[estimate.confidence.level], estimateStatusLabels[estimate.status],
    record.employee_scope, ...record.survey.open_questions, ...verifiedFacts(record),
    ...estimate.basis.map(item => item.statement),
    ...(record.sources || []).flatMap(source => [source.title, source.page_or_section])
  ].filter(Boolean).join(' '));
}
function sortValue(record, key) {
  if (key === 'company') return `${record.company_name_ja}-${record.stock_code}`;
  if (key === 'months') return record.estimate.months.central;
  if (key === 'amount') return record.estimate.amount_yen?.central ?? Number.NEGATIVE_INFINITY;
  if (key === 'sector') return `${record.estimate.sector_name_ja}-${record.company_name_ja}`;
  if (key === 'confidence') return record.estimate.confidence.score;
  return record.stock_code;
}
function sorted(records) {
  const direction = state.sortDirection === 'asc' ? 1 : -1;
  return [...records].sort((a,b) => {
    const av = sortValue(a,state.sortKey), bv = sortValue(b,state.sortKey);
    if (typeof av === 'string' || typeof bv === 'string') return String(av).localeCompare(String(bv),'ja') * direction;
    return (av - bv) * direction;
  });
}
function updateSortHeaders() {
  document.querySelectorAll('th[data-column]').forEach(header => {
    const active = header.dataset.column === state.sortKey;
    header.setAttribute('aria-sort', active ? (state.sortDirection === 'asc' ? 'ascending' : 'descending') : 'none');
  });
}
function renderCompanies() {
  if (!state.data) return;
  const query = normalize(state.query.trim());
  const records = sorted(state.data.records.filter(record => {
    const confidenceMatch = state.confidence === 'all' || record.estimate.confidence.level === state.confidence;
    return confidenceMatch && (!query || searchText(record).includes(query));
  }));
  document.querySelector('#company-body').innerHTML = records.map(companyRow).join('');
  document.querySelector('#result-count').textContent = `${records.length}社 / 全${state.data.records.length}社`;
  document.querySelector('#empty').hidden = records.length !== 0;
  updateSortHeaders();
}
function setMetrics(data) {
  document.querySelector('#as-of').textContent = `基準日 ${data.as_of}`;
  document.querySelector('#metric-quantified').textContent = `${data.summary.quantified_company_count} / ${data.summary.record_count}`;
  document.querySelector('#metric-median-months').textContent = months(data.summary.median_estimated_months);
  document.querySelector('#metric-median-amount').textContent = yen(data.summary.median_estimated_amount_yen);
  document.querySelector('#metric-confidence').textContent = percent(data.summary.average_estimate_confidence);
  document.querySelector('#metric-verified').textContent = data.summary.verified_record_count;
}
async function init() {
  try {
    const response = await fetch('./data/bonus.json', {cache:'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    document.querySelector('#benchmark-body').innerHTML = state.data.quantitative_benchmarks.map(benchmarkRow).join('');
    document.querySelector('#source-body').innerHTML = state.data.source_registry.map(sourceRow).join('');
    setMetrics(state.data);
    renderCompanies();
  } catch (error) {
    document.querySelector('#benchmark-body').innerHTML = `<tr><td colspan="8" class="load-error">データを読み込めませんでした: ${escapeHtml(error.message)}</td></tr>`;
    document.querySelector('#company-body').innerHTML = `<tr><td colspan="7" class="load-error">データを読み込めませんでした: ${escapeHtml(error.message)}</td></tr>`;
    document.querySelector('#result-count').textContent = '読み込み失敗';
  }
}
document.querySelector('#search').addEventListener('input', event => { state.query = event.target.value; renderCompanies(); });
document.querySelectorAll('.filter').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.filter').forEach(item => { const active = item === button; item.classList.toggle('active',active); item.setAttribute('aria-pressed',String(active)); });
  state.confidence = button.dataset.confidence; renderCompanies();
}));
document.querySelectorAll('.sort-button').forEach(button => button.addEventListener('click', () => {
  const key = button.dataset.sort;
  if (state.sortKey === key) state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
  else { state.sortKey = key; state.sortDirection = key === 'company' || key === 'sector' ? 'asc' : 'desc'; }
  renderCompanies();
}));
init();
