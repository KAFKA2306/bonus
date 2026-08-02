const state = { data: null, stage: 'all', query: '', sortKey: 'company', sortDirection: 'asc' };

const stageLabels = {
  evidence_found: '一次証拠あり',
  source_reviewed: '資料確認済み',
  queued: '未着手'
};
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
const stageOrder = { evidence_found: 0, source_reviewed: 1, queued: 2 };

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'})[char]);
}
function normalize(value = '') { return String(value).normalize('NFKC').toLocaleLowerCase('ja'); }
function list(items = []) { return items.length ? items.map(item => `<li>${escapeHtml(item)}</li>`).join('') : '<li>なし</li>'; }
function channelName(id) {
  return state.data?.source_registry.find(item => item.id === id)?.name_ja || id || '必須チャネル確認済み';
}
function verifiedFacts(record) {
  const facts = [];
  if (record.classification) facts.push(`方式: ${classificationLabels[record.classification] || record.classification}`);
  if (record.bonus?.frequency_per_year != null) facts.push(`支給回数: 年${record.bonus.frequency_per_year}回`);
  const annual = record.bonus?.annual_months;
  if (annual) {
    const label = annual.kind === 'range' ? `${annual.minimum}–${annual.maximum}か月` : `${annual.value}か月${annual.kind === 'minimum' ? '以上' : annual.kind === 'maximum' ? '以下' : ''}`;
    facts.push(`年換算月数: ${label}`);
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
function details(record) {
  const reviewed = record.survey.reviewed_channel_ids.map(channelName);
  return `<details class="row-details"><summary>調査内容</summary><div class="detail-panel">
    <section><h3>個社一次資料</h3>${sourceLinks(record)}</section>
    <section><h3>確認済みチャネル</h3><ul>${list(reviewed)}</ul></section>
    <section><h3>未解決の問い</h3><ul>${list(record.survey.open_questions)}</ul></section>
    <section><h3>対象範囲・注記</h3><p>${escapeHtml(record.employee_scope)}</p><ul>${list(record.notes || [])}</ul></section>
  </div></details>`;
}
function companyRow(record) {
  const survey = record.survey;
  const facts = verifiedFacts(record);
  return `<tr>
    <th scope="row" class="company-cell"><strong>${escapeHtml(record.company_name_ja)}</strong><span>${escapeHtml(record.stock_code)}</span></th>
    <td><span class="status status-${escapeHtml(survey.stage)}">${escapeHtml(stageLabels[survey.stage])}</span></td>
    <td><strong>${survey.reviewed_required_count} / ${survey.required_channel_count}</strong><small>${Math.round(survey.coverage_ratio * 100)}%</small></td>
    <td><strong>${escapeHtml(survey.next_channel_name_ja || '必須チャネル確認済み')}</strong></td>
    <td>${facts.length ? `<ul class="compact-list">${list(facts)}</ul>` : '<span class="muted">推定せず未確認</span>'}</td>
    <td>${details(record)}</td>
  </tr>`;
}
function searchText(record) {
  return normalize([
    record.company_name_ja, record.stock_code, stageLabels[record.survey.stage],
    record.survey.next_channel_name_ja, record.employee_scope,
    ...record.survey.open_questions, ...verifiedFacts(record),
    ...(record.sources || []).flatMap(source => [source.title, source.page_or_section])
  ].filter(Boolean).join(' '));
}
function sortValue(record, key) {
  if (key === 'company') return `${record.company_name_ja}-${record.stock_code}`;
  if (key === 'stage') return stageOrder[record.survey.stage] ?? 9;
  if (key === 'coverage') return record.survey.coverage_ratio;
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
    const stageMatch = state.stage === 'all' || record.survey.stage === state.stage;
    return stageMatch && (!query || searchText(record).includes(query));
  }));
  document.querySelector('#company-body').innerHTML = records.map(companyRow).join('');
  document.querySelector('#result-count').textContent = `${records.length}社 / 全${state.data.records.length}社`;
  document.querySelector('#empty').hidden = records.length !== 0;
  updateSortHeaders();
}
function setMetrics(data) {
  document.querySelector('#as-of').textContent = `基準日 ${data.as_of}`;
  document.querySelector('#metric-channels').textContent = data.summary.source_channel_count;
  document.querySelector('#metric-primary').textContent = data.summary.primary_channel_count;
  document.querySelector('#metric-verified').textContent = data.summary.verified_record_count;
  document.querySelector('#metric-coverage').textContent = `${Math.round(data.summary.research_coverage_ratio * 100)}%`;
}
async function init() {
  try {
    const response = await fetch('./data/bonus.json', {cache:'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    document.querySelector('#source-body').innerHTML = state.data.source_registry.map(sourceRow).join('');
    setMetrics(state.data);
    renderCompanies();
  } catch (error) {
    document.querySelector('#company-body').innerHTML = `<tr><td colspan="6" class="load-error">データを読み込めませんでした: ${escapeHtml(error.message)}</td></tr>`;
    document.querySelector('#result-count').textContent = '読み込み失敗';
  }
}
document.querySelector('#search').addEventListener('input', event => { state.query = event.target.value; renderCompanies(); });
document.querySelectorAll('.filter').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.filter').forEach(item => { const active = item === button; item.classList.toggle('active',active); item.setAttribute('aria-pressed',String(active)); });
  state.stage = button.dataset.stage; renderCompanies();
}));
document.querySelectorAll('.sort-button').forEach(button => button.addEventListener('click', () => {
  const key = button.dataset.sort;
  if (state.sortKey === key) state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
  else { state.sortKey = key; state.sortDirection = key === 'coverage' ? 'desc' : 'asc'; }
  renderCompanies();
}));
init();
