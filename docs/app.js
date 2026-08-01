const state = { data: null, status: 'all', query: '' };

const labels = {
  confirmed: '確認済み',
  partially_confirmed: '一部確認',
  unknown: '未確認',
  hybrid: 'ハイブリッド',
  performance_linked: '業績連動',
  base_salary_linked: '基本給連動',
  discretionary: '総合判断'
};
const accents = ['var(--blue)', 'var(--lav)', 'var(--mint)', 'var(--rose)', 'var(--apricot)'];

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}
function display(value, fallback = '未確認') { return value === null || value === undefined || value === '' ? fallback : value; }
function annualMonths(bonus) {
  const annual = bonus?.annual_months;
  if (!annual) return '未確認';
  if (annual.kind === 'minimum') return `${annual.value}か月以上`;
  if (annual.kind === 'maximum') return `${annual.value}か月以下`;
  if (annual.kind === 'range') return `${annual.minimum}–${annual.maximum}か月`;
  return `${annual.value}か月`;
}
function recordSummary(record) {
  const bonus = record.bonus || {};
  return bonus.pool_basis || bonus.allocation_logic || record.notes?.[0] || '一次情報で制度の詳細を確認できていません。';
}
function card(record, index) {
  const sources = (record.sources || []).map(source => `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(source.title)} ↗</a>`).join('');
  return `<article class="card" style="--accent:${accents[index % accents.length]}">
    <div class="card-top"><div><div class="company-code">${escapeHtml(record.stock_code)}</div><h3>${escapeHtml(record.company_name_ja)}</h3></div><span class="status">${labels[record.evidence_status] || record.evidence_status}</span></div>
    <p class="scope">${escapeHtml(record.employee_scope)}</p>
    <div class="facts">
      <div class="fact"><span>制度分類</span><strong>${escapeHtml(labels[record.classification] || '未確定')}</strong></div>
      <div class="fact"><span>支給回数</span><strong>${record.bonus?.frequency_per_year ? `${record.bonus.frequency_per_year}回 / 年` : '未確認'}</strong></div>
      <div class="fact"><span>年換算月数</span><strong>${escapeHtml(annualMonths(record.bonus))}</strong></div>
      <div class="fact"><span>基準日</span><strong>${escapeHtml(record.as_of)}</strong></div>
    </div>
    <p class="summary-text">${escapeHtml(recordSummary(record))}</p>
    <div class="source-list">${sources}</div>
  </article>`;
}
function render() {
  if (!state.data) return;
  const query = state.query.trim().toLowerCase();
  const records = state.data.records.filter(record => {
    const statusMatch = state.status === 'all' || record.evidence_status === state.status;
    const queryMatch = !query || `${record.company_name_ja} ${record.stock_code} ${record.classification || ''}`.toLowerCase().includes(query);
    return statusMatch && queryMatch;
  });
  document.querySelector('#cards').innerHTML = records.map(card).join('');
  document.querySelector('#empty').hidden = records.length !== 0;
}
function setMetrics(data) {
  document.querySelector('#as-of').textContent = `基準日 ${data.as_of}`;
  document.querySelector('#metric-records').textContent = data.summary.record_count;
  document.querySelector('#metric-confirmed').textContent = data.summary.confirmed_or_partial_count;
  document.querySelector('#metric-average').textContent = data.summary.explicit_point_months_average === null ? '—' : `${data.summary.explicit_point_months_average}月`;
}
async function init() {
  try {
    const response = await fetch('./data/bonus.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    setMetrics(state.data);
    render();
  } catch (error) {
    document.querySelector('#cards').innerHTML = `<p class="empty">データを読み込めませんでした: ${escapeHtml(error.message)}</p>`;
  }
}
document.querySelector('#search').addEventListener('input', event => { state.query = event.target.value; render(); });
document.querySelectorAll('.filter').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.filter').forEach(item => item.classList.remove('active'));
  button.classList.add('active'); state.status = button.dataset.status; render();
}));
init();
