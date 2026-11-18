const $ = (sel) => document.querySelector(sel);
const THUMB = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>';
const WINDOW = 10;
const SERIES = [['model', 'Model'], ['match', 'Prompt match only']];
// One key per control, by tile. The left hand covers the left column of tiles, the right hand the right.
const STAR_KEYS = ['qwert', 'yuiop', 'asdfg', 'hjkl;'];
const THUMB_KEYS = ['zx', 'nm', 'cv', ',.']; // down, up

let vocab = {};
let keymap = {}; // key -> the control it presses, rebuilt every round
let mode = {}; // {} = auto-rotate, {axis, value} = pinned, {text} = custom
let round = null;
let feedback = null;
let busy = false;

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `${res.status} ${res.statusText}`);
  return res.json();
}

function el(tag, attrs = {}, html = '') {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  node.innerHTML = html;
  return node;
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// -- conditions --------------------------------------------------------------

function fillValues(axis, selected) {
  $('#value').innerHTML = (vocab[axis] || []).map((v) => `<option${v === selected ? ' selected' : ''}>${esc(v)}</option>`).join('');
}

function renderCondition() {
  const c = round.condition;
  const s = round.state;
  $('#cond-axis').textContent = c.axis === 'custom' ? 'your own' : c.axis;
  $('#cond-value').textContent = c.value;
  paintCondition();
  document.querySelectorAll('.for-cond').forEach((n) => { n.textContent = `for ${c.value}`; });
  // Keep the menus in step with whatever is on screen.
  if (c.axis !== 'custom') {
    $('#axis').value = c.axis;
    fillValues(c.axis, c.value);
    $('#text').value = '';
  }
}

// With the condition switched off for this round, it is struck out and the feedback counts as plain taste.
function paintCondition() {
  const c = round.condition;
  const s = round.state;
  const how = c.labeled ? 'by label' : 'matched by CLIP';
  $('#cond-form').classList.toggle('free', feedback.free);
  $('#free').setAttribute('aria-pressed', String(feedback.free));
  $('#cond-meta').textContent = `Round ${s.counts.rounds + s.counts.skipped + 1} · ` + (feedback.free
    ? 'condition off · just pick what you like best'
    : `${c.auto ? 'auto-rotating' : 'pinned'} · ${s.members} items fit (${how})`);
}

async function pin(next) {
  mode = next;
  document.activeElement.blur(); // hand the keyboard back to the round
  await loadRound();
}

// -- the round ---------------------------------------------------------------

async function loadRound() {
  const query = new URLSearchParams(mode).toString();
  round = await api(`/api/round${query ? `?${query}` : ''}`);
  feedback = { pick: null, thumbs: {}, ratings: {}, free: false };
  renderCondition();
  const grid = $('#grid');
  grid.replaceChildren();
  keymap = {};
  // Every control is a button that shows its key; the key presses the button.
  const control = (key, attrs, html) => {
    const b = el('button', { type: 'button', ...attrs }, html);
    if (key) {
      b.setAttribute('aria-keyshortcuts', key);
      keymap[key] = b;
    }
    return b;
  };
  const hint = (key) => (key ? `<span class="key">${esc(key)}</span>` : '');
  round.items.forEach((item, i) => {
    const tile = el('figure', { class: 'tile' });
    const media = item.kind === 'video'
      ? `<video src="${item.url}" poster="${item.poster}" muted loop autoplay playsinline></video>`
      : `<img src="${item.url}" alt="">`;
    const pic = control(String(i + 1), { class: 'pic', 'aria-pressed': 'false', 'aria-label': `Item ${i + 1}: mark as favorite` },
      `${media}<span class="num">${i + 1}</span><span class="badge">Favorite</span>`);
    pic.onclick = () => { feedback.pick = feedback.pick === item.id ? null : item.id; paint(); };

    const thumbs = el('div', { class: 'thumbs' });
    [[-1, 'down'], [1, 'up']].forEach(([value, name], t) => {
      const key = THUMB_KEYS[i]?.[t];
      const b = control(key, { class: name, 'aria-pressed': 'false', 'aria-label': `Item ${i + 1}: thumbs ${name}` },
        `<span class="cap">${THUMB}</span>${hint(key)}`);
      b.onclick = () => {
        if (feedback.thumbs[item.id] === value) delete feedback.thumbs[item.id];
        else feedback.thumbs[item.id] = value;
        paint();
      };
      thumbs.append(b);
    });

    const stars = el('div', { class: 'stars' });
    for (let s = 1; s <= 5; s++) {
      const key = STAR_KEYS[i]?.[s - 1];
      const b = control(key, { 'aria-label': `Item ${i + 1}: ${s} star${s > 1 ? 's' : ''}` }, `<span class="glyph">★</span>${hint(key)}`);
      b.onclick = () => {
        if (feedback.ratings[item.id] === s) delete feedback.ratings[item.id];
        else feedback.ratings[item.id] = s;
        paint();
      };
      stars.append(b);
    }

    const controls = el('figcaption', { class: 'controls' });
    controls.append(thumbs, stars);
    tile.append(pic, controls);
    grid.append(tile);
  });
  paint();
  renderState(round.state);
}

function paint() {
  [...$('#grid').children].forEach((tile, i) => {
    const id = round.items[i].id;
    tile.querySelector('.pic').setAttribute('aria-pressed', String(feedback.pick === id));
    tile.querySelector('.down').setAttribute('aria-pressed', String(feedback.thumbs[id] === -1));
    tile.querySelector('.up').setAttribute('aria-pressed', String(feedback.thumbs[id] === 1));
    tile.querySelectorAll('.stars button').forEach((b, s) => b.classList.toggle('on', s < (feedback.ratings[id] || 0)));
  });
}

// Enter: sends whatever feedback was given, or skips the round if there was none.
async function submit() {
  if (busy || !round) return;
  busy = true;
  try {
    const result = await api('/api/feedback', { round_id: round.round_id, ...feedback });
    $('#status').textContent = result.model_hit === null ? ''
      : result.model_hit ? 'The model predicted that favorite.' : 'The model had guessed a different one.';
    await loadRound();
  } catch (err) {
    $('#status').textContent = `Could not submit: ${err.message}. Loading a fresh round.`;
    await loadRound().catch(() => {});
  } finally {
    busy = false;
  }
}

// -- side panel --------------------------------------------------------------

function renderState(state) {
  const strip = (items) => items.map((it) => `<img src="${it.poster}" alt="" loading="lazy">`).join('');
  $('#favorites').innerHTML = strip(state.favorites);
  $('#fav-note').textContent = state.fitted
    ? 'Top scores among items you have not seen under this condition.'
    : 'No feedback yet, so this is prompt match alone.';
  $('#least').innerHTML = state.least.length ? strip(state.least) : '<p class="empty">Nothing yet. Give some feedback first.</p>';

  const axes = Object.entries(state.by_condition);
  $('#by-condition').innerHTML = axes.length
    ? axes.map(([axis, rows]) => `<dt>${esc(axis)}</dt><dd>${rows.slice(0, 4).map((r) => `${esc(r.value)} <b>${r.stars.toFixed(1)}</b>`).join(' · ')}</dd>`).join('')
    : '<p class="empty">Star some items to fill this in.</p>';

  const names = (list) => list.map((c) => esc(c.value)).join(', ');
  $('#lean').innerHTML = state.fitted
    ? `<dt>Leans toward</dt><dd>${names(state.lean.likes)}</dd><dt>Leans away from</dt><dd>${names(state.lean.dislikes)}</dd>`
    : '<p class="empty">Nothing yet.</p>';

  const c = state.counts;
  $('#counts').innerHTML = [['Rounds', c.rounds], ['Conditions', c.conditions], ['Favorites', c.picks], ['Thumbs', c.thumbs], ['Stars', c.ratings]]
    .map(([label, n]) => `<div><dt>${label}</dt><dd>${n}</dd></div>`).join('');

  renderChart(state.accuracy, state.chance);
}

function renderChart(accuracy, chance) {
  const root = $('#chart');
  const n = accuracy.model.length;
  if (n < 2) {
    root.innerHTML = '<p class="empty">Pick a favorite in a few rounds to see whether the model predicts them better than prompt matching does.</p>';
    return;
  }
  // Trailing mean over up to WINDOW rounds, per series.
  const rolling = Object.fromEntries(SERIES.map(([key]) => [key, accuracy[key].map((_, i) => {
    const span = accuracy[key].slice(Math.max(0, i - WINDOW + 1), i + 1);
    return span.reduce((a, b) => a + b, 0) / span.length;
  })]));
  const W = 320, H = 132, left = 30, right = 12, top = 8, bottom = 20;
  const x = (i) => left + (i / (n - 1)) * (W - left - right);
  const y = (v) => top + (1 - v) * (H - top - bottom);
  const pct = (v) => `${Math.round(v * 100)}%`;
  const last = (key) => rolling[key].at(-1);

  root.innerHTML = `
    <div class="headline"><b>${pct(last('model'))}</b><span>of the last ${Math.min(WINDOW, n)} favorites predicted, against ${pct(last('match'))} for prompt match alone and ${pct(chance)} by chance</span></div>
    <div class="legend">${SERIES.map(([key, label]) => `<span><i class="${key}"></i>${label}</span>`).join('')}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Rolling guess accuracy over ${n} rounds: model ${pct(last('model'))}, prompt match alone ${pct(last('match'))}, chance ${pct(chance)}">
      ${[0, 0.5, 1].map((v) => `<line class="gridline" x1="${left}" x2="${W - right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left - 6}" y="${y(v) + 3}" text-anchor="end">${pct(v)}</text>`).join('')}
      <line class="chance" x1="${left}" x2="${W - right}" y1="${y(chance)}" y2="${y(chance)}"/>
      <text x="${left - 6}" y="${y(chance) + 3}" text-anchor="end">chance</text>
      <text x="${left}" y="${H - 4}">round 1</text>
      <text x="${W - right}" y="${H - 4}" text-anchor="end">round ${n}</text>
      <line class="cross" y1="${top}" y2="${H - bottom}" visibility="hidden"/>
      ${[...SERIES].reverse().map(([key]) => `<polyline class="line ${key}" points="${rolling[key].map((v, i) => `${x(i)},${y(v)}`).join(' ')}"/><circle class="dot ${key}" r="4" cx="${x(n - 1)}" cy="${y(last(key))}"/>`).join('')}
      <rect x="${left}" y="${top}" width="${W - left - right}" height="${H - top - bottom}" fill="transparent"/>
    </svg>
    <div class="tip" hidden></div>`;

  const svg = root.querySelector('svg');
  const cross = root.querySelector('.cross');
  const tip = root.querySelector('.tip');
  const place = (i) => SERIES.forEach(([key]) => {
    const dot = root.querySelector(`.dot.${key}`);
    dot.setAttribute('cx', x(i));
    dot.setAttribute('cy', y(rolling[key][i]));
  });
  svg.onpointermove = (ev) => {
    const box = svg.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - left) / (W - left - right)) * (n - 1))));
    place(i);
    cross.setAttribute('x1', x(i));
    cross.setAttribute('x2', x(i));
    cross.setAttribute('visibility', 'visible');
    tip.hidden = false;
    // Sits over the x-axis labels, which it supersedes while hovering.
    tip.style.top = `${box.bottom - root.getBoundingClientRect().top - 20}px`;
    tip.textContent = `Round ${i + 1} · model ${pct(rolling.model[i])} · prompt match ${pct(rolling.match[i])}`;
    const tipX = (x(i) / W) * box.width;
    tip.style.left = `${Math.max(0, Math.min(box.width - tip.offsetWidth, tipX - tip.offsetWidth / 2))}px`;
  };
  svg.onpointerleave = () => {
    place(n - 1);
    cross.setAttribute('visibility', 'hidden');
    tip.hidden = true;
  };
}

// -- wiring ------------------------------------------------------------------

// Two things stay out of the way until asked for, and are remembered: the key hints and the side panel.
function setting(name, button) {
  const set = (on) => {
    document.body.classList.toggle(name, on);
    button.setAttribute('aria-pressed', String(on));
    try { localStorage.setItem(name, on ? '1' : '0'); } catch { /* private window */ }
  };
  let on = false;
  try { on = localStorage.getItem(name) === '1'; } catch { /* private window */ }
  set(on);
  button.onclick = () => set(!document.body.classList.contains(name));
  return set;
}
const showKeys = setting('show-keys', $('#keys'));
const showPanel = setting('show-panel', $('#panel-toggle'));

// A new axis starts on its first value; the value menu then narrows it.
$('#axis').onchange = () => pin({ axis: $('#axis').value, value: vocab[$('#axis').value][0] });
$('#value').onchange = () => pin({ axis: $('#axis').value, value: $('#value').value });
$('#cond-form').onsubmit = (ev) => {
  ev.preventDefault();
  const text = $('#text').value.trim();
  if (text) pin({ text });
  else $('#text').blur();
};
$('#free').onclick = () => { feedback.free = !feedback.free; paintCondition(); };
$('#reset').onclick = async () => {
  if (!confirm('Archive the feedback so far and start from scratch?')) return;
  await api('/api/reset', {});
  $('#status').textContent = 'Reset. Previous feedback was archived, not deleted.';
  await loadRound();
};
document.addEventListener('keydown', (ev) => {
  if (ev.metaKey || ev.ctrlKey || ev.altKey || !round) return;
  if (ev.target.matches('#cond-form select, #cond-form input, #reset')) {
    // Typing a condition, or using a menu: keys belong to the field until Escape.
    if (ev.key === 'Escape') ev.target.blur();
    return;
  }
  const key = ev.key.toLowerCase();
  if (keymap[key]) {
    ev.preventDefault();
    if (!ev.repeat) keymap[key].click();
  } else if (key === 'enter') {
    // Enter always continues, even with focus on an item that was just clicked.
    ev.preventDefault();
    submit();
  } else if (key === '/') {
    ev.preventDefault();
    $('#text').focus();
  } else if (key === '0') {
    pin({});
  } else if (key === '-') {
    $('#free').click();
  } else if (key === '?') {
    $('#keys').click();
  } else if (key === '\\') {
    $('#panel-toggle').click();
  } else if (key === 'escape') {
    showPanel(false);
  }
});

(async () => {
  try {
    vocab = await api('/api/conditions');
    $('#axis').innerHTML = Object.keys(vocab).filter((a) => a !== 'custom').map((a) => `<option>${esc(a)}</option>`).join('');
    await loadRound();
  } catch (err) {
    $('#status').textContent = `Could not reach the server: ${err.message}`;
  }
})();
