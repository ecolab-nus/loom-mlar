function jsonForScript(value) {
  return JSON.stringify(value)
    .replaceAll('&', '\\u0026')
    .replaceAll('<', '\\u003c')
    .replaceAll('>', '\\u003e')
    .replaceAll('\u2028', '\\u2028')
    .replaceAll('\u2029', '\\u2029');
}

function htmlEscape(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function scopePath(scopeId, scopesById) {
  if (!scopeId) return [];
  const result = [];
  const visited = new Set();
  let cursor = scopesById.get(scopeId);
  while (cursor && !visited.has(cursor.id)) {
    visited.add(cursor.id);
    result.unshift(cursor.name);
    cursor = cursor.parent_scope ? scopesById.get(cursor.parent_scope) : null;
  }
  return result;
}

export function buildGalleryCatalog(document, diagrams) {
  const scopesById = new Map(document.scopes.map((scope) => [scope.id, scope]));
  const sectionLabels = {
    overview: 'System and subsystems',
    memory_reads: 'Memory reads',
    memory_writes: 'Memory writes',
    resources: 'Resource dependencies',
    networks: 'Network connections',
    other: 'Other views',
  };
  const sectionOrder = Object.keys(sectionLabels);
  const entries = diagrams.map((diagram) => {
    const scope = diagram.primaryScopeId ? scopesById.get(diagram.primaryScopeId) : null;
    return {
      id: diagram.id,
      title: diagram.title,
      section: diagram.section ?? 'other',
      section_label: sectionLabels[diagram.section] ?? sectionLabels.other,
      scope_id: scope?.id ?? null,
      scope_name: scope?.name ?? null,
      scope_path: scopePath(scope?.id, scopesById),
      is_root_scope: scope?.id === document.architecture.root_scope,
      html: `html/${diagram.id}.html`,
      component_count: diagram.componentIds.length,
      relationship_count: diagram.relationshipIds.length,
    };
  });
  entries.sort((left, right) => {
    const sectionDifference = sectionOrder.indexOf(left.section) - sectionOrder.indexOf(right.section);
    if (sectionDifference !== 0) return sectionDifference;
    if (left.is_root_scope !== right.is_root_scope) return left.is_root_scope ? -1 : 1;
    return left.title.localeCompare(right.title, 'en');
  });
  const defaultEntry =
    entries.find((entry) => entry.section === 'overview' && entry.is_root_scope) ?? entries[0];
  return {
    schema_version: 'mlar.archify-gallery.v1',
    architecture: document.architecture,
    source_schema_version: document.schema_version,
    language: 'en',
    default_diagram_id: defaultEntry?.id ?? null,
    sections: sectionOrder.map((id) => ({ id, label: sectionLabels[id] })),
    scopes: document.scopes.map((scope) => ({
      id: scope.id,
      name: scope.name,
      parent_scope: scope.parent_scope ?? null,
      path: scopePath(scope.id, scopesById),
      replication_factor: scope.replication_factor,
    })),
    diagrams: entries,
  };
}

export function renderGalleryHtml(document, diagrams) {
  const catalog = buildGalleryCatalog(document, diagrams);
  const labels = {
    appName: 'MLAR architecture explorer',
    diagrams: 'diagrams',
    search: 'Search views',
    allScopes: 'All scopes',
    noResults: 'No matching views',
    previous: 'Previous',
    next: 'Next',
    open: 'Open diagram',
    theme: 'Toggle shell theme',
    components: 'components',
    relationships: 'relationships',
    loading: 'Loading Archify diagram…',
    menu: 'Views',
  };
  const title = `${document.architecture.name} · ${labels.appName}`;
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>${htmlEscape(title)}</title>
  <style>
    :root { color-scheme: dark; --bg:#07111f; --panel:#0d1a2b; --panel2:#112238; --line:#253a54; --text:#edf5ff; --muted:#91a8c3; --accent:#5eead4; --accent2:#60a5fa; --shadow:0 18px 55px rgba(0,0,0,.28); }
    :root[data-theme="light"] { color-scheme: light; --bg:#eef4fa; --panel:#fff; --panel2:#f7fafc; --line:#d6e0ea; --text:#142033; --muted:#62748a; --accent:#0f766e; --accent2:#2563eb; --shadow:0 18px 55px rgba(30,64,100,.14); }
    * { box-sizing:border-box; }
    html, body { margin:0; min-height:100%; background:var(--bg); color:var(--text); font:14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    button, input, select { font:inherit; }
    button, select { color:var(--text); }
    .app { min-height:100vh; display:grid; grid-template-rows:68px minmax(0,1fr); }
    header { display:flex; align-items:center; justify-content:space-between; gap:20px; padding:0 24px; border-bottom:1px solid var(--line); background:color-mix(in srgb,var(--panel) 94%,transparent); position:relative; z-index:3; }
    .brand { display:flex; align-items:center; min-width:0; gap:13px; }
    .mark { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-weight:800; color:#05201d; background:linear-gradient(135deg,var(--accent),#93c5fd); box-shadow:0 8px 24px color-mix(in srgb,var(--accent) 24%,transparent); }
    .brand-copy { min-width:0; }
    .brand-copy strong, .brand-copy span { display:block; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .brand-copy strong { font-size:15px; letter-spacing:.01em; }
    .brand-copy span { color:var(--muted); font-size:12px; margin-top:2px; }
    .header-meta { display:flex; align-items:center; gap:10px; }
    .badge { border:1px solid var(--line); border-radius:999px; padding:5px 10px; color:var(--muted); background:var(--panel2); white-space:nowrap; }
    .layout { min-height:0; display:grid; grid-template-columns:310px minmax(0,1fr); }
    aside { min-height:0; overflow:auto; padding:18px 14px 26px; border-right:1px solid var(--line); background:var(--panel); }
    .filters { position:sticky; top:-18px; z-index:2; padding:18px 0 12px; background:var(--panel); }
    .search { width:100%; color:var(--text); background:var(--panel2); border:1px solid var(--line); border-radius:10px; padding:10px 12px; outline:none; }
    .search:focus, select:focus { border-color:var(--accent2); box-shadow:0 0 0 3px color-mix(in srgb,var(--accent2) 16%,transparent); }
    select { width:100%; margin-top:9px; background:var(--panel2); border:1px solid var(--line); border-radius:10px; padding:9px 11px; outline:none; }
    .section { margin:15px 0 4px; }
    .section-title { display:flex; justify-content:space-between; align-items:center; padding:0 8px 7px; color:var(--muted); font-size:11px; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
    .nav-item { width:100%; display:block; text-align:left; color:var(--text); border:1px solid transparent; border-radius:10px; padding:10px 11px; margin:3px 0; background:transparent; cursor:pointer; }
    .nav-item:hover { background:var(--panel2); border-color:var(--line); }
    .nav-item.active { background:color-mix(in srgb,var(--accent2) 15%,var(--panel2)); border-color:color-mix(in srgb,var(--accent2) 55%,var(--line)); }
    .nav-item strong { display:block; font-size:13px; font-weight:650; }
    .nav-item small { display:block; color:var(--muted); margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .empty { color:var(--muted); text-align:center; padding:32px 12px; }
    main { min-width:0; min-height:0; padding:18px; display:grid; grid-template-rows:auto minmax(0,1fr); gap:12px; }
    .toolbar { display:flex; align-items:center; justify-content:space-between; gap:16px; min-width:0; }
    .view-title { min-width:0; }
    .view-title h1 { font-size:18px; line-height:1.25; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .view-title p { margin:4px 0 0; color:var(--muted); font-size:12px; }
    .actions { display:flex; gap:7px; flex:0 0 auto; }
    .action { border:1px solid var(--line); border-radius:9px; padding:8px 10px; background:var(--panel); cursor:pointer; }
    .action:hover { border-color:var(--accent2); }
    .action:disabled { opacity:.4; cursor:not-allowed; }
    .viewer { min-height:560px; position:relative; overflow:hidden; border:1px solid var(--line); border-radius:15px; background:var(--panel2); box-shadow:var(--shadow); }
    iframe { display:block; width:100%; height:100%; min-height:560px; border:0; background:#fff; }
    .loading { position:absolute; inset:0; display:grid; place-items:center; color:var(--muted); background:var(--panel2); z-index:1; transition:opacity .18s ease; }
    .loading.hidden { opacity:0; pointer-events:none; }
    @media (max-width:900px) { .app { grid-template-rows:auto auto; } header { padding:14px 16px; } .header-meta .badge { display:none; } .layout { grid-template-columns:1fr; grid-template-rows:280px minmax(700px,1fr); } aside { border-right:0; border-bottom:1px solid var(--line); } main { padding:12px; } .viewer, iframe { min-height:640px; } .action span { display:none; } }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand"><div class="mark">M</div><div class="brand-copy"><strong id="architectureName"></strong><span>${htmlEscape(labels.appName)}</span></div></div>
      <div class="header-meta"><span class="badge" id="diagramCount"></span><span class="badge">Archify 2.14</span><button class="action" id="themeButton" type="button" title="${htmlEscape(labels.theme)}">◐</button></div>
    </header>
    <div class="layout">
      <aside aria-label="${htmlEscape(labels.menu)}">
        <div class="filters"><input class="search" id="search" type="search" placeholder="${htmlEscape(labels.search)}"><select id="scopeFilter"><option value="">${htmlEscape(labels.allScopes)}</option></select></div>
        <nav id="navigation"></nav>
      </aside>
      <main>
        <div class="toolbar">
          <div class="view-title"><h1 id="viewTitle"></h1><p id="viewMeta"></p></div>
          <div class="actions">
            <button class="action" id="previousButton" type="button" title="${htmlEscape(labels.previous)}">← <span>${htmlEscape(labels.previous)}</span></button>
            <button class="action" id="nextButton" type="button" title="${htmlEscape(labels.next)}"><span>${htmlEscape(labels.next)}</span> →</button>
            <button class="action" id="openButton" type="button" title="${htmlEscape(labels.open)}">↗ <span>${htmlEscape(labels.open)}</span></button>
          </div>
        </div>
        <section class="viewer"><div class="loading" id="loading">${htmlEscape(labels.loading)}</div><iframe id="diagramFrame" title="Archify diagram"></iframe></section>
      </main>
    </div>
  </div>
  <script>
    const catalog = ${jsonForScript(catalog)};
    const labels = ${jsonForScript(labels)};
    const byId = new Map(catalog.diagrams.map((diagram) => [diagram.id, diagram]));
    const architectureName = document.getElementById('architectureName');
    const diagramCount = document.getElementById('diagramCount');
    const search = document.getElementById('search');
    const scopeFilter = document.getElementById('scopeFilter');
    const navigation = document.getElementById('navigation');
    const viewTitle = document.getElementById('viewTitle');
    const viewMeta = document.getElementById('viewMeta');
    const frame = document.getElementById('diagramFrame');
    const loading = document.getElementById('loading');
    const previousButton = document.getElementById('previousButton');
    const nextButton = document.getElementById('nextButton');
    const openButton = document.getElementById('openButton');
    const themeButton = document.getElementById('themeButton');
    let activeId = null;
    let visible = catalog.diagrams;

    architectureName.textContent = catalog.architecture.name;
    diagramCount.textContent = catalog.diagrams.length + ' ' + labels.diagrams;
    for (const scope of catalog.scopes) {
      const option = document.createElement('option');
      option.value = scope.id;
      option.textContent = scope.path.join(' / ');
      scopeFilter.append(option);
    }

    function renderNavigation() {
      const query = search.value.trim().toLocaleLowerCase();
      const selectedScope = scopeFilter.value;
      visible = catalog.diagrams.filter((diagram) => {
        const searchable = [diagram.title, diagram.section_label, diagram.scope_name, ...diagram.scope_path].filter(Boolean).join(' ').toLocaleLowerCase();
        return (!query || searchable.includes(query)) && (!selectedScope || diagram.scope_id === selectedScope);
      });
      navigation.replaceChildren();
      for (const section of catalog.sections) {
        const items = visible.filter((diagram) => diagram.section === section.id);
        if (items.length === 0) continue;
        const wrapper = document.createElement('section');
        wrapper.className = 'section';
        const heading = document.createElement('div');
        heading.className = 'section-title';
        const headingText = document.createElement('span');
        headingText.textContent = section.label;
        const count = document.createElement('span');
        count.textContent = String(items.length);
        heading.append(headingText, count);
        wrapper.append(heading);
        for (const diagram of items) {
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'nav-item' + (diagram.id === activeId ? ' active' : '');
          button.dataset.diagramId = diagram.id;
          const title = document.createElement('strong');
          title.textContent = diagram.title;
          const detail = document.createElement('small');
          detail.textContent = diagram.scope_path.length ? diagram.scope_path.join(' / ') : diagram.section_label;
          button.append(title, detail);
          button.addEventListener('click', () => selectDiagram(diagram.id, true));
          wrapper.append(button);
        }
        navigation.append(wrapper);
      }
      if (visible.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'empty';
        empty.textContent = labels.noResults;
        navigation.append(empty);
      }
      updateActions();
    }

    function selectDiagram(id, updateHash) {
      const diagram = byId.get(id) ?? byId.get(catalog.default_diagram_id);
      if (!diagram) return;
      activeId = diagram.id;
      viewTitle.textContent = diagram.title;
      const scope = diagram.scope_path.length ? diagram.scope_path.join(' / ') + ' · ' : '';
      viewMeta.textContent = scope + diagram.component_count + ' ' + labels.components + ' · ' + diagram.relationship_count + ' ' + labels.relationships;
      loading.classList.remove('hidden');
      frame.src = diagram.html;
      frame.title = diagram.title;
      if (updateHash && window.location.hash !== '#' + encodeURIComponent(diagram.id)) {
        history.pushState(null, '', '#' + encodeURIComponent(diagram.id));
      }
      renderNavigation();
      document.querySelector('[data-diagram-id="' + CSS.escape(diagram.id) + '"]')?.scrollIntoView({ block:'nearest' });
    }

    function updateActions() {
      const index = visible.findIndex((diagram) => diagram.id === activeId);
      previousButton.disabled = index <= 0;
      nextButton.disabled = index < 0 || index >= visible.length - 1;
      openButton.disabled = !byId.has(activeId);
    }

    function move(offset) {
      const index = visible.findIndex((diagram) => diagram.id === activeId);
      const target = visible[index + offset];
      if (target) selectDiagram(target.id, true);
    }

    search.addEventListener('input', renderNavigation);
    scopeFilter.addEventListener('change', renderNavigation);
    frame.addEventListener('load', () => loading.classList.add('hidden'));
    previousButton.addEventListener('click', () => move(-1));
    nextButton.addEventListener('click', () => move(1));
    openButton.addEventListener('click', () => { const diagram = byId.get(activeId); if (diagram) window.open(diagram.html, '_blank', 'noopener'); });
    themeButton.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('mlar-gallery-theme', next); } catch {}
    });
    window.addEventListener('hashchange', () => selectDiagram(decodeURIComponent(location.hash.slice(1)), false));
    window.addEventListener('keydown', (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.key === 'ArrowLeft') move(-1);
      if (event.key === 'ArrowRight') move(1);
    });
    let savedTheme = null;
    try { savedTheme = localStorage.getItem('mlar-gallery-theme'); } catch {}
    if (savedTheme) document.documentElement.dataset.theme = savedTheme;
    const requested = decodeURIComponent(location.hash.slice(1));
    selectDiagram(byId.has(requested) ? requested : catalog.default_diagram_id, false);
  </script>
</body>
</html>
`;
}
