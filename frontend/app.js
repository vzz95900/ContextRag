// ── ContextAware RAG — Frontend App Logic (Document-Scoped Chat) ─────────────
const API = '';

// ── State ──────────────────────────────────────────────────
let messages = [];
let sessionId = null;
let documents = [];
let selectedDocId = null;
let selectedDocName = null;
let isLoading = false;

// ── Auth State ──────────────────────────────────────────────
let token = localStorage.getItem('token') || sessionStorage.getItem('token');
let username = localStorage.getItem('username') || sessionStorage.getItem('username');
let isGuest = localStorage.getItem('isGuest') === 'true' || sessionStorage.getItem('isGuest') === 'true';

// ── DOM Ready ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupInput();
  setupSidebarUpload();
  setupGatewayUpload();

  if (window.innerWidth < 768) {
    document.getElementById('sidebar').classList.add('collapsed');
  }

  // Check auth
  if (token) {
    showApp();
  } else {
    document.getElementById('authOverlay').classList.remove('hidden');
  }
});

// ── Auth Wrapper for Fetch ──────────────────────────────────
async function authFetch(url, options = {}) {
  options.headers = options.headers || {};
  if (token) {
    options.headers['Authorization'] = `Bearer ${token}`;
  }
  const resp = await fetch(url, options);
  if (resp.status === 401) {
    logout();
  }
  return resp;
}

// ── Auth UI & Handlers ──────────────────────────────────────
let currentAuthTab = 'login';

function switchAuthTab(tab) {
  currentAuthTab = tab;
  const tabLogin = document.getElementById('tabLogin');
  const tabRegister = document.getElementById('tabRegister');
  const submitBtn = document.getElementById('authSubmitBtn');
  const errorEl = document.getElementById('authError');
  
  errorEl.classList.add('hidden');
  
  if (tab === 'login') {
    tabLogin.className = 'flex-1 py-2 text-sm font-bold rounded-lg text-[#00e5ff] bg-[var(--glass-bg)] transition-all';
    tabRegister.className = 'flex-1 py-2 text-sm font-semibold rounded-lg text-[var(--text-muted)] hover:text-white transition-all';
    submitBtn.textContent = 'Sign In';
  } else {
    tabRegister.className = 'flex-1 py-2 text-sm font-bold rounded-lg text-[#b388ff] bg-[var(--glass-bg)] transition-all';
    tabLogin.className = 'flex-1 py-2 text-sm font-semibold rounded-lg text-[var(--text-muted)] hover:text-white transition-all';
    submitBtn.textContent = 'Create Account';
  }
}

async function handleAuthSubmit(e) {
  e.preventDefault();
  const usernameInput = document.getElementById('authUsername').value.trim();
  const passwordInput = document.getElementById('authPassword').value;
  const errorEl = document.getElementById('authError');
  
  errorEl.classList.add('hidden');
  
  const endpoint = currentAuthTab === 'login' ? '/api/auth/login' : '/api/auth/register';
  try {
    const resp = await fetch(`${API}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: usernameInput, password: passwordInput })
    });
    
    const data = await resp.json();
    if (resp.ok) {
      if (currentAuthTab === 'login') {
        token = data.token;
        username = data.username;
        isGuest = false;
        
        localStorage.setItem('token', token);
        localStorage.setItem('username', username);
        localStorage.setItem('isGuest', 'false');
        
        showApp();
      } else {
        switchAuthTab('login');
        errorEl.textContent = 'Registration successful! Please sign in.';
        errorEl.className = 'text-[#00e676] text-xs font-semibold mt-4 h-4 block';
        errorEl.classList.remove('hidden');
      }
    } else {
      errorEl.textContent = data.detail || 'Authentication failed';
      errorEl.className = 'text-red-400 text-xs font-semibold mt-4 h-4 block';
      errorEl.classList.remove('hidden');
    }
  } catch (err) {
    errorEl.textContent = 'Network error. Cannot reach backend.';
    errorEl.className = 'text-red-400 text-xs font-semibold mt-4 h-4 block';
    errorEl.classList.remove('hidden');
  }
}

async function handleGuestLogin() {
  const errorEl = document.getElementById('authError');
  errorEl.classList.add('hidden');
  
  try {
    const resp = await fetch(`${API}/api/auth/guest`, { method: 'POST' });
    const data = await resp.json();
    if (resp.ok) {
      token = data.token;
      username = data.username;
      isGuest = true;
      
      sessionStorage.setItem('token', token);
      sessionStorage.setItem('username', username);
      sessionStorage.setItem('isGuest', 'true');
      
      showApp();
    } else {
      errorEl.textContent = 'Failed to start guest session';
      errorEl.className = 'text-red-400 text-xs font-semibold mt-4 h-4 block';
      errorEl.classList.remove('hidden');
    }
  } catch (err) {
    errorEl.textContent = 'Network error. Cannot reach backend.';
    errorEl.className = 'text-red-400 text-xs font-semibold mt-4 h-4 block';
    errorEl.classList.remove('hidden');
  }
}

function showApp() {
  const authOverlay = document.getElementById('authOverlay');
  if (authOverlay) authOverlay.classList.add('hidden');
  const userDisp = document.getElementById('userDisplay');
  if (userDisp) userDisp.textContent = isGuest ? 'Guest' : username;
  loadDocuments();
}

function handleLogout() {
  if (token) {
    fetch(`${API}/api/auth/logout`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    }).catch(() => {});
  }
  logout();
}

function logout() {
  token = null;
  username = null;
  isGuest = false;
  
  localStorage.removeItem('token');
  localStorage.removeItem('username');
  localStorage.removeItem('isGuest');
  
  sessionStorage.removeItem('token');
  sessionStorage.removeItem('username');
  sessionStorage.removeItem('isGuest');
  
  const authOverlay = document.getElementById('authOverlay');
  if (authOverlay) authOverlay.classList.remove('hidden');
  exitDoc();
}

// Clean up guest sessions when closing the tab
window.addEventListener('beforeunload', () => {
  if (token && isGuest) {
    fetch(`${API}/api/auth/logout`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` },
      keepalive: true
    });
  }
});

// ═══════════════════════════════════════════════════════════
// DOCUMENT-FIRST FLOW
// ═══════════════════════════════════════════════════════════

/**
 * Activate a document — enters chat mode scoped to this doc.
 */
function activateDocument(docId, docName) {
  selectedDocId = docId;
  selectedDocName = docName;

  // Update header badge with exit logic inside
  document.getElementById('activeDocBadge').innerHTML = `
    <span class="text-xs font-bold text-[#b388ff] uppercase tracking-wider">Querying:</span>
    <span class="text-sm font-semibold text-white truncate max-w-[200px]">📄 ${escapeHtml(docName)}</span>
    <button onclick="exitDoc()" class="ml-2 w-5 h-5 flex items-center justify-center rounded bg-[rgba(255,82,82,0.1)] text-[#ff5252] hover:bg-[#ff5252] hover:text-white transition-colors" title="Change Document">✕</button>
  `;
  document.getElementById('optimizerBadge').style.display = '';

  // Hide gateway, set chat layout to welcome mode
  document.getElementById('docGateway').style.display = 'none';
  document.getElementById('chatInputArea').style.display = 'none';
  document.getElementById('chatHistorySection').style.display = '';

  // Show welcome with doc name
  const welcomeEl = document.getElementById('welcome');
  welcomeEl.style.display = 'flex';
  document.getElementById('welcomeDocName').textContent = `Querying: ${docName}`;

  // Clear messages for a fresh start
  messages = [];
  sessionId = null;
  document.getElementById('chatMessages').innerHTML = '';

  // Highlight active doc in sidebar
  highlightActiveDoc();

  // Load chat history for this doc
  loadChatHistory();
}

/**
 * Exit document mode — go back to the gateway.
 */
function exitDoc() {
  selectedDocId = null;
  selectedDocName = null;
  sessionId = null;
  messages = [];

  document.getElementById('activeDocBadge').innerHTML = `
    <span class="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">No document selected</span>
  `;
  document.getElementById('optimizerBadge').style.display = 'none';

  document.getElementById('docGateway').style.display = 'flex';
  document.getElementById('welcome').style.display = 'none';
  document.getElementById('chatInputArea').style.display = 'none';
  document.getElementById('chatHistorySection').style.display = 'none';
  document.getElementById('chatMessages').innerHTML = '';

  highlightActiveDoc();
  renderDocSearchList();
}

function highlightActiveDoc() {
  document.querySelectorAll('.doc-card').forEach(el => {
    el.classList.toggle('doc-active', el.dataset.docId === selectedDocId);
  });
}

// ═══════════════════════════════════════════════════════════
// CHAT
// ═══════════════════════════════════════════════════════════

function setupInput() {
  const bindTextarea = (inputId, btnId) => {
    const textarea = document.getElementById(inputId);
    const sendBtn = document.getElementById(btnId);
    if (!textarea || !sendBtn) return;

    textarea.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(inputId, btnId);
      }
    });

    textarea.addEventListener('input', () => {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
      sendBtn.disabled = !textarea.value.trim();
    });

    sendBtn.addEventListener('click', () => sendMessage(inputId, btnId));
  };

  bindTextarea('chatInput', 'sendBtn');
  bindTextarea('welcomeInput', 'welcomeSendBtn');
}

async function sendMessage(inputId = 'chatInput', btnId = 'sendBtn') {
  const textarea = document.getElementById(inputId);
  if(!textarea) return;
  const query = textarea.value.trim();
  if (!query || isLoading || !selectedDocId) return;

  hideWelcome();
  document.getElementById('chatInputArea').style.display = '';

  addMessage('user', query);
  textarea.value = '';
  textarea.style.height = 'auto';
  document.getElementById('sendBtn').disabled = true;
  const wBtn = document.getElementById('welcomeSendBtn');
  if(wBtn) wBtn.disabled = true;

  isLoading = true;
  const typingEl = showTypingIndicator();

  try {
    const payload = {
      query,
      session_id: sessionId,
      history: messages.slice(0, -1),
      filters: { doc_id: selectedDocId },
    };

    const resp = await authFetch(`${API}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    typingEl.remove();

    if (resp.ok) {
      const data = await resp.json();
      sessionId = data.session_id;
      addMessage('assistant', data.answer, data.sources, data.latency_ms, data.model, data.retrieval_mode);
      loadChatHistory();
    } else {
      // Show actual error detail from the API
      let errMsg = '❌ Error occurred while processing request.';
      try {
        const errData = await resp.json();
        if (errData.detail) errMsg = `❌ ${errData.detail}`;
      } catch (_) {}
      addMessage('assistant', errMsg);
    }
  } catch (e) {
    typingEl.remove();
    addMessage('assistant', '🔌 Cannot connect to API server. Ensure FastAPI is running.');
  } finally {
    isLoading = false;
  }
}

function addMessage(role, content, sources = null, latency = null, model = null, retrievalMode = null) {
  messages.push({ role, content, sources, latency, model });
  renderMessage(role, content, sources, latency, model, retrievalMode);
  scrollToBottom();
}

function renderMessage(role, content, sources, latency, model, retrievalMode) {
  const container = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatarText = role === 'user' ? 'U' : '🧠';
  let html = `
    <div class="message-avatar">${avatarText}</div>
    <div class="message-content">
      <div class="message-text">${formatMarkdown(content)}</div>
  `;

  if (sources && sources.length > 0) {
    const srcId = 'src-' + Date.now();
    html += `
      <button class="sources-toggle mt-3" onclick="toggleSources('${srcId}')">
        📎 ${sources.length} Cited Sources
      </button>
      <div class="sources-panel" id="${srcId}">
    `;
    sources.forEach(s => {
      html += `
        <div class="source-chip mt-2">
          <div class="source-meta">📄 ${s.document} · Page ${s.page} · Score: ${s.score.toFixed(3)}</div>
          <div class="text-[0.8rem] text-[var(--text-muted)] line-clamp-3">${escapeHtml(s.text.substring(0, 300))}...</div>
        </div>
      `;
    });
    html += '</div>';
  }

  if (latency != null) {
    html += `<div class="latency-badge">⚡ ${Math.round(latency)}ms generation</div>`;
  }

  html += '</div>';
  div.innerHTML = html;
  container.appendChild(div);
}

function toggleSources(id) {
  document.getElementById(id).classList.toggle('open');
}

function showTypingIndicator() {
  const container = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = `
    <div class="message-avatar">🧠</div>
    <div class="message-content" style="max-width:100px;">
      <div class="typing-indicator"><span></span><span></span><span></span></div>
    </div>
  `;
  container.appendChild(div);
  scrollToBottom();
  return div;
}

function hideWelcome() {
  const w = document.getElementById('welcome');
  if (w) w.style.display = 'none';
}

function scrollToBottom() {
  const c = document.getElementById('chatMessages');
  requestAnimationFrame(() => { c.scrollTop = c.scrollHeight; });
}

// ═══════════════════════════════════════════════════════════
// DOCUMENTS
// ═══════════════════════════════════════════════════════════

async function loadDocuments() {
  try {
    const resp = await authFetch(`${API}/api/documents`);
    if (!resp.ok) throw new Error('API error');
    const data = await resp.json();
    documents = data.documents || [];

    highlightActiveDoc();
    renderDocSearchList();
  } catch (e) {
    console.error('Failed to load documents:', e);
  }
}

function refreshGatewayDocList() {
  // Logic moved to renderDocSearchList for the new pop-up modal
}

// ═══════════════════════════════════════════════════════════
// DOCUMENT SEARCH MODAL (GATEWAY)
// ═══════════════════════════════════════════════════════════

function openDocSearchModal() {
  document.getElementById('docSearchModal').classList.remove('hidden');
  document.getElementById('modalSearchInput').value = '';
  document.getElementById('modalSearchInput').focus();
  renderDocSearchList();
}

function closeDocSearchModal(eventOrForce) {
  if (eventOrForce === true || eventOrForce?.target?.id === 'docSearchModal') {
    document.getElementById('docSearchModal').classList.add('hidden');
  }
}

function filterModalDocs(query) {
  renderDocSearchList(query.toLowerCase());
}

function renderDocSearchList(query = '') {
  const container = document.getElementById('modalDocList');
  if (!container) return;

  if (documents.length === 0) {
    container.innerHTML = `<div class="text-sm text-center py-8 text-[var(--text-muted)]">Knowledge base is empty. Upload a PDF first!</div>`;
    return;
  }

  const filtered = documents.filter(d => d.filename.toLowerCase().includes(query));

  if (filtered.length === 0) {
    container.innerHTML = `<div class="text-sm text-center py-8 text-[var(--text-muted)]">No documents match "${escapeHtml(query)}"</div>`;
    return;
  }

  container.innerHTML = filtered.map(d => `
    <div class="gateway-doc-item mb-2 group" onclick="closeDocSearchModal(true); activateDocument('${d.doc_id}', '${escapeAttr(d.filename)}')">
      <div class="flex items-center gap-4">
        <span class="text-2xl">📄</span>
        <div class="min-w-0 flex-1">
          <div class="text-sm font-bold text-[var(--text-main)] truncate">${escapeHtml(d.filename)}</div>
          <div class="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mt-0.5">${d.page_count} pages • ${d.chunk_count} chunks</div>
        </div>
      </div>
      <span class="text-xs px-3 py-1.5 rounded-lg bg-[rgba(0,229,255,0.12)] text-[#00e5ff] font-bold transition-opacity border border-[rgba(0,229,255,0.15)]">Launch</span>
    </div>
  `).join('');
}

async function deleteDoc(docId) {
  if (!confirm('Remove this document from the knowledge base?')) return;
  try {
    await authFetch(`${API}/api/documents/${docId}`, { method: 'DELETE' });
    if (selectedDocId === docId) exitDoc();
    loadDocuments();
  } catch (e) { /* ignore */ }
}

// ═══════════════════════════════════════════════════════════
// UPLOAD (sidebar + gateway)
// ═══════════════════════════════════════════════════════════

function setupSidebarUpload() {
  const zone = document.getElementById('uploadZone');
  const input = document.getElementById('fileInput');
  if (!zone || !input) return;

  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0], 'sidebar');
  });
  input.addEventListener('change', () => { if (input.files.length) uploadFile(input.files[0], 'sidebar'); });
}

function setupGatewayUpload() {
  const zone = document.getElementById('gatewayUploadZone');
  const input = document.getElementById('gatewayFileInput');
  if (!zone || !input) return;

  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0], 'gateway');
  });
  input.addEventListener('change', () => { if (input.files.length) uploadFile(input.files[0], 'gateway'); });
}

async function uploadFile(file, source = 'sidebar') {
  if (!file.name.toLowerCase().endsWith('.pdf')) { alert('Only PDFs Supported.'); return; }

  const statusId = source === 'gateway' ? 'gatewayUploadStatus' : 'uploadStatus';
  const status = document.getElementById(statusId);
  status.innerHTML = `<div class="mt-2 text-xs font-semibold text-[#00e5ff] animate-pulse text-center">Processing ${escapeHtml(file.name)}...</div>`;

  try {
    const form = new FormData();
    form.append('file', file);
    const resp = await authFetch(`${API}/api/upload`, { method: 'POST', body: form });
    if (resp.ok) {
      const data = await resp.json();
      status.innerHTML = `<div class="mt-2 text-xs font-bold text-[#00e676] text-center">✅ Indexed successfully</div>`;

      // Reload docs, then auto-activate the uploaded doc
      await loadDocuments();
      setTimeout(() => {
        status.innerHTML = '';
        activateDocument(data.doc_id, data.filename);
      }, 1000);
    } else {
      let errMsg = 'Upload failed';
      try { const errData = await resp.json(); if (errData.detail) errMsg = errData.detail; } catch (_) {}
      status.innerHTML = `<div class="mt-2 text-xs font-bold text-[#ff5252] text-center">❌ ${escapeHtml(errMsg)}</div>`;
    }
  } catch (e) {
    status.innerHTML = '<div class="mt-2 text-xs text-[#ff5252] text-center">Connection error</div>';
  }
}

// ═══════════════════════════════════════════════════════════
// CHAT HISTORY (scoped to active doc)
// ═══════════════════════════════════════════════════════════

async function loadChatHistory() {
  const list = document.getElementById('chatList');
  if (!selectedDocId) return [];

  try {
    const resp = await authFetch(`${API}/api/chats?doc_id=${encodeURIComponent(selectedDocId)}`);
    if (!resp.ok) return [];
    const data = await resp.json();
    const chats = data.sessions || [];

    if (chats.length) {
      list.innerHTML = chats.map(c => `
        <div class="chat-card p-2.5 flex justify-between items-center group" onclick="loadChat('${c.session_id}')">
          <div class="min-w-0 flex-1 truncate text-sm font-medium text-[var(--text-main)]">${escapeHtml(c.title)}</div>
          <button class="w-6 h-6 rounded bg-[rgba(255,82,82,0.1)] text-[#ff5252] hover:bg-[#ff5252] hover:text-white transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100 flex-shrink-0 ml-2" onclick="event.stopPropagation();deleteChat('${c.session_id}')">✕</button>
        </div>
      `).join('');
    } else {
      list.innerHTML = '<div class="text-xs text-center py-2 text-[var(--text-muted)]">No past chats</div>';
    }

    return chats;
  } catch (e) {
    return [];
  }
}

async function loadChat(id) {
  try {
    const resp = await authFetch(`${API}/api/chats/${id}`);
    if (!resp.ok) return;
    const data = await resp.json();
    sessionId = id;
    messages = data.messages || [];

    hideWelcome();
    document.getElementById('chatInputArea').style.display = '';
    const container = document.getElementById('chatMessages');
    container.innerHTML = '';
    messages.forEach(m => renderMessage(m.role, m.content, m.sources, m.latency, m.model));
    scrollToBottom();
  } catch (e) {}
}

async function deleteChat(id) {
  try {
    await authFetch(`${API}/api/chats/${id}`, { method: 'DELETE' });
    if (sessionId === id) newChat();
    loadChatHistory();
  } catch (e) {}
}

function newChat() {
  messages = [];
  sessionId = null;
  document.getElementById('chatMessages').innerHTML = '';
  document.getElementById('chatInputArea').style.display = 'none';
  const w = document.getElementById('welcome');
  if (w && selectedDocId) {
    w.style.display = 'flex';
    document.getElementById('welcomeDocName').textContent = `Querying: ${selectedDocName}`;
    const wInput = document.getElementById('welcomeInput');
    if (wInput) wInput.focus();
  }
}

function toggleSidebar() { document.getElementById('sidebar').classList.toggle('collapsed'); }

// ═══════════════════════════════════════════════════════════
// MARKDOWN HELPERS
// ═══════════════════════════════════════════════════════════
function formatMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.split('\n\n').map(p => `<p>${p}</p>`).join('');
  html = html.replace(/\n/g, '<br>');
  html = html.replace(/<br>- /g, '</p><ul><li>');
  html = html.replace(/<br>(\d+)\. /g, '</p><ol><li>');
  return html;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function escapeAttr(text) {
  return text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function askQuestion(q) {
  document.getElementById('chatInput').value = q;
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('chatInput').focus();
}
