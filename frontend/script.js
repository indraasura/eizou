// Change this to your Render URL when deploying (e.g., "https://my-backend.onrender.com")
const API_URL = "https://nexus-backend-service.onrender.com";

let authToken = null;
let currentRole = null;

// Helper function to inject the JWT token into API calls
async function apiCall(endpoint, options = {}) {
    if (authToken) {
        options.headers = { ...options.headers, 'Authorization': `Bearer ${authToken}` };
    }
    const response = await fetch(`${API_URL}${endpoint}`, options);
    if (response.status === 401) logout(); // Auto-logout if token expires
    return response;
}

// --- AUTHENTICATION ---
async function login() {
    const emailInput = document.getElementById('email').value;
    const passInput = document.getElementById('password').value;
    const errorDiv = document.getElementById('login-error');

    const formData = new FormData();
    formData.append("email", emailInput);
    formData.append("password", passInput);

    try {
        errorDiv.innerText = "Authenticating...";
        const res = await fetch(`${API_URL}/login`, { method: 'POST', body: formData });

        if (res.ok) {
            const data = await res.json();
            authToken = data.token;
            currentRole = data.role;

            // Set UI details
            document.getElementById('user-badge').innerText = data.email;
            document.getElementById('view-login').classList.remove('active');
            document.getElementById('view-app').classList.add('active');

            // Show Admin Panel option if applicable
            if (currentRole === 'admin') {
                document.querySelector('.admin-only').style.display = 'flex';
            } else {
                document.querySelector('.admin-only').style.display = 'none';
                switchTab('chat'); // Force users to chat tab
            }

            errorDiv.innerText = "";
            loadProjects();
        } else {
            errorDiv.innerText = "Invalid credentials. Please try again.";
        }
    } catch (e) {
        errorDiv.innerText = "Failed to connect to server.";
    }
}

function logout() {
    authToken = null;
    currentRole = null;
    document.getElementById('view-app').classList.remove('active');
    document.getElementById('view-login').classList.add('active');
    document.getElementById('email').value = '';
    document.getElementById('password').value = '';
}

// --- UI NAVIGATION ---
function switchTab(tabId) {
    document.getElementById('tab-chat').style.display = tabId === 'chat' ? 'flex' : 'none';
    document.getElementById('tab-admin').style.display = tabId === 'admin' ? 'block' : 'none';

    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    event.currentTarget.classList.add('active');
}

// --- ADMIN CAPABILITIES ---
async function loadProjects() {
    const res = await apiCall('/projects');
    if (!res.ok) return;

    const data = await res.json();

    const populateDropdown = (id) => {
        const el = document.getElementById(id);
        if (el) {
            el.innerHTML = '';
            if (data.projects.length === 0) {
                el.innerHTML = '<option value="">No projects found</option>';
            } else {
                data.projects.forEach(p => el.innerHTML += `<option value="${p.id}">${p.name}</option>`);
            }
        }
    };

    populateDropdown('project-selector');
    populateDropdown('upload-project-id');
    populateDropdown('assign-project-id');
    clearChat();
}

async function createProject() {
    const name = document.getElementById('new-project-name').value;
    if (!name) return alert("Please enter a project name.");

    const fd = new FormData(); fd.append("name", name);
    await apiCall('/admin/projects', { method: 'POST', body: fd });

    document.getElementById('new-project-name').value = '';
    alert("Project Created Successfully.");
    loadProjects();
}

async function createUser() {
    const e = document.getElementById('new-email').value;
    const p = document.getElementById('new-password').value;
    const r = document.getElementById('new-role').value;

    if (!e || !p) return alert("Email and Password are required.");

    const fd = new FormData();
    fd.append("email", e);
    fd.append("password", p);
    fd.append("role", r);

    const res = await apiCall('/admin/users', { method: 'POST', body: fd });
    if (res.ok) {
        alert("User Provisioned Successfully!");
        document.getElementById('new-email').value = '';
        document.getElementById('new-password').value = '';
    } else {
        const error = await res.json();
        alert(`Failed to create user: ${error.detail}`);
    }
}

async function assignUser() {
    const e = document.getElementById('assign-email').value;
    const pid = document.getElementById('assign-project-id').value;

    if (!e || !pid) return alert("Email and Project are required.");

    const fd = new FormData();
    fd.append("user_email", e);
    fd.append("project_id", pid);

    const res = await apiCall('/admin/assign', { method: 'POST', body: fd });
    if (res.ok) {
        alert("User Granted Access!");
        document.getElementById('assign-email').value = '';
    } else {
        alert("Assignment failed. Check if user exists or is already assigned.");
    }
}

async function uploadData() {
    const btn = document.getElementById('upload-btn');
    const status = document.getElementById('upload-status');
    const files = document.getElementById('file-upload').files;
    const pid = document.getElementById('upload-project-id').value;

    if (files.length === 0) return alert("Please select files to upload.");
    if (!pid) return alert("Please select a project.");

    btn.innerText = "Vectorizing...";
    status.innerText = "Processing files and sending to AI...";

    const fd = new FormData();
    for (let f of files) fd.append("files", f);
    fd.append("project_id", pid);
    fd.append("model", document.getElementById('upload-model').value);

    try {
        const res = await apiCall('/upload', { method: 'POST', body: fd });
        if (res.ok) {
            status.innerText = "✅ Knowledge Base Synced Successfully.";
            document.getElementById('file-upload').value = '';
        } else {
            const error = await res.json();
            status.innerText = `❌ Sync Failed: ${error.detail}`;
        }
    } catch (e) {
        status.innerText = "❌ Network Error.";
    } finally {
        btn.innerText = "Sync to Vector DB";
    }
}

// --- CHAT & AI RENDERING ---
function clearChat() {
    const select = document.getElementById('project-selector');
    const projectName = select.options[select.selectedIndex]?.text || "a project";
    document.getElementById('chat-container').innerHTML = `<div class="message ai"><div class="bubble">Context switched to <b>${projectName}</b>. How can I help you analyze this data?</div></div>`;
}

function appendMessage(role, text) {
    const container = document.getElementById('chat-container');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    let htmlContent = text;
    let chartConfigs = [];

    // Check if the AI output a Chart.js block
    const chartRegex = /```chart\n([\s\S]*?)\n```/g;
    htmlContent = htmlContent.replace(chartRegex, (match, jsonString) => {
        const chartId = 'chart-' + Math.random().toString(36).substr(2, 9);
        try {
            const config = JSON.parse(jsonString);
            chartConfigs.push({ id: chartId, config: config });
            return `<div class="chart-container"><canvas id="${chartId}"></canvas></div>`;
        } catch (e) {
            return `<div style="color:red;">⚠️ AI generated invalid chart data structure.</div>`;
        }
    });

    // Parse Markdown for tables and bold text
    htmlContent = marked.parse(htmlContent);
    msgDiv.innerHTML = `<div class="bubble">${htmlContent}</div>`;
    container.appendChild(msgDiv);

    // Render any charts found
    chartConfigs.forEach(chart => {
        const ctx = document.getElementById(chart.id).getContext('2d');
        new Chart(ctx, chart.config);
    });

    container.scrollTop = container.scrollHeight;
}

async function sendChat() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    const sendBtn = document.querySelector('.send-btn');

    if (!text) return;

    const projectId = document.getElementById('project-selector').value;
    if (!projectId) return alert("Please select a project to query.");

    appendMessage('user', text);
    input.value = '';
    sendBtn.disabled = true;

    // Loading indicator
    const container = document.getElementById('chat-container');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = `message ai loading-msg`;
    loadingDiv.innerHTML = `<div class="bubble" style="color: var(--text-secondary);">Analyzing data...</div>`;
    container.appendChild(loadingDiv);
    container.scrollTop = container.scrollHeight;

    const fd = new FormData();
    fd.append("message", text);
    fd.append("project_id", projectId);
    fd.append("model", document.getElementById('model-select').value);

    try {
        const res = await apiCall('/chat', { method: 'POST', body: fd });
        const data = await res.json();
        container.removeChild(loadingDiv);

        if (res.ok) {
            appendMessage('ai', data.answer);
        } else {
            appendMessage('ai', `⚠️ Error: ${data.detail}`);
        }
    } catch (e) {
        container.removeChild(loadingDiv);
        appendMessage('ai', "❌ Failed to connect to backend server.");
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}