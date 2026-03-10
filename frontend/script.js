// Change this to your Render URL when deploying https://nexus-backend-service.onrender.com
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

            document.getElementById('user-badge').innerText = data.email;
            document.getElementById('view-login').classList.remove('active');
            document.getElementById('view-app').classList.add('active');

            // Show Admin Panel option if applicable
            if (currentRole === 'admin') {
                document.querySelector('.admin-only').style.display = 'flex';
            } else {
                document.querySelector('.admin-only').style.display = 'none';
                switchTab('chat');
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
    document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

    document.getElementById(`tab-${tabId}`).style.display = tabId === 'chat' ? 'flex' : 'block';

    const target = document.getElementById(`tab-${tabId}`);
    if (target) {
        target.style.display = 'flex';
    }
    if (tabId === 'admin') {
        loadAdminData();
    }

    if (window.event && window.event.currentTarget) {
        window.event.currentTarget.classList.add('active');
    } else if (tabId === 'chat') {
        document.querySelectorAll('.nav-item')[0].classList.add('active');
    }
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

function appendMessage(role, text, sources = []) {
    const container = document.getElementById('chat-container');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    let htmlContent = text;
    let chartConfigs = [];

    // 1. --- Chart.js Regex Logic ---
    const chartRegex = /```chart\n([\s\S]*?)\n```/g;
    htmlContent = htmlContent.replace(chartRegex, (match, jsonString) => {
        const chartId = 'chart-' + Math.random().toString(36).substr(2, 9);
        try {
            const config = JSON.parse(jsonString);

            if (!config.options) config.options = {};
            config.options.responsive = true;
            config.options.maintainAspectRatio = false;

            chartConfigs.push({ id: chartId, config: config });

            return `<div class="chart-container" style="position:relative; height:450px; width:100%; background:var(--bg-main); padding:24px; border-radius:16px; border:1px solid var(--border); margin-top: 16px;"><canvas id="${chartId}"></canvas></div>`;
        } catch (e) {
            return `<div style="color:red;">⚠️ AI generated invalid chart data structure.</div>`;
        }
    });

    // 2. --- Markdown Parsing ---
    htmlContent = marked.parse(htmlContent);

    // 3. --- Material Design Source Pills ---
    let sourcesHtml = "";
    if (role === 'ai' && sources && sources.length > 0) {
        sourcesHtml = `<div class="sources-container"><div class="source-label">Documents referenced</div>`;
        
        sources.forEach(src => {
            // Check if URL is valid
            const url = (src.url && src.url !== "#") ? src.url : null;
            
            if (url) {
                sourcesHtml += `
                    <a href="${url}" 
                       target="_blank" 
                       rel="noopener noreferrer" 
                       class="source-pill" 
                       title="View ${src.name}">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                        </svg>
                        <span class="pill-text">${src.name}</span>
                    </a>`;
            } else {
                // If no URL found, show a disabled-looking pill
                sourcesHtml += `
                    <div class="source-pill disabled" style="opacity: 0.5; cursor: not-allowed;">
                        <span class="pill-text">${src.name} (Link Missing)</span>
                    </div>`;
            }
        });
        sourcesHtml += `</div>`;
    }

    // 4. --- Inject HTML into Bubble ---
    msgDiv.innerHTML = `<div class="bubble">${htmlContent}${sourcesHtml}</div>`;
    container.appendChild(msgDiv);

    // 5. --- Initialize Charts ---
    chartConfigs.forEach(chart => {
        const ctx = document.getElementById(chart.id).getContext('2d');
        new Chart(ctx, chart.config);
    });

    // 6. --- Scroll to View ---
    setTimeout(() => {
        msgDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }, 100);
}
async function sendChat() {
    const input = document.getElementById('user-input');
    const text = input.value.trim();
    const sendBtn = document.querySelector('.send-btn');
    const container = document.getElementById('chat-container');

    if (!text) return;

    // 1. Project Validation
    const projectId = document.getElementById('project-selector').value;
    if (!projectId) return alert("Please select a project.");

    // 2. UI Setup & User Message
    input.value = '';
    sendBtn.disabled = true;
    appendMessage('user', text);

    // 3. CREATE DYNAMIC LOADER (The "Thinking" state)
    const loaderDiv = document.createElement('div');
    loaderDiv.className = 'message ai loading-status';
    loaderDiv.innerHTML = `
        <div class="bubble" style="background: transparent; padding-left: 0;">
            <div class="thinking-steps">
                <div class="step-icon"></div>
                <span id="nexus-status-text">Searching Knowledge Base...</span>
            </div>
        </div>
    `;
    container.appendChild(loaderDiv);

    // Cycle through professional status updates
    const statuses = [
        "Analyzing relevant documents...",
        "Identifying key data points...",
        "Synthesizing enterprise insights...",
        "Finalizing response structure..."
    ];
    let statusIndex = 0;
    const statusInterval = setInterval(() => {
        statusIndex = (statusIndex + 1) % statuses.length;
        const statusEl = document.getElementById('nexus-status-text');
        if (statusEl) statusEl.innerText = statuses[statusIndex];
    }, 2000);

    // 4. API CALL
    const fd = new FormData();
    fd.append("message", text);
    fd.append("project_id", projectId);
    fd.append("model", document.getElementById('model-select').value);

    try {
        const res = await apiCall('/chat', { method: 'POST', body: fd });
        const data = await res.json();

        // Stop loader and remove it
        clearInterval(statusInterval);
        if (container.contains(loaderDiv)) container.removeChild(loaderDiv);

        if (res.ok) {
            // PASS BOTH THE ANSWER AND THE SOURCES ARRAY
            appendMessage('ai', data.answer, data.sources);
        } else {
            appendMessage('ai', `⚠️ Error: ${data.detail}`);
        }
    } catch (e) {
        clearInterval(statusInterval);
        if (container.contains(loaderDiv)) container.removeChild(loaderDiv);
        appendMessage('ai', "❌ Backend connection failed. Check your local server.");
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}
// --- Custom Dropdown Engine ---
function initializeNexusDropdowns() {
    document.querySelectorAll('.nexus-select').forEach(nativeSelect => {
        if (nativeSelect.nextElementSibling && nativeSelect.nextElementSibling.classList.contains('nexus-dropdown-wrapper')) {
            return;
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'nexus-dropdown-wrapper';
        nativeSelect.parentNode.insertBefore(wrapper, nativeSelect.nextSibling);
        wrapper.appendChild(nativeSelect);

        const trigger = document.createElement('div');
        trigger.className = 'nexus-dropdown-trigger';
        trigger.innerHTML = `
            <span class="trigger-text">Loading...</span>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>
        `;
        wrapper.appendChild(trigger);

        const optionsContainer = document.createElement('div');
        optionsContainer.className = 'nexus-dropdown-options';
        wrapper.appendChild(optionsContainer);

        const renderOptions = () => {
            optionsContainer.innerHTML = '';
            Array.from(nativeSelect.options).forEach(option => {
                const optDiv = document.createElement('div');
                optDiv.className = `nexus-dropdown-option ${option.selected ? 'selected' : ''}`;
                optDiv.dataset.value = option.value;
                optDiv.innerText = option.text;

                optDiv.addEventListener('click', () => {
                    nativeSelect.value = option.value;
                    trigger.querySelector('.trigger-text').innerText = option.text;

                    optionsContainer.querySelectorAll('.nexus-dropdown-option').forEach(el => el.classList.remove('selected'));
                    optDiv.classList.add('selected');
                    wrapper.classList.remove('open');

                    nativeSelect.dispatchEvent(new Event('change'));
                });
                optionsContainer.appendChild(optDiv);
            });
            trigger.querySelector('.trigger-text').innerText = nativeSelect.options[nativeSelect.selectedIndex]?.text || 'Select...';
        };

        renderOptions();

        const observer = new MutationObserver(renderOptions);
        observer.observe(nativeSelect, { childList: true });

        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            document.querySelectorAll('.nexus-dropdown-wrapper').forEach(w => {
                if (w !== wrapper) w.classList.remove('open');
            });
            wrapper.classList.toggle('open');
        });
    });

    document.addEventListener('click', () => {
        document.querySelectorAll('.nexus-dropdown-wrapper').forEach(w => w.classList.remove('open'));
    });
}

// --- Admin Panel Tab Switching ---
function switchAdminTab(tabName) {
    document.querySelectorAll('.admin-tab').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.admin-section').forEach(sec => sec.style.display = 'none');
    document.querySelector(`.admin-tab[onclick="switchAdminTab('${tabName}')"]`).classList.add('active');

    const targetSection = document.getElementById(`admin-sec-${tabName}`);
    if (targetSection) {
        targetSection.style.display = 'block';
    }
}

// --- ADMIN DATA LOADERS ---
async function loadAdminData() {
    await Promise.all([
        loadAdminProjects(),
        loadAdminUsers(),
        loadAdminDocuments()
    ]);
}

async function loadAdminProjects() {
    try {
        const res = await apiCall('/projects', { method: 'GET' });
        const data = await res.json();
        const tbody = document.getElementById('admin-project-list');
        tbody.innerHTML = '';

        if (data.projects) {
            data.projects.forEach(p => {
                tbody.innerHTML += `
                    <tr>
                        <td>${p.id}</td>
                        <td>${p.name}</td>
                        <td style="text-align: right;">
                            <button class="action-btn" onclick="editProject(${p.id}, '${p.name}')">Edit</button>
                            <button class="action-btn danger" onclick="deleteProject(${p.id})">Delete</button>
                        </td>
                    </tr>
                `;
            });
        }
    } catch (e) { console.error("Failed to load projects", e); }
}

async function loadAdminUsers() {
    try {
        const res = await apiCall('/admin/list_users', { method: 'GET' });
        const data = await res.json();
        const tbody = document.getElementById('admin-user-list');
        tbody.innerHTML = '';

        if (data.users) {
            data.users.forEach(u => {
                const roleClass = u.role === 'admin' ? 'admin' : '';
                tbody.innerHTML += `
                    <tr>
                        <td>${u.email}</td>
                        <td><span class="role-badge ${roleClass}">${u.role.toUpperCase()}</span></td>
                        <td>Managed via DB</td>
                        <td style="text-align: right;">
                            <button class="action-btn" onclick="toggleUserRole('${u.id}', '${u.role}')">Toggle Role</button>
                            <button class="action-btn danger" onclick="deleteUser('${u.id}')">Delete</button>
                        </td>
                    </tr>
                `;
            });
        }
    } catch (e) { console.error("Failed to load users", e); }
}

async function loadAdminDocuments() {
    try {
        const res = await apiCall('/admin/files', { method: 'GET' });
        const data = await res.json();
        const tbody = document.getElementById('admin-document-list');
        tbody.innerHTML = '';

        if (data.files) {
            data.files.forEach(f => {
                const date = new Date(f.uploaded_at).toLocaleDateString();
                tbody.innerHTML += `
                    <tr>
                        <td><a href="${f.file_url}" target="_blank" style="color: var(--accent); text-decoration: none;">${f.file_name}</a></td>
                        <td>Project ${f.project_id}</td>
                        <td>${date}</td>
                        <td style="text-align: right;">
                            <button class="action-btn danger" onclick="deleteDocument(${f.id})">Delete</button>
                        </td>
                    </tr>
                `;
            });
        }
    } catch (e) { console.error("Failed to load documents", e); }
}

// --- ADMIN ACTIONS ---
async function deleteProject(id) {
    if (!confirm("Are you sure? This will delete the project and cascade-delete all associated files and access mappings.")) return;
    await apiCall(`/admin/projects/${id}`, { method: 'DELETE' });
    loadAdminData(); // Refresh table
    fetchProjects(); // Refresh dropdowns
}

async function editProject(id, currentName) {
    const newName = prompt("Enter new project name:", currentName);
    if (!newName || newName === currentName) return;

    const fd = new FormData();
    fd.append("name", newName);
    await apiCall(`/admin/projects/${id}`, { method: 'PUT', body: fd });
    loadAdminData();
    fetchProjects();
}

async function deleteUser(id) {
    if (!confirm("Are you sure you want to permanently delete this user?")) return;
    await apiCall(`/admin/users/${id}`, { method: 'DELETE' });
    loadAdminData();
}

async function toggleUserRole(id, currentRole) {
    const newRole = currentRole === 'admin' ? 'user' : 'admin';
    if (!confirm(`Change user role to ${newRole.toUpperCase()}?`)) return;

    const fd = new FormData();
    fd.append("role", newRole);
    await apiCall(`/admin/users/${id}/role`, { method: 'PUT', body: fd });
    loadAdminData();
}

async function deleteDocument(id) {
    if (!confirm("Delete this document from the storage bucket and database?")) return;
    await apiCall(`/admin/files/${id}`, { method: 'DELETE' });
    loadAdminData();
}

// --- Mobile Sidebar Controls ---
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('sidebar-overlay');

    sidebar.classList.toggle('open');
    overlay.classList.toggle('active');
}

const originalSwitchTab = switchTab;
switchTab = function (tabId) {
    originalSwitchTab(tabId);
    if (window.innerWidth <= 768) {
        document.querySelector('.sidebar').classList.remove('open');
        document.getElementById('sidebar-overlay').classList.remove('active');
    }
};

// --- VOICE ENGINE ---
let isVoiceModeActive = false;
let recognition = null;
let synth = window.speechSynthesis;

// Initialize Speech Recognition
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
} else {
    console.warn("Speech recognition not supported in this browser.");
}

function toggleVoiceMode() {
    if (!recognition) {
        alert("Voice mode is not supported in this browser. Please use Chrome or Edge.");
        return;
    }

    const overlay = document.getElementById('voice-overlay');
    isVoiceModeActive = !isVoiceModeActive;

    if (isVoiceModeActive) {
        overlay.style.display = 'flex';
        startListening();
    } else {
        overlay.style.display = 'none';
        recognition.stop();
        synth.cancel();
    }
}

function setVoiceState(state, text) {
    const sphere = document.getElementById('voice-sphere');
    const statusText = document.getElementById('voice-status');
    const transcriptText = document.getElementById('voice-transcript');

    // Reset classes
    sphere.className = 'voice-sphere';
    sphere.classList.add(state);

    if (state === 'listening') statusText.innerText = "Listening...";
    if (state === 'thinking') statusText.innerText = "Thinking...";
    if (state === 'speaking') statusText.innerText = "Speaking...";

    if (text) transcriptText.innerText = text;
}

function startListening() {
    if (!isVoiceModeActive) return;
    synth.cancel();
    setVoiceState('listening', "I'm listening...");

    try {
        recognition.start();
    } catch (e) { console.log("Recognition already started"); }
}

recognition.onresult = (event) => {
    let interimTranscript = '';
    let finalTranscript = '';

    for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
        } else {
            interimTranscript += event.results[i][0].transcript;
        }
    }
    document.getElementById('voice-transcript').innerText = finalTranscript || interimTranscript;
};

recognition.onend = () => {
    if (!isVoiceModeActive) return;

    const finalQuestion = document.getElementById('voice-transcript').innerText;

    if (finalQuestion === "I'm listening..." || finalQuestion.trim() === "") {
        startListening();
        return;
    }

    processVoiceQuery(finalQuestion);
};

async function processVoiceQuery(question) {
    setVoiceState('thinking', question);

    const projectId = document.getElementById('project-selector').value;
    if (!projectId) {
        speakResponse("Please select a project first.");
        return;
    }

    const fd = new FormData();
    fd.append("message", question);
    fd.append("project_id", projectId);
    fd.append("model", document.getElementById('model-select').value || "gemini-2.5-flash");

    try {
        const res = await fetch(`${API_BASE_URL}/chat`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${localStorage.getItem('nexus_token')}` },
            body: fd
        });

        const data = await res.json();

        appendMessage('user', question);
        appendMessage('ai', data.answer);

        speakResponse(data.answer);

    } catch (error) {
        speakResponse("Sorry, I encountered an error checking the database.");
    }
}

function speakResponse(markdownText) {
    if (!isVoiceModeActive) return;

    const safeText = markdownText ? markdownText.toString() : "";

    const formatRegex = new RegExp('[*#_`~]', 'g');
    const citeRegex = new RegExp('\\', 'g');

    const cleanText = safeText.replace(formatRegex, '').replace(citeRegex, '');

    setVoiceState('speaking', cleanText);

    const utterance = new SpeechSynthesisUtterance(cleanText);

    const voices = synth.getVoices();
    const googleVoice = voices.find(v => v.name.includes('Google US English') || v.name.includes('Samantha'));
    if (googleVoice) utterance.voice = googleVoice;

    utterance.rate = 1.05;

    utterance.onend = () => {
        if (isVoiceModeActive) startListening();
    };

    synth.speak(utterance);
}

document.addEventListener('DOMContentLoaded', initializeNexusDropdowns);