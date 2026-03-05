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

            // NEW: Actually save the token to the browser's storage!
            localStorage.setItem('nexus_token', data.token);

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

    // Clear the active styling from all tabs
    document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

    document.getElementById(`tab-${tabId}`).style.display = 'block';
    event.currentTarget.classList.add('active');

    // THE TRIGGER: Load data when admin tab opens
    if (tabId === 'admin') {
        loadAdminData();
    }

    // Safely apply the active styling
    if (window.event && window.event.currentTarget) {
        // If triggered by a physical mouse click
        window.event.currentTarget.classList.add('active');
    } else if (tabId === 'chat') {
        // If triggered by code (like during a Standard User login)
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

            // FORCE the chart to be fully responsive and stretch to the container
            if (!config.options) config.options = {};
            config.options.responsive = true;
            config.options.maintainAspectRatio = false;

            chartConfigs.push({ id: chartId, config: config });

            // Remove the inline max-width and increase the height for a cinematic view
            return `<div class="chart-container" style="position:relative; height:450px; width:100%; background:var(--bg-main); padding:24px; border-radius:16px; border:1px solid var(--border); margin-top: 16px;"><canvas id="${chartId}"></canvas></div>`;
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
    const inputBoxContainer = document.querySelector('.input-box'); // Target the box for the glow

    if (!text) return;

    const projectId = document.getElementById('project-selector').value;
    if (!projectId || projectId === "") {
        return alert("Please select a valid project from the dropdown first.");
    }

    appendMessage('user', text);
    input.value = '';

    // --- UI LOCK & ANIMATION START ---
    sendBtn.disabled = true;
    input.disabled = true;
    input.placeholder = "AI is analyzing...";
    inputBoxContainer.classList.add('ai-thinking'); // Start the breathing glow

    const container = document.getElementById('chat-container');
    const fd = new FormData();
    fd.append("message", text);
    fd.append("project_id", projectId);
    fd.append("model", document.getElementById('model-select').value);

    try {
        const res = await apiCall('/chat', { method: 'POST', body: fd });
        const data = await res.json();

        if (res.ok) {
            appendMessage('ai', data.answer);
        } else {
            appendMessage('ai', `⚠️ Error: ${data.detail}`);
        }
    } catch (e) {
        appendMessage('ai', "❌ Failed to connect to backend server.");
    } finally {
        // --- UI UNLOCK & ANIMATION STOP ---
        inputBoxContainer.classList.remove('ai-thinking'); // Stop the glow
        input.disabled = false;
        input.placeholder = "Ask a question about the project data...";
        sendBtn.disabled = false;
        input.focus(); // Snap the cursor back for the next question
    }
}

// --- Custom Dropdown Engine ---
function initializeNexusDropdowns() {
    document.querySelectorAll('.nexus-select').forEach(nativeSelect => {
        // Prevent double initialization
        if (nativeSelect.nextElementSibling && nativeSelect.nextElementSibling.classList.contains('nexus-dropdown-wrapper')) {
            return;
        }

        // 1. Build the UI wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'nexus-dropdown-wrapper';
        nativeSelect.parentNode.insertBefore(wrapper, nativeSelect.nextSibling);
        wrapper.appendChild(nativeSelect);

        // 2. Build the visual button (trigger)
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

        // 3. The function that populates the custom options
        const renderOptions = () => {
            optionsContainer.innerHTML = '';
            Array.from(nativeSelect.options).forEach(option => {
                const optDiv = document.createElement('div');
                optDiv.className = `nexus-dropdown-option ${option.selected ? 'selected' : ''}`;
                optDiv.dataset.value = option.value;
                optDiv.innerText = option.text;

                // When a custom option is clicked...
                optDiv.addEventListener('click', () => {
                    nativeSelect.value = option.value; // Update the hidden real select
                    trigger.querySelector('.trigger-text').innerText = option.text; // Update text

                    optionsContainer.querySelectorAll('.nexus-dropdown-option').forEach(el => el.classList.remove('selected'));
                    optDiv.classList.add('selected');
                    wrapper.classList.remove('open'); // Close dropdown

                    // Fire the standard 'change' event so your existing logic (like clearChat) triggers
                    nativeSelect.dispatchEvent(new Event('change'));
                });
                optionsContainer.appendChild(optDiv);
            });
            // Set initial text
            trigger.querySelector('.trigger-text').innerText = nativeSelect.options[nativeSelect.selectedIndex]?.text || 'Select...';
        };

        // Render immediately
        renderOptions();

        // 4. Watch for dynamic API updates (Crucial for your project lists!)
        const observer = new MutationObserver(renderOptions);
        observer.observe(nativeSelect, { childList: true });

        // 5. Open/Close Logic
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            document.querySelectorAll('.nexus-dropdown-wrapper').forEach(w => {
                if (w !== wrapper) w.classList.remove('open');
            });
            wrapper.classList.toggle('open');
        });
    });

    // Close all if clicking outside
    document.addEventListener('click', () => {
        document.querySelectorAll('.nexus-dropdown-wrapper').forEach(w => w.classList.remove('open'));
    });
}

// --- Admin Panel Tab Switching ---
function switchAdminTab(tabName) {
    // 1. Remove active class from all tabs
    document.querySelectorAll('.admin-tab').forEach(tab => tab.classList.remove('active'));
    // 2. Hide all sections
    document.querySelectorAll('.admin-section').forEach(sec => sec.style.display = 'none');

    // 3. Activate the clicked tab and show the corresponding section
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

// Auto-close sidebar when switching tabs on mobile
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
    recognition.continuous = false; // Stop after they finish a sentence
    recognition.interimResults = true; // Show words as they speak
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
        synth.cancel(); // Stop AI speaking
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
    synth.cancel(); // Ensure AI stops talking if we start listening
    setVoiceState('listening', "I'm listening...");

    try {
        recognition.start();
    } catch (e) { console.log("Recognition already started"); }
}

// 1. As the user speaks, update the text on screen
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

// 2. When the user stops speaking, send it to the backend
recognition.onend = () => {
    if (!isVoiceModeActive) return;

    const finalQuestion = document.getElementById('voice-transcript').innerText;

    // If they didn't say anything, just start listening again
    if (finalQuestion === "I'm listening..." || finalQuestion.trim() === "") {
        startListening();
        return;
    }

    // Send the voice query to the AI
    processVoiceQuery(finalQuestion);
};

// 3. Send to API and handle response
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
        const res = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${localStorage.getItem('nexus_token')}` },
            body: fd
        });

        // NEW: Check if the server rejected the request before trying to parse JSON
        if (!res.ok) {
            const errorData = await res.text();
            console.error("Backend Error Details:", errorData);
            throw new Error(`Server responded with status: ${res.status}`);
        }

        const data = await res.json();
        
        // Inject into normal chat history
        appendMessage('user', question);
        appendMessage('ai', data.answer);
        
        // Speak the answer out loud
        speakResponse(data.answer);

   } catch (error) {
        console.error("Voice Query Failed:", error); 
        
        // Make the AI verbally tell you if you aren't logged in
        if (error.message.includes("401")) {
            speakResponse("Your session has expired. Please sign in to access the project data.");
        } else {
            speakResponse("Sorry, the server connection failed. Please check the console.");
        }
    }
}

// 4. Clean text and speak it (Safe Loop Version)
function speakResponse(markdownText) {
    if (!isVoiceModeActive) return;
    
    if (!markdownText) {
        setVoiceState('listening', "...");
        return;
    }
    
    let cleanText = markdownText.toString();
    
    // 1. Strip markdown
    cleanText = cleanText.split('*').join('');
    cleanText = cleanText.split('#').join('');
    cleanText = cleanText.split('_').join('');
    cleanText = cleanText.split('`').join('');
    cleanText = cleanText.split('~').join('');
    
    // 2. Strip tags without using while loops or regex
    let chunks = cleanText.split("[cite:");    

    for (let i = 1; i < chunks.length; i++) {
        let closeIndex = chunks[i].indexOf(']');
        if (closeIndex !== -1) {
            cleanText += chunks[i].substring(closeIndex + 1);
        } else {
            cleanText += chunks[i];
        }
    }
    
    // 3. Update UI
    setVoiceState('speaking', cleanText);
    
    // 4. Speak
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.85; 
    
    // Voice selection    
    const voices = window.speechSynthesis.getVoices();
    console.log("Available Voices: ", voices.map(v => v.name));

    // Look for clear English voice (Can change strings to try different voices)
    const preferredVoice = voices.find(v => 
        v.name.includes('Google US English') || 
        v.name.includes('Microsoft Zira') ||
        v.name.includes ('en-US')
    );

    if (preferredVoice) {
        utterance.voice = preferredVoice
    }

    // Typing animation sync

    // Clear UI before AI starts speaking
    setVoiceState('speaking', "");

    // Fire event exactly when AI begins speaking
    utterance.onboundary = (event) => {
        if (event.name === 'word'){
            const revealedText = cleanText.substring(0, event.charIndex + 10);
            setVoiceState('speaking', revealedText)
        }
    }

    // Failsafe: ensure that full text is shown when done
    utterance.onend = () => {
        setVoiceState('speaking', cleanText);
        if (isVoiceModeActive){
            startListening()
        }
    }
    
    window.speechSynthesis.speak(utterance);
}

// Ensure this runs when the page loads
document.addEventListener('DOMContentLoaded', initializeNexusDropdowns);