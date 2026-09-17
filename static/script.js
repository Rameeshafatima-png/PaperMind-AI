const chatArea = document.getElementById("chatArea");
const welcomeState = document.getElementById("welcomeState");
const fileInput = document.getElementById("fileInput");
const attachBtn = document.getElementById("attachBtn");
const sendBtn = document.getElementById("sendBtn");
const questionInput = document.getElementById("questionInput");
const uploadProgress = document.getElementById("uploadProgress");
const uploadPreview = document.getElementById("uploadPreview");
const alertBox = document.getElementById("alertBox");
const clearBtn = document.getElementById("clearBtn");
const documentList = document.getElementById("documentList");
const docCount = document.getElementById("docCount");
const topDocCount = document.getElementById("topDocCount");
const chunkCount = document.getElementById("chunkCount");
let alertTimer;

function esc(value) {
    return String(value ?? "").replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;");
}
function showAlert(message, type="error") {
    clearTimeout(alertTimer);
    alertBox.textContent = message;
    alertBox.className = `alert ${type}`;
    alertTimer = setTimeout(() => alertBox.classList.add("hidden"), 5000);
}
function hideWelcome() { document.getElementById("welcomeState")?.remove(); }

function renderDocuments(documents) {
    documentList.innerHTML = "";
    if (!documents?.length) {
        documentList.innerHTML = `<div class="empty-documents"><div class="empty-icon">+</div><span>No papers indexed yet.</span></div>`;
    } else {
        documents.forEach(doc => {
            const item = document.createElement("div");
            item.className = "document-item";
            item.innerHTML = `<div class="pdf-icon">PDF</div><span title="${esc(doc.name)}">${esc(doc.name)}</span><small>${esc(doc.pages)}p · ${esc(doc.chunks)}c</small>`;
            documentList.appendChild(item);
        });
    }
    const count = documents?.length || 0;
    docCount.textContent = count;
    topDocCount.textContent = count;
}

function addUserMessage(text) {
    hideWelcome();
    const el = document.createElement("div");
    el.className = "message user";
    el.innerHTML = `<div class="user-bubble">${esc(text)}</div>`;
    chatArea.appendChild(el);
    chatArea.scrollTop = chatArea.scrollHeight;
}
function addLoading() {
    const el = document.createElement("div");
    el.id = "loadingMessage";
    el.className = "message";
    el.innerHTML = `<div class="assistant-card loading-card"><div class="spinner"></div><span>Searching the indexed papers and preparing a grounded answer…</span></div>`;
    chatArea.appendChild(el);
    chatArea.scrollTop = chatArea.scrollHeight;
}
function removeLoading() { document.getElementById("loadingMessage")?.remove(); }

function addAssistantMessage(data) {
    removeLoading();
    const sources = (data.sources || []).map(source => `
        <div class="source-chip">
            <span>Source</span>
            <b>${esc(source.document)}</b><span>·</span><span>p. ${esc(source.page)}</span>
            ${source.similarity != null ? `<i style="--score:${Math.round(source.similarity*100)}%"></i>` : ""}
        </div>`).join("");
    const el = document.createElement("div");
    el.className = "message";
    el.innerHTML = `<div class="assistant-card"><div class="answer-label">Grounded answer</div><div class="answer-text">${esc(data.answer)}</div>${sources ? `<div class="sources">${sources}</div>` : ""}</div>`;
    chatArea.appendChild(el);
    chatArea.scrollTop = chatArea.scrollHeight;
}

attachBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
    const files = [...fileInput.files];
    if (!files.length) return;
    if (files.some(f => !f.name.toLowerCase().endsWith(".pdf"))) {
        showAlert("Only PDF files are supported.");
        fileInput.value = "";
        return;
    }
    uploadPreview.innerHTML = files.map(f => `<div class="file-pill"><span>PDF</span>${esc(f.name)}</div>`).join("");
    uploadPreview.classList.remove("hidden");
    uploadProgress.classList.remove("hidden");

    const form = new FormData();
    files.forEach(file => form.append("files", file));

    try {
        const response = await fetch("/api/upload", { method:"POST", body:form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Upload failed.");
        renderDocuments(data.documents || []);
        chunkCount.textContent = data.chunk_count ?? 0;

        if (data.uploaded?.length) {
            showAlert(`Indexed ${data.uploaded.length} paper${data.uploaded.length > 1 ? "s" : ""} successfully.`, "success");
        }
        if (data.errors?.length) showAlert(data.errors.join(" "), "error");
    } catch (error) {
        showAlert(error.message || "Could not upload the selected papers.");
    } finally {
        uploadProgress.classList.add("hidden");
        uploadPreview.classList.add("hidden");
        fileInput.value = "";
    }
});

async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) {
        showAlert("Please enter a question about your uploaded papers.");
        questionInput.focus();
        return;
    }
    addUserMessage(question);
    questionInput.value = "";
    questionInput.style.height = "auto";
    sendBtn.disabled = true;
    addLoading();

    try {
        const response = await fetch("/api/chat", {
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({question})
        });
        const data = await response.json();
        if (!response.ok) {
            removeLoading();
            showAlert(data.detail || "Could not generate an answer.");
            return;
        }
        addAssistantMessage(data);
    } catch (_) {
        removeLoading();
        showAlert("The server could not be reached. Make sure FastAPI is running.");
    } finally {
        sendBtn.disabled = false;
        questionInput.focus();
    }
}
sendBtn.addEventListener("click", askQuestion);
questionInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});
questionInput.addEventListener("input", () => {
    questionInput.style.height = "auto";
    questionInput.style.height = `${Math.min(questionInput.scrollHeight, 130)}px`;
});
document.querySelectorAll(".suggestion").forEach(button => {
    button.addEventListener("click", () => {
        questionInput.value = button.dataset.question;
        questionInput.dispatchEvent(new Event("input"));
        questionInput.focus();
    });
});

clearBtn.addEventListener("click", async () => {
    if (!confirm("Clear all uploaded papers and their indexed knowledge?")) return;
    try {
        const response = await fetch("/api/documents", {method:"DELETE"});
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not clear the knowledge base.");
        renderDocuments([]);
        chunkCount.textContent = "0";
        showAlert("Knowledge base cleared successfully.", "success");
    } catch (error) {
        showAlert(error.message || "Could not clear the knowledge base.");
    }
});

async function refreshStatus() {
    try {
        const response = await fetch("/api/status");
        if (!response.ok) return;
        const data = await response.json();
        renderDocuments(data.documents || []);
        chunkCount.textContent = data.chunk_count ?? 0;
    } catch (_) {}
}
refreshStatus();
