const chatEl = document.getElementById("chat");
const composerEl = document.getElementById("composer");
const messageEl = document.getElementById("message");
const sendBtnEl = document.getElementById("sendBtn");
const userIdEl = document.getElementById("userId");
const reloadBtnEl = document.getElementById("reloadBtn");
const memoryBtnEl = document.getElementById("memoryBtn");
const memoryDialogEl = document.getElementById("memoryDialog");
const memoryContentEl = document.getElementById("memoryContent");
const closeMemoryBtnEl = document.getElementById("closeMemoryBtn");

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = content;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function currentUserId() {
  const id = userIdEl.value.trim();
  return id || "default-user";
}

async function loadHistory() {
  const res = await fetch(`/api/history?user_id=${encodeURIComponent(currentUserId())}&limit=100`);
  const data = await res.json();
  chatEl.innerHTML = "";
  if (!data.messages.length) {
    addMessage("system", "会話履歴はまだありません。");
    return;
  }
  for (const msg of data.messages) {
    addMessage(msg.role === "assistant" ? "assistant" : "user", msg.content);
  }
}

async function showMemories() {
  const res = await fetch(`/api/memories?user_id=${encodeURIComponent(currentUserId())}&limit=40`);
  const data = await res.json();
  if (!data.memories.length) {
    memoryContentEl.textContent = "No memories.";
  } else {
    memoryContentEl.textContent = data.memories
      .map((m) => `#${m.id} [${m.memory_type}] (${m.emotion}, imp=${m.importance.toFixed(2)}) ${m.content}`)
      .join("\n");
  }
  memoryDialogEl.showModal();
}

async function sendMessage(text) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: currentUserId(),
      message: text,
    }),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText || "request failed");
  }
  return res.json();
}

messageEl.addEventListener("input", () => {
  messageEl.style.height = "auto";
  messageEl.style.height = `${Math.min(messageEl.scrollHeight, 200)}px`;
});

composerEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageEl.value.trim();
  if (!text) return;

  addMessage("user", text);
  messageEl.value = "";
  messageEl.style.height = "48px";
  sendBtnEl.disabled = true;
  addMessage("assistant", "考え中...");

  try {
    const data = await sendMessage(text);
    chatEl.lastChild.textContent = data.reply;
  } catch (error) {
    chatEl.lastChild.textContent = `エラー: ${error.message}`;
  } finally {
    sendBtnEl.disabled = false;
    messageEl.focus();
  }
});

reloadBtnEl.addEventListener("click", loadHistory);
memoryBtnEl.addEventListener("click", showMemories);
closeMemoryBtnEl.addEventListener("click", () => memoryDialogEl.close());

loadHistory();
