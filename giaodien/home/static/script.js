function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("active");
}

function handleKeyPress(e) {
  if (e.key === "Enter") sendMessage();
}

function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  const chatBox = document.getElementById("chat-box");

  // Tạo khối chứa cả user và bot message
  const messageBlock = document.createElement("div");
  messageBlock.className = "message-block";

  const userMsg = document.createElement("div");
  userMsg.className = "chat-message user";
  userMsg.innerText = message;
  messageBlock.appendChild(userMsg);

  const botMsg = document.createElement("div");
  botMsg.className = "chat-message bot";
  // -----------------------------------
  // Gọi API để lấy phản hồi từ bot
  botMsg.innerText = "Đang trả lời...";

  fetch("/chat/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message: message }),
  })
    .then((response) => response.json())
    .then((data) => {
      botMsg.innerText = data.response;
    })
    .catch((error) => {
      botMsg.innerText = "Lỗi kết nối đến server!";
      console.error("Lỗi:", error);
    });

  // -----------------------------------
  messageBlock.appendChild(botMsg);

  chatBox.appendChild(messageBlock);

  input.value = "";
  chatBox.scrollTop = chatBox.scrollHeight;
}