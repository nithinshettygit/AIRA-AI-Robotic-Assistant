document.addEventListener('DOMContentLoaded', () => {
  const chatWindow = document.getElementById('chat-window');
  const topicInput = document.getElementById('topic-input');
  const sendBtn = document.getElementById('send-btn');
  const pauseBtn = document.getElementById('pause-btn');

  const doubtModal = document.getElementById('doubt-modal');
  const doubtChat = document.getElementById('doubt-chat');
  const doubtInput = document.getElementById('doubt-input');
  const doubtSendBtn = document.getElementById('doubt-send-btn');
  const closeBtn = document.querySelector('.close-btn');

  let sessionId = null;

  function addMessage(sender, text, targetChat = chatWindow) {
      const messageDiv = document.createElement('div');
      messageDiv.className = `message ${sender.toLowerCase()}`;
      messageDiv.innerHTML = `<strong>${sender}:</strong> ${text}`;
      targetChat.appendChild(messageDiv);
      targetChat.scrollTop = targetChat.scrollHeight;
  }

  async function continueLesson() {
      if (!sessionId) return;
      const response = await fetch('/continue_lesson', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({session_id: sessionId})
      });
      if (!response.ok) {
          addMessage('System', 'Error continuing lesson');
          return;
      }
      const data = await response.json();
      addMessage('Teacher', data.message);
  }

  sendBtn.addEventListener('click', async () => {
      const topic = topicInput.value.trim();
      if (!topic && sendBtn.textContent === "Start Lesson") {
          alert("Please enter a topic to start the lesson.");
          return;
      }

      if (sendBtn.textContent === "Start Lesson") {
          addMessage('You', topic);
          try {
            const response = await fetch('/start_lesson', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({topic})
            });
            if (!response.ok) throw new Error("Failed to start lesson");
            const data = await response.json();
            sessionId = data.session_id;
            addMessage('Teacher', data.message);
            pauseBtn.style.display = 'inline-block';
            sendBtn.textContent = 'Continue Lesson';
            topicInput.value = '';
          } catch(e) {
            addMessage('System', e.message);
          }
      } else {
          await continueLesson();
      }
  });

  pauseBtn.addEventListener('click', () => {
      doubtModal.style.display = 'block';
      doubtChat.innerHTML = "";  // clear previous doubt chat
  });

  doubtSendBtn.addEventListener('click', async () => {
      const question = doubtInput.value.trim();
      if (!question || !sessionId) return;
      addMessage('You', question, doubtChat);
      doubtInput.value = '';

      try {
          const response = await fetch('/ask_doubt', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({session_id: sessionId, question})
          });
          if (!response.ok) throw new Error("Failed to get answer for doubt");
          const data = await response.json();
          addMessage('Teacher', data.message, doubtChat);
      } catch(e) {
          addMessage('System', e.message, doubtChat);
      }
  });

  closeBtn.addEventListener('click', async () => {
      doubtModal.style.display = 'none';
      try {
          await fetch('/resume_lesson', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({session_id: sessionId})
          });
          await continueLesson();
      } catch {
          addMessage('System', 'Failed to resume lesson');
      }
  });
});
