function uploadFile() {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('File uploaded and processed successfully');
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => console.error('Error:', error));
    } else {
        alert('Please select a file to upload');
    }
}

function askQuestion() {
    const userInput = document.getElementById('user-input');
    const question = userInput.value.trim();
    if (question) {
        addMessageToChat('You: ' + question);
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        })
        .then(response => response.json())
        .then(data => {
            if (data.answer) {
                addMessageToChat('AI: ' + data.answer);
            } else {
                addMessageToChat('AI: Error - ' + data.error);
            }
        })
        .catch(error => console.error('Error:', error));
        userInput.value = '';
    }
}

function addMessageToChat(message) {
    const chatHistory = document.getElementById('chat-history');
    const messageElement = document.createElement('p');
    messageElement.textContent = message;
    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}