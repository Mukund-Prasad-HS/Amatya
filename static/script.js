document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const questionForm = document.getElementById('question-form');
    const chatHistory = document.getElementById('chat-history');

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(uploadForm);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                alert(data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while uploading PDFs');
        });
    });

    questionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const questionInput = document.getElementById('question-input');
        const question = questionInput.value;

        // Add user message to chat history
        addMessageToChat('User', question);

        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({question: question})
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                addMessageToChat('Error', data.error);
            } else {
                addMessageToChat('Assistant', data.response);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addMessageToChat('Error', 'An error occurred while processing your question');
        });

        questionInput.value = '';
    });

    function addMessageToChat(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender.toLowerCase() + '-message');
        messageElement.textContent = `${sender}: ${message}`;
        chatHistory.appendChild(messageElement);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});