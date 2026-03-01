const video = document.getElementById('video');
const video_frame = document.getElementById('video_frame');
const errorMessage = document.getElementById('error-message');

// Check if getUserMedia is available
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    errorMessage.style.display = 'block';
    console.error('getUserMedia is not supported. This feature requires HTTPS.');
} else {
    // Access the device camera and stream to video element
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(error => {
            console.error('Error accessing camera:', error);
            errorMessage.style.display = 'block';
        });
}

let isProcessing = false;

// Function to capture a frame and send it to the server
function captureAndSendFrame() {
    if (isProcessing) return; // Skip if still processing previous frame
    
    isProcessing = true;
    const canvas = document.createElement('canvas');
    
    // Reduce resolution for faster upload/processing
    const maxWidth = 640;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    canvas.width = video.videoWidth * scale;
    canvas.height = video.videoHeight * scale;
    
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Use JPEG with lower quality for faster transfer
    const imageData = canvas.toDataURL('image/jpeg', 0.7);
    const base64ImageData = imageData.replace(/^data:image\/(png|jpg|jpeg);base64,/, '');

    sendImageToServer(base64ImageData);
}

// Send the captured image to the server
function sendImageToServer(imageData) {
    fetch('/emotion_pred', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(response => response.json())
    .then(data => {
        if (data['processed-image']) {
            video_frame.src = `data:image/png;base64,${data['processed-image']}`;
        } else {
            console.error('No processed image received from server');
        }
    })
    .catch(error => {
        console.error('Error sending image to server:', error);
    })
    .finally(() => {
        isProcessing = false;
    });
}

// Process frames as fast as possible (wait for previous to complete)
setInterval(captureAndSendFrame, 50); // Increased from 100ms to 50ms (20 FPS)
