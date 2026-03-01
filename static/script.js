const video = document.getElementById('video');
const video_frame = document.getElementById('video_frame');

// Access the device camera and stream to video element
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(error => {
        console.error('Error accessing camera:', error);
    });

// Function to capture a frame and send it to the server
function captureAndSendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/png');
    
    // Remove the prefix from the base64 string
    const base64ImageData = imageData.replace(/^data:image\/(png|jpg);base64,/, '');

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
    });
}

// Set an interval to capture and send frames every second
setInterval(captureAndSendFrame, 100); // Adjust the interval as needed (1000 ms = 1 second)
