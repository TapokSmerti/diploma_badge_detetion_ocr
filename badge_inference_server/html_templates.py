# html_templates.py
"""
HTML шаблоны для веб-интерфейса
"""

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Badge Detection - Remote Inference</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 20px;
            font-size: 2.5em;
            background: linear-gradient(135deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .main-content {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .video-panel {
            flex: 2;
            min-width: 640px;
            background: #0f0f1a;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .video-container {
            position: relative;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .video-container video,
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        
        canvas {
            display: none;
        }
        
        .controls-panel {
            flex: 1;
            min-width: 280px;
            background: #0f0f1a;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .control-group {
            margin-bottom: 25px;
        }
        
        .control-group label {
            display: block;
            margin-bottom: 8px;
            color: #aaa;
            font-size: 0.9em;
        }
        
        button {
            background: linear-gradient(135deg, #00ff88, #00cc66);
            color: #1a1a2e;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            border-radius: 8px;
            margin: 5px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,255,136,0.3);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button.danger {
            background: linear-gradient(135deg, #ff4444, #cc0000);
        }
        
        select, input {
            width: 100%;
            padding: 10px;
            background: #1a1a2e;
            border: 1px solid #333;
            color: white;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .stats {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #333;
        }
        
        .stat-item:last-child {
            border-bottom: none;
        }
        
        .stat-label {
            color: #aaa;
        }
        
        .stat-value {
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status.online {
            background: #00ff88;
            box-shadow: 0 0 10px #00ff88;
        }
        
        .status.offline {
            background: #ff4444;
        }
        
        .fps-value {
            color: #00ff88;
            font-size: 1.5em;
        }
        
        @media (max-width: 768px) {
            .video-panel {
                min-width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Badge Detection System</h1>
        
        <div class="main-content">
            <div class="video-panel">
                <div class="video-container">
                    <video id="webcam" autoplay playsinline></video>
                    <img id="result" style="display:none;">
                    <canvas id="canvas"></canvas>
                </div>
            </div>
            
            <div class="controls-panel">
                <div class="control-group">
                    <h3>🎮 Управление</h3>
                    <button onclick="startDetection()">▶ Запуск</button>
                    <button onclick="stopDetection()" class="danger">⏹ Стоп</button>
                    <button onclick="takeScreenshot()">📸 Скриншот</button>
                </div>
                
                <div class="control-group">
                    <label>📷 Камера</label>
                    <select id="camera"></select>
                </div>
                
                <div class="control-group">
                    <label>⚡ Частота (мс между кадрами)</label>
                    <input type="range" id="interval" min="30" max="200" value="50" step="10">
                    <span id="interval-value">50</span> мс
                </div>
                
                <div class="stats">
                    <h3>📊 Статистика</h3>
                    <div class="stat-item">
                        <span class="stat-label">Статус:</span>
                        <span class="stat-value">
                            <span class="status offline" id="status-led"></span>
                            <span id="status-text">Отключен</span>
                        </span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">FPS (клиент/сервер):</span>
                        <span class="stat-value fps-value" id="fps">0 / 0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Найдено бейджей:</span>
                        <span class="stat-value" id="badge-count">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Время инференса:</span>
                        <span class="stat-value" id="inf-time">0</span> мс
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Модель:</span>
                        <span class="stat-value" id="model-info">-</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const video = document.getElementById('webcam');
        const resultImg = document.getElementById('result');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        let isDetecting = false;
        let stream = null;
        let frameTimer = null;
        let fpsCounter = 0;
        let lastFpsTime = Date.now();
        let clientFps = 0;
        
        // Получение списка камер
        async function getCameras() {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');
            const select = document.getElementById('camera');
            select.innerHTML = '';
            videoDevices.forEach((device, idx) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${idx}`;
                select.appendChild(option);
            });
        }
        
        // Запуск камеры
        async function startCamera() {
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    deviceId: document.getElementById('camera').value ? { exact: document.getElementById('camera').value } : undefined
                }
            };
            
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            await video.play();
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }
        
        // Отправка кадра на сервер
        async function sendFrame() {
            if (!isDetecting || video.videoWidth === 0) return;
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
            const startTime = performance.now();
            const formData = new FormData();
            formData.append('file', blob);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const endTime = performance.now();
                const inferenceTime = (endTime - startTime).toFixed(1);
                document.getElementById('inf-time').innerText = inferenceTime;
                
                if (response.ok) {
                    const resultBlob = await response.blob();
                    const url = URL.createObjectURL(resultBlob);
                    resultImg.src = url;
                    resultImg.style.display = 'block';
                    video.style.display = 'none';
                    
                    // Обновление статистики из заголовков
                    const fps = response.headers.get('X-FPS');
                    const count = response.headers.get('X-Detections');
                    if (fps) {
                        document.getElementById('fps').innerHTML = `${clientFps.toFixed(1)} / ${parseFloat(fps).toFixed(1)}`;
                    }
                    if (count) document.getElementById('badge-count').innerText = count;
                    
                    // Обновление статуса
                    document.getElementById('status-led').className = 'status online';
                    document.getElementById('status-text').innerText = 'Работает';
                    
                    setTimeout(() => URL.revokeObjectURL(url), 100);
                }
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('status-led').className = 'status offline';
                document.getElementById('status-text').innerText = 'Ошибка';
            }
            
            // Подсчет клиентского FPS
            fpsCounter++;
            const now = Date.now();
            if (now - lastFpsTime >= 1000) {
                clientFps = fpsCounter * 1000 / (now - lastFpsTime);
                fpsCounter = 0;
                lastFpsTime = now;
                
                const currentFps = document.getElementById('fps').innerText;
                document.getElementById('fps').innerHTML = `${clientFps.toFixed(1)} / ${currentFps.split('/')[1] || '0'}`;
            }
            
            const interval = parseInt(document.getElementById('interval').value);
            frameTimer = setTimeout(sendFrame, interval);
        }
        
        function startDetection() {
            if (isDetecting) return;
            isDetecting = true;
            resultImg.style.display = 'none';
            video.style.display = 'block';
            sendFrame();
        }
        
        function stopDetection() {
            isDetecting = false;
            if (frameTimer) clearTimeout(frameTimer);
            resultImg.style.display = 'none';
            video.style.display = 'block';
            document.getElementById('status-led').className = 'status offline';
            document.getElementById('status-text').innerText = 'Остановлен';
        }
        
        async function takeScreenshot() {
            if (resultImg.src) {
                const link = document.createElement('a');
                link.download = `badge_${Date.now()}.jpg`;
                link.href = resultImg.src;
                link.click();
            }
        }
        
        // Обновление значения слайдера
        const intervalSlider = document.getElementById('interval');
        const intervalValue = document.getElementById('interval-value');
        intervalSlider.addEventListener('input', () => {
            intervalValue.innerText = intervalSlider.value;
        });
        
        // Инициализация
        getCameras();
        startCamera();
        
        // Получение информации о сервере
        fetch('/info').then(r => r.json()).then(data => {
            document.getElementById('model-info').innerText = `${data.model_type.toUpperCase()} on ${data.device.toUpperCase()}`;
        });
        
        // Переключение камеры
        document.getElementById('camera').addEventListener('change', () => {
            if (!isDetecting) {
                startCamera();
            }
        });
    </script>
</body>
</html>
"""

# Можно добавить дополнительные HTML шаблоны
HTML_SIMPLE = """
<!DOCTYPE html>
<html>
<head>
    <title>Badge Detection</title>
</head>
<body>
    <h1>Badge Detection</h1>
    <video id="webcam" autoplay></video>
    <img id="result">
    <button onclick="start()">Start</button>
    <button onclick="stop()">Stop</button>
    
    <script>
        const video = document.getElementById('webcam');
        const result = document.getElementById('result');
        let detecting = false;
        let timer = null;
        
        navigator.mediaDevices.getUserMedia({video: true}).then(stream => {
            video.srcObject = stream;
        });
        
        async function sendFrame() {
            if (!detecting) return;
            
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append('file', blob);
                const response = await fetch('/detect', {method: 'POST', body: formData});
                result.src = URL.createObjectURL(await response.blob());
                timer = setTimeout(sendFrame, 50);
            }, 'image/jpeg');
        }
        
        function start() { detecting = true; sendFrame(); }
        function stop() { detecting = false; clearTimeout(timer); }
    </script>
</body>
</html>
"""