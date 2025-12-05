// API Base URL
const API_BASE = '';

// Tab Management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// Form Handlers
document.getElementById('distance-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    await makeRequest('/predict/distance', {
        rssi: parseFloat(document.getElementById('distance-rssi').value),
        lqi: parseFloat(document.getElementById('distance-lqi').value),
        throughput: parseFloat(document.getElementById('distance-throughput').value)
    }, 'distance-result');
});

document.getElementById('human-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = {
        rssi: parseFloat(document.getElementById('human-rssi').value),
        lqi: parseFloat(document.getElementById('human-lqi').value),
        throughput: parseFloat(document.getElementById('human-throughput').value)
    };
    const timestamp = document.getElementById('human-timestamp').value;
    if (timestamp) data.timestamp = parseFloat(timestamp);
    
    await makeRequest('/detect/human-presence', data, 'human-result');
});

document.getElementById('location-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = {
        rssi: parseFloat(document.getElementById('location-rssi').value),
        lqi: parseFloat(document.getElementById('location-lqi').value),
        throughput: parseFloat(document.getElementById('location-throughput').value)
    };
    const rssiStddev = document.getElementById('location-rssi-stddev').value;
    if (rssiStddev) data.rssi_stddev = parseFloat(rssiStddev);
    
    await makeRequest('/classify/device-location', data, 'location-result');
});

document.getElementById('quality-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    await makeRequest('/score/signal-quality', {
        rssi: parseFloat(document.getElementById('quality-rssi').value),
        lqi: parseFloat(document.getElementById('quality-lqi').value),
        throughput: parseFloat(document.getElementById('quality-throughput').value)
    }, 'quality-result');
});

document.getElementById('anomaly-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const data = {
        rssi: parseFloat(document.getElementById('anomaly-rssi').value),
        lqi: parseFloat(document.getElementById('anomaly-lqi').value),
        throughput: parseFloat(document.getElementById('anomaly-throughput').value)
    };
    const timestamp = document.getElementById('anomaly-timestamp').value;
    if (timestamp) data.timestamp = parseFloat(timestamp);
    
    await makeRequest('/detect/anomaly', data, 'anomaly-result');
});

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const history = [];
    const historyItems = document.querySelectorAll('.history-item');
    
    historyItems.forEach(item => {
        const timestamp = item.querySelector('.history-timestamp').value;
        const rssi = item.querySelector('.history-rssi').value;
        const lqi = item.querySelector('.history-lqi').value;
        const throughput = item.querySelector('.history-throughput').value;
        
        if (timestamp && rssi && lqi && throughput) {
            history.push({
                timestamp: parseFloat(timestamp),
                rssi: parseFloat(rssi),
                lqi: parseFloat(lqi),
                throughput: parseFloat(throughput)
            });
        }
    });
    
    if (history.length < 2) {
        showError('prediction-result', 'At least 2 historical data points required!');
        return;
    }
    
    await makeRequest('/predict/signal-quality', {
        history: history,
        future_steps: parseInt(document.getElementById('future-steps').value)
    }, 'prediction-result');
});

// API Request Function
async function makeRequest(endpoint, data, resultId) {
    const resultDiv = document.getElementById(resultId);
    resultDiv.classList.remove('hidden');
    resultDiv.innerHTML = '<div class="spinner"></div> Processing...';
    
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displayResult(resultId, result, endpoint);
    } catch (error) {
        showError(resultId, `Error: ${error.message}`);
    }
}

// Display Result
function displayResult(resultId, data, endpoint) {
    const resultDiv = document.getElementById(resultId);
    let html = '';
    let resultClass = 'success';
    
    if (endpoint.includes('distance')) {
        html = `
            <div class="result ${resultClass}">
                <h3><i class="fas fa-ruler"></i> Distance Estimation Result</h3>
                <div class="result-content">
                    <div class="result-item">
                        <strong>Distance:</strong> ${data.distance} ${data.unit || 'meters'}
                    </div>
                    <div class="result-item">
                        <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        `;
    } else if (endpoint.includes('human-presence')) {
        resultClass = data.has_human ? 'warning' : 'success';
        html = `
            <div class="result ${resultClass}">
                <h3><i class="fas fa-user"></i> Human Presence Detection Result</h3>
                <div class="result-content">
                    <div class="result-item">
                        <strong>Human Present?</strong> 
                        <span class="badge ${data.has_human ? 'badge-warning' : 'badge-success'}">
                            ${data.has_human ? 'YES' : 'NO'}
                        </span>
                    </div>
                    <div class="result-item">
                        <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
        `;
    } else if (endpoint.includes('device-location')) {
        html = `
            <div class="result ${resultClass}">
                <h3><i class="fas fa-map-marker-alt"></i> Location Classification Result</h3>
                <div class="result-content">
                    <div class="result-item">
                        <strong>Location:</strong> 
                        <span class="badge badge-info">${data.location.toUpperCase()}</span>
                    </div>
                    <div class="result-item">
                        <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%
                    </div>
                    ${data.possible_locations ? `
                        <div class="result-item">
                            <strong>Probabilities:</strong><br>
                            ${Object.entries(data.possible_locations).map(([loc, prob]) => 
                                `• ${loc}: ${(prob * 100).toFixed(1)}%`
                            ).join('<br>')}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    } else if (endpoint.includes('signal-quality') && endpoint.includes('score')) {
        const gradeColors = {
            'excellent': 'badge-success',
            'good': 'badge-info',
            'fair': 'badge-warning',
            'poor': 'badge-danger'
        };
        html = `
            <div class="result ${resultClass}">
                <h3><i class="fas fa-star"></i> Signal Quality Score</h3>
                <div class="result-content">
                    <div class="result-item">
                        <strong>Overall Score:</strong> 
                        <span class="badge ${gradeColors[data.grade]}">${data.quality_score.toFixed(1)}/100</span>
                    </div>
                    <div class="result-item">
                        <strong>Grade:</strong> 
                        <span class="badge ${gradeColors[data.grade]}">${data.grade.toUpperCase()}</span>
                    </div>
                    ${data.breakdown ? `
                        <div class="result-item">
                            <strong>Breakdown:</strong><br>
                            • RSSI Score: ${data.breakdown.rssi_score.toFixed(1)}/100<br>
                            • LQI Score: ${data.breakdown.lqi_score.toFixed(1)}/100<br>
                            • THROUGHPUT Score: ${data.breakdown.throughput_score.toFixed(1)}/100
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    } else if (endpoint.includes('anomaly')) {
        resultClass = data.is_anomaly ? 'error' : 'success';
        html = `
            <div class="result ${resultClass}">
                <h3><i class="fas fa-exclamation-triangle"></i> Anomaly Detection Result</h3>
                <div class="result-content">
                    <div class="result-item">
                        <strong>Anomaly Detected?</strong> 
                        <span class="badge ${data.is_anomaly ? 'badge-danger' : 'badge-success'}">
                            ${data.is_anomaly ? 'YES' : 'NO'}
                        </span>
                    </div>
                    <div class="result-item">
                        <strong>Anomaly Score:</strong> ${(data.anomaly_score * 100).toFixed(1)}%
                    </div>
                    ${data.reason ? `
                        <div class="result-item">
                            <strong>Reason:</strong> ${data.reason}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    } else if (endpoint.includes('signal-quality') && endpoint.includes('predict')) {
        html = `
            <div class="result ${resultClass}">
                <h3><i class="fas fa-chart-line"></i> Signal Prediction Results</h3>
                <div class="result-content">
                    ${data.predictions.map((pred, idx) => `
                        <div class="result-item">
                            <strong>Prediction ${idx + 1} (t=${pred.timestamp}):</strong><br>
                            • RSSI: ${pred.rssi.toFixed(2)} dBm<br>
                            • LQI: ${pred.lqi.toFixed(2)}<br>
                            • THROUGHPUT: ${pred.throughput.toFixed(2)} bytes/s
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    resultDiv.innerHTML = html;
}

// Show Error
function showError(resultId, message) {
    const resultDiv = document.getElementById(resultId);
    resultDiv.classList.remove('hidden');
    resultDiv.innerHTML = `
        <div class="result error">
            <h3><i class="fas fa-exclamation-circle"></i> Error</h3>
            <div class="result-content">
                <p>${message}</p>
            </div>
        </div>
    `;
}

// Add History Item
function addHistoryItem() {
    const container = document.getElementById('history-inputs');
    const newItem = document.createElement('div');
    newItem.className = 'history-item';
    newItem.innerHTML = `
        <input type="number" placeholder="Timestamp" step="0.1" class="history-timestamp">
        <input type="number" placeholder="RSSI" step="0.1" class="history-rssi">
        <input type="number" placeholder="LQI" step="0.1" class="history-lqi">
        <input type="number" placeholder="THROUGHPUT" step="0.1" class="history-throughput">
    `;
    container.appendChild(newItem);
}

