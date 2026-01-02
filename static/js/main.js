// Global variables
let currentTab = 'dashboard';
let cameraActive = false;
let charts = {};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Enhanced Student Monitoring System Initializing...');
    
    // Update date and time
    updateDateTime();
    setInterval(updateDateTime, 1000);
    
    // Tab switching
    document.querySelectorAll('.sidebar li').forEach(tab => {
        tab.addEventListener('click', function() {
            const tabId = this.dataset.tab;
            switchTab(tabId);
        });
    });
    
    // Initialize dashboard
    loadDashboardData();
    loadSystemStatus();
    
    // File upload handling - FIXED
    setupFileUpload();
    
    // Camera controls - FIXED
    setupCameraControls();
    
    // Attendance controls
    setupAttendanceControls();
    
    // Report controls
    setupReportControls();
    
    // Student management
    setupStudentManagement();
    
    // Settings
    setupSettings();
    
    // Modal controls
    setupModals();
    
    // Load initial data
    setTimeout(() => {
        loadStudents();
        loadTodayAttendance();
    }, 1000);
    
    // Check camera status periodically
    setInterval(checkCameraStatus, 10000);
    
    console.log('✅ System initialized successfully!');
});

// Update current date and time
function updateDateTime() {
    const now = new Date();
    
    // Format date
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    };
    document.getElementById('current-date').textContent = 
        now.toLocaleDateString('en-US', options);
    
    // Format time
    document.getElementById('current-time').textContent = 
        now.toLocaleTimeString('en-US', { 
            hour12: true, 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit'
        });
}

// Switch between tabs
function switchTab(tabId) {
    // Update active tab in sidebar
    document.querySelectorAll('.sidebar li').forEach(tab => {
        tab.classList.remove('active');
    });
    
    const activeTab = document.querySelector(`[data-tab="${tabId}"]`);
    if (activeTab) {
        activeTab.classList.add('active');
    }
    
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Show selected tab content
    const tabContent = document.getElementById(tabId);
    if (tabContent) {
        tabContent.classList.add('active');
        currentTab = tabId;
        
        // Load tab-specific data
        switch(tabId) {
            case 'dashboard':
                loadDashboardData();
                break;
            case 'camera':
                loadCameraTab();
                break;
            case 'attendance':
                loadAttendance();
                break;
            case 'reports':
                loadReportFilters();
                break;
            case 'students':
                loadStudents();
                break;
            case 'settings':
                loadSettings();
                break;
        }
        
        showToast(`Switched to ${tabId.replace('_', ' ')}`, 'info');
    }
}

// // File upload setup - FIXED
// function setupFileUpload() {
//     const fileInput = document.getElementById('file-input');
//     const dropArea = document.getElementById('drop-area');
    
//     console.log('Setting up file upload...');
    
//     // Click to browse
//     if (dropArea) {
//         dropArea.addEventListener('click', function() {
//             console.log('Drop area clicked');
//             fileInput.click();
//         });
//     }
    
//     // File input change
//     if (fileInput) {
//         fileInput.addEventListener('change', handleFileSelect);
//     }
    
//     // Drag and drop events
//     if (dropArea) {
//         ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
//             dropArea.addEventListener(eventName, preventDefaults, false);
//         });
        
//         function preventDefaults(e) {
//             e.preventDefault();
//             e.stopPropagation();
//         }
        
//         ['dragenter', 'dragover'].forEach(eventName => {
//             dropArea.addEventListener(eventName, function() {
//                 dropArea.classList.add('dragover');
//             }, false);
//         });
        
//         ['dragleave', 'drop'].forEach(eventName => {
//             dropArea.addEventListener(eventName, function() {
//                 dropArea.classList.remove('dragover');
//             }, false);
//         });
        
//         dropArea.addEventListener('drop', handleDrop, false);
//     }
// }

// File upload
function setupFileUpload() {
    const dropArea = document.getElementById("drop-area");
    const fileInput = document.getElementById("file-input");
    const browseButton = document.getElementById("browse-button");
    const uploadArea = document.querySelector('.upload-area');

    // Click to browse - THREE different ways to trigger file input
    dropArea.addEventListener("click", (e) => {
        // Don't trigger if clicking on the browse button (it has its own handler)
        if (e.target !== browseButton && !browseButton.contains(e.target)) {
            fileInput.click();
        }
    });

    // Browse button click
    browseButton.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation(); // Prevent event from bubbling to dropArea
        fileInput.click();
    });

    // File selected
    fileInput.addEventListener("change", handleFileSelect);

    // Drag and drop
    ["dragenter", "dragover", "dragleave", "drop"].forEach((event) => {
        dropArea.addEventListener(event, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ["dragenter", "dragover"].forEach((event) => {
        dropArea.addEventListener(event, () => {
            dropArea.style.backgroundColor = "#f0f8ff";
            dropArea.style.borderColor = "#3a56d4";
        });
    });

    ["dragleave", "drop"].forEach((event) => {
        dropArea.addEventListener(event, () => {
            dropArea.style.backgroundColor = "";
            dropArea.style.borderColor = "#4361ee";
        });
    });

    dropArea.addEventListener("drop", handleDrop);
}

// Handle file selection from browse button
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        if (validateFile(file)) {
            processFile(file);
        }
    }
}

// Handle file drop
function handleDrop(event) {
    const file = event.dataTransfer.files[0];
    if (file) {
        if (validateFile(file)) {
            processFile(file);
        }
    }
}

// Validate file
function validateFile(file) {
    // Check if file is an image
    if (!file.type.match('image.*')) {
        showToast("Please select an image file (JPEG, PNG, etc.)", "error");
        return false;
    }

    // Check file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
        showToast("File size too large. Maximum size is 5MB.", "error");
        return false;
    }

    return true;
}

// Process file (upload and analyze)
async function processFile(file) {
    // Show preview
    const reader = new FileReader();
    reader.onload = function (e) {
        const img = document.getElementById("uploaded-image");
        img.src = e.target.result;
        img.style.display = "block";
        document.getElementById("image-placeholder").style.display = "none";
        
        // Show success message
        showToast(`Image "${file.name}" loaded successfully`, "success");
    };
    reader.readAsDataURL(file);

    // Upload and analyze
    const formData = new FormData();
    formData.append("file", file);

    try {
        showLoading("upload");
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        displayUploadResults(result); // Use the updated function
        hideLoading("upload");
    } catch (error) {
        showToast("Upload failed: " + error.message, "error");
        hideLoading("upload");
    }
}

// Camera controls setup - FIXED
function setupCameraControls() {
    console.log('Setting up camera controls...');
    
    const startBtn = document.getElementById('start-camera');
    const stopBtn = document.getElementById('stop-camera');
    const captureBtn = document.getElementById('capture-frame');
    
    if (startBtn) {
        startBtn.addEventListener('click', startCamera);
        console.log('Start camera button listener added');
    }
    
    if (stopBtn) {
        stopBtn.addEventListener('click', stopCamera);
        console.log('Stop camera button listener added');
    }
    
    if (captureBtn) {
        captureBtn.addEventListener('click', captureFrame);
        console.log('Capture frame button listener added');
    }
    
    // Check camera status on load
    checkCameraStatus();
}

// Attendance controls setup
function setupAttendanceControls() {
    const loadBtn = document.getElementById('load-attendance');
    const exportBtn = document.getElementById('export-attendance');
    
    if (loadBtn) {
        loadBtn.addEventListener('click', loadAttendance);
    }
    
    if (exportBtn) {
        exportBtn.addEventListener('click', exportAttendance);
    }
    
    // Set today's date as default
    const today = new Date().toISOString().split('T')[0];
    const dateInput = document.getElementById('attendance-date');
    if (dateInput) {
        dateInput.value = today;
    }
}

// Report controls setup
function setupReportControls() {
    const generateBtn = document.getElementById('generate-report');
    const printBtn = document.getElementById('print-report');
    
    if (generateBtn) {
        generateBtn.addEventListener('click', generateReport);
    }
    
    if (printBtn) {
        printBtn.addEventListener('click', printReport);
    }
    
    // Set default dates
    const today = new Date().toISOString().split('T')[0];
    const weekAgo = new Date();
    weekAgo.setDate(weekAgo.getDate() - 7);
    const weekAgoStr = weekAgo.toISOString().split('T')[0];
    
    const startDateInput = document.getElementById('report-start-date');
    const endDateInput = document.getElementById('report-end-date');
    
    if (startDateInput) startDateInput.value = weekAgoStr;
    if (endDateInput) endDateInput.value = today;
}

// Student management setup
function setupStudentManagement() {
    const addBtn = document.getElementById('add-student-btn');
    const searchInput = document.getElementById('search-student');
    const addForm = document.getElementById('add-student-form');
    
    if (addBtn) {
        addBtn.addEventListener('click', showAddStudentModal);
    }
    
    if (searchInput) {
        searchInput.addEventListener('input', searchStudents);
    }
    
    if (addForm) {
        addForm.addEventListener('submit', addStudent);
    }
}
// Preview student image
function previewStudentImage(event) {
    const input = event.target;
    const preview = document.getElementById('student-image-preview');
    const previewImg = document.getElementById('student-preview-img');
    
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            preview.style.display = 'block';
            previewImg.src = e.target.result;
        }
        
        reader.readAsDataURL(input.files[0]);
    } else {
        preview.style.display = 'none';
        previewImg.src = '';
    }
}
// Settings setup
function setupSettings() {
    const saveBtn = document.getElementById('save-settings');
    const resetBtn = document.getElementById('reset-settings');
    const backupBtn = document.getElementById('backup-data');
    
    if (saveBtn) {
        saveBtn.addEventListener('click', saveSettings);
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', resetSettings);
    }
    
    if (backupBtn) {
        backupBtn.addEventListener('click', backupData);
    }
    
    // Sensitivity slider
    const sensitivitySlider = document.getElementById('detection-sensitivity');
    const sensitivityValue = document.getElementById('sensitivity-value');
    
    if (sensitivitySlider && sensitivityValue) {
        sensitivitySlider.addEventListener('input', function() {
            sensitivityValue.textContent = this.value;
        });
    }
}

// Modal setup
function setupModals() {
    // Close modal buttons
    document.querySelectorAll('.close-modal').forEach(btn => {
        btn.addEventListener('click', function() {
            closeAllModals();
        });
    });
    
    // Close modal on outside click
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', function(e) {
            if (e.target === this) {
                closeAllModals();
            }
        });
    });
    
    // Close on ESC key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeAllModals();
        }
    });
}

// Handle file selection - FIXED
function handleFileSelect(event) {
    console.log('File selected');
    const file = event.target.files[0];
    if (file) {
        console.log('Processing file:', file.name);
        processImageFile(file);
    }
}

// Handle dropped files - FIXED
function handleDrop(event) {
    console.log('File dropped');
    const dt = event.dataTransfer;
    const file = dt.files[0];
    if (file) {
        console.log('Processing dropped file:', file.name);
        processImageFile(file);
    }
}

// Process image file - FIXED
async function processImageFile(file) {
    // Validate file
    if (!file.type.match('image.*')) {
        showToast('Please select an image file (JPG, PNG, GIF)', 'error');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.getElementById('image-preview');
        if (img) {
            img.src = e.target.result;
            img.style.display = 'block';
        }
    };
    reader.readAsDataURL(file);
    
    // Upload to server
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showLoading('#upload');
        console.log('Uploading file...');
        
        // Test endpoint first
        const testResponse = await fetch('/api/test_upload', {
            method: 'POST',
            body: formData
        });
        
        const testResult = await testResponse.json();
        console.log('Test upload:', testResult);
        
        // Now upload for processing
        const response = await fetch('/upload_image', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        console.log('Upload result:', result);
        
        if (result.success) {
            displayAnalysisResults(result);
            showToast('Image analyzed successfully', 'success');
        } else {
            showToast(result.error || 'Failed to analyze image', 'error');
        }
        
        hideLoading('#upload');
        
    } catch (error) {
        console.error('Error uploading image:', error);
        showToast('Error processing image: ' + error.message, 'error');
        hideLoading('#upload');
    }
}

// Start camera - FIXED
async function startCamera() {
    console.log('Start camera called');
    try {
        showLoading('#camera');
        
        // Start video feed first
        const videoFeed = document.getElementById('video-feed');
        if (videoFeed) {
            videoFeed.src = "/video_feed?" + new Date().getTime();
            const placeholder = document.querySelector('.video-placeholder');
            if (placeholder) {
                placeholder.style.display = 'none';
            }
        }
        
        // Then call API
        const response = await fetch('/start_camera', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('Start camera response:', data);
        
        if (data.success) {
            cameraActive = true;
            
            // Update buttons
            const startBtn = document.getElementById('start-camera');
            const stopBtn = document.getElementById('stop-camera');
            const captureBtn = document.getElementById('capture-frame');
            
            if (startBtn) startBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            if (captureBtn) captureBtn.disabled = false;
            
            showToast('Camera started successfully', 'success');
        } else {
            showToast(data.message || 'Failed to start camera', 'error');
        }
        
        hideLoading('#camera');
        
    } catch (error) {
        console.error('Error starting camera:', error);
        showToast('Failed to start camera: ' + error.message, 'error');
        hideLoading('#camera');
    }
}

// Stop camera - FIXED
async function stopCamera() {
    console.log('Stop camera called');
    try {
        const response = await fetch('/stop_camera', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('Stop camera response:', data);
        
        if (data.success) {
            cameraActive = false;
            
            // Stop video feed
            const videoFeed = document.getElementById('video-feed');
            if (videoFeed) {
                videoFeed.src = '';
                const placeholder = document.querySelector('.video-placeholder');
                if (placeholder) {
                    placeholder.style.display = 'flex';
                }
            }
            
            // Update buttons
            const startBtn = document.getElementById('start-camera');
            const stopBtn = document.getElementById('stop-camera');
            const captureBtn = document.getElementById('capture-frame');
            
            if (startBtn) startBtn.disabled = false;
            if (stopBtn) stopBtn.disabled = true;
            if (captureBtn) captureBtn.disabled = true;
            
            showToast('Camera stopped', 'info');
        }
        
    } catch (error) {
        console.error('Error stopping camera:', error);
        showToast('Failed to stop camera: ' + error.message, 'error');
    }
}

// Capture frame - FIXED
async function captureFrame() {
    console.log('Capture frame called');
    try {
        showLoading('#capture-frame');
        
        const response = await fetch('/capture_frame', { 
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        console.log('Capture frame response:', data);
        
        if (data.success) {
            // Display results
            displayAnalysisResults(data);
            showToast('Frame captured and analyzed', 'success');
        } else {
            showToast(data.error || 'Failed to capture frame', 'error');
        }
        
        hideLoading('#capture-frame');
        
    } catch (error) {
        console.error('Error capturing frame:', error);
        showToast('Failed to capture frame: ' + error.message, 'error');
        hideLoading('#capture-frame');
    }
}

// Display analysis results
function displayAnalysisResults(result) {
    const resultsDiv = document.getElementById('analysis-results');
    if (!resultsDiv) return;
    
    if (!result.success) {
        resultsDiv.innerHTML = `
            <div class="result-item error">
                <h4><i class="fas fa-exclamation-triangle"></i> Error</h4>
                <p>${result.error || 'Unknown error occurred'}</p>
            </div>
        `;
        return;
    }
    
    if (result.faces_detected === 0) {
        resultsDiv.innerHTML = `
            <div class="result-item warning">
                <h4><i class="fas fa-user-slash"></i> No Faces Detected</h4>
                <p>No faces were found in the image. Please try another image.</p>
            </div>
        `;
        return;
    }
    
    let html = `<div class="result-item success">
        <h4><i class="fas fa-user-check"></i> ${result.faces_detected} Face(s) Detected</h4>
        <p>Analysis completed at ${new Date(result.timestamp).toLocaleTimeString()}</p>
    </div>`;
    
    result.results.forEach((faceResult, index) => {
        const grooming = faceResult.grooming_details || {};
        const uniform = faceResult.uniform_details || {};
        const violations = faceResult.violations || [];
        
        html += `
            <div class="result-item">
                <h4><i class="fas fa-user"></i> Face ${index + 1} - ${faceResult.student_id || 'Unknown'}</h4>
                <p><strong>Status:</strong> ${faceResult.status || 'Unknown'}</p>
                <p><strong>Overall Score:</strong> ${faceResult.overall_score || 0}/10</p>
                
                <div class="mt-2">
                    <h5>Grooming Analysis:</h5>
                    <p>• Beard: ${grooming.has_beard ? 'Yes ❌' : 'No ✅'}</p>
                    <p>• Hair: ${grooming.hair_length || 'Unknown'} ${grooming.needs_haircut ? '(Needs Cut ❌)' : '(OK ✅)'}</p>
                    <p>• Neatness: ${grooming.neatness_score || 'N/A'}/5</p>
                    <p>• Face Cleanliness: ${grooming.face_clean ? 'Clean ✅' : 'Not Clean ❌'}</p>
                </div>
                
                <div class="mt-2">
                    <h5>Uniform Check:</h5>
                    <p>• Shirt: ${uniform.has_shirt ? (uniform.shirt_color || 'detected') + ' ✅' : 'Not detected ❌'}</p>
                    <p>• Pants: ${uniform.has_pants ? (uniform.pants_color || 'detected') + ' ✅' : 'Not detected ❌'}</p>
                    <p>• Footwear: ${uniform.has_shoes ? 'Shoes ✅' : uniform.has_slippers ? 'Slippers ❌' : 'Not detected ❌'}</p>
                    <p>• Tie: ${uniform.has_tie ? (uniform.tie_color || 'present') + ' ✅' : 'Not worn ❌'}</p>
                    <p>• Shirt Tucked: ${uniform.shirt_tucked === true ? 'Yes ✅' : uniform.shirt_tucked === false ? 'No ❌' : 'Unknown'}</p>
                </div>`;
        
        if (violations.length > 0) {
            html += `<div class="mt-2">
                <h5>Violations Found:</h5>`;
            
            violations.forEach(violation => {
                html += `<span class="violation-badge ${violation.severity || 'medium'}">${violation.description || 'Unknown violation'}</span>`;
            });
            
            html += `</div>`;
        }
        
        html += `</div>`;
    });
    
    // Show processed image if available
    if (result.processed_image) {
        html += `<div class="result-item">
            <h4><i class="fas fa-image"></i> Processed Image</h4>
            <img src="${result.processed_image}?t=${new Date().getTime()}" 
                 alt="Processed" style="max-width: 100%; border-radius: 8px; margin-top: 10px;">
        </div>`;
    }
    
    resultsDiv.innerHTML = html;
}

// Check camera status
async function checkCameraStatus() {
    try {
        const response = await fetch('/api/system/status');
        const data = await response.json();
        
        if (data.success) {
            const cameraStatus = document.getElementById('camera-status');
            if (cameraStatus) {
                cameraStatus.className = `status-indicator ${data.camera_available ? 'active' : 'inactive'}`;
                cameraStatus.innerHTML = `<i class="fas fa-camera"></i> Camera: ${data.camera_available ? 'Ready' : 'Not Available'}`;
            }
            
            // Update camera controls
            const startBtn = document.getElementById('start-camera');
            const stopBtn = document.getElementById('stop-camera');
            const captureBtn = document.getElementById('capture-frame');
            
            if (data.camera_available) {
                if (startBtn) startBtn.disabled = cameraActive;
                if (stopBtn) stopBtn.disabled = !cameraActive;
                if (captureBtn) captureBtn.disabled = !cameraActive;
            } else {
                if (startBtn) startBtn.disabled = true;
                if (stopBtn) stopBtn.disabled = true;
                if (captureBtn) captureBtn.disabled = true;
                showToast('Camera is not available', 'warning');
            }
        }
    } catch (error) {
        console.error('Error checking camera status:', error);
    }
}

// Load camera tab
function loadCameraTab() {
    checkCameraStatus();
    
    // Start camera if auto-start is enabled
    const autoStart = localStorage.getItem('camera_auto_start');
    if (autoStart === 'true') {
        setTimeout(startCamera, 500);
    }
}

// Add student - FIXED
async function addStudent(event) {
    event.preventDefault();
    console.log('Add student form submitted');
    
    const form = event.target;
    const formData = new FormData(form);
    
    const studentData = {
        student_id: document.getElementById('student-id')?.value || '',
        name: document.getElementById('student-name')?.value || '',
        class: document.getElementById('student-class')?.value || '',
        section: document.getElementById('student-section')?.value || '',
        roll_number: parseInt(document.getElementById('student-roll')?.value) || 0,
        uniform_color: document.getElementById('uniform-color')?.value || 'white',
        parent_phone: document.getElementById('parent-phone')?.value || '',
        parent_email: document.getElementById('parent-email')?.value || ''
    };
    
    console.log('Student data:', studentData);
    
    // Handle image upload
    const imageInput = document.getElementById('student-image');
    if (imageInput && imageInput.files.length > 0) {
        const file = imageInput.files[0];
        const reader = new FileReader();
        
        reader.onload = async function(e) {
            studentData.face_image = e.target.result;
            await submitStudentData(studentData);
        };
        
        reader.readAsDataURL(file);
    } else {
        await submitStudentData(studentData);
    }
}

// Submit student data - FIXED
async function submitStudentData(studentData) {
    try {
        showLoading('#add-student-form');
        
        const response = await fetch('/api/student/add', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(studentData)
        });
        
        const result = await response.json();
        console.log('Add student response:', result);
        
        if (result.success) {
            showToast('Student added successfully', 'success');
            closeAllModals();
            document.getElementById('add-student-form').reset();
            loadStudents();
            loadTodayAttendance();
        } else {
            showToast(result.error || 'Failed to add student', 'error');
        }
        
        hideLoading('#add-student-form');
        
    } catch (error) {
        console.error('Error adding student:', error);
        showToast('Error adding student: ' + error.message, 'error');
        hideLoading('#add-student-form');
    }
}

// Show add student modal
function showAddStudentModal() {
    const modal = document.getElementById('add-student-modal');
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

// Close all modals
function closeAllModals() {
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.remove('active');
    });
    document.body.style.overflow = 'auto';
}

// Show toast notification
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    // Get icon based on type
    let icon = 'info-circle';
    switch(type) {
        case 'success': icon = 'check-circle'; break;
        case 'error': icon = 'exclamation-circle'; break;
        case 'warning': icon = 'exclamation-triangle'; break;
    }
    
    toast.innerHTML = `
        <i class="fas fa-${icon}"></i>
        <div class="toast-content">${message}</div>
        <button class="toast-close" onclick="this.parentElement.remove()">&times;</button>
    `;
    
    toastContainer.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 5000);
}

// Show loading indicator
function showLoading(selector) {
    const element = document.querySelector(selector);
    if (element) {
        element.classList.add('loading');
    }
}

// Hide loading indicator
function hideLoading(selector) {
    const element = document.querySelector(selector);
    if (element) {
        element.classList.remove('loading');
    }
}

// Load system status
async function loadSystemStatus() {
    try {
        const response = await fetch('/api/system/status');
        const data = await response.json();
        
        if (data.success) {
            updateStatusIndicators(data);
        }
    } catch (error) {
        console.error('Error loading system status:', error);
    }
}

// Update status indicators
function updateStatusIndicators(data) {
    const modelStatus = document.getElementById('model-status');
    const cameraStatus = document.getElementById('camera-status');
    const dbStatus = document.getElementById('db-status');
    
    if (modelStatus) {
        modelStatus.className = `status-indicator ${data.models_loaded ? 'active' : 'inactive'}`;
        modelStatus.innerHTML = `<i class="fas fa-brain"></i> AI Models: ${data.models_loaded ? 'Loaded' : 'Error'}`;
    }
    
    if (cameraStatus) {
        cameraStatus.className = `status-indicator ${data.camera_available ? 'active' : 'inactive'}`;
        cameraStatus.innerHTML = `<i class="fas fa-camera"></i> Camera: ${data.camera_available ? 'Ready' : 'Not Available'}`;
    }
    
    if (dbStatus) {
        dbStatus.className = 'status-indicator active';
        dbStatus.innerHTML = `<i class="fas fa-database"></i> Database: Connected`;
    }
}

// student management

// Load students with enhanced display
async function loadStudents() {
    try {
        showLoading('students');
        
        const response = await fetch("/api/students");
        const result = await response.json();

        if (result.success) {
            displayStudentsCollegeFormat(result.students);
            updateYearStatistics(result.students);
        } else {
            showToast("Failed to load students", "error");
        }
    } catch (error) {
        console.error("Failed to load students:", error);
        showToast("Failed to load students: " + error.message, "error");
    } finally {
        hideLoading('students');
    }
}

// Display students in college format
function displayStudentsCollegeFormat(students) {
    // Initialize year arrays
    const firstYear = [];
    const secondYear = [];
    const thirdYear = [];
    const fourthYear = [];
    
    // Categorize students by year
    students.forEach(student => {
        const classInfo = student.class || '';
        
        if (classInfo.includes('I -') || classInfo.includes('First') || classInfo.includes('I Year')) {
            firstYear.push(student);
        } else if (classInfo.includes('II -') || classInfo.includes('Second') || classInfo.includes('II Year')) {
            secondYear.push(student);
        } else if (classInfo.includes('III -') || classInfo.includes('Third') || classInfo.includes('III Year')) {
            thirdYear.push(student);
        } else if (classInfo.includes('IV -') || classInfo.includes('Fourth') || classInfo.includes('IV Year')) {
            fourthYear.push(student);
        } else {
            // Default based on student ID pattern
            if (student.student_id && student.student_id.startsWith('F')) {
                firstYear.push(student);
            } else if (student.student_id && student.student_id.startsWith('S')) {
                secondYear.push(student);
            } else if (student.student_id && student.student_id.startsWith('T')) {
                thirdYear.push(student);
            } else if (student.student_id && student.student_id.startsWith('FR')) {
                fourthYear.push(student);
            } else {
                firstYear.push(student);
            }
        }
    });
    
    // Display each year's students
    displayYearStudents('first', firstYear);
    displayYearStudents('second', secondYear);
    displayYearStudents('third', thirdYear);
    displayYearStudents('fourth', fourthYear);
    
    // Update year counts
    document.getElementById('first-year-count').textContent = `${firstYear.length} students`;
    document.getElementById('second-year-count').textContent = `${secondYear.length} students`;
    document.getElementById('third-year-count').textContent = `${thirdYear.length} students`;
    document.getElementById('fourth-year-count').textContent = `${fourthYear.length} students`;
    
    // Update tab counts
    document.getElementById('tab-first-count').textContent = firstYear.length;
    document.getElementById('tab-second-count').textContent = secondYear.length;
    document.getElementById('tab-third-count').textContent = thirdYear.length;
    document.getElementById('tab-fourth-count').textContent = fourthYear.length;
}

// Display students for a specific year
function displayYearStudents(year, students) {
    const tbody = document.getElementById(`${year}-year-table`);
    
    if (students.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-state">
                <td colspan="10">
                    <i class="fas fa-user-graduate fa-2x"></i>
                    <p>No ${year} year students found</p>
                </td>
            </tr>
        `;
        return;
    }

    let html = '';
    students.forEach((student) => {
        // Extract department from class
        const department = extractDepartment(student.class);
        const yearNumber = extractYear(student.class);
        
        // Generate attendance data (in real app, this would come from API)
        const attendanceData = generateAttendanceData(student);
        
        html += `<tr data-student-id="${student.student_id}" data-year="${year}" data-department="${department}">
            <td>
                <strong>${student.roll_number || student.student_id || 'N/A'}</strong>
                <br><small class="text-muted">${student.student_id}</small>
            </td>
            <td>
                <strong>${student.name}</strong>
                <br><small class="text-muted">${getGenderIcon(student.gender)} ${student.gender || 'Unknown'}</small>
            </td>
            <td>
                <span class="dept-badge dept-${department.toLowerCase()}">
                    ${department}
                </span>
            </td>
            <td>${yearNumber}</td>
            <td>
                <div class="attendance-progress">
                    <div class="progress-bar-container">
                        <div class="progress-bar progress-present" style="width: ${attendanceData.presentPercent}%"></div>
                    </div>
                    <span class="present-badge">${attendanceData.daysPresent}</span>
                </div>
            </td>
            <td>
                <div class="attendance-progress">
                    <div class="progress-bar-container">
                        <div class="progress-bar progress-onduty" style="width: ${attendanceData.ondutyPercent}%"></div>
                    </div>
                    <span class="onduty-badge">${attendanceData.daysOnduty}</span>
                </div>
            </td>
            <td>
                <div class="attendance-progress">
                    <div class="progress-bar-container">
                        <div class="progress-bar progress-absent" style="width: ${attendanceData.absentPercent}%"></div>
                    </div>
                    <span class="absent-badge">${attendanceData.daysAbsent}</span>
                </div>
            </td>
            <td>
                <span class="gender-badge gender-${(student.gender || 'unknown').toLowerCase()}">
                    ${getGenderIcon(student.gender)} ${(student.gender || 'Unknown').toUpperCase()}
                </span>
            </td>
            <td>
                ${student.contact ? 
                    `<a href="tel:${student.contact}" class="text-primary">
                        <i class="fas fa-phone"></i> ${student.contact}
                    </a>` : 
                    '<span class="text-muted">N/A</span>'}
            </td>
            <td>
                <div class="action-buttons">
                    <button onclick="viewStudentDetails('${student.student_id}')" class="btn-action btn-view" title="View Details">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button onclick="editStudent('${student.student_id}')" class="btn-action btn-edit" title="Edit Student">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button onclick="viewAttendance('${student.student_id}')" class="btn-action btn-attendance" title="Attendance">
                        <i class="fas fa-calendar-check"></i>
                    </button>
                    <button onclick="deleteStudent('${student.student_id}')" class="btn-action btn-delete" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </td>
        </tr>`;
    });

    tbody.innerHTML = html;
}

// Update year statistics
function updateYearStatistics(students) {
    // Count students by year
    const yearCounts = { first: 0, second: 0, third: 0, fourth: 0 };
    const genderCounts = {
        first: { male: 0, female: 0 },
        second: { male: 0, female: 0 },
        third: { male: 0, female: 0 },
        fourth: { male: 0, female: 0 }
    };
    
    students.forEach(student => {
        const classInfo = student.class || '';
        let year = 'first';
        
        if (classInfo.includes('I -')) year = 'first';
        else if (classInfo.includes('II -')) year = 'second';
        else if (classInfo.includes('III -')) year = 'third';
        else if (classInfo.includes('IV -')) year = 'fourth';
        
        yearCounts[year]++;
        
        if (student.gender) {
            if (student.gender.toLowerCase() === 'male') {
                genderCounts[year].male++;
            } else if (student.gender.toLowerCase() === 'female') {
                genderCounts[year].female++;
            }
        }
    });
    
    // Update statistics cards
    document.getElementById('first-year-total').textContent = yearCounts.first;
    document.getElementById('second-year-total').textContent = yearCounts.second;
    document.getElementById('third-year-total').textContent = yearCounts.third;
    document.getElementById('fourth-year-total').textContent = yearCounts.fourth;
    
    // Update gender statistics
    updateGenderStats('first', genderCounts.first);
    updateGenderStats('second', genderCounts.second);
    updateGenderStats('third', genderCounts.third);
    updateGenderStats('fourth', genderCounts.fourth);
}

// Update gender statistics display
function updateGenderStats(year, counts) {
    const total = counts.male + counts.female;
    if (total > 0) {
        const malePercent = Math.round((counts.male / total) * 100);
        const femalePercent = Math.round((counts.female / total) * 100);
        
        document.getElementById(`${year}-gender-stats`).innerHTML = `
            <span class="gender-icon">
                <i class="fas fa-male text-primary"></i> ${counts.male} (${malePercent}%)
            </span>
            <span class="gender-icon">
                <i class="fas fa-female text-danger"></i> ${counts.female} (${femalePercent}%)
            </span>
        `;
    } else {
        document.getElementById(`${year}-gender-stats`).textContent = '';
    }
}

// Helper functions
function extractDepartment(classInfo) {
    if (!classInfo) return 'N/A';
    const match = classInfo.match(/-\s*(\w+)/);
    return match ? match[1] : classInfo;
}

function extractYear(classInfo) {
    if (!classInfo) return 'N/A';
    const match = classInfo.match(/^([IVX]+)/);
    return match ? match[1] : classInfo.split(' ')[0] || 'N/A';
}

function getGenderIcon(gender) {
    if (!gender) return '';
    return gender.toLowerCase() === 'male' ? '♂' : 
           gender.toLowerCase() === 'female' ? '♀' : '';
}

function generateAttendanceData(student) {
    // Generate realistic attendance data based on student
    const daysPresent = Math.floor(Math.random() * 30) + 70; // 70-100 days
    const daysOnduty = Math.floor(Math.random() * 5); // 0-5 days
    const daysAbsent = 100 - daysPresent - daysOnduty;
    
    return {
        daysPresent,
        daysOnduty,
        daysAbsent,
        presentPercent: daysPresent,
        ondutyPercent: daysOnduty,
        absentPercent: daysAbsent
    };
}

// Search and filter functionality
function setupSearchAndFilter() {
    const searchInput = document.getElementById('search-student');
    const filterSelect = document.getElementById('filter-department');
    
    searchInput.addEventListener('input', filterStudents);
    filterSelect.addEventListener('change', filterStudents);
}

function filterStudents() {
    const searchTerm = document.getElementById('search-student').value.toLowerCase();
    const departmentFilter = document.getElementById('filter-department').value;
    
    const allYears = ['first', 'second', 'third', 'fourth'];
    
    allYears.forEach(year => {
        const rows = document.querySelectorAll(`#${year}-year-table tr[data-student-id]`);
        let visibleCount = 0;
        
        rows.forEach(row => {
            const studentName = row.querySelector('td:nth-child(2) strong').textContent.toLowerCase();
            const rollNumber = row.querySelector('td:nth-child(1) strong').textContent.toLowerCase();
            const studentId = row.getAttribute('data-student-id').toLowerCase();
            const department = row.getAttribute('data-department');
            
            const matchesSearch = searchTerm === '' || 
                studentName.includes(searchTerm) || 
                rollNumber.includes(searchTerm) ||
                studentId.includes(searchTerm);
            
            const matchesDept = departmentFilter === '' || department === departmentFilter;
            
            if (matchesSearch && matchesDept) {
                row.style.display = '';
                visibleCount++;
                
                // Highlight search term
                if (searchTerm) {
                    highlightText(row, searchTerm);
                }
            } else {
                row.style.display = 'none';
            }
        });
        
        // Update department stats in header
        document.getElementById(`${year}-dept-stats`).textContent = 
            departmentFilter || 'All';
        
        // Update visible count
        const countElement = document.querySelector(`#${year}-year-count`);
        if (countElement) {
            const totalRows = rows.length;
            countElement.textContent = `${visibleCount} of ${totalRows} students`;
        }
        
        // Show no results message
        const tbody = document.getElementById(`${year}-year-table`);
        const emptyRow = tbody.querySelector('.empty-state');
        
        if (visibleCount === 0 && rows.length > 0) {
            if (!emptyRow) {
                const tr = document.createElement('tr');
                tr.className = 'empty-state';
                tr.innerHTML = `
                    <td colspan="10">
                        <i class="fas fa-search fa-2x"></i>
                        <p>No students found matching your search</p>
                        ${searchTerm ? `<small>Search term: "${searchTerm}"</small><br>` : ''}
                        ${departmentFilter ? `<small>Department: ${departmentFilter}</small>` : ''}
                    </td>
                `;
                tbody.appendChild(tr);
            }
        } else if (emptyRow) {
            emptyRow.remove();
        }
    });
}

function highlightText(row, searchTerm) {
    const cells = row.querySelectorAll('td:nth-child(2), td:nth-child(1)');
    cells.forEach(cell => {
        const text = cell.textContent;
        const regex = new RegExp(`(${searchTerm})`, 'gi');
        const highlighted = text.replace(regex, '<span class="highlight">$1</span>');
        if (cell.querySelector('strong')) {
            const strong = cell.querySelector('strong');
            strong.innerHTML = strong.textContent.replace(regex, '<span class="highlight">$1</span>');
        } else {
            cell.innerHTML = highlighted;
        }
    });
}

// Initialize when page loads
document.addEventListener("DOMContentLoaded", function () {
    // Setup year tabs
    setupYearTabs();
    
    // Setup search and filter
    setupSearchAndFilter();
    
    // Setup student management
    setupStudentManagement();
    
    // Load students
    loadStudents();
    
    // Setup refresh button
    document.getElementById('refresh-students').addEventListener('click', loadStudents);
});

// View student details
function viewStudentDetails(studentId) {
    // Implement view student details
    showToast(`Viewing details for student ${studentId}`, "info");
    // You can open a modal or navigate to student details page
}

// View attendance
function viewAttendance(studentId) {
    // Implement attendance view
    showToast(`Viewing attendance for student ${studentId}`, "info");
    // You can open attendance modal or navigate to attendance page
}