import requests
import os

def test_endpoints():
    base_url = "http://127.0.0.1:5000"
    
    print("Testing endpoints...")
    
    # Test 1: Upload endpoint
    print("\n1. Testing upload endpoint...")
    test_file = {'file': open('test_image.jpg', 'rb')} if os.path.exists('test_image.jpg') else None
    
    if test_file:
        response = requests.post(f"{base_url}/upload_image", files=test_file)
        print(f"Upload response: {response.status_code}")
        print(f"Upload result: {response.json()}")
    else:
        print("No test image found. Create a 'test_image.jpg' for testing.")
    
    # Test 2: Camera endpoints
    print("\n2. Testing camera endpoints...")
    response = requests.post(f"{base_url}/start_camera")
    print(f"Start camera: {response.status_code}, {response.json()}")
    
    response = requests.post(f"{base_url}/stop_camera")
    print(f"Stop camera: {response.status_code}, {response.json()}")
    
    # Test 3: Student API
    print("\n3. Testing student API...")
    student_data = {
        "student_id": "TEST001",
        "name": "Test Student",
        "class": "10",
        "section": "A",
        "roll_number": 99,
        "uniform_color": "white"
    }
    
    response = requests.post(f"{base_url}/api/student/add", json=student_data)
    print(f"Add student: {response.status_code}, {response.json()}")
    
    # Test 4: System status
    print("\n4. Testing system status...")
    response = requests.get(f"{base_url}/api/system/status")
    print(f"System status: {response.status_code}, {response.json()}")

if __name__ == "__main__":
    test_endpoints()