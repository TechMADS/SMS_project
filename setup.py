import os
import sqlite3
import urllib.request

def setup_project():
    print("=" * 60)
    print("🎓 STUDENT MONITORING SYSTEM - SETUP")
    print("=" * 60)
    
    # Create directories
    directories = [
        'static/uploads',
        'database',
        'utils',
        'templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    # Create init files
    init_files = [
        'utils/__init__.py'
    ]
    
    for file_path in init_files:
        with open(file_path, 'w') as f:
            f.write('')
        print(f"✅ Created: {file_path}")
    
    # Initialize database
    try:
        conn = sqlite3.connect('database/students.db')
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE,
                name TEXT,
                class TEXT,
                roll_number INTEGER,
                registered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                date DATE,
                check_in TIME,
                grooming_status TEXT,
                uniform_status TEXT,
                violations TEXT
            )
        ''')
        
        # Add sample data
        c.execute("SELECT COUNT(*) FROM students")
        if c.fetchone()[0] == 0:
            sample_students = [
                ('S001', 'John Doe', '10', 1),
                ('S002', 'Jane Smith', '10', 2),
                ('S003', 'Robert Johnson', '11', 1),
                ('S004', 'Emily Davis', '11', 3),
                ('S005', 'Michael Wilson', '12', 5),
            ]
            
            for student in sample_students:
                c.execute('INSERT INTO students (student_id, name, class, roll_number) VALUES (?, ?, ?, ?)', student)
            
            print("✅ Added 5 sample students")
        
        conn.commit()
        conn.close()
        print("✅ Database initialized")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("\n📋 NEXT STEPS:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Place your 'grooming_model_with_gender.pth' in project root")
    print("3. Run the system: python app.py")
    print("4. Open: http://127.0.0.1:5000")
    print("\n🌟 ALL FEATURES READY:")
    print("   • Photo upload with detailed analysis")
    print("   • Live camera with start/stop/capture")
    print("   • Add/delete students")
    print("   • Beard/Tie/Shoes detection")
    print("   • Attendance tracking")
    print("   • Responsive design")
    print("=" * 60)

if __name__ == '__main__':
    setup_project()