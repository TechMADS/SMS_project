import sqlite3
from datetime import datetime, date, timedelta

class AttendanceSystem:
    def __init__(self, db_path='database/students.db'):
        self.db_path = db_path
    
    def mark_attendance(self, student_id, status='Present', grooming_score=0, uniform_check='OK'):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        today = date.today()
        current_time = datetime.now().time()
        
        c.execute('''
            INSERT OR REPLACE INTO attendance 
            (student_id, date, status, check_in, grooming_score, uniform_check)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (student_id, today, status, current_time, grooming_score, uniform_check))
        
        conn.commit()
        conn.close()
    
    def get_daily_report(self, report_date=None):
        if report_date is None:
            report_date = date.today()
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('''
            SELECT s.name, s.student_id, s.class, 
                   a.status, a.check_in, a.check_out,
                   a.grooming_score, a.uniform_check
            FROM students s
            LEFT JOIN attendance a ON s.student_id = a.student_id AND a.date = ?
            ORDER BY s.class, s.name
        ''', (report_date,))
        
        rows = c.fetchall()
        report = [dict(row) for row in rows]
        
        conn.close()
        
        return {
            'date': report_date,
            'total_students': len(report),
            'present_count': len([r for r in report if r['status'] == 'Present']),
            'absent_count': len([r for r in report if r['status'] == 'Absent']),
            'late_count': len([r for r in report if r['check_in'] and 
                              datetime.strptime(r['check_in'], '%H:%M:%S').time() > 
                              datetime.strptime('08:30:00', '%H:%M:%S').time()]),
            'details': report
        }
    
    def generate_monthly_report(self, month=None, year=None):
        if month is None:
            month = date.today().month
        if year is None:
            year = date.today().year
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year+1}-01-01"
        else:
            end_date = f"{year}-{month+1:02d}-01"
        
        c.execute('''
            SELECT s.student_id, s.name, s.class,
                   COUNT(a.date) as days_present,
                   AVG(a.grooming_score) as avg_grooming_score,
                   GROUP_CONCAT(DISTINCT a.uniform_check) as uniform_issues
            FROM students s
            LEFT JOIN attendance a ON s.student_id = a.student_id 
                AND a.date >= ? AND a.date < ?
            GROUP BY s.student_id
        ''', (start_date, end_date))
        
        rows = c.fetchall()
        
        monthly_report = []
        for row in rows:
            report = dict(row)
            # Calculate attendance percentage
            total_days = 30  # Assuming 30 days in month
            report['attendance_percentage'] = (report['days_present'] / total_days) * 100
            
            monthly_report.append(report)
        
        conn.close()
        
        return {
            'month': month,
            'year': year,
            'reports': monthly_report
        }