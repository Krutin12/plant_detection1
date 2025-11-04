"""
Detection History Module
=======================
Manages disease detection history and analysis.
"""

import json
import csv
from datetime import datetime
from pathlib import Path

class DetectionHistoryModule:
    """Manages detection history and related analytics"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / 'detection_history'
        self.data_path.mkdir(exist_ok=True)
        
        self.detection_records = []
        self.load_detection_history()
        
    def load_detection_history(self):
        """Load existing detection history from file"""
        history_file = self.data_path / 'detection_history.json'
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for record in data:
                        # Convert timestamp string back to datetime
                        record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                        self.detection_records.append(record)
                print(f"Loaded {len(self.detection_records)} detection records")
            except Exception as e:
                print(f"Error loading detection history: {e}")
    
    def save_detection_history(self):
        """Save detection history to file"""
        try:
            history_file = self.data_path / 'detection_history.json'
            with open(history_file, 'w') as f:
                # Convert datetime to string for JSON serialization
                data = [{**record, 'timestamp': record['timestamp'].isoformat()} 
                       for record in self.detection_records]
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving detection history: {e}")
    
    def add_detection(self, detection_record):
        """Add new detection record"""
        detection_record['id'] = len(self.detection_records) + 1
        if 'timestamp' not in detection_record:
            detection_record['timestamp'] = datetime.now()
        
        self.detection_records.append(detection_record)
        self.save_detection_history()
        
        print(f"Detection record added with ID: {detection_record['id']}")
    
    def view_detection_history(self):
        """Display detection history"""
        if not self.detection_records:
            print("No detection records found.")
            return
        
        print(f"\n--- DETECTION HISTORY ({len(self.detection_records)} records) ---")
        
        # Show recent records first
        for record in reversed(self.detection_records[-10:]):  # Show last 10
            print(f"ID: {record['id']} | Date: {record['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            print(f"Disease: {record.get('predicted_disease', 'Unknown')}")
            print(f"Confidence: {record.get('confidence', 0):.1%}")
            
            if 'image_path' in record:
                import os
                print(f"Image: {os.path.basename(record['image_path'])}")
            
            print("-" * 50)
    
    def get_detection_summary(self):
        """Get summary statistics of detections"""
        if not self.detection_records:
            return {}
        
        total_detections = len(self.detection_records)
        
        # Disease distribution
        diseases = [record.get('predicted_disease', 'Unknown') for record in self.detection_records]
        disease_counts = {}
        for disease in diseases:
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        # Confidence statistics
        confidences = [record.get('confidence', 0) for record in self.detection_records]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Date range
        if self.detection_records:
            dates = [record['timestamp'] for record in self.detection_records]
            date_range = f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"
        else:
            date_range = "No records"
        
        return {
            'total_detections': total_detections,
            'disease_distribution': disease_counts,
            'average_confidence': avg_confidence,
            'date_range': date_range
        }
    
    def find_similar_cases(self, disease_name):
        """Find similar disease cases"""
        similar_cases = [
            record for record in self.detection_records 
            if record.get('predicted_disease', '').lower() == disease_name.lower()
        ]
        
        return similar_cases
    
    def search_by_date(self, start_date, end_date):
        """Search detections by date range"""
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        filtered_records = [
            record for record in self.detection_records 
            if start_date <= record['timestamp'] <= end_date
        ]
        
        return filtered_records
    
    def export_to_csv(self):
        """Export detection history to CSV"""
        if not self.detection_records:
            print("No records to export.")
            return
        
        csv_file = self.data_path / f'detection_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['id', 'timestamp', 'predicted_disease', 'confidence', 'image_path']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in self.detection_records:
                    row = {
                        'id': record.get('id', ''),
                        'timestamp': record['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'predicted_disease': record.get('predicted_disease', ''),
                        'confidence': record.get('confidence', 0),
                        'image_path': record.get('image_path', '')
                    }
                    writer.writerow(row)
            
            print(f"Detection history exported to: {csv_file}")
            return csv_file
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return None
    
    def delete_record(self, record_id):
        """Delete a detection record"""
        for i, record in enumerate(self.detection_records):
            if record.get('id') == record_id:
                deleted_record = self.detection_records.pop(i)
                self.save_detection_history()
                print(f"Deleted record: {deleted_record.get('predicted_disease', 'Unknown')}")
                return True
        
        print(f"Record with ID {record_id} not found.")
        return False
    
    def run(self):
        """Main detection history interface"""
        while True:
            print(f"\n--- DETECTION HISTORY MODULE ---")
            print("1. View Detection History")
            print("2. Search Records")
            print("3. View Summary Statistics")
            print("4. Export to CSV")
            print("5. Delete Record")
            print("6. Return to Main Menu")
            
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                self.view_detection_history()
                
            elif choice == '2':
                self.search_interface()
                
            elif choice == '3':
                self.display_summary()
                
            elif choice == '4':
                self.export_to_csv()
                
            elif choice == '5':
                self.delete_interface()
                
            elif choice == '6':
                break
                
            else:
                print("Invalid choice. Please select 1-6.")
    
    def search_interface(self):
        """Interactive search interface"""
        print("\n--- SEARCH DETECTION RECORDS ---")
        print("1. Search by Disease Name")
        print("2. Search by Date Range")
        print("3. Search by Confidence Level")
        
        search_choice = input("Select search type (1-3): ").strip()
        
        if search_choice == '1':
            disease_name = input("Enter disease name to search: ").strip()
            results = self.find_similar_cases(disease_name)
            
            if results:
                print(f"\nFound {len(results)} records for '{disease_name}':")
                for record in results:
                    print(f"ID: {record['id']} | Date: {record['timestamp'].strftime('%Y-%m-%d')}")
                    print(f"Confidence: {record.get('confidence', 0):.1%}")
            else:
                print(f"No records found for '{disease_name}'")
        
        elif search_choice == '2':
            try:
                start_date = input("Enter start date (YYYY-MM-DD): ").strip()
                end_date = input("Enter end date (YYYY-MM-DD): ").strip()
                
                results = self.search_by_date(start_date, end_date)
                
                if results:
                    print(f"\nFound {len(results)} records between {start_date} and {end_date}:")
                    for record in results:
                        print(f"ID: {record['id']} | Date: {record['timestamp'].strftime('%Y-%m-%d')}")
                        print(f"Disease: {record.get('predicted_disease', 'Unknown')}")
                else:
                    print(f"No records found in the specified date range")
                    
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD")
        
        elif search_choice == '3':
            try:
                min_confidence = float(input("Enter minimum confidence (0.0-1.0): "))
                
                results = [
                    record for record in self.detection_records 
                    if record.get('confidence', 0) >= min_confidence
                ]
                
                if results:
                    print(f"\nFound {len(results)} records with confidence >= {min_confidence:.1%}:")
                    for record in results:
                        print(f"ID: {record['id']} | Confidence: {record.get('confidence', 0):.1%}")
                        print(f"Disease: {record.get('predicted_disease', 'Unknown')}")
                else:
                    print(f"No records found with confidence >= {min_confidence:.1%}")
                    
            except ValueError:
                print("Invalid confidence value. Please enter a number between 0.0 and 1.0")
    
    def display_summary(self):
        """Display summary statistics"""
        summary = self.get_detection_summary()
        
        if not summary:
            print("No detection records available.")
            return
        
        print(f"\n--- DETECTION SUMMARY STATISTICS ---")
        print(f"Total Detections: {summary['total_detections']}")
        print(f"Average Confidence: {summary['average_confidence']:.1%}")
        print(f"Date Range: {summary['date_range']}")
        
        print(f"\nDisease Distribution:")
        for disease, count in sorted(summary['disease_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
            percentage = (count / summary['total_detections']) * 100
            print(f"  {disease}: {count} cases ({percentage:.1f}%)")
    
    def delete_interface(self):
        """Interactive delete interface"""
        if not self.detection_records:
            print("No records to delete.")
            return
        
        print(f"\n--- DELETE DETECTION RECORD ---")
        print("Recent records:")
        
        for record in self.detection_records[-5:]:  # Show last 5
            print(f"ID: {record['id']} | {record['timestamp'].strftime('%Y-%m-%d')} | {record.get('predicted_disease', 'Unknown')}")
        
        try:
            record_id = int(input("Enter record ID to delete: "))
            
            confirm = input(f"Are you sure you want to delete record {record_id}? (y/N): ").lower()
            if confirm == 'y':
                self.delete_record(record_id)
            else:
                print("Deletion cancelled.")
                
        except ValueError:
            print("Invalid record ID.")