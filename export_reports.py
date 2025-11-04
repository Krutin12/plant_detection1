"""
Export Reports Module
====================
Handles comprehensive data export and report generation.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class ExportReportsModule:
    """Comprehensive report generation and data export"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.exports_path = self.base_path / 'exports'
        self.exports_path.mkdir(exist_ok=True)
        
        # Create subdirectories for different export types
        (self.exports_path / 'csv').mkdir(exist_ok=True)
        (self.exports_path / 'json').mkdir(exist_ok=True)
        (self.exports_path / 'charts').mkdir(exist_ok=True)
        (self.exports_path / 'reports').mkdir(exist_ok=True)
    
    def export_detection_history(self):
        """Export detection history in multiple formats"""
        print("\n--- EXPORT DETECTION HISTORY ---")
        print("1. CSV Format")
        print("2. JSON Format") 
        print("3. Summary Report")
        
        choice = input("Select export format (1-3): ").strip()
        
        if choice == '1':
            return self.export_detection_csv()
        elif choice == '2':
            return self.export_detection_json()
        elif choice == '3':
            return self.export_detection_summary()
        else:
            print("Invalid choice.")
            return None
    
    def export_detection_csv(self):
        """Export detection history to CSV"""
        detection_file = self.base_path / 'detection_history' / 'detection_history.json'
        
        if not detection_file.exists():
            print("No detection history found.")
            return None
        
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                print("No detection records to export.")
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self.exports_path / 'csv' / f'detection_history_{timestamp}.csv'
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['id', 'timestamp', 'predicted_disease', 'confidence', 'image_path']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in data:
                    row = {
                        'id': record.get('id', ''),
                        'timestamp': record.get('timestamp', ''),
                        'predicted_disease': record.get('predicted_disease', ''),
                        'confidence': record.get('confidence', 0),
                        'image_path': record.get('image_path', '')
                    }
                    writer.writerow(row)
            
            print(f"Detection history exported to CSV: {csv_file}")
            return csv_file
            
        except Exception as e:
            print(f"Error exporting detection CSV: {e}")
            return None
    
    def export_detection_json(self):
        """Export detection history to JSON"""
        detection_file = self.base_path / 'detection_history' / 'detection_history.json'
        
        if not detection_file.exists():
            print("No detection history found.")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = self.exports_path / 'json' / f'detection_export_{timestamp}.json'
            
            # Copy the file with additional metadata
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_records': len(data),
                'data': data
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Detection history exported to JSON: {export_file}")
            return export_file
            
        except Exception as e:
            print(f"Error exporting detection JSON: {e}")
            return None
    
    def export_detection_summary(self):
        """Export detection summary report"""
        detection_file = self.base_path / 'detection_history' / 'detection_history.json'
        
        if not detection_file.exists():
            print("No detection history found.")
            return None
        
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.exports_path / 'reports' / f'detection_summary_{timestamp}.txt'
            
            with open(report_file, 'w') as f:
                f.write("DETECTION HISTORY SUMMARY REPORT\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Records: {len(data)}\n\n")
                
                if data:
                    # Disease distribution
                    diseases = [record.get('predicted_disease', 'Unknown') for record in data]
                    disease_counts = {}
                    for disease in diseases:
                        disease_counts[disease] = disease_counts.get(disease, 0) + 1
                    
                    f.write("DISEASE DISTRIBUTION:\n")
                    f.write("-" * 20 + "\n")
                    for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(diseases)) * 100
                        f.write(f"{disease}: {count} ({percentage:.1f}%)\n")
                    
                    # Confidence statistics
                    confidences = [record.get('confidence', 0) for record in data]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    f.write(f"\nCONFIDENCE STATISTICS:\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Average Confidence: {avg_confidence:.1%}\n")
                    f.write(f"Highest Confidence: {max(confidences):.1%}\n")
                    f.write(f"Lowest Confidence: {min(confidences):.1%}\n")
                    
                    # Date range
                    timestamps = [record.get('timestamp', '') for record in data if record.get('timestamp')]
                    if timestamps:
                        f.write(f"\nDATE RANGE:\n")
                        f.write("-" * 20 + "\n")
                        f.write(f"From: {min(timestamps)}\n")
                        f.write(f"To: {max(timestamps)}\n")
            
            print(f"Detection summary report exported: {report_file}")
            return report_file
            
        except Exception as e:
            print(f"Error exporting detection summary: {e}")
            return None
    
    def export_treatment_history(self):
        """Export treatment history"""
        print("\n--- EXPORT TREATMENT HISTORY ---")
        print("1. CSV Format")
        print("2. JSON Format")
        print("3. Treatment Effectiveness Report")
        
        choice = input("Select export format (1-3): ").strip()
        
        if choice == '1':
            return self.export_treatment_csv()
        elif choice == '2':
            return self.export_treatment_json()
        elif choice == '3':
            return self.export_treatment_effectiveness()
        else:
            print("Invalid choice.")
            return None
    
    def export_treatment_csv(self):
        """Export treatment history to CSV"""
        treatment_file = self.base_path / 'treatment_history' / 'treatment_history.json'
        
        if not treatment_file.exists():
            print("No treatment history found.")
            return None
        
        try:
            with open(treatment_file, 'r') as f:
                data = json.load(f)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = self.exports_path / 'csv' / f'treatment_history_{timestamp}.csv'
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['id', 'timestamp', 'disease', 'treatment_type', 'status', 'planned_date']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in data:
                    row = {
                        'id': record.get('id', ''),
                        'timestamp': record.get('timestamp', ''),
                        'disease': record.get('disease', ''),
                        'treatment_type': record.get('treatment_type', ''),
                        'status': record.get('status', ''),
                        'planned_date': record.get('planned_date', '')
                    }
                    writer.writerow(row)
            
            print(f"Treatment history exported to CSV: {csv_file}")
            return csv_file
            
        except Exception as e:
            print(f"Error exporting treatment CSV: {e}")
            return None
    
    def export_treatment_json(self):
        """Export treatment history to JSON"""
        treatment_file = self.base_path / 'treatment_history' / 'treatment_history.json'
        
        if not treatment_file.exists():
            print("No treatment history found.")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = self.exports_path / 'json' / f'treatment_export_{timestamp}.json'
            
            with open(treatment_file, 'r') as f:
                data = json.load(f)
            
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_records': len(data),
                'data': data
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"Treatment history exported to JSON: {export_file}")
            return export_file
            
        except Exception as e:
            print(f"Error exporting treatment JSON: {e}")
            return None
    
    def export_treatment_effectiveness(self):
        """Export treatment effectiveness report"""
        treatment_file = self.base_path / 'treatment_history' / 'treatment_history.json'
        
        if not treatment_file.exists():
            print("No treatment history found.")
            return None
        
        try:
            with open(treatment_file, 'r') as f:
                data = json.load(f)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.exports_path / 'reports' / f'treatment_effectiveness_{timestamp}.txt'
            
            with open(report_file, 'w') as f:
                f.write("TREATMENT EFFECTIVENESS REPORT\n")
                f.write("=" * 35 + "\n\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Treatments: {len(data)}\n\n")
                
                if data:
                    # Status distribution
                    status_counts = {}
                    for record in data:
                        status = record.get('status', 'unknown')
                        status_counts[status] = status_counts.get(status, 0) + 1
                    
                    f.write("TREATMENT STATUS DISTRIBUTION:\n")
                    f.write("-" * 30 + "\n")
                    for status, count in status_counts.items():
                        percentage = (count / len(data)) * 100
                        f.write(f"{status.title()}: {count} ({percentage:.1f}%)\n")
                    
                    # Treatment type effectiveness
                    type_effectiveness = {}
                    for record in data:
                        t_type = record.get('treatment_type', 'unknown')
                        status = record.get('status', 'unknown')
                        
                        if t_type not in type_effectiveness:
                            type_effectiveness[t_type] = {'total': 0, 'completed': 0}
                        
                        type_effectiveness[t_type]['total'] += 1
                        if status == 'completed':
                            type_effectiveness[t_type]['completed'] += 1
                    
                    f.write(f"\nTREATMENT TYPE EFFECTIVENESS:\n")
                    f.write("-" * 30 + "\n")
                    for t_type, stats in type_effectiveness.items():
                        success_rate = (stats['completed'] / stats['total']) * 100 if stats['total'] > 0 else 0
                        f.write(f"{t_type.title()}: {success_rate:.1f}% ({stats['completed']}/{stats['total']})\n")
            
            print(f"Treatment effectiveness report exported: {report_file}")
            return report_file
            
        except Exception as e:
            print(f"Error exporting treatment effectiveness: {e}")
            return None
    
    def create_analytics_charts(self):
        """Create analytics charts from data"""
        print("\n--- CREATE ANALYTICS CHARTS ---")
        print("1. Disease Distribution Chart")
        print("2. Treatment Status Chart")
        print("3. Detection Timeline Chart")
        print("4. All Charts")
        
        choice = input("Select chart type (1-4): ").strip()
        
        created_charts = []
        
        if choice in ['1', '4']:
            chart = self.create_disease_distribution_chart()
            if chart:
                created_charts.append(chart)
        
        if choice in ['2', '4']:
            chart = self.create_treatment_status_chart()
            if chart:
                created_charts.append(chart)
        
        if choice in ['3', '4']:
            chart = self.create_detection_timeline_chart()
            if chart:
                created_charts.append(chart)
        
        if created_charts:
            print(f"\nCreated {len(created_charts)} charts:")
            for chart in created_charts:
                print(f"  - {chart.name}")
        else:
            print("No charts were created.")
    
    def create_disease_distribution_chart(self):
        """Create disease distribution pie chart"""
        detection_file = self.base_path / 'detection_history' / 'detection_history.json'
        
        if not detection_file.exists():
            print("No detection data available for chart.")
            return None
        
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                return None
            
            # Count diseases
            diseases = [record.get('predicted_disease', 'Unknown') for record in data]
            disease_counts = {}
            for disease in diseases:
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
            
            # Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(disease_counts.values(), labels=disease_counts.keys(), autopct='%1.1f%%')
            plt.title('Disease Distribution')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = self.exports_path / 'charts' / f'disease_distribution_{timestamp}.png'
            
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Disease distribution chart created: {chart_file}")
            return chart_file
            
        except Exception as e:
            print(f"Error creating disease distribution chart: {e}")
            return None
    
    def create_treatment_status_chart(self):
        """Create treatment status bar chart"""
        treatment_file = self.base_path / 'treatment_history' / 'treatment_history.json'
        
        if not treatment_file.exists():
            print("No treatment data available for chart.")
            return None
        
        try:
            with open(treatment_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                return None
            
            # Count statuses
            statuses = [record.get('status', 'unknown') for record in data]
            status_counts = {}
            for status in statuses:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Create bar chart
            plt.figure(figsize=(10, 6))
            plt.bar(status_counts.keys(), status_counts.values())
            plt.title('Treatment Status Distribution')
            plt.xlabel('Status')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = self.exports_path / 'charts' / f'treatment_status_{timestamp}.png'
            
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Treatment status chart created: {chart_file}")
            return chart_file
            
        except Exception as e:
            print(f"Error creating treatment status chart: {e}")
            return None
    
    def create_detection_timeline_chart(self):
        """Create detection timeline chart"""
        detection_file = self.base_path / 'detection_history' / 'detection_history.json'
        
        if not detection_file.exists():
            print("No detection data available for chart.")
            return None
        
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                return None
            
            # Group by date
            from datetime import datetime
            
            dates = []
            for record in data:
                timestamp_str = record.get('timestamp', '')
                if timestamp_str:
                    try:
                        dt = datetime.fromisoformat(timestamp_str)
                        dates.append(dt.date())
                    except:
                        continue
            
            if not dates:
                print("No valid timestamps found for timeline chart.")
                return None
            
            # Count detections per date
            date_counts = {}
            for date in dates:
                date_counts[date] = date_counts.get(date, 0) + 1
            
            # Sort by date
            sorted_dates = sorted(date_counts.keys())
            counts = [date_counts[date] for date in sorted_dates]
            
            # Create line chart
            plt.figure(figsize=(12, 6))
            plt.plot(sorted_dates, counts, marker='o')
            plt.title('Detection Timeline')
            plt.xlabel('Date')
            plt.ylabel('Number of Detections')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_file = self.exports_path / 'charts' / f'detection_timeline_{timestamp}.png'
            
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Detection timeline chart created: {chart_file}")
            return chart_file
            
        except Exception as e:
            print(f"Error creating detection timeline chart: {e}")
            return None
    
    def create_comprehensive_report(self):
        """Create comprehensive system report"""
        print("\n--- CREATING COMPREHENSIVE REPORT ---")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.exports_path / 'reports' / f'comprehensive_report_{timestamp}.txt'
        
        try:
            with open(report_file, 'w') as f:
                f.write("COMPREHENSIVE FARM REPORT\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Detection summary
                detection_summary = self.get_detection_summary()
                f.write("DETECTION SUMMARY:\n")
                f.write("-" * 18 + "\n")
                f.write(f"Total Detections: {detection_summary.get('total_detections', 0)}\n")
                f.write(f"Average Confidence: {detection_summary.get('average_confidence', 0):.1%}\n")
                
                if detection_summary.get('disease_distribution'):
                    f.write("\nTop Diseases:\n")
                    for disease, count in list(detection_summary['disease_distribution'].items())[:5]:
                        f.write(f"  {disease}: {count}\n")
                
                # Treatment summary
                treatment_summary = self.get_treatment_summary()
                f.write(f"\nTREATMENT SUMMARY:\n")
                f.write("-" * 18 + "\n")
                f.write(f"Total Treatments: {treatment_summary.get('total_treatments', 0)}\n")
                
                if treatment_summary.get('status_distribution'):
                    f.write("Status Distribution:\n")
                    for status, count in treatment_summary['status_distribution'].items():
                        f.write(f"  {status.title()}: {count}\n")
                
                # System statistics
                system_stats = self.get_system_statistics()
                f.write(f"\nSYSTEM STATISTICS:\n")
                f.write("-" * 18 + "\n")
                f.write(f"Total Data Size: {system_stats.get('total_size_mb', 0):.1f} MB\n")
                f.write(f"Available Backups: {system_stats.get('backup_count', 0)}\n")
                
                # Recommendations
                f.write(f"\nRECOMMENDATIONS:\n")
                f.write("-" * 15 + "\n")
                recommendations = self.generate_recommendations(detection_summary, treatment_summary)
                for rec in recommendations:
                    f.write(f"• {rec}\n")
            
            print(f"Comprehensive report created: {report_file}")
            return report_file
            
        except Exception as e:
            print(f"Error creating comprehensive report: {e}")
            return None
    
    def get_detection_summary(self):
        """Get detection data summary"""
        detection_file = self.base_path / 'detection_history' / 'detection_history.json'
        
        if not detection_file.exists():
            return {}
        
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            diseases = [record.get('predicted_disease', 'Unknown') for record in data]
            disease_counts = {}
            for disease in diseases:
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
            
            confidences = [record.get('confidence', 0) for record in data]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'total_detections': len(data),
                'disease_distribution': dict(sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)),
                'average_confidence': avg_confidence
            }
            
        except Exception as e:
            print(f"Error getting detection summary: {e}")
            return {}
    
    def get_treatment_summary(self):
        """Get treatment data summary"""
        treatment_file = self.base_path / 'treatment_history' / 'treatment_history.json'
        
        if not treatment_file.exists():
            return {}
        
        try:
            with open(treatment_file, 'r') as f:
                data = json.load(f)
            
            status_counts = {}
            for record in data:
                status = record.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'total_treatments': len(data),
                'status_distribution': status_counts
            }
            
        except Exception as e:
            print(f"Error getting treatment summary: {e}")
            return {}
    
    def get_system_statistics(self):
        """Get system statistics"""
        total_size = 0
        
        # Calculate total size
        for folder in self.base_path.iterdir():
            if folder.is_dir():
                for file_path in folder.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        
        # Count backups
        backup_path = self.base_path / 'backups'
        backup_count = len(list(backup_path.glob('*.zip'))) if backup_path.exists() else 0
        
        return {
            'total_size_mb': total_size / (1024 * 1024),
            'backup_count': backup_count
        }
    
    def generate_recommendations(self, detection_summary, treatment_summary):
        """Generate actionable recommendations"""
        recommendations = []
        
        total_detections = detection_summary.get('total_detections', 0)
        avg_confidence = detection_summary.get('average_confidence', 0)
        
        if total_detections == 0:
            recommendations.append("Start using disease detection to build historical data")
        elif avg_confidence < 0.7:
            recommendations.append("Improve image quality for better detection accuracy")
        
        total_treatments = treatment_summary.get('total_treatments', 0)
        status_dist = treatment_summary.get('status_distribution', {})
        
        if total_treatments == 0:
            recommendations.append("Create treatment plans for detected diseases")
        elif status_dist.get('completed', 0) / total_treatments < 0.7:
            recommendations.append("Follow through on treatment plans for better results")
        
        # Default recommendations
        if len(recommendations) == 0:
            recommendations = [
                "Continue regular monitoring and detection",
                "Maintain detailed treatment records",
                "Create system backups regularly"
            ]
        
        return recommendations
    
    def run(self):
        """Main export reports interface"""
        while True:
            print(f"\n--- EXPORT REPORTS MODULE ---")
            print("1. Export Detection History")
            print("2. Export Treatment History")
            print("3. Create Analytics Charts")
            print("4. Create Comprehensive Report")
            print("5. Bulk Export (All Data)")
            print("6. Custom Export")
            print("7. Return to Main Menu")
            
            choice = input("Select option (1-7): ").strip()
            
            if choice == '1':
                self.export_detection_history()
                
            elif choice == '2':
                self.export_treatment_history()
                
            elif choice == '3':
                self.create_analytics_charts()
                
            elif choice == '4':
                self.create_comprehensive_report()
                
            elif choice == '5':
                self.bulk_export()
                
            elif choice == '6':
                self.custom_export()
                
            elif choice == '7':
                break
                
            else:
                print("Invalid choice. Please select 1-7.")
    
    def bulk_export(self):
        """Export all data in multiple formats"""
        print("\n--- BULK EXPORT ---")
        print("Exporting all available data...")
        
        exported_files = []
        
        # Export detection history
        csv_file = self.export_detection_csv()
        if csv_file:
            exported_files.append(csv_file)
        
        json_file = self.export_detection_json()
        if json_file:
            exported_files.append(json_file)
        
        # Export treatment history
        treatment_csv = self.export_treatment_csv()
        if treatment_csv:
            exported_files.append(treatment_csv)
        
        # Create charts
        disease_chart = self.create_disease_distribution_chart()
        if disease_chart:
            exported_files.append(disease_chart)
        
        status_chart = self.create_treatment_status_chart()
        if status_chart:
            exported_files.append(status_chart)
        
        # Create comprehensive report
        report = self.create_comprehensive_report()
        if report:
            exported_files.append(report)
        
        print(f"\nBulk export completed. {len(exported_files)} files created:")
        for file_path in exported_files:
            print(f"  - {file_path.name}")
    
    def custom_export(self):
        """Create custom export based on user preferences"""
        print("\n--- CUSTOM EXPORT ---")
        print("Configure your custom export:")
        
        # Get date range
        start_date = input("Start date (YYYY-MM-DD) or press Enter for all: ").strip()
        end_date = input("End date (YYYY-MM-DD) or press Enter for all: ").strip()
        
        # Get disease filter
        disease_filter = input("Filter by disease (or press Enter for all): ").strip()
        
        # Get export format
        print("Export formats:")
        print("1. CSV only")
        print("2. JSON only")
        print("3. Both CSV and JSON")
        print("4. Summary report")
        
        format_choice = input("Select format (1-4): ").strip()
        
        print(f"\nCreating custom export...")
        print(f"Date range: {start_date or 'All'} to {end_date or 'All'}")
        print(f"Disease filter: {disease_filter or 'All diseases'}")
        
        # This would implement the actual filtering and export logic
        print("Custom export feature would be implemented here with the specified filters.")
        print("For now, please use the standard export options.")