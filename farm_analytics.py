"""
Farm Analytics Module
====================
Provides comprehensive analytics and insights for farm data.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class FarmAnalyticsModule:
    """Comprehensive farm analytics and reporting"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.analytics_path = self.base_path / 'analytics'
        self.analytics_path.mkdir(exist_ok=True)
        
        # Load data from other modules
        self.detection_data = self.load_detection_data()
        self.treatment_data = self.load_treatment_data()
        self.fertilizer_data = self.load_fertilizer_data()
    
    def load_detection_data(self):
        """Load detection history data"""
        detection_file = self.base_path / 'detection_history' / 'detection_history.json'
        
        if detection_file.exists():
            try:
                with open(detection_file, 'r') as f:
                    data = json.load(f)
                    # Convert timestamps
                    for record in data:
                        record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                    return data
            except Exception as e:
                print(f"Error loading detection data: {e}")
        
        return []
    
    def load_treatment_data(self):
        """Load treatment history data"""
        treatment_file = self.base_path / 'treatment_history' / 'treatment_history.json'
        
        if treatment_file.exists():
            try:
                with open(treatment_file, 'r') as f:
                    data = json.load(f)
                    # Convert timestamps
                    for record in data:
                        record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                        if 'planned_date' in record and record['planned_date']:
                            record['planned_date'] = datetime.fromisoformat(record['planned_date'])
                    return data
            except Exception as e:
                print(f"Error loading treatment data: {e}")
        
        return []
    
    def load_fertilizer_data(self):
        """Load fertilizer plan data"""
        fertilizer_file = self.base_path / 'fertilizer_data' / 'fertilizer_history.json'
        
        if fertilizer_file.exists():
            try:
                with open(fertilizer_file, 'r') as f:
                    data = json.load(f)
                    # Convert timestamps
                    for record in data:
                        record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                    return data
            except Exception as e:
                print(f"Error loading fertilizer data: {e}")
        
        return []
    
    def get_detection_analytics(self):
        """Analyze detection patterns"""
        if not self.detection_data:
            return {}
        
        # Time-based analysis
        dates = [record['timestamp'].date() for record in self.detection_data]
        date_counts = {}
        for date in dates:
            date_counts[date] = date_counts.get(date, 0) + 1
        
        # Disease distribution
        diseases = [record.get('predicted_disease', 'Unknown') for record in self.detection_data]
        disease_counts = {}
        for disease in diseases:
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        # Confidence analysis
        confidences = [record.get('confidence', 0) for record in self.detection_data]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Monthly trends
        monthly_counts = {}
        for record in self.detection_data:
            month_key = record['timestamp'].strftime('%Y-%m')
            monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
        
        return {
            'total_detections': len(self.detection_data),
            'daily_distribution': date_counts,
            'disease_distribution': disease_counts,
            'average_confidence': avg_confidence,
            'monthly_trends': monthly_counts,
            'most_common_disease': max(disease_counts.items(), key=lambda x: x[1])[0] if disease_counts else 'None'
        }
    
    def get_treatment_analytics(self):
        """Analyze treatment effectiveness and patterns"""
        if not self.treatment_data:
            return {}
        
        # Status distribution
        status_counts = {}
        for record in self.treatment_data:
            status = record.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Treatment type effectiveness
        type_counts = {}
        for record in self.treatment_data:
            t_type = record.get('treatment_type', 'unknown')
            type_counts[t_type] = type_counts.get(t_type, 0) + 1
        
        # Success rate calculation
        completed = status_counts.get('completed', 0)
        total_treatments = len(self.treatment_data)
        success_rate = (completed / total_treatments) * 100 if total_treatments > 0 else 0
        
        return {
            'total_treatments': total_treatments,
            'status_distribution': status_counts,
            'type_distribution': type_counts,
            'success_rate': success_rate,
            'completion_rate': (completed / total_treatments * 100) if total_treatments > 0 else 0
        }
    
    def get_fertilizer_analytics(self):
        """Analyze fertilizer usage patterns"""
        if not self.fertilizer_data:
            return {}
        
        # Crop distribution
        crops = []
        total_area = 0
        npk_totals = {'N': 0, 'P': 0, 'K': 0}
        
        for record in self.fertilizer_data:
            plan = record.get('plan', {})
            crop = plan.get('crop_type', 'Unknown')
            crops.append(crop)
            
            area = plan.get('area_hectares', 0)
            total_area += area
            
            requirements = plan.get('requirements', {})
            for nutrient in ['N', 'P', 'K']:
                npk_totals[nutrient] += requirements.get(nutrient, 0)
        
        crop_counts = {}
        for crop in crops:
            crop_counts[crop] = crop_counts.get(crop, 0) + 1
        
        return {
            'total_plans': len(self.fertilizer_data),
            'total_area_covered': total_area,
            'crop_distribution': crop_counts,
            'total_nutrient_requirements': npk_totals,
            'average_area_per_plan': total_area / len(self.fertilizer_data) if self.fertilizer_data else 0
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive farm analytics report"""
        detection_analytics = self.get_detection_analytics()
        treatment_analytics = self.get_treatment_analytics()
        fertilizer_analytics = self.get_fertilizer_analytics()
        
        report = {
            'generated_at': datetime.now(),
            'detection_analytics': detection_analytics,
            'treatment_analytics': treatment_analytics,
            'fertilizer_analytics': fertilizer_analytics,
            'insights': self.generate_insights(detection_analytics, treatment_analytics, fertilizer_analytics)
        }
        
        # Save report
        report_file = self.analytics_path / f'comprehensive_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        try:
            with open(report_file, 'w') as f:
                # Convert datetime for JSON serialization
                report_copy = report.copy()
                report_copy['generated_at'] = report['generated_at'].isoformat()
                json.dump(report_copy, f, indent=2)
            print(f"Comprehensive report saved: {report_file}")
        except Exception as e:
            print(f"Error saving report: {e}")
        
        return report
    
    def generate_insights(self, detection_analytics, treatment_analytics, fertilizer_analytics):
        """Generate actionable insights from analytics"""
        insights = []
        
        # Detection insights
        if detection_analytics:
            total_detections = detection_analytics.get('total_detections', 0)
            avg_confidence = detection_analytics.get('average_confidence', 0)
            most_common_disease = detection_analytics.get('most_common_disease', 'None')
            
            if total_detections > 0:
                insights.append(f"You have performed {total_detections} disease detections with an average confidence of {avg_confidence:.1%}")
                
                if avg_confidence < 0.7:
                    insights.append("Consider improving image quality for better detection confidence")
                
                if most_common_disease != 'Healthy Plant' and most_common_disease != 'None':
                    insights.append(f"Most common issue: {most_common_disease} - consider preventive measures")
        
        # Treatment insights
        if treatment_analytics:
            success_rate = treatment_analytics.get('success_rate', 0)
            total_treatments = treatment_analytics.get('total_treatments', 0)
            
            if total_treatments > 0:
                insights.append(f"Treatment success rate: {success_rate:.1f}% from {total_treatments} treatments")
                
                if success_rate < 70:
                    insights.append("Consider reviewing treatment effectiveness and follow-up procedures")
        
        # Fertilizer insights
        if fertilizer_analytics:
            total_area = fertilizer_analytics.get('total_area_covered', 0)
            total_plans = fertilizer_analytics.get('total_plans', 0)
            
            if total_plans > 0:
                insights.append(f"Fertilizer management covers {total_area:.1f} hectares across {total_plans} plans")
        
        # General insights
        if len(insights) == 0:
            insights.append("Start using the system regularly to generate meaningful insights")
        
        return insights
    
    def display_analytics_dashboard(self):
        """Display comprehensive analytics dashboard"""
        print("\n" + "="*60)
        print("FARM ANALYTICS DASHBOARD")
        print("="*60)
        
        detection_analytics = self.get_detection_analytics()
        treatment_analytics = self.get_treatment_analytics()
        fertilizer_analytics = self.get_fertilizer_analytics()
        
        # Detection Analytics
        if detection_analytics:
            print("\nDETECTION ANALYTICS:")
            print(f"Total Detections: {detection_analytics['total_detections']}")
            print(f"Average Confidence: {detection_analytics['average_confidence']:.1%}")
            print(f"Most Common Disease: {detection_analytics['most_common_disease']}")
            
            print("\nDisease Distribution:")
            for disease, count in sorted(detection_analytics['disease_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
                percentage = (count / detection_analytics['total_detections']) * 100
                print(f"  {disease}: {count} ({percentage:.1f}%)")
        
        # Treatment Analytics
        if treatment_analytics:
            print(f"\nTREATMENT ANALYTICS:")
            print(f"Total Treatments: {treatment_analytics['total_treatments']}")
            print(f"Success Rate: {treatment_analytics['success_rate']:.1f}%")
            print(f"Completion Rate: {treatment_analytics['completion_rate']:.1f}%")
            
            print("\nTreatment Types:")
            for t_type, count in treatment_analytics['type_distribution'].items():
                percentage = (count / treatment_analytics['total_treatments']) * 100
                print(f"  {t_type.title()}: {count} ({percentage:.1f}%)")
        
        # Fertilizer Analytics
        if fertilizer_analytics:
            print(f"\nFERTILIZER ANALYTICS:")
            print(f"Total Plans: {fertilizer_analytics['total_plans']}")
            print(f"Total Area Covered: {fertilizer_analytics['total_area_covered']:.1f} hectares")
            print(f"Average Area per Plan: {fertilizer_analytics['average_area_per_plan']:.1f} hectares")
            
            print("\nCrop Distribution:")
            for crop, count in fertilizer_analytics['crop_distribution'].items():
                percentage = (count / fertilizer_analytics['total_plans']) * 100
                print(f"  {crop}: {count} ({percentage:.1f}%)")
        
        # Insights
        insights = self.generate_insights(detection_analytics, treatment_analytics, fertilizer_analytics)
        if insights:
            print(f"\nKEY INSIGHTS:")
            for i, insight in enumerate(insights, 1):
                print(f"  {i}. {insight}")
    
    def create_trend_analysis(self):
        """Create trend analysis for detections over time"""
        if not self.detection_data:
            print("No detection data available for trend analysis.")
            return
        
        print("\n--- TREND ANALYSIS ---")
        
        # Group by month
        monthly_data = {}
        for record in self.detection_data:
            month_key = record['timestamp'].strftime('%Y-%m')
            if month_key not in monthly_data:
                monthly_data[month_key] = {'total': 0, 'diseases': {}}
            
            monthly_data[month_key]['total'] += 1
            disease = record.get('predicted_disease', 'Unknown')
            monthly_data[month_key]['diseases'][disease] = monthly_data[month_key]['diseases'].get(disease, 0) + 1
        
        # Display trends
        sorted_months = sorted(monthly_data.keys())
        
        print("Monthly Detection Trends:")
        for month in sorted_months[-6:]:  # Show last 6 months
            data = monthly_data[month]
            print(f"\n{month}: {data['total']} detections")
            
            # Show top diseases for the month
            top_diseases = sorted(data['diseases'].items(), key=lambda x: x[1], reverse=True)[:3]
            for disease, count in top_diseases:
                print(f"  - {disease}: {count}")
        
        # Calculate growth rate
        if len(sorted_months) >= 2:
            recent_month = monthly_data[sorted_months[-1]]['total']
            previous_month = monthly_data[sorted_months[-2]]['total']
            
            if previous_month > 0:
                growth_rate = ((recent_month - previous_month) / previous_month) * 100
                print(f"\nMonth-over-month change: {growth_rate:+.1f}%")
    
    def export_analytics_csv(self):
        """Export analytics data to CSV format"""
        try:
            import csv
            
            # Export detection summary
            if self.detection_data:
                csv_file = self.analytics_path / f'detection_analytics_{datetime.now().strftime("%Y%m%d")}.csv'
                
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Date', 'Disease', 'Confidence'])
                    
                    for record in self.detection_data:
                        writer.writerow([
                            record['timestamp'].strftime('%Y-%m-%d'),
                            record.get('predicted_disease', 'Unknown'),
                            record.get('confidence', 0)
                        ])
                
                print(f"Detection analytics exported to: {csv_file}")
            
            # Export treatment summary
            if self.treatment_data:
                csv_file = self.analytics_path / f'treatment_analytics_{datetime.now().strftime("%Y%m%d")}.csv'
                
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Date', 'Disease', 'Treatment_Type', 'Status'])
                    
                    for record in self.treatment_data:
                        writer.writerow([
                            record['timestamp'].strftime('%Y-%m-%d'),
                            record.get('disease', 'Unknown'),
                            record.get('treatment_type', 'Unknown'),
                            record.get('status', 'Unknown')
                        ])
                
                print(f"Treatment analytics exported to: {csv_file}")
            
        except Exception as e:
            print(f"Error exporting CSV: {e}")
    
    def run(self):
        """Main farm analytics interface"""
        while True:
            print(f"\n--- FARM ANALYTICS MODULE ---")
            print("1. View Analytics Dashboard")
            print("2. Generate Comprehensive Report")
            print("3. Trend Analysis")
            print("4. Disease Pattern Analysis")
            print("5. Treatment Effectiveness Analysis")
            print("6. Export Analytics Data")
            print("7. Return to Main Menu")
            
            choice = input("Select option (1-7): ").strip()
            
            if choice == '1':
                self.display_analytics_dashboard()
                
            elif choice == '2':
                report = self.generate_comprehensive_report()
                print("Comprehensive report generated and saved.")
                
            elif choice == '3':
                self.create_trend_analysis()
                
            elif choice == '4':
                self.analyze_disease_patterns()
                
            elif choice == '5':
                self.analyze_treatment_effectiveness()
                
            elif choice == '6':
                self.export_analytics_csv()
                
            elif choice == '7':
                break
                
            else:
                print("Invalid choice. Please select 1-7.")
    
    def analyze_disease_patterns(self):
        """Analyze disease occurrence patterns"""
        if not self.detection_data:
            print("No detection data available for disease pattern analysis.")
            return
        
        print("\n--- DISEASE PATTERN ANALYSIS ---")
        
        # Seasonal analysis
        seasonal_data = {'Spring': {}, 'Summer': {}, 'Autumn': {}, 'Winter': {}}
        season_map = {3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
                     12: 'Winter', 1: 'Winter', 2: 'Winter'}
        
        for record in self.detection_data:
            month = record['timestamp'].month
            season = season_map.get(month, 'Unknown')
            disease = record.get('predicted_disease', 'Unknown')
            
            if disease not in seasonal_data[season]:
                seasonal_data[season][disease] = 0
            seasonal_data[season][disease] += 1
        
        print("Seasonal Disease Distribution:")
        for season, diseases in seasonal_data.items():
            if diseases:
                print(f"\n{season}:")
                sorted_diseases = sorted(diseases.items(), key=lambda x: x[1], reverse=True)
                for disease, count in sorted_diseases[:3]:  # Top 3 per season
                    print(f"  {disease}: {count} cases")
    
    def analyze_treatment_effectiveness(self):
        """Analyze treatment effectiveness by type and disease"""
        if not self.treatment_data:
            print("No treatment data available for effectiveness analysis.")
            return
        
        print("\n--- TREATMENT EFFECTIVENESS ANALYSIS ---")
        
        # Group by treatment type and status
        effectiveness = {}
        for record in self.treatment_data:
            t_type = record.get('treatment_type', 'unknown')
            status = record.get('status', 'unknown')
            disease = record.get('disease', 'unknown')
            
            if t_type not in effectiveness:
                effectiveness[t_type] = {'total': 0, 'completed': 0, 'diseases': {}}
            
            effectiveness[t_type]['total'] += 1
            if status == 'completed':
                effectiveness[t_type]['completed'] += 1
            
            if disease not in effectiveness[t_type]['diseases']:
                effectiveness[t_type]['diseases'][disease] = {'total': 0, 'completed': 0}
            
            effectiveness[t_type]['diseases'][disease]['total'] += 1
            if status == 'completed':
                effectiveness[t_type]['diseases'][disease]['completed'] += 1
        
        # Display effectiveness rates
        for t_type, data in effectiveness.items():
            success_rate = (data['completed'] / data['total']) * 100 if data['total'] > 0 else 0
            print(f"\n{t_type.title()} Treatment:")
            print(f"  Overall Success Rate: {success_rate:.1f}% ({data['completed']}/{data['total']})")
            
            # Show effectiveness by disease
            print(f"  Effectiveness by Disease:")
            for disease, disease_data in data['diseases'].items():
                disease_rate = (disease_data['completed'] / disease_data['total']) * 100 if disease_data['total'] > 0 else 0
                print(f"    {disease}: {disease_rate:.1f}% ({disease_data['completed']}/{disease_data['total']})")