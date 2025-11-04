"""
Treatment History Module
========================
Manages treatment plans and application history.
"""

import json
from datetime import datetime
from pathlib import Path

class TreatmentHistoryModule:
    """Manages treatment history and planning"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / 'treatment_history'
        self.data_path.mkdir(exist_ok=True)
        
        self.treatment_records = []
        self.treatment_templates = self.load_treatment_templates()
        self.load_treatment_history()
    
    def load_treatment_templates(self):
        """Load treatment templates for different diseases"""
        return {
            "Fungal_Disease": {
                "chemical": {
                    "products": ["Copper Sulfate", "Mancozeb", "Chlorothalonil"],
                    "dosage": "2-3g per liter of water",
                    "frequency": "Every 7-10 days",
                    "duration": "3-4 applications"
                },
                "organic": {
                    "products": ["Neem Oil", "Baking Soda Solution", "Copper Fungicide"],
                    "dosage": "5ml neem oil per liter",
                    "frequency": "Every 5-7 days",
                    "duration": "Until symptoms subside"
                }
            },
            "Bacterial_Disease": {
                "chemical": {
                    "products": ["Copper Bactericide", "Streptomycin"],
                    "dosage": "2g per liter of water",
                    "frequency": "Every 5-7 days",
                    "duration": "3-4 applications"
                },
                "organic": {
                    "products": ["Copper Soap", "Hydrogen Peroxide", "Plant Extract"],
                    "dosage": "10ml per liter",
                    "frequency": "Every 3-5 days",
                    "duration": "2-3 weeks"
                }
            },
            "Viral_Disease": {
                "chemical": {
                    "products": ["No direct chemical cure"],
                    "dosage": "Focus on vector control",
                    "frequency": "Preventive only",
                    "duration": "Ongoing"
                },
                "organic": {
                    "products": ["Remove infected plants", "Boost plant immunity"],
                    "dosage": "Complete removal recommended",
                    "frequency": "Immediate action",
                    "duration": "Preventive measures ongoing"
                }
            },
            "Healthy_Plant": {
                "chemical": {
                    "products": ["Preventive fungicide", "Balanced fertilizer"],
                    "dosage": "Follow label instructions",
                    "frequency": "Monthly",
                    "duration": "Throughout growing season"
                },
                "organic": {
                    "products": ["Compost", "Neem oil", "Beneficial microorganisms"],
                    "dosage": "As per organic guidelines",
                    "frequency": "Bi-weekly",
                    "duration": "Continuous"
                }
            }
        }
    
    def load_treatment_history(self):
        """Load existing treatment history"""
        history_file = self.data_path / 'treatment_history.json'
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for record in data:
                        record['timestamp'] = datetime.fromisoformat(record['timestamp'])
                        if 'planned_date' in record and record['planned_date']:
                            record['planned_date'] = datetime.fromisoformat(record['planned_date'])
                        self.treatment_records.append(record)
                print(f"Loaded {len(self.treatment_records)} treatment records")
            except Exception as e:
                print(f"Error loading treatment history: {e}")
    
    def save_treatment_history(self):
        """Save treatment history to file"""
        try:
            history_file = self.data_path / 'treatment_history.json'
            with open(history_file, 'w') as f:
                data = []
                for record in self.treatment_records:
                    record_copy = record.copy()
                    record_copy['timestamp'] = record['timestamp'].isoformat()
                    if 'planned_date' in record_copy and record_copy['planned_date']:
                        record_copy['planned_date'] = record_copy['planned_date'].isoformat()
                    data.append(record_copy)
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving treatment history: {e}")
    
    def add_treatment_plan(self, detection_record):
        """Add treatment plan based on detection"""
        disease = detection_record.get('predicted_disease', 'Unknown')
        
        print(f"\n--- CREATE TREATMENT PLAN FOR {disease} ---")
        
        # Get treatment type preference
        print("Select treatment approach:")
        print("1. Chemical Treatment")
        print("2. Organic Treatment")
        print("3. Integrated Approach")
        
        treatment_choice = input("Select option (1-3): ").strip()
        
        if treatment_choice == '1':
            treatment_type = 'chemical'
        elif treatment_choice == '2':
            treatment_type = 'organic'
        else:
            treatment_type = 'integrated'
        
        # Get treatment template
        template_key = self.get_template_key(disease)
        template = self.treatment_templates.get(template_key, self.treatment_templates['Fungal_Disease'])
        
        # Create treatment plan
        treatment_plan = {
            'id': len(self.treatment_records) + 1,
            'timestamp': datetime.now(),
            'detection_id': detection_record.get('id'),
            'disease': disease,
            'treatment_type': treatment_type,
            'status': 'planned',
            'template_used': template_key
        }
        
        if treatment_type in template:
            treatment_plan.update(template[treatment_type])
        
        # Get additional details
        notes = input("Additional notes (optional): ").strip()
        if notes:
            treatment_plan['notes'] = notes
        
        # Get planned application date
        planned_date = input("Planned application date (YYYY-MM-DD) or press Enter for today: ").strip()
        if planned_date:
            try:
                treatment_plan['planned_date'] = datetime.strptime(planned_date, '%Y-%m-%d')
            except ValueError:
                treatment_plan['planned_date'] = datetime.now()
        else:
            treatment_plan['planned_date'] = datetime.now()
        
        self.treatment_records.append(treatment_plan)
        self.save_treatment_history()
        
        print(f"\nTreatment plan created with ID: {treatment_plan['id']}")
        self.display_treatment_plan(treatment_plan)
    
    def get_template_key(self, disease):
        """Determine template key based on disease name"""
        disease_lower = disease.lower()
        
        if any(keyword in disease_lower for keyword in ['fungal', 'rust', 'blight', 'rot', 'mold', 'spot']):
            return 'Fungal_Disease'
        elif any(keyword in disease_lower for keyword in ['bacterial', 'bacteria']):
            return 'Bacterial_Disease'
        elif any(keyword in disease_lower for keyword in ['viral', 'virus', 'mosaic']):
            return 'Viral_Disease'
        elif 'healthy' in disease_lower:
            return 'Healthy_Plant'
        else:
            return 'Fungal_Disease'  # Default
    
    def display_treatment_plan(self, plan):
        """Display treatment plan details"""
        print(f"\n--- TREATMENT PLAN #{plan['id']} ---")
        print(f"Disease: {plan['disease']}")
        print(f"Treatment Type: {plan['treatment_type'].title()}")
        print(f"Status: {plan['status'].title()}")
        print(f"Planned Date: {plan.get('planned_date', 'Not specified').strftime('%Y-%m-%d') if isinstance(plan.get('planned_date'), datetime) else 'Not specified'}")
        
        if 'products' in plan:
            print(f"Products: {', '.join(plan['products'])}")
        if 'dosage' in plan:
            print(f"Dosage: {plan['dosage']}")
        if 'frequency' in plan:
            print(f"Frequency: {plan['frequency']}")
        if 'duration' in plan:
            print(f"Duration: {plan['duration']}")
        if 'notes' in plan:
            print(f"Notes: {plan['notes']}")
    
    def view_treatment_history(self):
        """Display treatment history"""
        if not self.treatment_records:
            print("No treatment records found.")
            return
        
        print(f"\n--- TREATMENT HISTORY ({len(self.treatment_records)} records) ---")
        
        for record in reversed(self.treatment_records[-10:]):  # Show last 10
            self.display_treatment_plan(record)
            print("-" * 50)
    
    def update_treatment_status(self):
        """Update status of treatment plans"""
        if not self.treatment_records:
            print("No treatment plans to update.")
            return
        
        print("\n--- UPDATE TREATMENT STATUS ---")
        
        # Show recent treatment plans
        recent_plans = [r for r in self.treatment_records if r['status'] in ['planned', 'in_progress']][-5:]
        
        if not recent_plans:
            print("No active treatment plans to update.")
            return
        
        for i, plan in enumerate(recent_plans, 1):
            print(f"{i}. ID: {plan['id']} | Disease: {plan['disease']} | Status: {plan['status']}")
        
        try:
            choice = int(input("Select plan to update (number): ")) - 1
            if 0 <= choice < len(recent_plans):
                selected_plan = recent_plans[choice]
                
                print("New status options:")
                print("1. In Progress")
                print("2. Completed")
                print("3. Cancelled")
                
                status_choice = input("Select new status (1-3): ").strip()
                
                status_map = {'1': 'in_progress', '2': 'completed', '3': 'cancelled'}
                new_status = status_map.get(status_choice, selected_plan['status'])
                
                # Update the status in the original record
                for record in self.treatment_records:
                    if record['id'] == selected_plan['id']:
                        record['status'] = new_status
                        record['status_updated'] = datetime.now()
                        break
                
                self.save_treatment_history()
                print(f"Treatment plan {selected_plan['id']} status updated to: {new_status}")
                
            else:
                print("Invalid selection.")
                
        except ValueError:
            print("Invalid input.")
    
    def get_treatment_statistics(self):
        """Get treatment statistics"""
        if not self.treatment_records:
            return {}
        
        # Status distribution
        status_counts = {}
        for record in self.treatment_records:
            status = record.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Treatment type distribution
        type_counts = {}
        for record in self.treatment_records:
            t_type = record.get('treatment_type', 'unknown')
            type_counts[t_type] = type_counts.get(t_type, 0) + 1
        
        # Disease treated
        disease_counts = {}
        for record in self.treatment_records:
            disease = record.get('disease', 'unknown')
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        return {
            'total_treatments': len(self.treatment_records),
            'status_distribution': status_counts,
            'type_distribution': type_counts,
            'disease_distribution': disease_counts
        }
    
    def run(self):
        """Main treatment history interface"""
        while True:
            print(f"\n--- TREATMENT HISTORY MODULE ---")
            print("1. View Treatment History")
            print("2. Create New Treatment Plan")
            print("3. Update Treatment Status")
            print("4. Treatment Statistics")
            print("5. Search Treatments")
            print("6. Return to Main Menu")
            
            choice = input("Select option (1-6): ").strip()
            
            if choice == '1':
                self.view_treatment_history()
                
            elif choice == '2':
                self.create_manual_treatment()
                
            elif choice == '3':
                self.update_treatment_status()
                
            elif choice == '4':
                self.display_statistics()
                
            elif choice == '5':
                self.search_treatments()
                
            elif choice == '6':
                break
                
            else:
                print("Invalid choice. Please select 1-6.")
    
    def create_manual_treatment(self):
        """Create treatment plan manually"""
        print("\n--- CREATE MANUAL TREATMENT PLAN ---")
        
        disease = input("Enter disease name: ").strip()
        if not disease:
            print("Disease name is required.")
            return
        
        # Create fake detection record for manual entry
        detection_record = {
            'id': 'manual',
            'predicted_disease': disease
        }
        
        self.add_treatment_plan(detection_record)
    
    def display_statistics(self):
        """Display treatment statistics"""
        stats = self.get_treatment_statistics()
        
        if not stats:
            print("No treatment statistics available.")
            return
        
        print(f"\n--- TREATMENT STATISTICS ---")
        print(f"Total Treatments: {stats['total_treatments']}")
        
        print(f"\nStatus Distribution:")
        for status, count in stats['status_distribution'].items():
            percentage = (count / stats['total_treatments']) * 100
            print(f"  {status.title()}: {count} ({percentage:.1f}%)")
        
        print(f"\nTreatment Type Distribution:")
        for t_type, count in stats['type_distribution'].items():
            percentage = (count / stats['total_treatments']) * 100
            print(f"  {t_type.title()}: {count} ({percentage:.1f}%)")
        
        print(f"\nDiseases Treated:")
        for disease, count in sorted(stats['disease_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / stats['total_treatments']) * 100
            print(f"  {disease}: {count} ({percentage:.1f}%)")
    
    def search_treatments(self):
        """Search treatments by various criteria"""
        if not self.treatment_records:
            print("No treatment records to search.")
            return
        
        print("\n--- SEARCH TREATMENTS ---")
        print("1. Search by Disease")
        print("2. Search by Status")
        print("3. Search by Treatment Type")
        print("4. Search by Date Range")
        
        search_choice = input("Select search type (1-4): ").strip()
        
        results = []
        
        if search_choice == '1':
            disease = input("Enter disease name to search: ").strip()
            results = [r for r in self.treatment_records 
                      if disease.lower() in r.get('disease', '').lower()]
        
        elif search_choice == '2':
            status = input("Enter status (planned/in_progress/completed/cancelled): ").strip()
            results = [r for r in self.treatment_records 
                      if r.get('status', '').lower() == status.lower()]
        
        elif search_choice == '3':
            t_type = input("Enter treatment type (chemical/organic/integrated): ").strip()
            results = [r for r in self.treatment_records 
                      if r.get('treatment_type', '').lower() == t_type.lower()]
        
        elif search_choice == '4':
            try:
                start_date = input("Enter start date (YYYY-MM-DD): ").strip()
                end_date = input("Enter end date (YYYY-MM-DD): ").strip()
                
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                results = [r for r in self.treatment_records 
                          if start_dt <= r['timestamp'] <= end_dt]
            except ValueError:
                print("Invalid date format.")
                return
        
        if results:
            print(f"\nFound {len(results)} matching treatments:")
            for record in results[-10:]:  # Show last 10 results
                print(f"ID: {record['id']} | Disease: {record['disease']} | "
                      f"Status: {record['status']} | Date: {record['timestamp'].strftime('%Y-%m-%d')}")
        else:
            print("No matching treatments found.")