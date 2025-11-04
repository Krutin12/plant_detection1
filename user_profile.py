"""
User Profile Module
==================
Manages user profiles and farm information.
"""

import json
from datetime import datetime
from pathlib import Path

class UserProfileModule:
    """Manages user profiles and farm information"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.profile_path = self.base_path / 'user_profiles'
        self.profile_path.mkdir(exist_ok=True)
        
        self.current_profile = {}
        self.load_profile()
    
    def load_profile(self):
        """Load existing user profile"""
        profile_file = self.profile_path / 'current_profile.json'
        
        if profile_file.exists():
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                    if 'registration_date' in data:
                        data['registration_date'] = datetime.fromisoformat(data['registration_date'])
                    self.current_profile = data
                print(f"Profile loaded: {self.current_profile.get('farmer_name', 'Unknown')}")
            except Exception as e:
                print(f"Error loading profile: {e}")
                self.current_profile = self.create_default_profile()
        else:
            self.current_profile = self.create_default_profile()
    
    def create_default_profile(self):
        """Create default user profile"""
        return {
            'farmer_id': '',
            'farmer_name': '',
            'farm_name': '',
            'location': '',
            'phone': '',
            'email': '',
            'total_area': 0.0,
            'primary_crops': [],
            'farming_experience': 0,
            'farming_type': 'mixed',  # organic, conventional, mixed
            'registration_date': datetime.now(),
            'preferences': {
                'language': 'English',
                'units': 'metric',
                'notifications': True,
                'treatment_preference': 'integrated'  # organic, chemical, integrated
            }
        }
    
    def save_profile(self):
        """Save current profile to file"""
        try:
            profile_file = self.profile_path / 'current_profile.json'
            profile_copy = self.current_profile.copy()
            profile_copy['registration_date'] = self.current_profile['registration_date'].isoformat()
            
            with open(profile_file, 'w') as f:
                json.dump(profile_copy, f, indent=2)
            
            print("Profile saved successfully.")
            
        except Exception as e:
            print(f"Error saving profile: {e}")
    
    def display_profile(self):
        """Display current profile information"""
        print("\n" + "="*50)
        print("USER PROFILE INFORMATION")
        print("="*50)
        
        print(f"Farmer ID: {self.current_profile.get('farmer_id', 'Not set')}")
        print(f"Name: {self.current_profile.get('farmer_name', 'Not set')}")
        print(f"Farm Name: {self.current_profile.get('farm_name', 'Not set')}")
        print(f"Location: {self.current_profile.get('location', 'Not set')}")
        print(f"Phone: {self.current_profile.get('phone', 'Not set')}")
        print(f"Email: {self.current_profile.get('email', 'Not set')}")
        print(f"Total Farm Area: {self.current_profile.get('total_area', 0)} hectares")
        print(f"Primary Crops: {', '.join(self.current_profile.get('primary_crops', []))}")
        print(f"Farming Experience: {self.current_profile.get('farming_experience', 0)} years")
        print(f"Farming Type: {self.current_profile.get('farming_type', 'Not set')}")
        print(f"Registration Date: {self.current_profile['registration_date'].strftime('%Y-%m-%d')}")
        
        # Preferences
        prefs = self.current_profile.get('preferences', {})
        print(f"\nPreferences:")
        print(f"Language: {prefs.get('language', 'English')}")
        print(f"Units: {prefs.get('units', 'metric')}")
        print(f"Notifications: {'Enabled' if prefs.get('notifications', True) else 'Disabled'}")
        print(f"Treatment Preference: {prefs.get('treatment_preference', 'integrated')}")
    
    def update_basic_info(self):
        """Update basic profile information"""
        print("\n--- UPDATE BASIC INFORMATION ---")
        
        # Get updates
        farmer_id = input(f"Farmer ID [{self.current_profile.get('farmer_id', '')}]: ").strip()
        if farmer_id:
            self.current_profile['farmer_id'] = farmer_id
        
        farmer_name = input(f"Farmer Name [{self.current_profile.get('farmer_name', '')}]: ").strip()
        if farmer_name:
            self.current_profile['farmer_name'] = farmer_name
        
        farm_name = input(f"Farm Name [{self.current_profile.get('farm_name', '')}]: ").strip()
        if farm_name:
            self.current_profile['farm_name'] = farm_name
        
        location = input(f"Location [{self.current_profile.get('location', '')}]: ").strip()
        if location:
            self.current_profile['location'] = location
        
        phone = input(f"Phone [{self.current_profile.get('phone', '')}]: ").strip()
        if phone:
            self.current_profile['phone'] = phone
        
        email = input(f"Email [{self.current_profile.get('email', '')}]: ").strip()
        if email:
            self.current_profile['email'] = email
        
        print("Basic information updated.")
    
    def update_farm_info(self):
        """Update farm-specific information"""
        print("\n--- UPDATE FARM INFORMATION ---")
        
        # Total area
        area_input = input(f"Total Farm Area in hectares [{self.current_profile.get('total_area', 0)}]: ").strip()
        if area_input:
            try:
                self.current_profile['total_area'] = float(area_input)
            except ValueError:
                print("Invalid area value, keeping current value.")
        
        # Primary crops
        print(f"Current primary crops: {', '.join(self.current_profile.get('primary_crops', []))}")
        crops_input = input("Enter primary crops (comma-separated): ").strip()
        if crops_input:
            crops = [crop.strip() for crop in crops_input.split(',')]
            self.current_profile['primary_crops'] = crops
        
        # Farming experience
        exp_input = input(f"Farming Experience in years [{self.current_profile.get('farming_experience', 0)}]: ").strip()
        if exp_input:
            try:
                self.current_profile['farming_experience'] = int(exp_input)
            except ValueError:
                print("Invalid experience value, keeping current value.")
        
        # Farming type
        print("Farming type options: 1) Organic 2) Conventional 3) Mixed")
        type_choice = input("Select farming type (1-3): ").strip()
        type_map = {'1': 'organic', '2': 'conventional', '3': 'mixed'}
        if type_choice in type_map:
            self.current_profile['farming_type'] = type_map[type_choice]
        
        print("Farm information updated.")
    
    def update_preferences(self):
        """Update user preferences"""
        print("\n--- UPDATE PREFERENCES ---")
        
        prefs = self.current_profile.get('preferences', {})
        
        # Language
        print("Language options: 1) English 2) Spanish 3) French 4) Hindi")
        lang_choice = input("Select language (1-4): ").strip()
        lang_map = {'1': 'English', '2': 'Spanish', '3': 'French', '4': 'Hindi'}
        if lang_choice in lang_map:
            prefs['language'] = lang_map[lang_choice]
        
        # Units
        print("Unit system: 1) Metric 2) Imperial")
        unit_choice = input("Select unit system (1-2): ").strip()
        if unit_choice == '1':
            prefs['units'] = 'metric'
        elif unit_choice == '2':
            prefs['units'] = 'imperial'
        
        # Notifications
        notif_choice = input("Enable notifications? (y/N): ").lower()
        prefs['notifications'] = notif_choice == 'y'
        
        # Treatment preference
        print("Treatment preference: 1) Organic 2) Chemical 3) Integrated")
        treat_choice = input("Select treatment preference (1-3): ").strip()
        treat_map = {'1': 'organic', '2': 'chemical', '3': 'integrated'}
        if treat_choice in treat_map:
            prefs['treatment_preference'] = treat_map[treat_choice]
        
        self.current_profile['preferences'] = prefs
        print("Preferences updated.")
    
    def export_profile(self):
        """Export profile to different formats"""
        print("\n--- EXPORT PROFILE ---")
        print("1. Export as JSON")
        print("2. Export as Text Summary")
        
        choice = input("Select export format (1-2): ").strip()
        
        if choice == '1':
            self.export_as_json()
        elif choice == '2':
            self.export_as_text()
        else:
            print("Invalid choice.")
    
    def export_as_json(self):
        """Export profile as JSON file"""
        try:
            export_file = self.profile_path / f'profile_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            profile_copy = self.current_profile.copy()
            profile_copy['registration_date'] = self.current_profile['registration_date'].isoformat()
            
            with open(export_file, 'w') as f:
                json.dump(profile_copy, f, indent=2)
            
            print(f"Profile exported as JSON: {export_file}")
            
        except Exception as e:
            print(f"Error exporting JSON: {e}")
    
    def export_as_text(self):
        """Export profile as text summary"""
        try:
            export_file = self.profile_path / f'profile_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            
            with open(export_file, 'w') as f:
                f.write("FARMER PROFILE SUMMARY\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"Farmer ID: {self.current_profile.get('farmer_id', 'Not set')}\n")
                f.write(f"Name: {self.current_profile.get('farmer_name', 'Not set')}\n")
                f.write(f"Farm Name: {self.current_profile.get('farm_name', 'Not set')}\n")
                f.write(f"Location: {self.current_profile.get('location', 'Not set')}\n")
                f.write(f"Contact: {self.current_profile.get('phone', 'Not set')}\n")
                f.write(f"Email: {self.current_profile.get('email', 'Not set')}\n")
                f.write(f"Total Area: {self.current_profile.get('total_area', 0)} hectares\n")
                f.write(f"Primary Crops: {', '.join(self.current_profile.get('primary_crops', []))}\n")
                f.write(f"Experience: {self.current_profile.get('farming_experience', 0)} years\n")
                f.write(f"Farming Type: {self.current_profile.get('farming_type', 'Not set')}\n")
                f.write(f"Registration: {self.current_profile['registration_date'].strftime('%Y-%m-%d')}\n")
                
                prefs = self.current_profile.get('preferences', {})
                f.write(f"\nPreferences:\n")
                f.write(f"Language: {prefs.get('language', 'English')}\n")
                f.write(f"Units: {prefs.get('units', 'metric')}\n")
                f.write(f"Treatment Preference: {prefs.get('treatment_preference', 'integrated')}\n")
            
            print(f"Profile exported as text: {export_file}")
            
        except Exception as e:
            print(f"Error exporting text: {e}")
    
    def reset_profile(self):
        """Reset profile to default values"""
        print("\n--- RESET PROFILE ---")
        confirm = input("Are you sure you want to reset all profile data? (type 'RESET' to confirm): ")
        
        if confirm == 'RESET':
            self.current_profile = self.create_default_profile()
            self.save_profile()
            print("Profile has been reset to default values.")
        else:
            print("Reset cancelled.")
    
    def get_profile_summary(self):
        """Get a summary of profile completeness"""
        required_fields = ['farmer_id', 'farmer_name', 'location', 'total_area']
        completed_fields = 0
        
        for field in required_fields:
            if self.current_profile.get(field):
                completed_fields += 1
        
        completeness = (completed_fields / len(required_fields)) * 100
        
        return {
            'completeness_percentage': completeness,
            'completed_fields': completed_fields,
            'total_fields': len(required_fields),
            'missing_fields': [field for field in required_fields if not self.current_profile.get(field)]
        }
    
    def run(self):
        """Main user profile interface"""
        while True:
            print(f"\n--- USER PROFILE MODULE ---")
            
            # Show profile completeness
            summary = self.get_profile_summary()
            print(f"Profile Completeness: {summary['completeness_percentage']:.0f}%")
            
            print("1. View Profile")
            print("2. Update Basic Information")
            print("3. Update Farm Information")
            print("4. Update Preferences")
            print("5. Export Profile")
            print("6. Reset Profile")
            print("7. Profile Statistics")
            print("8. Return to Main Menu")
            
            choice = input("Select option (1-8): ").strip()
            
            if choice == '1':
                self.display_profile()
                
            elif choice == '2':
                self.update_basic_info()
                self.save_profile()
                
            elif choice == '3':
                self.update_farm_info()
                self.save_profile()
                
            elif choice == '4':
                self.update_preferences()
                self.save_profile()
                
            elif choice == '5':
                self.export_profile()
                
            elif choice == '6':
                self.reset_profile()
                
            elif choice == '7':
                self.display_profile_statistics()
                
            elif choice == '8':
                break
                
            else:
                print("Invalid choice. Please select 1-8.")
    
    def display_profile_statistics(self):
        """Display profile statistics and completeness"""
        summary = self.get_profile_summary()
        
        print(f"\n--- PROFILE STATISTICS ---")
        print(f"Profile Completeness: {summary['completeness_percentage']:.0f}%")
        print(f"Completed Fields: {summary['completed_fields']}/{summary['total_fields']}")
        
        if summary['missing_fields']:
            print(f"Missing Fields: {', '.join(summary['missing_fields'])}")
        
        # Additional statistics
        if self.current_profile.get('primary_crops'):
            print(f"Number of Primary Crops: {len(self.current_profile['primary_crops'])}")
        
        reg_date = self.current_profile['registration_date']
        days_since_reg = (datetime.now() - reg_date).days
        print(f"Days Since Registration: {days_since_reg}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if summary['completeness_percentage'] < 100:
            print("- Complete missing profile fields for better recommendations")
        if not self.current_profile.get('primary_crops'):
            print("- Add primary crops for crop-specific advice")
        if self.current_profile.get('total_area', 0) == 0:
            print("- Set farm area for accurate fertilizer calculations")