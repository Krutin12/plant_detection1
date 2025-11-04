"""
Fertilizer Calculator Module
===========================
Handles crop-specific fertilizer recommendations and calculations.
"""

import json
from datetime import datetime
from pathlib import Path

class FertilizerCalculatorModule:
    """Fertilizer calculation and recommendation system"""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / 'fertilizer_data'
        self.data_path.mkdir(exist_ok=True)
        
        self.fertilizer_history = []
        self.crop_requirements = self.load_crop_requirements()
        
    def load_crop_requirements(self):
        """Load crop-specific fertilizer requirements"""
        return {
            'Apple': {
                'base_npk': {'N': 150, 'P': 75, 'K': 200},
                'micronutrients': ['Calcium', 'Magnesium', 'Boron', 'Zinc'],
                'ph_range': (6.0, 7.0),
                'growth_stages': {
                    'flowering': {'N': 1.2, 'P': 1.5, 'K': 1.0},
                    'fruiting': {'N': 0.8, 'P': 1.3, 'K': 1.8},
                    'dormant': {'N': 0.5, 'P': 0.7, 'K': 0.6}
                }
            },
            'Tomato': {
                'base_npk': {'N': 120, 'P': 100, 'K': 150},
                'micronutrients': ['Calcium', 'Magnesium', 'Boron'],
                'ph_range': (6.0, 7.0),
                'growth_stages': {
                    'seedling': {'N': 0.8, 'P': 1.2, 'K': 0.9},
                    'flowering': {'N': 1.0, 'P': 1.5, 'K': 1.2},
                    'fruiting': {'N': 0.9, 'P': 1.1, 'K': 1.8}
                }
            },
            'Potato': {
                'base_npk': {'N': 140, 'P': 60, 'K': 180},
                'micronutrients': ['Sulfur', 'Calcium', 'Magnesium'],
                'ph_range': (5.0, 6.5),
                'growth_stages': {
                    'planting': {'N': 0.5, 'P': 1.0, 'K': 0.6},
                    'tuber_formation': {'N': 1.2, 'P': 1.3, 'K': 1.5},
                    'bulking': {'N': 0.8, 'P': 1.0, 'K': 1.8}
                }
            },
            'Corn': {
                'base_npk': {'N': 180, 'P': 80, 'K': 120},
                'micronutrients': ['Zinc', 'Iron', 'Manganese'],
                'ph_range': (6.0, 6.8),
                'growth_stages': {
                    'planting': {'N': 0.3, 'P': 1.0, 'K': 0.8},
                    'vegetative': {'N': 1.5, 'P': 1.2, 'K': 1.0},
                    'tasseling': {'N': 1.2, 'P': 1.0, 'K': 1.3}
                }
            }
        }
    
    def calculate_fertilizer_plan(self, crop_type, area_hectares, growth_stage='vegetative', 
                                soil_ph=6.5, disease_present=None):
        """Calculate fertilizer requirements"""
        
        if crop_type not in self.crop_requirements:
            crop_type = 'Tomato'  # Default
        
        crop_data = self.crop_requirements[crop_type]
        base_npk = crop_data['base_npk'].copy()
        
        # Apply growth stage multipliers
        stage_multipliers = crop_data['growth_stages'].get(growth_stage, 
                                                         {'N': 1.0, 'P': 1.0, 'K': 1.0})
        
        # Calculate requirements
        requirements = {}
        for nutrient in ['N', 'P', 'K']:
            requirements[nutrient] = base_npk[nutrient] * stage_multipliers[nutrient] * area_hectares
        
        # Adjust for disease
        if disease_present and 'disease' in disease_present.lower():
            requirements['K'] *= 1.3  # Increase potassium for disease resistance
            requirements['N'] *= 0.8  # Reduce nitrogen for some diseases
        
        # Generate recommendations
        recommendations = self.generate_recommendations(crop_type, requirements, 
                                                      crop_data['micronutrients'])
        
        fertilizer_plan = {
            'crop_type': crop_type,
            'area_hectares': area_hectares,
            'growth_stage': growth_stage,
            'soil_ph': soil_ph,
            'requirements': requirements,
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }
        
        return fertilizer_plan
    
    def generate_recommendations(self, crop_type, requirements, micronutrients):
        """Generate specific fertilizer recommendations"""
        recommendations = []
        
        # NPK fertilizer
        n_needed = requirements['N']
        p_needed = requirements['P']
        k_needed = requirements['K']
        
        recommendations.append({
            'type': 'NPK Fertilizer',
            'product': f'NPK 20-20-20',
            'quantity': f'{(n_needed + p_needed + k_needed) / 3:.0f} kg',
            'application': 'Apply in 2-3 split doses during growing season'
        })
        
        # Additional nitrogen if needed
        if n_needed > 120:
            recommendations.append({
                'type': 'Nitrogen Supplement',
                'product': 'Urea (46% N)',
                'quantity': f'{(n_needed - 120) / 0.46:.0f} kg',
                'application': 'Top dressing during vegetative growth'
            })
        
        # Micronutrients
        recommendations.append({
            'type': 'Micronutrients',
            'product': 'Micronutrient Mix',
            'quantity': '10-15 kg per hectare',
            'application': f'Contains {", ".join(micronutrients[:3])}'
        })
        
        return recommendations
    
    def display_fertilizer_plan(self, plan):
        """Display fertilizer plan in formatted way"""
        print(f"\n" + "="*50)
        print("FERTILIZER RECOMMENDATION PLAN")
        print("="*50)
        print(f"Crop: {plan['crop_type']}")
        print(f"Area: {plan['area_hectares']} hectares")
        print(f"Growth Stage: {plan['growth_stage']}")
        print(f"Soil pH: {plan['soil_ph']}")
        
        print(f"\nNUTRIENT REQUIREMENTS:")
        for nutrient, amount in plan['requirements'].items():
            print(f"{nutrient}: {amount:.0f} kg")
        
        print(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(plan['recommendations'], 1):
            print(f"{i}. {rec['type']}")
            print(f"   Product: {rec['product']}")
            print(f"   Quantity: {rec['quantity']}")
            print(f"   Application: {rec['application']}")
            print()
    
    def save_fertilizer_plan(self, plan):
        """Save fertilizer plan to history"""
        plan_record = {
            'id': len(self.fertilizer_history) + 1,
            'timestamp': datetime.now(),
            'plan': plan
        }
        
        self.fertilizer_history.append(plan_record)
        
        # Save to file
        try:
            history_file = self.data_path / 'fertilizer_history.json'
            with open(history_file, 'w') as f:
                json.dump([{**record, 'timestamp': record['timestamp'].isoformat()} 
                          for record in self.fertilizer_history], f, indent=2)
        except Exception as e:
            print(f"Error saving fertilizer history: {e}")
        
        return plan_record['id']
    
    def run_calculator(self, detected_disease=None):
        """Run interactive fertilizer calculator"""
        print("\n--- FERTILIZER CALCULATOR ---")
        
        # Get crop information
        crop_types = list(self.crop_requirements.keys())
        print("Available crop types:")
        for i, crop in enumerate(crop_types, 1):
            print(f"{i}. {crop}")
        
        try:
            crop_choice = int(input("Select crop type (1-{}): ".format(len(crop_types)))) - 1
            crop_type = crop_types[crop_choice] if 0 <= crop_choice < len(crop_types) else 'Tomato'
        except:
            crop_type = 'Tomato'
        
        # Get area
        try:
            area = float(input("Enter area in hectares: ") or "1.0")
        except:
            area = 1.0
        
        # Get growth stage
        stages = ['vegetative', 'flowering', 'fruiting']
        print("Growth stages: 1) Vegetative 2) Flowering 3) Fruiting")
        try:
            stage_choice = int(input("Select growth stage (1-3): ")) - 1
            growth_stage = stages[stage_choice] if 0 <= stage_choice < len(stages) else 'vegetative'
        except:
            growth_stage = 'vegetative'
        
        # Get soil pH
        try:
            soil_ph = float(input("Enter soil pH (4.0-8.0): ") or "6.5")
        except:
            soil_ph = 6.5
        
        # Calculate fertilizer plan
        plan = self.calculate_fertilizer_plan(crop_type, area, growth_stage, 
                                            soil_ph, detected_disease)
        
        # Display plan
        self.display_fertilizer_plan(plan)
        
        # Save plan
        save_plan = input("\nSave this fertilizer plan? (y/N): ").lower() == 'y'
        if save_plan:
            plan_id = self.save_fertilizer_plan(plan)
            print(f"Fertilizer plan saved with ID: {plan_id}")
    
    def view_history(self):
        """View fertilizer plan history"""
        if not self.fertilizer_history:
            print("No fertilizer plans in history.")
            return
        
        print(f"\n--- FERTILIZER PLAN HISTORY ---")
        for record in reversed(self.fertilizer_history[-10:]):  # Show last 10
            plan = record['plan']
            print(f"ID: {record['id']} | Date: {record['timestamp'].strftime('%Y-%m-%d')}")
            print(f"Crop: {plan['crop_type']} | Area: {plan['area_hectares']} ha")
            print(f"Stage: {plan['growth_stage']}")
            print(f"NPK Requirements: N={plan['requirements']['N']:.0f}, "
                  f"P={plan['requirements']['P']:.0f}, K={plan['requirements']['K']:.0f}")
            print("-" * 50)
    
    def run(self):
        """Main fertilizer calculator interface"""
        while True:
            print(f"\n--- FERTILIZER CALCULATOR MODULE ---")
            print("1. Calculate New Fertilizer Plan")
            print("2. View Fertilizer History")
            print("3. Cost Estimation")
            print("4. Return to Main Menu")
            
            choice = input("Select option (1-4): ").strip()
            
            if choice == '1':
                self.run_calculator()
            elif choice == '2':
                self.view_history()
            elif choice == '3':
                self.estimate_costs()
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please select 1-4.")
    
    def estimate_costs(self):
        """Estimate fertilizer costs"""
        if not self.fertilizer_history:
            print("No fertilizer plans available for cost estimation.")
            return
        
        # Simple cost estimation (prices in local currency)
        fertilizer_prices = {
            'NPK': 25,  # per kg
            'Urea': 15,  # per kg
            'Micronutrient': 100  # per kg
        }
        
        print("\n--- COST ESTIMATION ---")
        latest_plan = self.fertilizer_history[-1]['plan']
        
        total_cost = 0
        for rec in latest_plan['recommendations']:
            # Extract quantity (simplified)
            quantity_str = rec['quantity']
            try:
                quantity = float(quantity_str.split()[0])
                if 'NPK' in rec['product']:
                    cost = quantity * fertilizer_prices['NPK']
                elif 'Urea' in rec['product']:
                    cost = quantity * fertilizer_prices['Urea']
                else:
                    cost = quantity * fertilizer_prices['Micronutrient']
                
                total_cost += cost
                print(f"{rec['type']}: {quantity} kg × {cost/quantity:.0f} = {cost:.2f}")
            except:
                print(f"{rec['type']}: Cost calculation unavailable")
        
        print(f"\nTotal Estimated Cost: {total_cost:.2f} (local currency)")