"""
Plant Disease Detection System - Streamlit Application
====================================================
A comprehensive web application for plant disease detection, treatment management,
fertilizer recommendations, and farm analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
import zipfile
import shutil

# Import your existing modules
import sys
sys.path.append('.')

# Try to import the existing modules
try:
    from user_profile import UserProfileModule
    from treatment_history import TreatmentHistoryModule
    from fertilizer_calculator import FertilizerCalculatorModule
    from farm_analytics import FarmAnalyticsModule
    from export_reports import ExportReportsModule
    from detection_history import DetectionHistoryModule
    from data_management import DataManagementModule
    from disease_detection_streamlit import DiseaseDetectionStreamlit
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: black;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .disease-card {
        background-color: black;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 0.5rem 0;
    }
    .healthy-card {
        background-color: black;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #32cd32;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.base_path = Path('.')
    st.session_state.user_profile = None
    st.session_state.detection_history = None
    st.session_state.treatment_history = None
    st.session_state.fertilizer_calc = None
    st.session_state.farm_analytics = None
    st.session_state.export_reports = None
    st.session_state.data_management = None

# Initialize modules
@st.cache_resource
def initialize_modules():
    """Initialize all system modules"""
    base_path = Path('.')
    return {
        'user_profile': UserProfileModule(base_path),
        'detection_history': DetectionHistoryModule(base_path),
        'treatment_history': TreatmentHistoryModule(base_path),
        'fertilizer_calc': FertilizerCalculatorModule(base_path),
        'farm_analytics': FarmAnalyticsModule(base_path),
        'export_reports': ExportReportsModule(base_path),
        'data_management': DataManagementModule(base_path),
        'disease_detector': DiseaseDetectionStreamlit()
    }

modules = initialize_modules()

# Disease classes from your model
DISEASE_CLASSES = [
    
    "Early_Disease_Symptoms", "Severe_Plant_Disease", "Plant_Healthy_Condition",
    "Moderate_Fungal_Disease", "Severe_Plant_Stress"
]

# Main navigation
def main():
    st.markdown('<h1 class="main-header">🌱 Plant Disease Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        [
            "🏠 Dashboard",
            "🔍 Disease Detection", 
            "👤 User Profile",
            "💊 Treatment History",
            "🌾 Fertilizer Calculator",
            "📊 Farm Analytics",
            "📄 Export Reports",
            "🗄️ Data Management"
        ]
    )
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "🔍 Disease Detection":
        disease_detection_page()
    elif page == "👤 User Profile":
        user_profile_page()
    elif page == "💊 Treatment History":
        treatment_history_page()
    elif page == "🌾 Fertilizer Calculator":
        fertilizer_calculator_page()
    elif page == "📊 Farm Analytics":
        farm_analytics_page()
    elif page == "📄 Export Reports":
        export_reports_page()
    elif page == "🗄️ Data Management":
        data_management_page()

# Dashboard Page
def dashboard_page():
    st.header("📊 System Dashboard")
    
    # Get analytics data
    detection_analytics = modules['farm_analytics'].get_detection_analytics()
    treatment_analytics = modules['farm_analytics'].get_treatment_analytics()
    fertilizer_analytics = modules['farm_analytics'].get_fertilizer_analytics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Detections",
            value=detection_analytics.get('total_detections', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Average Confidence",
            value=f"{detection_analytics.get('average_confidence', 0):.1%}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Total Treatments",
            value=treatment_analytics.get('total_treatments', 0),
            delta=None
        )
    
    with col4:
        st.metric(
            label="Success Rate",
            value=f"{treatment_analytics.get('success_rate', 0):.1f}%",
            delta=None
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Disease Distribution")
        if detection_analytics.get('disease_distribution'):
            disease_data = detection_analytics['disease_distribution']
            fig = px.pie(
                values=list(disease_data.values()),
                names=list(disease_data.keys()),
                title="Disease Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No detection data available")
    
    with col2:
        st.subheader("Treatment Status")
        if treatment_analytics.get('status_distribution'):
            status_data = treatment_analytics['status_distribution']
            fig = px.bar(
                x=list(status_data.keys()),
                y=list(status_data.values()),
                title="Treatment Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No treatment data available")
    
    # Recent activity
    st.subheader("Recent Activity")
    if modules['detection_history'].detection_records:
        recent_detections = modules['detection_history'].detection_records[-5:]
        for record in reversed(recent_detections):
            with st.container():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{record.get('predicted_disease', 'Unknown')}**")
                with col2:
                    st.write(f"Confidence: {record.get('confidence', 0):.1%}")
                with col3:
                    st.write(record['timestamp'].strftime('%Y-%m-%d'))
                st.divider()

# Disease Detection Page
def disease_detection_page():
    st.header("🔍 Disease Detection")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload a plant image for disease detection",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a plant leaf or affected area"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                # Use actual disease detection model
                predicted_disease, confidence, additional_info = modules['disease_detector'].predict_disease(image)
                
                # Display results
                st.success("Analysis Complete!")
                
                # Results card
                if "Healthy" in predicted_disease:
                    st.markdown(f"""
                    <div class="healthy-card">
                        <h3>✅ {predicted_disease}</h3>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p>Your plant appears to be healthy! Continue with regular care.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="disease-card">
                        <h3>⚠️ {predicted_disease}</h3>
                        <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        <p>Disease detected. Please check the treatment recommendations below.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display additional analysis
                if additional_info and 'color_ratios' in additional_info:
                    st.subheader("🔬 Image Analysis")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Green Coverage", f"{additional_info['color_ratios'].get('green', 0):.1%}")
                    with col2:
                        st.metric("Brown Areas", f"{additional_info['color_ratios'].get('brown', 0):.1%}")
                    with col3:
                        st.metric("Yellow Areas", f"{additional_info['color_ratios'].get('yellow', 0):.1%}")
                
                # Treatment recommendations
                st.subheader("💊 Treatment Recommendations")
                disease_info = modules['disease_detector'].get_disease_info(predicted_disease)
                if disease_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Symptoms:** {disease_info['symptoms']}")
                        st.write(f"**Severity:** {disease_info['severity']}")
                        st.write(f"**Urgency:** {disease_info['urgency']}")
                    with col2:
                        st.write(f"**Treatment:** {disease_info['treatment']}")
                        st.write(f"**Prevention:** {disease_info['prevention']}")
                
                # Treatment type selection
                st.subheader("🎯 Treatment Plan")
                treatment_type = st.selectbox("Select Treatment Approach", ["integrated", "organic", "chemical"])
                
                if st.button("Get Detailed Treatment Plan"):
                    treatment_plan = modules['disease_detector'].get_treatment_recommendations(predicted_disease, treatment_type)
                    
                    st.write("**Recommended Products:**")
                    for product in treatment_plan['recommended_products']:
                        st.write(f"- {product}")
                    
                    st.write(f"**Application:** {treatment_plan['application_notes']}")
                    st.write(f"**Frequency:** {treatment_plan['frequency']}")
                    st.write(f"**Safety Notes:** {treatment_plan['safety_notes']}")
                
                # Save detection record
                image_path = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                detection_record = {
                    'timestamp': datetime.now(),
                    'predicted_disease': predicted_disease,
                    'confidence': confidence,
                    'image_path': image_path,
                    'additional_info': additional_info
                }
                modules['detection_history'].add_detection(detection_record)
                
                # Action buttons
                # Buttons section
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📝 Create Treatment Plan"):
        st.session_state.create_treatment = predicted_disease
        st.rerun()

with col2:
    if st.button("🌾 Fertilizer Recommendation"):
        st.session_state.fertilizer_recommendation = predicted_disease
        st.rerun()

with col3:
    if st.button("📊 View Similar Cases"):
        st.session_state.view_similar_cases = predicted_disease
        st.rerun()

# Display results OUTSIDE the columns (after the columns section)

# Show Treatment Plan
if hasattr(st.session_state, 'create_treatment') and st.session_state.create_treatment:
    st.subheader("📝 Treatment Plan")
    treatment_plan = modules['treatment_plans'].generate_plan(st.session_state.create_treatment)
    st.write(treatment_plan)
    if st.button("Close Treatment Plan"):
        st.session_state.create_treatment = None
        st.rerun()

# Show Fertilizer Recommendation
if hasattr(st.session_state, 'fertilizer_recommendation') and st.session_state.fertilizer_recommendation:
    st.subheader("🌾 Fertilizer Recommendation")
    fertilizer = modules['fertilizer'].get_recommendation(st.session_state.fertilizer_recommendation)
    st.write(fertilizer)
    if st.button("Close Fertilizer Recommendation"):
        st.session_state.fertilizer_recommendation = None
        st.rerun()

# Show Similar Cases
if hasattr(st.session_state, 'view_similar_cases') and st.session_state.view_similar_cases:
    st.subheader("📊 Similar Cases")
    similar_cases = modules['detection_history'].find_similar_cases(st.session_state.view_similar_cases)
    if similar_cases:
        st.write(f"Found {len(similar_cases)} similar cases")
        for case in similar_cases[:3]:
            st.write(f"- {case['timestamp'].strftime('%Y-%m-%d')}: {case.get('predicted_disease', 'Unknown')}")
    else:
        st.info("No similar cases found")
    if st.button("Close Similar Cases"):
        st.session_state.view_similar_cases = None
        st.rerun()


# User Profile Page
def user_profile_page():
    st.header("👤 User Profile Management")
    
    # Load current profile
    profile = modules['user_profile'].current_profile
    
    # Profile form
    with st.form("user_profile_form"):
        st.subheader("Basic Information")
        
        col1, col2 = st.columns(2)
        with col1:
            farmer_id = st.text_input("Farmer ID", value=profile.get('farmer_id', ''))
            farmer_name = st.text_input("Farmer Name", value=profile.get('farmer_name', ''))
            farm_name = st.text_input("Farm Name", value=profile.get('farm_name', ''))
        
        with col2:
            location = st.text_input("Location", value=profile.get('location', ''))
            phone = st.text_input("Phone", value=profile.get('phone', ''))
            email = st.text_input("Email", value=profile.get('email', ''))
        
        st.subheader("Farm Information")
        col1, col2 = st.columns(2)
        with col1:
            total_area = st.number_input("Total Area (hectares)", value=profile.get('total_area', 0.0), min_value=0.0)
            farming_experience = st.number_input("Farming Experience (years)", value=profile.get('farming_experience', 0), min_value=0)
        
        with col2:
            primary_crops = st.multiselect(
                "Primary Crops",
                ["Apple", "Tomato", "Potato", "Corn", "Grape", "Wheat", "Rice"],
                default=profile.get('primary_crops', [])
            )
            farming_type = st.selectbox(
                "Farming Type",
                ["organic", "conventional", "mixed"],
                index=["organic", "conventional", "mixed"].index(profile.get('farming_type', 'mixed'))
            )
        
        st.subheader("Preferences")
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Language", ["English", "Spanish", "French", "Hindi"], 
                                  index=["English", "Spanish", "French", "Hindi"].index(profile.get('preferences', {}).get('language', 'English')))
            units = st.selectbox("Units", ["metric", "imperial"], 
                               index=["metric", "imperial"].index(profile.get('preferences', {}).get('units', 'metric')))
        
        with col2:
            notifications = st.checkbox("Enable Notifications", value=profile.get('preferences', {}).get('notifications', True))
            treatment_preference = st.selectbox(
                "Treatment Preference",
                ["organic", "chemical", "integrated"],
                index=["organic", "chemical", "integrated"].index(profile.get('preferences', {}).get('treatment_preference', 'integrated'))
            )
        
        # Submit button
        if st.form_submit_button("💾 Save Profile", type="primary"):
            # Update profile
            modules['user_profile'].current_profile.update({
                'farmer_id': farmer_id,
                'farmer_name': farmer_name,
                'farm_name': farm_name,
                'location': location,
                'phone': phone,
                'email': email,
                'total_area': total_area,
                'primary_crops': primary_crops,
                'farming_experience': farming_experience,
                'farming_type': farming_type,
                'preferences': {
                    'language': language,
                    'units': units,
                    'notifications': notifications,
                    'treatment_preference': treatment_preference
                }
            })
            
            modules['user_profile'].save_profile()
            st.success("Profile saved successfully!")
            st.rerun()
    
    # Profile statistics
    st.subheader("Profile Statistics")
    summary = modules['user_profile'].get_profile_summary()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Profile Completeness", f"{summary['completeness_percentage']:.0f}%")
    with col2:
        st.metric("Completed Fields", f"{summary['completed_fields']}/{summary['total_fields']}")
    with col3:
        st.metric("Primary Crops", len(profile.get('primary_crops', [])))

# Treatment History Page
def treatment_history_page():
    st.header("💊 Treatment History")
    
    # Add new treatment
    with st.expander("➕ Add New Treatment Plan"):
        with st.form("new_treatment_form"):
            col1, col2 = st.columns(2)
            with col1:
                disease = st.selectbox("Disease", DISEASE_CLASSES)
                treatment_type = st.selectbox("Treatment Type", ["chemical", "organic", "integrated"])
            with col2:
                status = st.selectbox("Status", ["planned", "in_progress", "completed", "cancelled"])
                planned_date = st.date_input("Planned Date", value=datetime.now().date())
            
            notes = st.text_area("Notes (optional)")
            
            if st.form_submit_button("Add Treatment Plan"):
                # Create treatment plan
                treatment_plan = {
                    'timestamp': datetime.now(),
                    'disease': disease,
                    'treatment_type': treatment_type,
                    'status': status,
                    'planned_date': planned_date,
                    'notes': notes
                }
                
                # Add to treatment history
                modules['treatment_history'].treatment_records.append(treatment_plan)
                modules['treatment_history'].save_treatment_history()
                st.success("Treatment plan added successfully!")
                st.rerun()
    
    # View treatment history
    st.subheader("Treatment History")
    
    if modules['treatment_history'].treatment_records:
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All"] + list(set(record.get('status', '') for record in modules['treatment_history'].treatment_records)))
        with col2:
            disease_filter = st.selectbox("Filter by Disease", ["All"] + list(set(record.get('disease', '') for record in modules['treatment_history'].treatment_records)))
        with col3:
            treatment_type_filter = st.selectbox("Filter by Type", ["All"] + list(set(record.get('treatment_type', '') for record in modules['treatment_history'].treatment_records)))
        
        # Filter records
        filtered_records = modules['treatment_history'].treatment_records
        if status_filter != "All":
            filtered_records = [r for r in filtered_records if r.get('status') == status_filter]
        if disease_filter != "All":
            filtered_records = [r for r in filtered_records if r.get('disease') == disease_filter]
        if treatment_type_filter != "All":
            filtered_records = [r for r in filtered_records if r.get('treatment_type') == treatment_type_filter]
        
        # Display records
        for i, record in enumerate(reversed(filtered_records[-10:])):  # Show last 10
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.write(f"**{record.get('disease', 'Unknown')}**")
                    if record.get('notes'):
                        st.caption(record['notes'])
                with col2:
                    st.write(f"Type: {record.get('treatment_type', 'Unknown')}")
                with col3:
                    st.write(f"Status: {record.get('status', 'Unknown')}")
                with col4:
                    st.write(record['timestamp'].strftime('%Y-%m-%d'))
                st.divider()
    else:
        st.info("No treatment records found. Add a new treatment plan above.")

# Fertilizer Calculator Page
def fertilizer_calculator_page():
    st.header("🌾 Fertilizer Calculator")
    
    with st.form("fertilizer_form"):
        st.subheader("Crop Information")
        
        col1, col2 = st.columns(2)
        with col1:
            crop_type = st.selectbox("Crop Type", ["Apple", "Tomato", "Potato", "Corn"])
            area_hectares = st.number_input("Area (hectares)", min_value=0.1, value=1.0, step=0.1)
        
        with col2:
            growth_stage = st.selectbox("Growth Stage", ["vegetative", "flowering", "fruiting"])
            soil_ph = st.slider("Soil pH", min_value=4.0, max_value=8.0, value=6.5, step=0.1)
        
        disease_present = st.text_input("Disease Present (optional)", placeholder="Enter disease name if any")
        
        if st.form_submit_button("Calculate Fertilizer Plan", type="primary"):
            # Calculate fertilizer plan
            plan = modules['fertilizer_calc'].calculate_fertilizer_plan(
                crop_type, area_hectares, growth_stage, soil_ph, disease_present
            )
            
            # Display results
            st.success("Fertilizer plan calculated successfully!")
            
            # Display plan
            st.subheader("Fertilizer Recommendation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Crop:** {plan['crop_type']}")
                st.write(f"**Area:** {plan['area_hectares']} hectares")
                st.write(f"**Growth Stage:** {plan['growth_stage']}")
                st.write(f"**Soil pH:** {plan['soil_ph']}")
            
            with col2:
                st.write("**Nutrient Requirements:**")
                for nutrient, amount in plan['requirements'].items():
                    st.write(f"- {nutrient}: {amount:.0f} kg")
            
            st.subheader("Recommendations")
            for i, rec in enumerate(plan['recommendations'], 1):
                with st.container():
                    st.write(f"**{i}. {rec['type']}**")
                    st.write(f"Product: {rec['product']}")
                    st.write(f"Quantity: {rec['quantity']}")
                    st.write(f"Application: {rec['application']}")
                    st.divider()
            
            # Save plan
            if st.button("Save Fertilizer Plan"):
                plan_id = modules['fertilizer_calc'].save_fertilizer_plan(plan)
                st.success(f"Fertilizer plan saved with ID: {plan_id}")

# Farm Analytics Page
def farm_analytics_page():
    st.header("📊 Farm Analytics")
    
    # Get analytics data
    detection_analytics = modules['farm_analytics'].get_detection_analytics()
    treatment_analytics = modules['farm_analytics'].get_treatment_analytics()
    fertilizer_analytics = modules['farm_analytics'].get_fertilizer_analytics()
    
    # Detection Analytics
    st.subheader("Detection Analytics")
    
    if detection_analytics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Detections", detection_analytics.get('total_detections', 0))
            st.metric("Average Confidence", f"{detection_analytics.get('average_confidence', 0):.1%}")
        
        with col2:
            st.metric("Most Common Disease", detection_analytics.get('most_common_disease', 'None'))
        
        # Disease distribution chart
        if detection_analytics.get('disease_distribution'):
            disease_data = detection_analytics['disease_distribution']
            fig = px.bar(
                x=list(disease_data.keys()),
                y=list(disease_data.values()),
                title="Disease Distribution"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Treatment Analytics
    st.subheader("Treatment Analytics")
    
    if treatment_analytics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Treatments", treatment_analytics.get('total_treatments', 0))
        with col2:
            st.metric("Success Rate", f"{treatment_analytics.get('success_rate', 0):.1f}%")
        with col3:
            st.metric("Completion Rate", f"{treatment_analytics.get('completion_rate', 0):.1f}%")
        
        # Treatment type distribution
        if treatment_analytics.get('type_distribution'):
            type_data = treatment_analytics['type_distribution']
            fig = px.pie(
                values=list(type_data.values()),
                names=list(type_data.keys()),
                title="Treatment Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Fertilizer Analytics
    st.subheader("Fertilizer Analytics")
    
    if fertilizer_analytics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Plans", fertilizer_analytics.get('total_plans', 0))
            st.metric("Total Area Covered", f"{fertilizer_analytics.get('total_area_covered', 0):.1f} hectares")
        
        with col2:
            st.metric("Average Area per Plan", f"{fertilizer_analytics.get('average_area_per_plan', 0):.1f} hectares")
        
        # Crop distribution
        if fertilizer_analytics.get('crop_distribution'):
            crop_data = fertilizer_analytics['crop_distribution']
            fig = px.bar(
                x=list(crop_data.keys()),
                y=list(crop_data.values()),
                title="Crop Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

# Export Reports Page
def export_reports_page():
    st.header("📄 Export Reports")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Detection History")
        if st.button("Export Detection Data (CSV)"):
            with st.spinner("Exporting..."):
                csv_file = modules['export_reports'].export_detection_csv()
                if csv_file:
                    st.success(f"Exported to: {csv_file}")
        
        if st.button("Export Detection Data (JSON)"):
            with st.spinner("Exporting..."):
                json_file = modules['export_reports'].export_detection_json()
                if json_file:
                    st.success(f"Exported to: {json_file}")
    
    with col2:
        st.subheader("Export Treatment History")
        if st.button("Export Treatment Data (CSV)"):
            with st.spinner("Exporting..."):
                csv_file = modules['export_reports'].export_treatment_csv()
                if csv_file:
                    st.success(f"Exported to: {csv_file}")
        
        if st.button("Export Treatment Data (JSON)"):
            with st.spinner("Exporting..."):
                json_file = modules['export_reports'].export_treatment_json()
                if json_file:
                    st.success(f"Exported to: {json_file}")
    
    # Charts
    st.subheader("Generate Charts")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Disease Distribution Chart"):
            with st.spinner("Generating chart..."):
                chart_file = modules['export_reports'].create_disease_distribution_chart()
                if chart_file:
                    st.success(f"Chart created: {chart_file}")
    
    with col2:
        if st.button("Treatment Status Chart"):
            with st.spinner("Generating chart..."):
                chart_file = modules['export_reports'].create_treatment_status_chart()
                if chart_file:
                    st.success(f"Chart created: {chart_file}")
    
    with col3:
        if st.button("Detection Timeline Chart"):
            with st.spinner("Generating chart..."):
                chart_file = modules['export_reports'].create_detection_timeline_chart()
                if chart_file:
                    st.success(f"Chart created: {chart_file}")
    
    # Comprehensive report
    st.subheader("Comprehensive Report")
    if st.button("Generate Comprehensive Report", type="primary"):
        with st.spinner("Generating comprehensive report..."):
            report_file = modules['export_reports'].create_comprehensive_report()
            if report_file:
                st.success(f"Comprehensive report created: {report_file}")

# Data Management Page
def data_management_page():
    st.header("🗄️ Data Management")
    
    # System status
    st.subheader("System Status")
    stats = modules['data_management'].get_system_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Size", f"{stats['total_size_mb']:.1f} MB")
    with col2:
        st.metric("Health Score", f"{modules['data_management'].calculate_health_score(stats)}/100")
    
    # Backup operations
    st.subheader("Backup Operations")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Create System Backup", type="primary"):
            with st.spinner("Creating backup..."):
                backup_file = modules['data_management'].create_system_backup()
                if backup_file:
                    st.success(f"Backup created: {backup_file}")
    
    with col2:
        if st.button("View Available Backups"):
            backup_files = list(modules['data_management'].backup_path.glob('*.zip'))
            if backup_files:
                for backup_file in sorted(backup_files, reverse=True)[:5]:
                    st.write(f"- {backup_file.name}")
            else:
                st.info("No backups available")
    
    # Cleanup operations
    st.subheader("System Cleanup")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clean Old Backups"):
            cleaned = modules['data_management'].clean_old_backups()
            st.success(f"Cleaned {cleaned} old backup files")
    
    with col2:
        if st.button("Clean Temp Files"):
            cleaned = modules['data_management'].clean_temp_files()
            st.success(f"Cleaned {cleaned} temporary files")
    
    with col3:
        if st.button("Clean Empty Folders"):
            cleaned = modules['data_management'].clean_empty_folders()
            st.success(f"Cleaned {cleaned} empty folders")
    
    # Database maintenance
    st.subheader("Database Maintenance")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Verify Data Integrity"):
            modules['data_management'].verify_data_integrity()
    
    with col2:
        if st.button("Optimize Data Storage"):
            modules['data_management'].optimize_data_storage()

if __name__ == "__main__":
    main()
