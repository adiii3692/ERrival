import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_synthetic_triage_data(n_samples=5000):
    """
    Generate synthetic ER triage data that matches the database schema
    and produces realistic medical scenarios for ML training.
    """
    
    # Define realistic symptom categories and their typical triage levels
    symptom_patterns = {
        # Critical (Level 1) - Life threatening
        'critical': {
            'symptoms': ['chest_pain', 'difficulty_breathing', 'severe_bleeding', 'unconscious', 'cardiac_arrest'],
            'temp_range': (98.0, 104.0),
            'pulse_range': (40, 180),
            'bp_sys_range': (60, 200),
            'bp_dias_range': (40, 120),
            'resp_range': (8, 40),
            'o2_sat_range': (70, 95),
            'pain_range': (7, 10),
            'triage_level': 1
        },
        # Urgent (Level 2) - Serious but stable
        'urgent': {
            'symptoms': ['severe_pain', 'high_fever', 'vomiting', 'severe_headache', 'allergic_reaction'],
            'temp_range': (99.0, 103.0),
            'pulse_range': (60, 140),
            'bp_sys_range': (90, 180),
            'bp_dias_range': (60, 110),
            'resp_range': (12, 30),
            'o2_sat_range': (88, 98),
            'pain_range': (5, 9),
            'triage_level': 2
        },
        # Semi-urgent (Level 3) - Moderate concern
        'semi_urgent': {
            'symptoms': ['moderate_pain', 'fever', 'nausea', 'dizziness', 'minor_bleeding'],
            'temp_range': (98.5, 101.5),
            'pulse_range': (70, 120),
            'bp_sys_range': (100, 160),
            'bp_dias_range': (65, 100),
            'resp_range': (14, 24),
            'o2_sat_range': (92, 99),
            'pain_range': (3, 7),
            'triage_level': 3
        },
        # Less urgent (Level 4) - Mild symptoms
        'less_urgent': {
            'symptoms': ['mild_pain', 'cold_symptoms', 'minor_cut', 'rash', 'sore_throat'],
            'temp_range': (98.0, 100.5),
            'pulse_range': (60, 100),
            'bp_sys_range': (110, 140),
            'bp_dias_range': (70, 90),
            'resp_range': (16, 22),
            'o2_sat_range': (95, 100),
            'pain_range': (1, 5),
            'triage_level': 4
        },
        # Non-urgent (Level 5) - Routine care
        'non_urgent': {
            'symptoms': ['check_up', 'prescription_refill', 'minor_rash', 'follow_up', 'vaccination'],
            'temp_range': (97.5, 99.5),
            'pulse_range': (60, 90),
            'bp_sys_range': (110, 130),
            'bp_dias_range': (70, 85),
            'resp_range': (16, 20),
            'o2_sat_range': (97, 100),
            'pain_range': (0, 3),
            'triage_level': 5
        }
    }
    
    # Age distribution affects vital signs and triage decisions
    age_groups = {
        'pediatric': {'age_range': (0, 17), 'weight': 0.15},
        'adult': {'age_range': (18, 64), 'weight': 0.65},
        'elderly': {'age_range': (65, 95), 'weight': 0.20}
    }
    
    data = []
    
    for i in range(n_samples):
        # Choose severity category based on realistic ER distributions
        severity_weights = [0.05, 0.15, 0.30, 0.35, 0.15]  # Critical to Non-urgent
        severity_category = np.random.choice(list(symptom_patterns.keys()), 
                                           p=severity_weights)
        pattern = symptom_patterns[severity_category]
        
        # Choose age group
        age_group = np.random.choice(list(age_groups.keys()), 
                                   p=[ag['weight'] for ag in age_groups.values()])
        age = np.random.randint(*age_groups[age_group]['age_range'])
        
        # Age-adjusted vital signs
        age_factor = 1.0
        if age < 18:  # Pediatric adjustments
            age_factor = 0.8 + (age / 18) * 0.2  # Gradual increase
        elif age > 65:  # Elderly adjustments
            age_factor = 1.0 + (age - 65) * 0.005  # Slight increase with age
        
        # Generate vital signs within pattern ranges, adjusted for age
        temperature = np.random.uniform(*pattern['temp_range'])
        pulse = int(np.random.uniform(*pattern['pulse_range']) * age_factor)
        bp_systolic = int(np.random.uniform(*pattern['bp_sys_range']) * age_factor)
        bp_diastolic = int(np.random.uniform(*pattern['bp_dias_range']) * age_factor)
        respiratory_rate = int(np.random.uniform(*pattern['resp_range']))
        oxygen_saturation = np.random.uniform(*pattern['o2_sat_range'])
        pain_scale = int(np.random.uniform(*pattern['pain_range']))
        
        # Add some realistic noise/variation
        temperature += np.random.normal(0, 0.2)
        pulse += int(np.random.normal(0, 5))
        bp_systolic += int(np.random.normal(0, 10))
        bp_diastolic += int(np.random.normal(0, 5))
        oxygen_saturation += np.random.normal(0, 1)
        
        # Clamp values to realistic ranges
        temperature = np.clip(temperature, 95.0, 108.0)
        pulse = np.clip(pulse, 30, 200)
        bp_systolic = np.clip(bp_systolic, 60, 220)
        bp_diastolic = np.clip(bp_diastolic, 40, 130)
        oxygen_saturation = np.clip(oxygen_saturation, 70, 100)
        respiratory_rate = np.clip(respiratory_rate, 8, 50)
        
        # Choose primary symptom
        primary_symptom = np.random.choice(pattern['symptoms'])
        
        # Generate arrival time (simulate different times of day)
        base_time = datetime.now() - timedelta(days=np.random.randint(0, 30))
        arrival_hour = np.random.choice(range(24), p=get_hourly_distribution())
        arrival_time = base_time.replace(hour=arrival_hour, 
                                       minute=np.random.randint(0, 60),
                                       second=np.random.randint(0, 60))
        
        # Create feature vector for ML model
        row = {
            # Demographics
            'age': age,
            'gender': np.random.choice(['M', 'F']),
            
            # Vital signs - primary features for ML
            'temperature': round(temperature, 1),
            'pulse': pulse,
            'blood_pressure_systolic': bp_systolic,
            'blood_pressure_diastolic': bp_diastolic,
            'respiratory_rate': respiratory_rate,
            'oxygen_saturation': round(oxygen_saturation, 1),
            'pain_scale': pain_scale,
            
            # Time-based features
            'arrival_hour': arrival_hour,
            'is_weekend': arrival_time.weekday() >= 5,
            
            # Symptom encoding (simplified - in reality you'd use more sophisticated NLP)
            'primary_symptom': primary_symptom,
            
            # Derived features
            'pulse_pressure': bp_systolic - bp_diastolic,
            'shock_index': pulse / bp_systolic if bp_systolic > 0 else 0,
            'modified_early_warning_score': calculate_mews(temperature, pulse, bp_systolic, 
                                                          respiratory_rate, oxygen_saturation),
            
            # Target variable
            'triage_level': pattern['triage_level'],
            
            # Additional fields for completeness
            'arrival_time': arrival_time,
            'chief_complaint': f"Patient presents with {primary_symptom.replace('_', ' ')}"
        }
        
        data.append(row)
    
    return pd.DataFrame(data)

def get_hourly_distribution():
    """Return realistic hourly distribution for ER arrivals"""
    # Higher during evening hours, lower during early morning
    hours = np.array([0.02, 0.015, 0.01, 0.01, 0.015, 0.02, 0.03, 0.04, 
                     0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08,
                     0.085, 0.09, 0.095, 0.08, 0.07, 0.06, 0.04, 0.03])
    return hours / hours.sum()

def calculate_mews(temp, pulse, bp_sys, resp_rate, o2_sat):
    """Calculate Modified Early Warning Score - a real clinical scoring system"""
    score = 0
    
    # Temperature scoring
    if temp <= 35.0: score += 2
    elif temp >= 38.5: score += 1
    
    # Pulse scoring
    if pulse <= 40: score += 2
    elif pulse <= 50: score += 1
    elif pulse >= 130: score += 2
    elif pulse >= 110: score += 1
    
    # Blood pressure scoring
    if bp_sys <= 90: score += 2
    elif bp_sys <= 100: score += 1
    elif bp_sys >= 200: score += 2
    
    # Respiratory rate scoring
    if resp_rate <= 8: score += 2
    elif resp_rate >= 25: score += 2
    elif resp_rate >= 21: score += 1
    
    # Oxygen saturation scoring
    if o2_sat <= 91: score += 2
    elif o2_sat <= 93: score += 1
    
    return score

def create_feature_matrix(df):
    """Create feature matrix suitable for ML training"""
    # Encode categorical variables
    df_encoded = df.copy()
    
    # One-hot encode gender
    df_encoded['gender_M'] = (df_encoded['gender'] == 'M').astype(int)
    df_encoded['gender_F'] = (df_encoded['gender'] == 'F').astype(int)
    
    # One-hot encode primary symptoms (top symptoms only)
    top_symptoms = df['primary_symptom'].value_counts().head(10).index
    for symptom in top_symptoms:
        df_encoded[f'symptom_{symptom}'] = (df_encoded['primary_symptom'] == symptom).astype(int)
    
    # Select features for ML model
    feature_columns = [
        'age', 'temperature', 'pulse', 'blood_pressure_systolic', 
        'blood_pressure_diastolic', 'respiratory_rate', 'oxygen_saturation',
        'pain_scale', 'arrival_hour', 'is_weekend', 'pulse_pressure',
        'shock_index', 'modified_early_warning_score', 'gender_M'
    ]
    
    # Add symptom features
    feature_columns.extend([col for col in df_encoded.columns if col.startswith('symptom_')])
    
    X = df_encoded[feature_columns].fillna(0)
    y = df_encoded['triage_level']
    
    return X, y, feature_columns

if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic ER triage dataset...")
    df = generate_synthetic_triage_data(n_samples=10000)
    
    # Display basic statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTriage level distribution:")
    print(df['triage_level'].value_counts().sort_index())
    
    print(f"\nVital signs summary:")
    vital_cols = ['temperature', 'pulse', 'blood_pressure_systolic', 'oxygen_saturation']
    print(df[vital_cols].describe())
    
    # Save full dataset
    df.to_csv('er_triage_dataset.csv', index=False)
    print(f"\nFull dataset saved to 'er_triage_dataset.csv'")
    
    # Create ML-ready features
    X, y, feature_names = create_feature_matrix(df)
    
    # Save ML-ready dataset
    ml_df = pd.concat([X, y], axis=1)
    ml_df.to_csv('er_triage_ml_dataset.csv', index=False)
    print(f"ML-ready dataset saved to 'er_triage_ml_dataset.csv'")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    print("\n✅ Dataset generation complete!")
    print("You can now use this data to train your Random Forest model.")