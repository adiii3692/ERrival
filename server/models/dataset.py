import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_er_triage_dataset(n_samples=5000):
    """
    Generate synthetic ER triage dataset with realistic medical parameters
    """
    
    # Define symptom categories matching common ER chief complaints
    symptoms = {
        'chest_pain': {
            'weight': 0.8, 
            'keywords': ['chest pain', 'chest pressure', 'heart attack', 'angina', 'cardiac', 'substernal pain'],
            'chief_complaints': ['Chest pain', 'Chest pressure', 'Heart attack symptoms', 'Cardiac chest pain']
        },
        'shortness_of_breath': {
            'weight': 0.7, 
            'keywords': ['shortness of breath', 'difficulty breathing', 'dyspnea', 'asthma', 'pneumonia', 'respiratory distress'],
            'chief_complaints': ['Shortness of breath', 'Difficulty breathing', 'Respiratory distress', 'Asthma exacerbation']
        },
        'severe_headache': {
            'weight': 0.6, 
            'keywords': ['severe headache', 'migraine', 'head injury', 'stroke', 'neurological', 'worst headache'],
            'chief_complaints': ['Severe headache', 'Migraine', 'Head injury', 'Neurological symptoms']
        },
        'abdominal_pain': {
            'weight': 0.5, 
            'keywords': ['abdominal pain', 'stomach pain', 'appendicitis', 'nausea', 'vomiting', 'gastrointestinal'],
            'chief_complaints': ['Abdominal pain', 'Stomach pain', 'Nausea and vomiting', 'GI symptoms']
        },
        'fever': {
            'weight': 0.4, 
            'keywords': ['fever', 'high temperature', 'infection', 'flu', 'chills', 'sepsis'],
            'chief_complaints': ['Fever', 'High fever', 'Flu-like symptoms', 'Infection symptoms']
        },
        'trauma_injury': {
            'weight': 0.6, 
            'keywords': ['trauma', 'injury', 'fracture', 'laceration', 'burn', 'motor vehicle accident', 'fall'],
            'chief_complaints': ['Trauma', 'Motor vehicle accident', 'Fall injury', 'Laceration', 'Fracture']
        },
        'altered_mental_status': {
            'weight': 0.7, 
            'keywords': ['confusion', 'altered mental status', 'unconscious', 'seizure', 'stroke', 'overdose'],
            'chief_complaints': ['Altered mental status', 'Confusion', 'Seizure', 'Unconscious']
        },
        'psychiatric_emergency': {
            'weight': 0.4, 
            'keywords': ['suicidal ideation', 'psychiatric emergency', 'anxiety', 'panic attack', 'depression', 'psychosis'],
            'chief_complaints': ['Suicidal ideation', 'Psychiatric emergency', 'Panic attack', 'Anxiety']
        },
        'back_pain': {
            'weight': 0.3, 
            'keywords': ['back pain', 'spine injury', 'sciatica', 'muscle strain', 'herniated disc'],
            'chief_complaints': ['Back pain', 'Lower back pain', 'Spine injury', 'Sciatica']
        },
        'allergic_reaction': {
            'weight': 0.5, 
            'keywords': ['allergic reaction', 'anaphylaxis', 'rash', 'hives', 'swelling', 'food allergy'],
            'chief_complaints': ['Allergic reaction', 'Anaphylaxis', 'Severe rash', 'Drug reaction']
        },
        'overdose_poisoning': {
            'weight': 0.8, 
            'keywords': ['overdose', 'poisoning', 'drug overdose', 'alcohol poisoning', 'toxic ingestion'],
            'chief_complaints': ['Drug overdose', 'Poisoning', 'Alcohol poisoning', 'Toxic ingestion']
        },
        'eye_problem': {
            'weight': 0.3, 
            'keywords': ['eye pain', 'vision loss', 'eye injury', 'foreign body', 'eye infection'],
            'chief_complaints': ['Eye pain', 'Vision problems', 'Eye injury', 'Foreign body in eye']
        },
        'ear_problem': {
            'weight': 0.2, 
            'keywords': ['ear pain', 'hearing loss', 'ear infection', 'tinnitus', 'ear discharge'],
            'chief_complaints': ['Ear pain', 'Ear infection', 'Hearing loss', 'Ear discharge']
        },
        'urinary_problem': {
            'weight': 0.3, 
            'keywords': ['urinary retention', 'kidney stone', 'UTI', 'blood in urine', 'painful urination'],
            'chief_complaints': ['Urinary retention', 'Kidney stone', 'UTI symptoms', 'Blood in urine']
        },
        'minor_complaint': {
            'weight': 0.1, 
            'keywords': ['minor cut', 'cold symptoms', 'prescription refill', 'routine', 'follow-up'],
            'chief_complaints': ['Minor cut', 'Cold symptoms', 'Prescription refill', 'Routine follow-up']
        }
    }
    
    data = []
    
    for i in range(n_samples):
        # Basic demographics
        age = np.random.normal(45, 20)
        age = max(0, min(100, age))  # Constrain to realistic range
        
        # Gender (affects some vital sign ranges)
        gender = np.random.choice(['M', 'F'], p=[0.48, 0.52])
        
        # Time of arrival (affects severity distribution)
        hour = np.random.randint(0, 24)
        is_night = 1 if hour >= 22 or hour <= 6 else 0
        is_weekend = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Primary symptom category with realistic ER distribution
        symptom_cat = np.random.choice(list(symptoms.keys()), 
                                     p=[0.12, 0.10, 0.08, 0.18, 0.12, 0.15, 0.06, 0.05, 0.08, 0.03, 0.02, 0.01, 0.01, 0.03, 0.05])
        
        # Generate vitals based on age, gender, and symptom severity
        base_severity = symptoms[symptom_cat]['weight']
        
        # Temperature (98.6°F normal, higher for infections/fever)
        if symptom_cat in ['fever', 'allergic_reaction']:
            temperature = np.random.normal(101.5, 2.0)
        elif symptom_cat in ['overdose_poisoning', 'altered_mental_status']:
            temperature = np.random.normal(97.8, 1.5)  # Often hypothermic
        else:
            temperature = np.random.normal(98.6, 0.8)
        temperature = max(95, min(108, temperature))
        
        # Heart rate (60-100 normal)
        if symptom_cat in ['chest_pain', 'shortness_of_breath', 'allergic_reaction', 'overdose_poisoning']:
            pulse = np.random.normal(105, 20)
        elif symptom_cat in ['altered_mental_status', 'psychiatric_emergency']:
            pulse = np.random.normal(85, 15)
        elif age > 65:
            pulse = np.random.normal(75, 12)
        else:
            pulse = np.random.normal(78, 10)
        pulse = max(40, min(180, pulse))
        
        # Blood pressure (systolic/diastolic)
        if symptom_cat in ['chest_pain', 'severe_headache'] or age > 60:
            systolic_bp = np.random.normal(145, 25)
            diastolic_bp = np.random.normal(90, 12)
        elif symptom_cat in ['overdose_poisoning', 'trauma_injury']:
            systolic_bp = np.random.normal(105, 20)  # Often hypotensive
            diastolic_bp = np.random.normal(65, 10)
        else:
            systolic_bp = np.random.normal(120, 15)
            diastolic_bp = np.random.normal(80, 8)
        
        systolic_bp = max(70, min(220, systolic_bp))
        diastolic_bp = max(40, min(130, diastolic_bp))
        
        # Respiratory rate (12-20 normal)
        if symptom_cat in ['shortness_of_breath', 'chest_pain', 'overdose_poisoning']:
            respiratory_rate = np.random.normal(24, 5)
        elif symptom_cat in ['altered_mental_status']:
            respiratory_rate = np.random.normal(10, 3)  # Often depressed
        else:
            respiratory_rate = np.random.normal(16, 3)
        respiratory_rate = max(6, min(45, respiratory_rate))
        
        # Oxygen saturation (95-100% normal)
        if symptom_cat in ['shortness_of_breath', 'overdose_poisoning']:
            oxygen_sat = np.random.normal(90, 4)
        elif symptom_cat in ['chest_pain', 'trauma_injury']:
            oxygen_sat = np.random.normal(94, 3)
        else:
            oxygen_sat = np.random.normal(98, 1.5)
        oxygen_sat = max(70, min(100, oxygen_sat))
        
        # Pain scale (0-10)
        if symptom_cat in ['chest_pain', 'severe_headache', 'trauma_injury', 'back_pain']:
            pain_scale = np.random.randint(7, 11)
        elif symptom_cat in ['abdominal_pain', 'urinary_problem', 'eye_problem']:
            pain_scale = np.random.randint(5, 9)
        elif symptom_cat in ['altered_mental_status', 'overdose_poisoning']:
            pain_scale = np.random.randint(0, 4)  # Often unable to assess
        else:
            pain_scale = np.random.randint(2, 7)
        
        # Duration of symptoms (hours)
        if symptom_cat in ['chest_pain', 'shortness_of_breath', 'allergic_reaction', 'overdose_poisoning']:
            symptom_duration = np.random.exponential(3)  # Acute symptoms
        elif symptom_cat in ['back_pain', 'minor_complaint']:
            symptom_duration = np.random.exponential(48)  # Can be chronic
        else:
            symptom_duration = np.random.exponential(12)
        symptom_duration = min(336, symptom_duration)  # Cap at 2 weeks
        
        # Generate symptom keywords and chief complaint
        selected_keywords = random.sample(symptoms[symptom_cat]['keywords'], 
                                        min(3, len(symptoms[symptom_cat]['keywords'])))
        symptom_keywords = ', '.join(selected_keywords)
        chief_complaint = random.choice(symptoms[symptom_cat]['chief_complaints'])
        
        # Previous ER visits (frequent flyers)
        prev_visits = np.random.poisson(1.5)
        prev_visits = min(20, prev_visits)
        
        # Calculate triage level (1=Critical, 5=Non-urgent)
        severity_score = 0
        
        # Age factor
        if age < 2 or age > 80:
            severity_score += 0.3
        elif age > 65:
            severity_score += 0.2
        
        # Vital signs factors
        if temperature > 103 or temperature < 96:
            severity_score += 0.4
        elif temperature > 100.5:
            severity_score += 0.2
            
        if pulse > 120 or pulse < 50:
            severity_score += 0.4
        elif pulse > 100:
            severity_score += 0.2
            
        if systolic_bp > 180 or systolic_bp < 90:
            severity_score += 0.4
        elif systolic_bp > 140:
            severity_score += 0.2
            
        if respiratory_rate > 24 or respiratory_rate < 12:
            severity_score += 0.3
            
        if oxygen_sat < 90:
            severity_score += 0.5
        elif oxygen_sat < 95:
            severity_score += 0.3
            
        # Symptom-specific factors
        severity_score += base_severity
        
        # Critical symptoms get immediate boost
        if symptom_cat in ['overdose_poisoning', 'altered_mental_status']:
            severity_score += 0.6
        elif symptom_cat in ['chest_pain', 'trauma_injury']:
            severity_score += 0.4
        
        # Pain factor
        if pain_scale >= 8:
            severity_score += 0.3
        elif pain_scale >= 6:
            severity_score += 0.2
            
        # Duration factor (acute symptoms more urgent)
        if symptom_duration < 2:
            severity_score += 0.2
            
        # Time factor (night/weekend can be more urgent)
        if is_night:
            severity_score += 0.1
            
        # Convert severity score to triage level
        if severity_score >= 1.2:
            triage_level = 1  # Critical
        elif severity_score >= 0.8:
            triage_level = 2  # High
        elif severity_score >= 0.5:
            triage_level = 3  # Medium
        elif severity_score >= 0.3:
            triage_level = 4  # Low
        else:
            triage_level = 5  # Non-urgent
            
        # Add some randomness to make it more realistic
        if np.random.random() < 0.1:  # 10% chance of random adjustment
            triage_level = min(5, max(1, triage_level + np.random.choice([-1, 1])))
        
        # Compile record
        record = {
            'patient_id': f'P{i+1:05d}',
            'age': round(age, 1),
            'gender': gender,
            'temperature': round(temperature, 1),
            'pulse': round(pulse),
            'systolic_bp': round(systolic_bp),
            'diastolic_bp': round(diastolic_bp),
            'respiratory_rate': round(respiratory_rate),
            'oxygen_saturation': round(oxygen_sat, 1),
            'pain_scale': pain_scale,
            'symptom_category': symptom_cat,
            'symptom_keywords': symptom_keywords,
            'chief_complaint': chief_complaint,
            'symptom_duration_hours': round(symptom_duration, 1),
            'previous_visits': prev_visits,
            'arrival_hour': hour,
            'is_weekend': is_weekend,
            'triage_level': triage_level
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

def prepare_features_for_ml(df):
    """
    Prepare features for machine learning model
    """
    # Create a copy for processing
    df_processed = df.copy()
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    df_processed['gender_encoded'] = le_gender.fit_transform(df_processed['gender'])
    
    le_symptom = LabelEncoder()
    df_processed['symptom_encoded'] = le_symptom.fit_transform(df_processed['symptom_category'])
    
    # Feature engineering
    df_processed['bp_ratio'] = df_processed['systolic_bp'] / df_processed['diastolic_bp']
    df_processed['age_group'] = pd.cut(df_processed['age'], 
                                     bins=[0, 18, 35, 50, 65, 100], 
                                     labels=['child', 'young_adult', 'adult', 'senior', 'elderly'])
    df_processed['age_group_encoded'] = le_age = LabelEncoder().fit_transform(df_processed['age_group'])
    
    # Vital signs severity indicators
    df_processed['temp_abnormal'] = ((df_processed['temperature'] > 100.4) | 
                                   (df_processed['temperature'] < 96.0)).astype(int)
    df_processed['pulse_abnormal'] = ((df_processed['pulse'] > 100) | 
                                    (df_processed['pulse'] < 60)).astype(int)
    df_processed['bp_high'] = (df_processed['systolic_bp'] > 140).astype(int)
    df_processed['oxygen_low'] = (df_processed['oxygen_saturation'] < 95).astype(int)
    
    # Select features for model
    feature_columns = [
        'age', 'gender_encoded', 'temperature', 'pulse', 'systolic_bp', 
        'diastolic_bp', 'respiratory_rate', 'oxygen_saturation', 'pain_scale',
        'symptom_encoded', 'symptom_duration_hours', 'previous_visits',
        'arrival_hour', 'is_weekend', 'bp_ratio', 'age_group_encoded',
        'temp_abnormal', 'pulse_abnormal', 'bp_high', 'oxygen_low'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed['triage_level']
    
    # Store encoders for later use
    encoders = {
        'gender': le_gender,
        'symptom': le_symptom,
        'feature_columns': feature_columns
    }
    
    return X, y, encoders

def train_random_forest_model(X, y):
    """
    Train Random Forest model for triage classification
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return rf_model, accuracy, feature_importance

def save_model_and_data(model, encoders, df, feature_importance):
    """
    Save the trained model and associated data
    """
    # Save model
    joblib.dump(model, 'er_triage_model.pkl')
    
    # Save encoders
    joblib.dump(encoders, 'er_triage_encoders.pkl')
    
    # Save dataset
    df.to_csv('er_triage_dataset.csv', index=False)
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    print("\nModel and data saved successfully!")
    print("Files created:")
    print("- er_triage_model.pkl")
    print("- er_triage_encoders.pkl") 
    print("- er_triage_dataset.csv")
    print("- feature_importance.csv")

def main():
    """
    Main function to generate data and train model
    """
    print("Generating ER Triage Dataset...")
    
    # Generate dataset
    df = generate_er_triage_dataset(n_samples=5000)
    
    print(f"Dataset generated with {len(df)} samples")
    print(f"Triage level distribution:")
    print(df['triage_level'].value_counts().sort_index())
    
    # Prepare features
    print("\nPreparing features for machine learning...")
    X, y, encoders = prepare_features_for_ml(df)
    
    # Train model
    print("\nTraining Random Forest model...")
    model, accuracy, feature_importance = train_random_forest_model(X, y)
    
    # Save everything
    save_model_and_data(model, encoders, df, feature_importance)
    
    return model, encoders, df, feature_importance

if __name__ == "__main__":
    model, encoders, df, feature_importance = main()