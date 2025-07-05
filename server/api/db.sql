-- ============================================================================
-- ER TRIAGE SYSTEM - DATABASE SCHEMAS
-- ============================================================================

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- PATIENTS TABLE - Core patient information
-- ============================================================================
CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Basic Demographics
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10) CHECK (gender IN ('male', 'female', 'other', 'unknown')),
    
    -- Contact Information
    phone VARCHAR(20),
    email VARCHAR(100),
    address TEXT,
    city VARCHAR(50),
    state VARCHAR(20),
    zip_code VARCHAR(10),
    
    -- Emergency Contact
    emergency_contact_name VARCHAR(100),
    emergency_contact_phone VARCHAR(20),
    emergency_contact_relationship VARCHAR(50),
    
    -- Medical Information
    blood_type VARCHAR(5) CHECK (blood_type IN ('A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-')),
    allergies TEXT,
    medical_history TEXT,
    current_medications TEXT,
    
    -- Insurance
    insurance_provider VARCHAR(100),
    insurance_policy_number VARCHAR(50),
    insurance_group_number VARCHAR(50),
    
    -- System Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for performance
    CONSTRAINT patients_email_unique UNIQUE (email),
    CONSTRAINT patients_phone_unique UNIQUE (phone)
);

-- ============================================================================
-- VISITS TABLE - Each ER visit
-- ============================================================================
CREATE TABLE visits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    
    -- Visit Details
    visit_number VARCHAR(20) UNIQUE NOT NULL, -- Human-readable visit ID
    arrival_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    departure_time TIMESTAMP NULL,
    
    -- Visit Status
    status VARCHAR(20) NOT NULL DEFAULT 'waiting' 
        CHECK (status IN ('waiting', 'in_triage', 'assigned', 'in_treatment', 'discharged', 'admitted', 'transferred', 'left_ama')),
    
    -- Chief Complaint
    chief_complaint TEXT NOT NULL,
    pain_level INTEGER CHECK (pain_level >= 0 AND pain_level <= 10),
    
    -- Arrival Method
    arrival_method VARCHAR(20) CHECK (arrival_method IN ('walk_in', 'ambulance', 'helicopter', 'police', 'other')),
    
    -- Final Outcome
    discharge_disposition VARCHAR(30) CHECK (discharge_disposition IN 
        ('home', 'admitted', 'transferred', 'ama', 'deceased', 'other')),
    discharge_instructions TEXT,
    
    -- System Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- STAFF TABLE - Hospital staff information
-- ============================================================================
CREATE TABLE staff (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Personal Information
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    employee_id VARCHAR(20) UNIQUE NOT NULL,
    
    -- Contact Information
    phone VARCHAR(20),
    email VARCHAR(100) UNIQUE,
    
    -- Professional Information
    role VARCHAR(30) NOT NULL CHECK (role IN 
        ('nurse', 'doctor', 'physician_assistant', 'nurse_practitioner', 'tech', 'admin', 'security', 'other')),
    department VARCHAR(30),
    license_number VARCHAR(50),
    specialization VARCHAR(50),
    
    -- Work Information
    hire_date DATE,
    employment_status VARCHAR(20) DEFAULT 'active' CHECK (employment_status IN ('active', 'inactive', 'terminated')),
    
    -- System Access
    can_triage BOOLEAN DEFAULT FALSE,
    can_assign_rooms BOOLEAN DEFAULT FALSE,
    can_override_triage BOOLEAN DEFAULT FALSE,
    is_admin BOOLEAN DEFAULT FALSE,
    
    -- System Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TRIAGE_RECORDS TABLE - Triage assessments and ML predictions
-- ============================================================================
CREATE TABLE triage_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    visit_id UUID NOT NULL REFERENCES visits(id) ON DELETE CASCADE,
    
    -- Vital Signs
    temperature DECIMAL(4,1), -- Fahrenheit
    blood_pressure_systolic INTEGER,
    blood_pressure_diastolic INTEGER,
    heart_rate INTEGER,
    respiratory_rate INTEGER,
    oxygen_saturation INTEGER CHECK (oxygen_saturation >= 0 AND oxygen_saturation <= 100),
    
    -- Assessment Time
    assessment_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- ML Model Inputs
    age INTEGER NOT NULL,
    symptom_keywords TEXT, -- Comma-separated keywords
    severity_indicators TEXT, -- JSON string of extracted features
    
    -- ML Model Outputs
    ml_triage_level INTEGER CHECK (ml_triage_level >= 1 AND ml_triage_level <= 5),
    ml_confidence_score DECIMAL(3,2) CHECK (ml_confidence_score >= 0 AND ml_confidence_score <= 1),
    ml_model_version VARCHAR(20),
    
    -- Final Triage Decision
    final_triage_level INTEGER NOT NULL CHECK (final_triage_level >= 1 AND final_triage_level <= 5),
    triage_nurse_id UUID REFERENCES staff(id),
    manual_override BOOLEAN DEFAULT FALSE,
    override_reason TEXT,
    
    -- Triage Notes
    nursing_assessment TEXT,
    acuity_factors TEXT, -- JSON string of factors affecting severity
    
    -- System Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- ROOMS TABLE - ER room information
-- ============================================================================
CREATE TABLE rooms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Room Details
    room_number VARCHAR(10) UNIQUE NOT NULL,
    room_name VARCHAR(50),
    
    -- Room Type and Capacity
    room_type VARCHAR(20) NOT NULL CHECK (room_type IN 
        ('trauma', 'resuscitation', 'critical_care', 'general', 'pediatric', 'psychiatric', 'isolation', 'triage')),
    capacity INTEGER NOT NULL DEFAULT 1,
    
    -- Equipment and Features
    equipment_level VARCHAR(20) CHECK (equipment_level IN ('basic', 'intermediate', 'advanced', 'trauma')),
    has_isolation BOOLEAN DEFAULT FALSE,
    has_cardiac_monitor BOOLEAN DEFAULT FALSE,
    has_ventilator BOOLEAN DEFAULT FALSE,
    has_iv_pump BOOLEAN DEFAULT FALSE,
    
    -- Availability
    is_active BOOLEAN DEFAULT TRUE,
    is_available BOOLEAN DEFAULT TRUE,
    maintenance_notes TEXT,
    
    -- Location
    floor_number INTEGER,
    wing VARCHAR(20),
    department VARCHAR(30),
    
    -- System Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- ROOM_ASSIGNMENTS TABLE - Patient room assignments
-- ============================================================================
CREATE TABLE room_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    visit_id UUID NOT NULL REFERENCES visits(id) ON DELETE CASCADE,
    room_id UUID NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    
    -- Assignment Details
    assigned_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    assigned_by UUID REFERENCES staff(id),
    unassigned_at TIMESTAMP NULL,
    
    -- Assignment Status
    status VARCHAR(20) NOT NULL DEFAULT 'assigned' 
        CHECK (status IN ('assigned', 'occupied', 'discharged', 'transferred', 'cancelled')),
    
    -- Priority and Reasoning
    assignment_priority INTEGER CHECK (assignment_priority >= 1 AND assignment_priority <= 5),
    assignment_reason TEXT,
    
    -- Notes
    special_requirements TEXT,
    assignment_notes TEXT,
    
    -- System Fields
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Patients table indexes
CREATE INDEX idx_patients_name ON patients(last_name, first_name);
CREATE INDEX idx_patients_dob ON patients(date_of_birth);

-- Visits table indexes
CREATE INDEX idx_visits_patient ON visits(patient_id);
CREATE INDEX idx_visits_arrival ON visits(arrival_time);
CREATE INDEX idx_visits_status ON visits(status);

-- Triage records indexes
CREATE INDEX idx_triage_visit ON triage_records(visit_id);
CREATE INDEX idx_triage_level ON triage_records(final_triage_level);
CREATE INDEX idx_triage_time ON triage_records(assessment_time);

-- Room assignments indexes
CREATE INDEX idx_room_assignments_visit ON room_assignments(visit_id);
CREATE INDEX idx_room_assignments_room ON room_assignments(room_id);
CREATE INDEX idx_room_assignments_status ON room_assignments(status);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers to relevant tables
CREATE TRIGGER update_patients_updated_at BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_visits_updated_at BEFORE UPDATE ON visits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_triage_records_updated_at BEFORE UPDATE ON triage_records
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rooms_updated_at BEFORE UPDATE ON rooms
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_room_assignments_updated_at BEFORE UPDATE ON room_assignments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_staff_updated_at BEFORE UPDATE ON staff
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SAMPLE DATA VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Active patients in ER
CREATE VIEW active_patients AS
SELECT 
    p.id,
    p.first_name,
    p.last_name,
    v.visit_number,
    v.arrival_time,
    v.status,
    v.chief_complaint,
    tr.final_triage_level,
    ra.room_id,
    r.room_number
FROM patients p
JOIN visits v ON p.id = v.patient_id
LEFT JOIN triage_records tr ON v.id = tr.visit_id
LEFT JOIN room_assignments ra ON v.id = ra.visit_id AND ra.status = 'assigned'
LEFT JOIN rooms r ON ra.room_id = r.id
WHERE v.status IN ('waiting', 'in_triage', 'assigned', 'in_treatment')
ORDER BY tr.final_triage_level ASC, v.arrival_time ASC;

-- Room occupancy status
CREATE VIEW room_status AS
SELECT 
    r.id,
    r.room_number,
    r.room_type,
    r.capacity,
    r.is_available,
    COUNT(ra.id) as current_occupancy,
    (r.capacity - COUNT(ra.id)) as available_beds
FROM rooms r
LEFT JOIN room_assignments ra ON r.id = ra.room_id AND ra.status = 'assigned'
WHERE r.is_active = TRUE
GROUP BY r.id, r.room_number, r.room_type, r.capacity, r.is_available;

-- ============================================================================
-- SAMPLE QUERIES FOR COMMON OPERATIONS
-- ============================================================================

-- Find available rooms for specific triage level
/*
SELECT r.*, rs.available_beds
FROM rooms r
JOIN room_status rs ON r.id = rs.id
WHERE rs.available_beds > 0
  AND r.room_type IN ('trauma', 'critical_care') -- For high-priority patients
ORDER BY rs.available_beds DESC;
*/

-- Get patient's complete visit history
/*
SELECT 
    v.visit_number,
    v.arrival_time,
    v.chief_complaint,
    tr.final_triage_level,
    r.room_number,
    v.status
FROM visits v
LEFT JOIN triage_records tr ON v.id = tr.visit_id
LEFT JOIN room_assignments ra ON v.id = ra.visit_id
LEFT JOIN rooms r ON ra.room_id = r.id
WHERE v.patient_id = 'patient-uuid-here'
ORDER BY v.arrival_time DESC;
*/

-- Current ER dashboard metrics
/*
SELECT 
    COUNT(CASE WHEN v.status = 'waiting' THEN 1 END) as waiting_patients,
    COUNT(CASE WHEN v.status = 'in_treatment' THEN 1 END) as patients_in_treatment,
    AVG(EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - v.arrival_time))/60) as avg_wait_time_minutes,
    COUNT(CASE WHEN tr.final_triage_level = 1 THEN 1 END) as critical_patients,
    COUNT(CASE WHEN tr.final_triage_level = 2 THEN 1 END) as urgent_patients
FROM visits v
LEFT JOIN triage_records tr ON v.id = tr.visit_id
WHERE v.arrival_time::date = CURRENT_DATE
  AND v.status IN ('waiting', 'in_triage', 'assigned', 'in_treatment');
*/