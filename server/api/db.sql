-- Emergency Room Triage System Database Schema
-- Designed for single nurse managing multiple patients across multiple rooms

-- Rooms table - manages physical room availability
CREATE TABLE rooms (
    id SERIAL PRIMARY KEY,
    room_number VARCHAR(10) NOT NULL UNIQUE,
    room_type VARCHAR(50) NOT NULL, -- 'general', 'trauma', 'isolation', 'observation'
    capacity INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    equipment_available TEXT[], -- Array of available equipment
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Patients table - core patient information
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(20) NOT NULL UNIQUE, -- Hospital patient ID
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10),
    phone VARCHAR(20),
    emergency_contact VARCHAR(200),
    allergies TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Triage entries - main workflow table
CREATE TABLE triage_entries (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id) ON DELETE CASCADE,
    room_id INTEGER REFERENCES rooms(id),
    
    -- Arrival and triage info
    arrival_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chief_complaint TEXT NOT NULL,
    symptoms TEXT[], -- Array of symptoms
    pain_scale INTEGER CHECK (pain_scale >= 0 AND pain_scale <= 10),
    
    -- Vital signs
    temperature DECIMAL(4,2),
    pulse INTEGER,
    blood_pressure_systolic INTEGER,
    blood_pressure_diastolic INTEGER,
    respiratory_rate INTEGER,
    oxygen_saturation DECIMAL(5,2),
    
    -- Triage classification
    triage_level INTEGER NOT NULL CHECK (triage_level >= 1 AND triage_level <= 5),
    -- 1 = Critical, 2 = Urgent, 3 = Semi-urgent, 4 = Less urgent, 5 = Non-urgent
    triage_score DECIMAL(5,2), -- ML model confidence score
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'waiting' CHECK (status IN ('waiting', 'in_room', 'in_treatment', 'discharged', 'transferred')),
    assigned_time TIMESTAMP,
    treatment_start_time TIMESTAMP,
    discharge_time TIMESTAMP,
    
    -- Additional info
    notes TEXT,
    discharge_instructions TEXT,
    follow_up_required BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vital signs history - for tracking patient condition over time
CREATE TABLE vital_signs_history (
    id SERIAL PRIMARY KEY,
    triage_entry_id INTEGER REFERENCES triage_entries(id) ON DELETE CASCADE,
    temperature DECIMAL(4,2),
    pulse INTEGER,
    blood_pressure_systolic INTEGER,
    blood_pressure_diastolic INTEGER,
    respiratory_rate INTEGER,
    oxygen_saturation DECIMAL(5,2),
    pain_scale INTEGER CHECK (pain_scale >= 0 AND pain_scale <= 10),
    notes TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Room assignments - tracks room occupancy over time
CREATE TABLE room_assignments (
    id SERIAL PRIMARY KEY,
    triage_entry_id INTEGER REFERENCES triage_entries(id) ON DELETE CASCADE,
    room_id INTEGER REFERENCES rooms(id),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    released_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- System configuration - for dashboard settings
CREATE TABLE system_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_triage_entries_status ON triage_entries(status);
CREATE INDEX idx_triage_entries_triage_level ON triage_entries(triage_level);
CREATE INDEX idx_triage_entries_arrival_time ON triage_entries(arrival_time);
CREATE INDEX idx_triage_entries_patient_id ON triage_entries(patient_id);
CREATE INDEX idx_room_assignments_active ON room_assignments(is_active);
CREATE INDEX idx_vital_signs_triage_entry ON vital_signs_history(triage_entry_id);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_triage_entries_updated_at 
    BEFORE UPDATE ON triage_entries 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Sample data for testing
INSERT INTO rooms (room_number, room_type, capacity) VALUES 
('ER-001', 'general', 1),
('ER-002', 'general', 1),
('ER-003', 'trauma', 1),
('ER-004', 'isolation', 1),
('ER-005', 'observation', 2);

INSERT INTO system_config (config_key, config_value, description) VALUES
('max_waiting_time_critical', '5', 'Maximum waiting time for critical patients (minutes)'),
('max_waiting_time_urgent', '30', 'Maximum waiting time for urgent patients (minutes)'),
('auto_discharge_enabled', 'false', 'Enable automatic discharge suggestions'),
('vitals_update_interval', '15', 'Interval for vital signs updates (minutes)');

-- Useful views for the dashboard
CREATE VIEW current_patient_status AS
SELECT 
    t.id as triage_id,
    p.patient_id,
    p.first_name,
    p.last_name,
    r.room_number,
    t.triage_level,
    t.status,
    t.arrival_time,
    t.chief_complaint,
    t.temperature,
    t.pulse,
    t.blood_pressure_systolic,
    t.blood_pressure_diastolic,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - t.arrival_time))/60 as waiting_time_minutes
FROM triage_entries t
JOIN patients p ON t.patient_id = p.id
LEFT JOIN rooms r ON t.room_id = r.id
WHERE t.status IN ('waiting', 'in_room', 'in_treatment')
ORDER BY t.triage_level ASC, t.arrival_time ASC;

CREATE VIEW room_occupancy AS
SELECT 
    r.id as room_id,
    r.room_number,
    r.room_type,
    r.capacity,
    COUNT(ra.id) as current_occupancy,
    r.capacity - COUNT(ra.id) as available_capacity
FROM rooms r
LEFT JOIN room_assignments ra ON r.id = ra.room_id AND ra.is_active = TRUE
WHERE r.is_active = TRUE
GROUP BY r.id, r.room_number, r.room_type, r.capacity
ORDER BY r.room_number;