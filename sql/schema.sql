-- Schema for bank_reviews database

CREATE TABLE banks ( 
	bank_id SERIAL PRIMARY KEY, 
	bank_name VARCHAR(255) NOT NULL, 
	app_name VARCHAR(255) NOT NULL 
); 
CREATE TABLE reviews ( 
	review_id SERIAL PRIMARY KEY, 
	bank_id INT NOT NULL REFERENCES banks(bank_id), 
	review_text TEXT NOT NULL, 
	rating SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5), 
	review_date DATE, sentiment_label VARCHAR(20), -- 'positive' / 'negative' / NULL 
	sentiment_score NUMERIC(5,4) CHECK (sentiment_score BETWEEN -1 AND 1), 
	sentiment_confidence NUMERIC(5,4), -- confidence score (optional but nice) 
	processed_review TEXT, -- cleaned version 
	theme VARCHAR(100), -- e.g. "App Reliability & Stability" 
	source VARCHAR(100) DEFAULT 'Google Play Store'
); 


INSERT INTO banks (bank_name, app_name)
VALUES
('Commercial Bank of Ethiopia', 'CBE Mobile'),
('Bank of Abyssinia', 'BOA Mobile'),
('Dashen Bank', 'Dashen Bank');
