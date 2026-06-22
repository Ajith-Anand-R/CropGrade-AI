-- Schema file for AgriGrade AI crop_batches table on Supabase

CREATE TABLE IF NOT EXISTS public.crop_batches (
    id VARCHAR(50) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    produce_type VARCHAR(50) DEFAULT 'Tomato',
    total_count INTEGER DEFAULT 0,
    grade_a_count INTEGER DEFAULT 0,
    grade_b_count INTEGER DEFAULT 0,
    grade_c_count INTEGER DEFAULT 0,
    ripeness_red INTEGER DEFAULT 0,
    ripeness_pink INTEGER DEFAULT 0,
    ripeness_green INTEGER DEFAULT 0,
    ripeness_overripe INTEGER DEFAULT 0,
    defect_cracked INTEGER DEFAULT 0,
    defect_rotten INTEGER DEFAULT 0,
    defect_bruised INTEGER DEFAULT 0,
    avg_size INTEGER DEFAULT 0,
    shelf_life_days INTEGER DEFAULT 0,
    market_value NUMERIC DEFAULT 0.0,
    recommended_buyer VARCHAR(100),
    batch_grade VARCHAR(10)
);

-- Enable RLS (Row Level Security)
ALTER TABLE public.crop_batches ENABLE ROW LEVEL SECURITY;

-- Allow public read/write access (safe for hackathon demonstration)
CREATE POLICY "Allow public select" ON public.crop_batches FOR SELECT USING (true);
CREATE POLICY "Allow public insert" ON public.crop_batches FOR INSERT WITH CHECK (true);
