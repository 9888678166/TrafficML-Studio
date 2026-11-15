"""
Sample Data Generator for Road Segment Congestion Pattern Discovery
Generates realistic 15-minute interval urban traffic flow data matching
the Kaggle 'Traffic Prediction Dataset' schema.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_STATE = 42

def generate_traffic_data(num_days: int = 31, output_path: str = "data/raw/Traffic.csv"):
    np.random.seed(RANDOM_STATE)
    
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    records = []
    
    # Start on a Monday
    start_date = datetime(2025, 10, 6, 0, 0, 0)
    
    for day_idx in range(num_days):
        current_day_date = start_date + timedelta(days=day_idx)
        day_of_month = (day_idx % 31) + 1
        day_name = days_of_week[current_day_date.weekday()]
        is_weekend = current_day_date.weekday() >= 5
        
        # 96 intervals of 15 minutes each
        for step in range(96):
            time_obj = current_day_date + timedelta(minutes=step * 15)
            # Format time matching Kaggle style e.g. "12:00:00 AM", "01:15:00 PM"
            time_str = time_obj.strftime("%I:%M:%S %p")
            hour = time_obj.hour + time_obj.minute / 60.0
            
            # Base diurnal patterns
            if not is_weekend:
                # Weekday patterns
                if 0 <= hour < 6:
                    # Late night / early dawn
                    cars = np.random.poisson(lam=8)
                    bikes = np.random.poisson(lam=2)
                    buses = np.random.poisson(lam=1)
                    trucks = np.random.poisson(lam=7) # Overnight freight
                elif 6 <= hour < 10:
                    # Morning commute peak
                    peak_factor = np.exp(-((hour - 8.5) ** 2) / 1.5)
                    cars = int(np.random.normal(loc=55 + 50 * peak_factor, scale=8))
                    bikes = int(np.random.normal(loc=18 + 15 * peak_factor, scale=4))
                    buses = int(np.random.normal(loc=12 + 10 * peak_factor, scale=3))
                    trucks = int(np.random.poisson(lam=4)) # Restricted during peak
                elif 10 <= hour < 16:
                    # Midday commercial & freight
                    cars = int(np.random.normal(loc=40, scale=6))
                    bikes = int(np.random.normal(loc=12, scale=3))
                    buses = int(np.random.normal(loc=8, scale=2))
                    trucks = int(np.random.normal(loc=18, scale=4)) # High commercial deliveries
                elif 16 <= hour < 20:
                    # Evening rush peak
                    peak_factor = np.exp(-((hour - 18.0) ** 2) / 1.8)
                    cars = int(np.random.normal(loc=65 + 55 * peak_factor, scale=9))
                    bikes = int(np.random.normal(loc=20 + 16 * peak_factor, scale=4))
                    buses = int(np.random.normal(loc=14 + 10 * peak_factor, scale=3))
                    trucks = int(np.random.poisson(lam=5))
                else:
                    # Evening taper (20 - 24)
                    cars = int(np.random.normal(loc=25, scale=5))
                    bikes = int(np.random.normal(loc=7, scale=2))
                    buses = int(np.random.normal(loc=4, scale=1))
                    trucks = int(np.random.poisson(lam=8))
            else:
                # Weekend patterns (shifted, leisure-driven)
                if 0 <= hour < 8:
                    cars = np.random.poisson(lam=6)
                    bikes = np.random.poisson(lam=1)
                    buses = np.random.poisson(lam=1)
                    trucks = np.random.poisson(lam=3)
                elif 8 <= hour < 14:
                    cars = int(np.random.normal(loc=35, scale=6))
                    bikes = int(np.random.normal(loc=14, scale=3))
                    buses = int(np.random.normal(loc=5, scale=2))
                    trucks = int(np.random.poisson(lam=4))
                elif 14 <= hour < 21:
                    cars = int(np.random.normal(loc=45, scale=7))
                    bikes = int(np.random.normal(loc=16, scale=4))
                    buses = int(np.random.normal(loc=6, scale=2))
                    trucks = int(np.random.poisson(lam=3))
                else:
                    cars = int(np.random.normal(loc=20, scale=4))
                    bikes = int(np.random.normal(loc=5, scale=2))
                    buses = int(np.random.normal(loc=2, scale=1))
                    trucks = int(np.random.poisson(lam=2))
            
            # Non-negative clamping
            cars = max(0, cars)
            bikes = max(0, bikes)
            buses = max(0, buses)
            trucks = max(0, trucks)
            total = cars + bikes + buses + trucks
            
            # Congestion label assignment (ground truth proxy)
            # Thresholds aligned with typical domain distribution
            if total < 30:
                situation = "low"
            elif total < 70:
                situation = "normal"
            elif total < 110:
                situation = "high"
            else:
                situation = "heavy"
                
            records.append({
                "Time": time_str,
                "Date": day_of_month,
                "Day of the week": day_name,
                "CarCount": cars,
                "BikeCount": bikes,
                "BusCount": buses,
                "TruckCount": trucks,
                "Total": total,
                "Traffic Situation": situation
            })
            
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records saved to {output_path}")
    print(df.head())
    print("\nTraffic Situation distribution:")
    print(df["Traffic Situation"].value_counts(normalize=True))
    return df

if __name__ == "__main__":
    generate_traffic_data()

