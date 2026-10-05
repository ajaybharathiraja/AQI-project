def calculate_indian_aqi(pm25, pm10, no2, so2, co, ozone, nh3=0, pb=0, co2=0, voc=0):
    """
    Calculates the Indian Air Quality Index (AQI) based on the sub-indices 
    of 6 major pollutants. 
    
    The overall AQI is determined by the MAXIMUM sub-index among all the pollutants.
    
    Parameters:
    - pm25: Particulate Matter < 2.5 µg/m³ (24-hour average)
    - pm10: Particulate Matter < 10 µg/m³ (24-hour average)
    - no2: Nitrogen Dioxide µg/m³ (24-hour average)
    - so2: Sulfur Dioxide µg/m³ (24-hour average)
    - co: Carbon Monoxide mg/m³ (8-hour average)
    - ozone: Ozone µg/m³ (8-hour average)
    - nh3: Ammonia µg/m³ (24-hour average)
    - pb: Lead µg/m³ (24-hour average)
    - co2: Carbon Dioxide ppm (24-hour average)
    - voc: Volatile Organic Compounds µg/m³ (24-hour average)
    
    Returns:
    - aqi_value (float): The final calculated AQI score.
    - category (str): The health category (Good, Satisfactory, Moderate, etc.)
    """
    
    # Step 1: Calculate the sub-index for each pollutant based on standard thresholds.
    # The formula is simplified here as a linear percentage against the Indian standard.
    # (Standard Limits: PM2.5=60, PM10=100, NO2=80, SO2=80, CO=2.0, Ozone=100, NH3=400, Pb=1.0)
    
    sub_indices = [
        (pm25 or 0) * (100.0 / 60.0),   # PM2.5 Sub-Index
        (pm10 or 0) * (100.0 / 100.0),  # PM10 Sub-Index
        (no2 or 0)  * (100.0 / 80.0),   # NO2 Sub-Index
        (so2 or 0)  * (100.0 / 80.0),   # SO2 Sub-Index
        (co or 0)   * (100.0 / 2.0),    # CO Sub-Index
        (ozone or 0)* (100.0 / 100.0),  # Ozone Sub-Index
        (nh3 or 0)  * (100.0 / 400.0),  # NH3 Sub-Index
        (pb or 0)   * (100.0 / 1.0),    # Pb Sub-Index
        (co2 or 0)  * (100.0 / 1000.0), # CO2 Sub-Index (1000 ppm standard)
        (voc or 0)  * (100.0 / 250.0)   # VOC Sub-Index (250 µg/m³ standard)
    ]
    
    # Step 2: The overall AQI is the highest value among all sub-indices.
    # If no data is available for any pollutant, the AQI is 0.
    aqi_val = max(sub_indices) if any(sub_indices) else 0
    
    # Step 3: Map the final AQI score to its corresponding health category.
    if aqi_val <= 50:
        category = "Good"
    elif aqi_val <= 100:
        category = "Satisfactory"
    elif aqi_val <= 200:
        category = "Moderate"
    elif aqi_val <= 300:
        category = "Poor"
    elif aqi_val <= 400:
        category = "Very Poor"
    else:
        category = "Severe"
        
    return aqi_val, category
