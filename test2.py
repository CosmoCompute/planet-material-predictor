import math

def convert_earth_seismic_to_mars(
    earth_velocity: float,
    wave_type: str,
    earth_layer: str = None
) -> dict:
    """
    Estimates a seismic wave velocity on Mars based on an Earth seismic velocity.

    This function provides a conceptual conversion, as precise physical
    transformation is highly complex and depends on detailed planetary models.
    It applies heuristic adjustments based on general differences in planetary
    interiors (size, pressure, temperature, composition).

    Args:
        earth_velocity (float): The seismic wave velocity on Earth in km/s.
        wave_type (str): The type of seismic wave ('P' for Primary/Compressional,
                         'S' for Secondary/Shear). Case-insensitive.
        earth_layer (str, optional): The Earth layer where the velocity
                                     originates ('crust', 'mantle', 'core').
                                     If None, the function will attempt to infer
                                     the layer based on typical Earth velocity ranges.

    Returns:
        dict: A dictionary containing:
              - 'mars_velocity_km_s' (float): The estimated seismic velocity on Mars.
              - 'inferred_earth_layer' (str): The Earth layer inferred or provided.
              - 'notes' (str): Explanations about the conversion and assumptions.
    """

    wave_type = wave_type.upper()
    mars_velocity = 0.0
    notes = ""
    inferred_layer = earth_layer

    # Define typical Earth seismic velocity ranges for inference (km/s)
    EARTH_P_WAVE_RANGES = {
        "crust": (3.0, 7.0),
        "mantle": (7.0, 14.0),
        "outer_core": (8.0, 10.0),
        "inner_core": (10.0, 12.0)
    }

    EARTH_S_WAVE_RANGES = {
        "crust": (1.5, 4.0),
        "mantle": (4.0, 7.5),
        "outer_core": (0.0, 0.0),  # S-waves don't travel through liquid outer core
        "inner_core": (3.0, 4.5)
    }

    # Heuristic adjustment factors for Mars relative to Earth
    # These are illustrative and not derived from complex physical models.
    # Factors account for lower pressure/density on Mars, and potentially
    # different compositions/temperatures.
    ADJUSTMENT_FACTORS = {
        "P": {
            "crust": 0.95,  # Slightly lower due to less compression
            "mantle": 0.98, # Very similar ranges observed, slight reduction for overall lower pressure
            "core": 0.90   # Lower for Mars' liquid core compared to Earth's outer core
        },
        "S": {
            "crust": 0.95,
            "mantle": 0.98,
            "core": 0.0    # S-waves do not propagate through Mars' liquid core
        }
    }

    if wave_type not in ["P", "S"]:
        return {
            "mars_velocity_km_s": None,
            "inferred_earth_layer": None,
            "notes": "Invalid wave_type. Please use 'P' or 'S'."
        }

    # If earth_layer is not provided, try to infer it
    if inferred_layer is None:
        if wave_type == "P":
            for layer, (min_v, max_v) in EARTH_P_WAVE_RANGES.items():
                if min_v <= earth_velocity <= max_v:
                    inferred_layer = layer
                    break
        elif wave_type == "S":
            for layer, (min_v, max_v) in EARTH_S_WAVE_RANGES.items():
                if min_v <= earth_velocity <= max_v:
                    inferred_layer = layer
                    break

        if inferred_layer is None:
            notes += (f"Could not precisely infer Earth layer for {wave_type}-wave velocity "
                      f"{earth_velocity} km/s. Applying a general adjustment. "
                      "Providing 'earth_layer' (e.g., 'crust', 'mantle', 'core') "
                      "will yield a more specific estimation."
                     )
            # Fallback to a general adjustment if layer cannot be inferred
            if wave_type == "P":
                mars_velocity = earth_velocity * ADJUSTMENT_FACTORS["P"]["mantle"]
            else: # S-wave
                mars_velocity = earth_velocity * ADJUSTMENT_FACTORS["S"]["mantle"]
            return {
                "mars_velocity_km_s": round(mars_velocity, 2),
                "inferred_earth_layer": "unknown (general estimate)",
                "notes": notes
            }

    # Apply conversion based on inferred or provided layer
    if inferred_layer in ["crust", "mantle"]:
        mars_velocity = earth_velocity * ADJUSTMENT_FACTORS[wave_type][inferred_layer]
        notes += (f"Estimated for {inferred_layer} layer. "
                  "Mars is generally less compressed than Earth, leading to slightly lower velocities."
                 )
    elif inferred_layer in ["outer_core", "inner_core"]:
        if wave_type == "P":
            # Mars has a liquid core. Compare to Earth's liquid outer core.
            mars_velocity = earth_velocity * ADJUSTMENT_FACTORS["P"]["core"]
            notes += ("Estimated for core layer. Mars has a liquid core, "
                      "which is generally less dense than Earth's core, "
                      "leading to a lower P-wave velocity."
                     )
        elif wave_type == "S":
            # S-waves do not propagate through liquid cores. Mars' core is liquid.
            mars_velocity = 0.0
            notes += ("S-waves do not propagate through liquid layers. "
                      "Mars' core is believed to be entirely liquid, hence S-wave velocity is 0 km/s."
                     )
    else:
        notes += (f"Invalid or unhandled Earth layer '{inferred_layer}'. "
                  "No specific conversion applied. Returning original velocity for now. "
                  "Please use 'crust', 'mantle', or 'core'."
                 )
        mars_velocity = earth_velocity # Return original if layer is unhandled

    return {
        "mars_velocity_km_s": round(mars_velocity, 2),
        "inferred_earth_layer": inferred_layer,
        "notes": n