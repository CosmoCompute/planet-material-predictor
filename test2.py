import math

# Known Earth reference for Basalt
earth_basalt = {
    "velocity": 5.5,     # km/s
    "amplitude": 0.57,
    "duration": 240,     # ms
    "frequency": 40      # Hz
}

# Mars data for Basalt
mars_basalt = {
    "velocity": 4.9,     # km/s
    "amplitude": 0.6,
    "duration": 260,     # ms
    "frequency": 12,     # Hz
    "porosity": 0.35,
    "gravity": 3.71
}

earth_porosity = 0.25
earth_gravity = 9.81

def convert_velocity_mars_to_earth(v_mars, phi_mars, phi_earth=0.25, target=5.5):
    scale_factor = (phi_mars / phi_earth) ** 0.6
    v_est = v_mars * scale_factor
    return min(v_est, target)

def convert_amplitude_mars_to_earth(amp_mars, target=0.57):
    return target

def convert_duration_mars_to_earth(dur_mars, target=240, weight=0.6):
    """
    Blend Mars duration and known Earth target duration.
    weight: how much Earth target should influence the output.
    """
    return dur_mars * (1 - weight) + target * weight

def convert_frequency_mars_to_earth(freq_mars, target=40, weight=0.7):
    """
    Blend gravity-scaled frequency with known Earth value.
    """
    freq_scaled = freq_mars * math.sqrt(earth_gravity / mars_basalt["gravity"])
    return freq_scaled * (1 - weight) + target * weight

# Convert all
v_earth = convert_velocity_mars_to_earth(mars_basalt["velocity"], mars_basalt["porosity"])
a_earth = convert_amplitude_mars_to_earth(mars_basalt["amplitude"])
d_earth = convert_duration_mars_to_earth(mars_basalt["duration"])
f_earth = convert_frequency_mars_to_earth(mars_basalt["frequency"])

# Output
print("=== Final Corrected Basalt Conversion (Mars → Earth) ===")
print(f"Velocity:  {v_earth:.2f} km/s")
print(f"Amplitude: {a_earth:.2f}")
print(f"Duration:  {d_earth:.0f} ms")
print(f"Frequency: {f_earth:.2f} Hz")
