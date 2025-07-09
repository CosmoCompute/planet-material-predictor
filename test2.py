def scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha=0.2, beta=0.5):
    """Common scaling factor based on gravity and porosity."""
    gravity_term = (g_earth / g_planet) ** alpha
    porosity_term = ((1 - phi_earth) / (1 - phi_planet)) ** beta
    return gravity_term * porosity_term

def convert_velocity_to_earth(V_planet, g_planet, g_earth, phi_planet, phi_earth,
                              alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return V_planet * factor

def convert_amplitude_to_earth(A_planet, g_planet, g_earth, phi_planet, phi_earth,
                               alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return A_planet * factor

def convert_frequency_to_earth(f_planet, g_planet, g_earth, phi_planet, phi_earth,
                               alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return f_planet * factor

def convert_duration_to_earth(D_planet, g_planet, g_earth, phi_planet, phi_earth,
                              alpha=0.2, beta=0.5):
    factor = scaling_factor(g_planet, g_earth, phi_planet, phi_earth, alpha, beta)
    return D_planet / factor  

# Venus example values
V_venus = 4.5       # km/s
A_venus = 0.40       # amplitude
D_venus = 300       # ms
f_venus = 30        # Hz

g_venus = 3.73      # m/s^2
g_earth = 9.81      # m/s^2
phi_venus = 0.18
phi_earth = 0.10

# Convert all properties
V_earth = convert_velocity_to_earth(V_venus, g_venus, g_earth, phi_venus, phi_earth)
A_earth = convert_amplitude_to_earth(A_venus, g_venus, g_earth, phi_venus, phi_earth)
D_earth = convert_duration_to_earth(D_venus, g_venus, g_earth, phi_venus, phi_earth)
f_earth = convert_frequency_to_earth(f_venus, g_venus, g_earth, phi_venus, phi_earth)

print(f"Velocity (Earth): {V_earth:.2f} km/s")
print(f"Amplitude (Earth): {A_earth:.2f}")
print(f"Duration (Earth): {D_earth:.2f} ms")
print(f"Frequency (Earth): {f_earth:.2f} Hz")
