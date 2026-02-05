import numpy as np
from datetime import datetime, timedelta
import json
from astropy.time import Time
import astropy.coordinates as coord
import astropy.units as u
from datetime import datetime, timedelta
from scipy.spatial.transform import Rotation as R
import numpy as np
import poliastro
from scipy.integrate import solve_ivp
import matplotlib as mpl
from matplotlib import cm


R_EARTH = 6378.137  # km
MU = 398600.4418  # km^3/s^2
OMEGA_EARTH = 7.2921159e-5  # rad/s

# rotation from ECI to Three.js
R_eci_to_threejs = R.from_matrix([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

def get_orbit_period(state):
    """Compute orbit period from initial state [r, v]."""
    r = np.linalg.norm(state[0:3])
    v = np.linalg.norm(state[3:6])
    a = 1 / (2 / r - v**2 / MU)
    return 2 * np.pi * np.sqrt(a**3 / MU)

def convert_vector_eci_to_threejs(v):
    """Convert ECI vector [x, y, z] → Three.js [x, z, -y]."""
    x, y, z = v
    return [y, z, x]
    # return [x,y,z]


def convert_quaternion_eci_to_threejs(q):
    """Convert quaternion from ECI frame to Three.js Y-up frame."""
    qx, qy, qz, qw = q  # assuming [qx, qy, qz, qw] input
    R_eci = R.from_quat([qx, qy, qz, qw])
    R_threejs = R_eci_to_threejs * R_eci
    return R_threejs.as_quat()  # [x, y, z, w]

def eci_to_ecef(r_eci, dt_utc):
    """
    Convert ECI (GCRS) coordinates to ECEF (ITRS).

    Parameters:
        r_eci : array-like, shape (3,) in km
            ECI position vector [x, y, z].
        dt_utc : datetime.datetime
            UTC time corresponding to r_eci.

    Returns:
        np.ndarray, shape (3,)
            ECEF position vector [x, y, z] in km
    """
    # Astropy time object
    t_ast = Time(dt_utc, scale="utc")
    
    # GCRS coordinate (ECI)
    gcrs = coord.GCRS(x=r_eci[0]*u.km, y=r_eci[1]*u.km, z=r_eci[2]*u.km, obstime=t_ast)
    
    # Transform to ITRS (ECEF)
    itrs = gcrs.transform_to(coord.ITRS(obstime=t_ast))
    
    return np.array([itrs.x.value, itrs.y.value, itrs.z.value])


class TrajectoryGenerator:

    def __init__(self):
        pass

    def __init__(self, initial_state, start_time, maneuvers=None, eci=True):
        self.initial_state = initial_state

        # Support both list input and ISO string
        if isinstance(start_time, list):
            self.start_time = datetime(*start_time)
        else:
            self.start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%SZ")

        self.maneuvers = maneuvers if maneuvers else []
        self.eci = eci

    def get_moon_position_eci(self, t_now):
        """
        Compute Moon position in ECI frame using astropy.
        Returns position vector [x, y, z] in km.
        """
        current_time = Time(t_now, scale="utc")
        moon_pos = coord.get_body('moon', current_time).transform_to(coord.GCRS(obstime=current_time))
        r_moon = moon_pos.cartesian.xyz.to(u.km).value
        return r_moon.tolist()

    def get_moon_positions_eci_3js(self, duration, dt):
        """
        Compute Moon positions in ECI frame using astropy.
        Returns dict with 'times' and 'positions'.
        """
        num_steps = int(duration // dt) + 1
        times = [self.start_time + timedelta(seconds=i * dt) for i in range(num_steps)]
        positions = []

        for t_now in times:
            current_time = Time(t_now, scale="utc")
            moon_pos = coord.get_body('moon', current_time).transform_to(coord.GCRS(obstime=current_time))
            r_moon = moon_pos.cartesian.xyz.to(u.km).value
            positions.append(convert_vector_eci_to_threejs(r_moon.tolist()))

        return {"times": times, "positions": positions}
    
    def get_sun_positions_unit_vec_eci_3js(self, duration, dt):
        """
        Compute Sun unit direction vectors in ECI frame using astropy.
        Returns dict with 'times' and 'unit_vectors'.
        """
        num_steps = int(duration // dt) + 1
        times = [self.start_time + timedelta(seconds=i * dt) for i in range(num_steps)]
        unit_vectors = []

        for t_now in times:
            current_time = Time(t_now, scale="utc")
            sun_pos = coord.get_sun(current_time).transform_to(coord.GCRS(obstime=current_time))
            r_sun = sun_pos.cartesian.xyz.to(u.km).value
            sun_vec = r_sun / np.linalg.norm(r_sun)
            unit_vectors.append(convert_vector_eci_to_threejs(sun_vec.tolist()))

        return {"times": times, "unit_vectors": unit_vectors}
    
    def compute_initial_earth_rotation_angle(self):
        """
        Compute the initial Earth Rotation Angle (ERA) at start_time.
        Returns ERA in radians.
        """

        t_now = self.start_time
        # Convert to astropy Time (UT1 scale)
        t_ast = Time(t_now, scale="utc")
        jd_ut1 = t_ast.ut1.jd  # UT1 Julian Date
        # IAU 2000 formula for ERA (Earth Rotation Angle)
        era = (2.0 * np.pi * (0.7790572732640 + 1.00273781191135448 * (jd_ut1 - 2451545.0))) % (2.0 * np.pi)
        era = era + np.pi/2


        return {"earth_rotation_angle": era}

    def create_json(self, dict_data, filename):
        with open(f"{filename}.json", "w") as f:
            json.dump(dict_data, f, indent=2, default=str)

    def add_burn(self, time, delta_v):
        self.maneuvers.append((time, delta_v))

    def sat_perturbations(self, t, state):
        perturbation = np.zeros(3)
        perturbation += self.sat_j2_perturbation(t, state)
        perturbation += self.sat_drag_perturbation(t, state)
        perturbation += self.sat_srp_perturbation(t, state)
        return perturbation

    def sat_j2_perturbation(self, t, state):
        perturbation = np.zeros(3)
        perturbation += poliastro.core.perturbations.J2_perturbation(t, state, 3.986e5, 1.08262668e-3, 6378.137)
        return perturbation

    # using atmospheric_drag_exponential
    def sat_drag_perturbation(self, t, state):
        C_d = 2.2
        A_m = 0.01      # km^2/kg
        rho_0 = 3.614e-13  # kg/km^3
        H = 88.667      # km

        return poliastro.core.perturbations.atmospheric_drag_exponential(
            t, state,
            k=3.986e5,
            R=6378.137,
            C_D=C_d,
            A_over_m=A_m,
            H0=H,
            rho0=rho_0
        )


    def sat_srp_perturbation(self, t, state):
        perturbation = np.zeros(3)
        return perturbation

    def include_ground_sites(self, sites : list):
        # sites = [("name", lat, lon, alt_m), ...]
        self.ground_sites = {}
        self.ground_sites["names"] = []
        self.ground_sites["lats"] = []
        self.ground_sites["lons"] = []
        self.ground_sites["alts_m"] = []
        for site in sites:
            name, lat, lon, alt_m = site
            self.ground_sites["names"].append(name)
            self.ground_sites["lats"].append(lat)
            self.ground_sites["lons"].append(lon)
            self.ground_sites["alts_m"].append(alt_m)

    def sat_diff_eq(self, t, state):
        r = state[0:3]
        v = state[3:6]
        r_norm = np.linalg.norm(r)
        a_gravity = -MU * r / r_norm**3
        return np.hstack((v, a_gravity))
    
    # segment organization:
    # start_time : date and time of the trajectory start
    # earth_rotation_angle : initial rotation at this epoch
    # t : array of times in seconds from start_time
    # segments : array of segments
    # - t : array of times
    # - position_eci : array of position vectors in ECI frame
    # - velocity_eci : array of velocity vectors in ECI frame
    # - 

    def lvlh_to_eci_delta_v(self, r, v, dv_lvlh):
        # Unit vectors
        r_hat = r / np.linalg.norm(r)
        h_hat = np.cross(r, v)
        h_hat /= np.linalg.norm(h_hat)
        t_hat = np.cross(h_hat, r_hat)

        # LVLH components
        dv_r, dv_t, dv_n = dv_lvlh

        return dv_r * r_hat + dv_t * t_hat + dv_n * h_hat

    

    def generate_trajectory(self, filename, duration, dt, sites = []):
        num_steps = int(duration / dt) + 1
        trajectory = {}
        state = self.initial_state.copy()
        segment = 0
        t_arr, pos_arr, vel_arr = [], [], []
        segment_dict = {}
        self.include_ground_sites(sites)
        spring_cmap = mpl.colormaps['spring']

        data = np.linspace(0, 1, len(self.maneuvers)+1)

        colors = spring_cmap(data)

        maneuver_colors = [colors[i][:3] for i in range(len(colors))]

        for i in range(num_steps):
            t = i * dt
            # current_time = self.start_time + timedelta(seconds=t)

            # Apply any maneuvers
            # print(self.maneuvers)
            for maneuver_time, delta_v in self.maneuvers:
                if abs(t - maneuver_time) < dt/2:
                    dv_eci = self.lvlh_to_eci_delta_v(state[0:3], state[3:6], delta_v)
                    print(dv_eci)
                    state[3:6] += dv_eci
                    segment_dict["segment_id"] = segment
                    segment_dict["color"] = [int(c*255) for c in maneuver_colors[segment]]
                    segment_dict["t"] = t_arr
                    segment_dict["position_eci"] = pos_arr
                    segment_dict["velocity_eci"] = vel_arr
                    
                    trajectory.setdefault("segments", []).append(segment_dict)
                    t_arr, pos_arr, vel_arr = [], [], []
                    segment_dict = {}
                    segment += 1
            

            # Integrate motion (simple Euler)
            # r = state[0:3]
            # v = state[3:6]
            # r_norm = np.linalg.norm(r)
            # a_gravity = -MU * r / r_norm**3
            # state[0:3] += v * dt
            # state[3:6] += a_gravity * dt
            # ode45 to integrate
            sol = solve_ivp(self.sat_diff_eq, [t, t+dt], state, method='RK45', rtol=1e-8)
            state = sol.y[:, -1]
            
            # print(r)
            if (np.linalg.norm(state[0:3]) < R_EARTH):
                print("Warning: Satellite has crashed into Earth at time ", t)
                # should stop simulation and not read more
                break

            t_arr.append(t)
            # segment_arr.append(segment)
            pos_arr.append(convert_vector_eci_to_threejs(state[0:3].tolist()))
            vel_arr.append(convert_vector_eci_to_threejs(state[3:6].tolist()))

        # Generate Sun unit vectors (same time base)
        sun_data = self.get_sun_positions_unit_vec_eci_3js(duration, dt)
        moon_data = self.get_moon_positions_eci_3js(duration, dt)
        earth_rotation_data = self.compute_initial_earth_rotation_angle()

        trajectory["start_time"] = self.start_time.isoformat() + "Z"
        trajectory["earth_rotation_angle"] = earth_rotation_data["earth_rotation_angle"]
        trajectory["t"] = t_arr
        trajectory["position_eci"] = pos_arr
        trajectory["velocity_eci"] = vel_arr
        trajectory["sun_times"] = [s.isoformat() + "Z" for s in sun_data["times"]][0:len(t_arr)]
        trajectory["sun_unit_vectors_eci"] = sun_data["unit_vectors"][0:len(t_arr)]
        trajectory["moon_positions_eci"] = moon_data["positions"][0:len(t_arr)]
        trajectory["ground sites"] = self.ground_sites if hasattr(self, 'ground_sites') else {}
        # finalize last segment
        segment_dict["segment_id"] = segment
        segment_dict["t"] = t_arr
        segment_dict["position_eci"] = pos_arr
        segment_dict["velocity_eci"] = vel_arr
        segment_dict["color"] = [int(c*255) for c in maneuver_colors[segment]]
        
        trajectory.setdefault("segments", []).append(segment_dict)

        # print(len(trajectory["t"]))
        # print(len(trajectory["position_eci"]))
        # print(len(trajectory["velocity_eci"]))
        # print(len(trajectory["sun_unit_vectors_eci"]))
        # print(len(trajectory["sun_times"]))
        # print(len(trajectory["moon_positions_eci"]))
        
        assert(len(trajectory["t"]) == len(trajectory["position_eci"]) == 
               len(trajectory["velocity_eci"]) == len(trajectory["sun_unit_vectors_eci"]) == 
               len(trajectory["sun_times"]))

        # Write to file
        self.create_json(trajectory, filename)
        return trajectory

# -------------------- Main --------------------
if __name__ == "__main__":
    initial_state = np.array([R_EARTH + 1500, 0, 0, 0, 7.12, 0])
    orbit_period = get_orbit_period(initial_state)
    traj_gen = TrajectoryGenerator(
        initial_state,
        start_time=[2026,2,4,0,0,0],
        eci=True
    )
    sim_til = orbit_period*.5
    # ---------------- Maneuvers ----------------
    traj_gen.add_burn(orbit_period*.5,     np.array([0.0, 0.75, 0.0]))  # raise apogee
    orbit_period = get_orbit_period(np.array([R_EARTH + 1500, 0, 0, 0, 7.12 + 0.75, 0]))
    sim_til += orbit_period*.5
    traj_gen.add_burn(sim_til,   np.array([0.0, 0.65, 0.0]))  # circularize
    orbit_period = get_orbit_period(np.array([R_EARTH + 1500, 0, 0, 0, 7.12 + 0.75 + 0.65, 0]))
    sim_til += orbit_period
    # traj_gen.add_burn(5*3600,   np.array([0.0, 0.0, 0.03]))  # small plane trim
    # -------------------------------------------

    trajectory = traj_gen.generate_trajectory(
        filename="trajectory2",
        duration=sim_til,
        dt=10
    )
