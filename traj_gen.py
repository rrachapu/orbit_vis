import numpy as np
from datetime import datetime, timedelta
import json
from astropy.time import Time
import astropy.coordinates as coord
import astropy.units as u
from datetime import datetime, timedelta
from scipy.spatial.transform import Rotation as R
import numpy as np


R_EARTH = 6371.0  # km
MU = 398600.4418  # km^3/s^2
ALTITUDE = 1000.0  # km
OMEGA_EARTH = 7.2921159e-5  # rad/s

# rotation from ECI to Three.js
R_eci_to_threejs = R.from_matrix([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

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

    def get_sun_positions_eci(self, duration, dt):
        """
        Compute Sun positions in ECI frame using astropy.
        Returns dict with 'times' and 'positions'.
        """
        num_steps = int(duration // dt) + 1
        times = [self.start_time + timedelta(seconds=i * dt) for i in range(num_steps)]
        positions = []

        for t_now in times:
            current_time = Time(t_now, scale="utc")
            sun_pos = coord.get_sun(current_time).transform_to(coord.GCRS(obstime=current_time))
            r_sun = sun_pos.cartesian.xyz.to(u.km).value
            print(r_sun)
            positions.append(convert_vector_eci_to_threejs(r_sun.tolist()))

        return {"times": times, "positions": positions}
    
    def get_sun_positions_unit_vec_eci(self, duration, dt):
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
    

    # def sun_direction_quaternions_eci(self, duration, dt):
    #     """
    #     Compute quaternions from ECI to Sun-pointing frame using astropy.
    #     Returns dict with 'times' and 'quaternions'.
    #     """
    #     num_steps = int(duration // dt) + 1
    #     times = [self.start_time + timedelta(seconds=i * dt) for i in range(num_steps)]
    #     quaternions = []

    #     for t_now in times:
    #         current_time = Time(t_now, scale="utc")
    #         sun_pos = get_sun(current_time).transform_to(GCRS(obstime=current_time))
    #         r_sun = sun_pos.cartesian.xyz.to(u.km).value
    #         sun_vec = r_sun / np.linalg.norm(r_sun)

    #         # Define frame: Z → Sun
    #         z_axis = sun_vec
    #         x_axis = np.array([1.0, 0.0, 0.0])
    #         if np.allclose(z_axis, x_axis):
    #             x_axis = np.array([0.0, 1.0, 0.0])
    #         y_axis = np.cross(z_axis, x_axis)
    #         y_axis /= np.linalg.norm(y_axis)
    #         x_axis = np.cross(y_axis, z_axis)

    #         R_mat = np.column_stack((x_axis, y_axis, z_axis))
    #         q = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]
    #         q_threejs = convert_quaternion_eci_to_threejs(q)
    #         quaternions.append(q_threejs.tolist())
    #         print(q_threejs)

    #     return {"times": times, "quaternions": quaternions}

    def create_json(self, dict_data, filename):
        with open(f"{filename}.json", "w") as f:
            json.dump(dict_data, f, indent=2, default=str)

    def add_burn(self, time, delta_v):
        self.maneuvers.append((time, delta_v))

    def generate_trajectory(self, filename, duration, dt):
        num_steps = int(duration / dt) + 1
        trajectory = {}
        state = self.initial_state.copy()

        t_arr, pos_arr, vel_arr = [], [], []

        for i in range(num_steps):
            t = i * dt
            current_time = self.start_time + timedelta(seconds=t)

            # Apply any maneuvers
            for maneuver_time, delta_v in self.maneuvers:
                if np.isclose(t, maneuver_time):
                    state[3:6] += delta_v

            # Integrate motion (simple Euler)
            r = state[0:3]
            v = state[3:6]
            r_norm = np.linalg.norm(r)
            a_gravity = -MU * r / r_norm**3
            state[0:3] += v * dt
            state[3:6] += a_gravity * dt

            # Store
            t_arr.append(t)
            # print(r)
            pos_arr.append(convert_vector_eci_to_threejs(r.tolist()))
            vel_arr.append(convert_vector_eci_to_threejs(v.tolist()))

        # Generate Sun unit vectors (same time base)
        sun_data = self.get_sun_positions_unit_vec_eci(duration, dt)

        trajectory["start_time"] = self.start_time.isoformat() + "Z"
        trajectory["t"] = t_arr
        trajectory["position_eci"] = pos_arr
        trajectory["velocity_eci"] = vel_arr
        trajectory["sun_times"] = [s.isoformat() + "Z" for s in sun_data["times"]]
        trajectory["sun_unit_vectors_eci"] = sun_data["unit_vectors"]

        assert(len(trajectory["t"]) == len(trajectory["position_eci"]) == len(trajectory["velocity_eci"]) == len(trajectory["sun_unit_vectors_eci"]))

        # Write to file
        self.create_json(trajectory, filename)
        return trajectory

# -------------------- Main --------------------
if __name__ == "__main__":
    initial_state = np.array([R_EARTH + 500, 0, 0, 0, 7.6, 0])
    traj_gen = TrajectoryGenerator(initial_state, start_time=[2025,10,1,0,0,0], eci=True)
    traj_gen.add_burn(600, np.array([0, 1, 0]))
    trajectory = traj_gen.generate_trajectory(filename = "trajectory2", duration=30*3600, dt=60)
    
    # sun_positions = traj_gen.get_sun_positions_eci(3600*1440, 60)


    
