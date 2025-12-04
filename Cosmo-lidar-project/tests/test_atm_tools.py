import unittest
from src.cosmo_lidar.atm_tools import Calcul_T_ant_1_el, Calcul_T_ant_2

class TestAtmTools(unittest.TestCase):

    def test_Calcul_T_ant_1_el(self):
        frequency = [1e9, 2e9]  # Example frequencies in Hz
        theta_b = [0.1, 0.2]  # Example beam angles in radians
        altitudes = [1000, 2000]  # Example altitudes in meters
        N = 10  # Number of integration points
        Temperature = [300, 290]  # Example temperatures in Kelvin
        Pressure = [1013, 1000]  # Example pressures in hPa
        P_water = [10, 5]  # Example water vapor pressures in hPa
        elevation = 45  # Example elevation in degrees

        result = Calcul_T_ant_1_el(frequency, theta_b, altitudes, N, Temperature, Pressure, P_water, elevation)
        self.assertIsInstance(result, (list, np.ndarray))  # Check if result is a list or numpy array

    def test_Calcul_T_ant_2(self):
        frequency = [1e9, 2e9]  # Example frequencies in Hz
        elevation = 45  # Example elevation in degrees
        obs_alt = 0  # Example observation altitude in km

        result = Calcul_T_ant_2(frequency, elevation, obs_alt)
        self.assertIsInstance(result, (list, np.ndarray))  # Check if result is a list or numpy array

if __name__ == '__main__':
    unittest.main()