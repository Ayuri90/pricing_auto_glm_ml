TEST_SIZE = 0.2
RANDOM_STATE = 42

# Exposition maximale : le dataset freMTPL contient des expositions > 1
# (jusqu'à 1.99), anomalie connue que l'on écrête à 1 an.
MAX_EXPOSURE = 1.0

N_BINS = {
    "Density": 5,
    "DriverAge": 5,
    "CarAge": 2
}
