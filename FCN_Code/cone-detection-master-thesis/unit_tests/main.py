from tqdm import tqdm

# Tests
from unit_tests.tests_loss import test_difference, test_equivalence
from unit_tests.tests_particles import test_conversion, test_generation

# List of tests
tests = [
    test_equivalence,
    test_difference,
    test_conversion,
    test_generation
]

def do_tests():
    success, failure = 0, 0
    for test in tqdm(tests):
        if test():
            success += 1
        else:
            failure += 1
    print(f"Unit tests: {success} succeeded, {failure} failed")
        
if __name__ == "__main__":
    do_tests()