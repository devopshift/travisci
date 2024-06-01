import unittest
from main import train_model

class TestMain(unittest.TestCase):
    def test_train_model(self):
        prediction = train_model()
        self.assertTrue(prediction is not None)
        self.assertAlmostEqual(prediction[0], 16, delta=0.1)

if __name__ == '__main__':
    unittest.main()
