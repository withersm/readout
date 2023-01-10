import unittest
import logging
import os
import kidpylib.config as kplc


def setUpModule():
    """Disable logging while doing these tests."""
    logging.disable()


class TestDefaultConfigNoFile(unittest.TestCase):
    def setUp(self):
        self.conf = kplc.GeneralConfig("kidpylib_test/not_a_real_config_file.cfg")
        self.conf2 = kplc.GeneralConfig("kidpylib_test/not_a_real_config_file.cfg")

    def test_0_obj_not_none_(self):
        self.assertIsNotNone(self.conf)

    def test_1_singlet_instance(self):
        self.assertIsNotNone(self.conf)
        self.assertIsNotNone(self.conf2)
        self.assertEqual(self.conf, self.conf2)

    def test_2_singlet_canary(self):
        self.assertEqual(self.conf.cfg.canary, "cat")
        self.conf.cfg.canary = "dog"
        self.assertEqual(
            self.conf2.cfg.canary, "dog", "Singlet instance, expected canary = dog"
        )

    def test_default_parameters_set(self):
        self.assertEqual(self.conf.cfg.redis_port, "6379")

    def test_write_new_file(self):
        self.conf.write_config()
        a = os.path.exists("kidpylib_test/not_a_real_config_file.cfg")
        self.assertEqual(a, True, "failed to create file")


class TestModifiedConfig(unittest.TestCase):
    def setUp(self):
        self.conf = kplc.GeneralConfig("kidpylib_test/alternate_config.cfg")

    def test_0_obj_not_none_(self):
        self.assertIsNotNone(self.conf)

    def test_canary_is_pizza(self):
        self.assertEqual(self.conf.cfg.canary, "pizza")

    def test_new_ip_isset(self):
        self.assertEqual(self.conf.cfg.redis_host, "10.0.0.69")


if __name__ == "__main__":

    unittest.main()
