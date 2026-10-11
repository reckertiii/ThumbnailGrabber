"""Captures must not be written into the container's own filesystem.

The failure is silent and convincing: the app creates its output folder, writes
to it, serves the files back and reports success, while the host sees an empty
dataset. It happens whenever a bind mount misses -- most easily by mounting a
pool root and expecting child ZFS datasets to come with it, which a
non-recursive bind mount does not do.
"""

import unittest
from unittest import mock

import web.server as ws

# A container's mount table: image root, the usual pseudo-filesystems, and one
# bind mount for the dataset.
MOUNTED = {"/", "/proc", "/sys", "/dev", "/etc/hosts", "/mnt/main/apps"}

# The same container with the bind mount missing -- what actually happened.
UNMOUNTED = {"/", "/proc", "/sys", "/dev", "/etc/hosts"}


class TestEphemeralDetection(unittest.TestCase):
    def _ephemeral(self, path, points):
        with mock.patch.object(ws, "_mount_points", return_value=points):
            return ws.is_ephemeral(path)

    def test_path_on_a_bind_mount_is_persistent(self):
        self.assertFalse(self._ephemeral("/mnt/main/apps/capture", MOUNTED))
        self.assertFalse(self._ephemeral("/mnt/main/apps", MOUNTED))

    def test_same_path_without_the_mount_is_ephemeral(self):
        # The exact case that lost the captures.
        self.assertTrue(self._ephemeral("/mnt/main/apps/capture", UNMOUNTED))

    def test_a_sibling_dataset_without_its_own_mount_is_ephemeral(self):
        # Mounting /mnt/main/apps does nothing for another dataset.
        self.assertTrue(self._ephemeral("/mnt/main/media/slides", MOUNTED))

    def test_app_internal_paths_are_ephemeral(self):
        self.assertTrue(self._ephemeral("/app/captures", MOUNTED))

    def test_unknown_mount_table_reports_nothing(self):
        # No procfs (Windows, or a stripped container): never cry wolf.
        self.assertFalse(self._ephemeral("/mnt/main/apps/capture", set()))

    def test_warning_text_names_the_path_and_the_cause(self):
        with mock.patch.object(ws, "_mount_points", return_value=UNMOUNTED):
            msg = ws._persistence_warning("/mnt/main/apps/capture")
        self.assertIn("/mnt/main/apps/capture", msg)
        self.assertIn("lost", msg)
        # The child-dataset trap is the likely cause, so it has to be named.
        self.assertIn("child dataset", msg)

    def test_no_warning_when_the_folder_is_fine(self):
        with mock.patch.object(ws, "_mount_points", return_value=MOUNTED):
            self.assertEqual(ws._persistence_warning("/mnt/main/apps/capture"), "")


if __name__ == "__main__":
    unittest.main()
