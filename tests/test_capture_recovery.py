"""Offline capture recovery preserves originals and publishes only complete projects."""

import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from kinect_scanner.capture_spool import CaptureSpool
from kinect_scanner.capture_recovery import export_recovery_project
from shared.recording import RecordingWriter
from shared.settings import ScanSettings


class CaptureRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rgb = np.full((480, 640, 3), [20, 90, 150], np.uint8)
        self.depth = np.full((480, 640), 1000, np.uint16)
        self.settings = ScanSettings().to_dict()

    def spool(self):
        spool = CaptureSpool(self.root / 'buffers', 'original-session', self.settings, receipts=True, base_index=5)
        for index in range(2):
            spool.append(self.rgb + index, self.depth + index,
                         {'frame_id': f'run:{index}', 'timestamp_s': 100 + index, 'custom': {'retain': True}})
        return spool

    def test_lossless_pending_only_project_preserves_settings_metadata_and_original_buffer(self):
        spool = self.spool()
        originals = {path: path.read_bytes() for path in spool.pending_paths}
        destination = self.root / 'recovery.zip'
        self.assertEqual(destination, export_recovery_project(spool, destination))
        with zipfile.ZipFile(destination) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            self.assertEqual(3, manifest['version'])
            self.assertEqual(json.loads(json.dumps(self.settings)), manifest['settings'])
            self.assertTrue(manifest['recovery']['partial'])
            self.assertEqual('original-session', manifest['recovery']['source_session_id'])
            self.assertNotIn('mesh', manifest)
            self.assertNotIn('reconstruction', manifest)
            for index, frame in enumerate(manifest['frames']):
                np.testing.assert_array_equal(self.rgb + index, np.asarray(Image.open(io.BytesIO(archive.read(frame['rgb'])))))
                np.testing.assert_array_equal(self.depth + index, np.asarray(Image.open(io.BytesIO(archive.read(frame['depth'])))))
                self.assertEqual(spool.load(spool.pending_paths[index])[2], frame['metadata'])
        self.assertEqual(2, spool.pending_count)
        for path, data in originals.items():
            self.assertEqual(data, path.read_bytes())

    def test_corruption_and_replace_failure_keep_existing_destination(self):
        spool = self.spool()
        destination = self.root / 'existing.zip'
        destination.write_bytes(b'previous complete project')
        with patch('kinect_scanner.capture_recovery.os.replace', side_effect=OSError('replace failed')):
            with self.assertRaisesRegex(OSError, 'replace failed'):
                export_recovery_project(spool, destination)
        self.assertEqual(b'previous complete project', destination.read_bytes())
        spool.pending_paths[1].write_bytes(b'corrupt capture')
        with self.assertRaises((OSError, ValueError, zipfile.BadZipFile)):
            export_recovery_project(spool, destination)
        self.assertEqual(b'previous complete project', destination.read_bytes())
        self.assertEqual([], list(self.root.glob('.capture-recovery-*')))
        self.assertEqual(2, spool.pending_count)

    def test_legacy_integer_ids_allow_export_but_timestamps_alone_do_not_prove_order(self):
        directory = self.root / 'legacy'
        directory.mkdir()
        (directory / 'session.json').write_text(json.dumps({'settings': self.settings, 'session_id': 'old'}))
        for name, stamp in (('a', 10), ('b', 2)):
            with (directory / (name + '.npz')).open('wb') as output:
                np.savez(output, rgb=self.rgb, depth=self.depth, metadata=json.dumps({'timestamp_s': 12 - stamp, 'frame_id': stamp}))
        spool = CaptureSpool.open(directory)
        self.assertEqual(['b.npz', 'a.npz'], [p.name for p in spool.pending_paths])
        self.assertFalse(spool.retry_safe)
        export_recovery_project(spool, self.root / 'legacy.zip')
        with (directory / 'a.npz').open('wb') as output:
            np.savez(output, rgb=self.rgb, depth=self.depth, metadata=json.dumps({'timestamp_s': 3}))
        uncertain = CaptureSpool.open(directory)
        self.assertTrue(uncertain.ordering_error)
        with self.assertRaisesRegex(ValueError, 'order cannot be established'):
            export_recovery_project(uncertain, self.root / 'unknown.zip')
        self.assertEqual(2, uncertain.pending_count)
        for name, identifier, stamp in (("a", "first-run:1", 10), ("b", "second-run:2", 2)):
            with (directory / (name + '.npz')).open('wb') as output:
                np.savez(output, rgb=self.rgb, depth=self.depth,
                         metadata=json.dumps({'frame_id': identifier, 'timestamp_s': stamp}))
        self.assertTrue(CaptureSpool.open(directory).ordering_error)

    def test_recording_retry_is_pixel_verified_and_conflicting_identity_does_not_overwrite(self):
        writer = RecordingWriter(self.root / 'recording', self.settings, session_id='session')
        metadata = {'server_index': 3, 'frame_id': 'run:3'}
        self.assertTrue(writer.append(self.rgb, self.depth, metadata))
        previous = {path: path.read_bytes() for path in writer.path.rglob('*') if path.is_file()}
        self.assertFalse(writer.append(self.rgb.copy(), self.depth.copy(), metadata))
        self.assertEqual(1, len(writer.manifest['frames']))
        for changed_rgb, changed_metadata in ((self.rgb + 1, metadata),
                                             (self.rgb, {'server_index': 4, 'frame_id': 'run:3'}),
                                             (self.rgb, {'server_index': 3, 'frame_id': 'other'})):
            with self.assertRaisesRegex(ValueError, 'Conflicting local capture'):
                writer.append(changed_rgb, self.depth, changed_metadata)
        for path, data in previous.items():
            self.assertEqual(data, path.read_bytes())
        # for_session deliberately discovers only scan-* recordings.
        writer.path.rename(self.root / 'scan-recording')
        reopened = RecordingWriter.for_session(self.root, 'session')
        self.assertFalse(reopened.append(self.rgb, self.depth, metadata))
        self.assertEqual(1, len(reopened.manifest['frames']))

    def test_malformed_recording_rows_are_reported_without_overwriting_evidence(self):
        for index, malformed in enumerate((None, {'metadata': None, 'rgb': 'rgb/000000.png', 'depth': 'depth/000000.png'}, {})):
            writer = RecordingWriter(self.root / f'scan-malformed-{index}', self.settings, session_id='session')
            writer.manifest['frames'] = [malformed]
            writer._save_manifest()
            previous = (writer.path / 'manifest.json').read_bytes()
            with self.subTest(malformed=malformed), self.assertRaisesRegex(ValueError, 'Invalid local recording'):
                writer.append(self.rgb, self.depth, {'server_index': 0, 'frame_id': 'run:0'})
            self.assertEqual(previous, (writer.path / 'manifest.json').read_bytes())
            self.assertEqual([], list((writer.path / 'rgb').iterdir()))


if __name__ == '__main__':
    unittest.main()
