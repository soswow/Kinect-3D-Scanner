import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KINECT_BLOCK_COUNT", "5000")
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from scanner_server.engine import ScanEngine
from scanner_server.motion_evidence import MotionEvidence
from scanner_server.session import export_session, load_session
from shared.inertial import calibration_profile
from shared.motion_journal import acceleration_journal, validate_journal, merge_journals
from shared.settings import CameraCalibration
from tests.test_motion_history import observation, frame
from kinect_scanner.server_task_worker import ServerTaskWorker, ServerTask, ServerTaskType


def journal():
    profile = calibration_profile({"id":"measured","verified":True,"bias_m_s2":[0.,0.,0.],
                                   "scale":[1.,1.,1.],"sensor_to_camera":np.eye(3).tolist()})
    return {"version":1,"segments":[{"capture_generation":"camera","recording_segment":"recording",
            "calibration":profile,"status":{"closed":True,"complete":True,"dropped_reads":0},
            "samples":[observation(i*.05,i) for i in range(1,41)]}]}


class MotionJournalTests(unittest.TestCase):
    def test_invalid_shapes_and_reused_recording_identities_are_rejected(self):
        for payload in ([], None, 1, "journal"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_journal(payload)
        payload = journal()
        payload["segments"].append(payload["segments"][0])
        with self.assertRaises(ValueError):
            validate_journal(payload)
        previous = validate_journal(journal())
        incoming = validate_journal(journal())
        incoming["segments"][0].update(capture_generation="different", samples=[])
        with self.assertRaises(ValueError):
            merge_journals(previous, incoming)

    def test_journal_cannot_claim_completeness_after_drops_or_change_a_read(self):
        payload = journal()
        payload["segments"][0]["status"]["dropped_reads"] = 1
        with self.assertRaisesRegex(ValueError, "complete"):
            validate_journal(payload)
        payload = journal()
        other = json.loads(json.dumps(payload["segments"][0]))
        other["recording_segment"] = "another-segment"
        other["samples"][0]["acceleration_m_s2"][0] += 1.
        payload["segments"].append(other)
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_journal(payload)

    def test_full_journal_supplies_gravity_when_capture_interval_history_is_missing(self):
        rows = [frame(.5),frame(1.5)]
        inverted = np.diag([1.,-1.,-1.,1.])
        self.assertTrue(MotionEvidence(rows,CameraCalibration()).check(1,0,inverted)["accepted"])
        evidence = MotionEvidence(rows,CameraCalibration(),journal=validate_journal(journal()))
        self.assertFalse(evidence.check(1,0,inverted)["accepted"])

    def test_immutable_prefix_is_merged_without_losing_later_reads(self):
        payload = validate_journal(journal())
        shorter = validate_journal(journal());shorter["segments"][0]["samples"]=shorter["segments"][0]["samples"][:10]
        self.assertEqual(payload,merge_journals(payload,shorter))
        changed = validate_journal(journal());changed["segments"][0]["samples"][0]["acceleration_m_s2"][0]=1.
        with self.assertRaises(ValueError):merge_journals(payload,changed)

    def test_invalid_upload_does_not_change_the_active_journal_and_export_preserves_it(self):
        engine = ScanEngine(device="cpu");restored=ScanEngine(device="cpu")
        self.addCleanup(engine.shutdown);self.addCleanup(restored.shutdown)
        payload = validate_journal(journal());engine.store_motion_journal(payload)
        changed = journal();changed["segments"][0]["samples"][0]["read_end_s"]=-1.
        with self.assertRaises(ValueError):engine.store_motion_journal(changed)
        self.assertEqual(payload,engine.motion_journal)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"session.zip";export_session(engine,path);load_session(restored,path)
        self.assertEqual(payload,restored.motion_journal)

    def test_finish_uploads_checkpointed_acceleration_before_requesting_the_build(self):
        payload=journal();segment=payload["segments"][0]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/"recording";folder.mkdir()
            data=b"".join((json.dumps(s)+"\n").encode() for s in segment["samples"])
            (folder/"accelerometer.jsonl").write_bytes(data)
            # No RGB/depth files exist: Finish must upload only the small journal.
            (folder/"configuration.json").write_text(json.dumps({"capture_generation":"camera","settings":{"accelerometer_calibration":segment["calibration"]}}))
            snapshot={"root":str(root),"segments":[{"generation":"recording","status":{"closed":True,"dropped":{"accelerometer":0},"index_bytes":{"accelerometer":len(data)}}}]}
            order=[];client=Mock();client.request_build.side_effect=lambda:order.append("build") or {"success":True}
            client.send_motion_journal.side_effect=lambda j:order.append("motion") or {"success":True}
            recorder=SimpleNamespace(completed_sensor_snapshot=lambda path:snapshot)
            ServerTaskWorker(client)._dispatch(ServerTask(ServerTaskType.BUILD_MESH,{"sensor_recorder":recorder,"sensor_path":str(root)}))
            self.assertEqual(["motion","build"],order)
            self.assertEqual(40,len(client.send_motion_journal.call_args.args[0]["segments"][0]["samples"]))
            self.assertEqual(payload["segments"][0]["samples"],acceleration_journal(snapshot)["segments"][0]["samples"])


if __name__ == "__main__":
    unittest.main()
