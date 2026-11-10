import csv
import random
import unittest
from datetime import datetime, timezone
from tempfile import TemporaryDirectory

from git2kg.datapoint import (
    ActorInfo,
    CommitInfo,
    DatapointList,
    EditDatapoint,
)
from git2kg.persistence.persistencemanager import (
    CSVPersistenceManager,
)


def random_hash():
    hash = random.getrandbits(160)
    return "%032x" % hash


class PersistenceManagerTests(unittest.TestCase):
    def setUp(self):
        self.datapoints = DatapointList(
            [
                EditDatapoint(
                    author=ActorInfo(name="Greg House", email="g.house@ppth.com"),
                    prev_author=ActorInfo(name="Lisa Cuddy", email="boss@ppth.com"),
                    commit=CommitInfo(random_hash()),
                    prev_commit=CommitInfo(random_hash()),
                    new_path="prescription/vicodin",
                    old_path="prescription/vicodin",
                    line_number=32,
                    new_line="10000mg",
                    edit_type="added",
                    timestamp=datetime.now(timezone.utc)
                ),
                EditDatapoint(
                    author=ActorInfo(name="James Wilson", email="j.wilson@ppth.com"),
                    prev_author=ActorInfo(name="Greg House", email="g.house@ppth.com"),
                    commit=CommitInfo(random_hash()),
                    prev_commit=CommitInfo(random_hash()),
                    new_path="oncology/patient_notes.txt",
                    old_path="oncology/patient_notes.txt",
                    line_number=12,
                    new_line="Recommend follow-up MRI",
                    edit_type="added",
                    timestamp=datetime.now(timezone.utc)
                ),
                EditDatapoint(
                    author=ActorInfo(name="Robert Chase", email="r.chase@ppth.com"),
                    prev_author=ActorInfo(
                        name="Eric Foreman", email="e.foreman@ppth.com"
                    ),
                    commit=CommitInfo(random_hash()),
                    prev_commit=CommitInfo(random_hash()),
                    new_path="surgery/protocol.md",
                    old_path="surgery/protocol.md",
                    line_number=78,
                    new_line="Adjust incision angle by 2 degrees",
                    edit_type="deleted",
                    timestamp=datetime.now(timezone.utc)
                ),
                EditDatapoint(
                    author=ActorInfo(name="Eric Foreman", email="e.foreman@ppth.com"),
                    prev_author=ActorInfo(
                        name="Allison Cameron", email="a.cameron@ppth.com"
                    ),
                    commit=CommitInfo(random_hash()),
                    prev_commit=CommitInfo(random_hash()),
                    new_path="diagnostics/logs.csv",
                    old_path="diagnostics/logs.csv",
                    line_number=3,
                    new_line="blood_pressure,120/80",
                    edit_type="added",
                    timestamp=datetime.now(timezone.utc)
                ),
                EditDatapoint(
                    author=ActorInfo(
                        name="Allison Cameron", email="a.cameron@ppth.com"
                    ),
                    prev_author=ActorInfo(
                        name="Robert Chase", email="r.chase@ppth.com"
                    ),
                    commit=CommitInfo(random_hash()),
                    prev_commit=CommitInfo(random_hash()),
                    new_path="research/notes.txt",
                    old_path="research/notes.txt",
                    line_number=5,
                    new_line="Patient exhibits rare autoimmune markers",
                    edit_type="deleted",
                    timestamp=datetime.now(timezone.utc)
                ),
            ]
        )

    def test_save_csv(self):

        with TemporaryDirectory() as tmp_dir:
            persistence_manager = CSVPersistenceManager(
                repository_id="ppth", base_path=tmp_dir
            )
            persistence_manager.save(self.datapoints)
            # Read csv with python built-in csv module to verify content

            file_path = f"{tmp_dir}/ppth_EditDatapoint.csv"
            with open(file_path, mode="r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                self.assertEqual(len(rows), len(self.datapoints))
                for row, dp in zip(rows, self.datapoints, strict=True):
                    self.assertEqual(row["author_name"], dp.author.name)
                    self.assertEqual(row["author_email"], dp.author.email)
                    self.assertEqual(row["prev_author_name"], dp.prev_author.name)
                    self.assertEqual(row["prev_author_email"], dp.prev_author.email)
                    self.assertEqual(row["commit_hash"], dp.commit.hash)
                    self.assertEqual(row["new_path"], dp.new_path)
                    self.assertEqual(row["old_path"], dp.old_path)
                    self.assertEqual(int(row["line_number"]), dp.line_number)
                    self.assertEqual(row["new_line"], dp.new_line)
                    self.assertEqual(row["edit_type"], dp.edit_type)
                    # Convert loaded timestamp string back to datetime for comparison
                    loaded_timestamp = datetime.fromisoformat(row["timestamp"])
                    self.assertEqual(loaded_timestamp, dp.timestamp)

    def test_save_and_load_csv(self):
        with TemporaryDirectory() as tmp_dir:
            persistence_manager = CSVPersistenceManager(
                repository_id="ppth", base_path=tmp_dir
            )
            persistence_manager.save(self.datapoints)
            loaded_datapoints = persistence_manager.load(EditDatapoint)
            self.assertEqual(len(loaded_datapoints), len(self.datapoints))
            for loaded_dp, original_dp in zip(loaded_datapoints, self.datapoints, strict=True):
                self.assertEqual(loaded_dp.author.name, original_dp.author.name)
                self.assertEqual(loaded_dp.author.email, original_dp.author.email)
                self.assertEqual(
                    loaded_dp.prev_author.name, original_dp.prev_author.name
                )
                self.assertEqual(
                    loaded_dp.prev_author.email, original_dp.prev_author.email
                )
                self.assertEqual(loaded_dp.commit.hash, original_dp.commit.hash)
                self.assertEqual(loaded_dp.new_path, original_dp.new_path)
                self.assertEqual(loaded_dp.old_path, original_dp.old_path)
                self.assertEqual(loaded_dp.line_number, original_dp.line_number)
                self.assertEqual(loaded_dp.new_line, original_dp.new_line)
                self.assertEqual(loaded_dp.edit_type, original_dp.edit_type)
                self.assertEqual(loaded_dp.timestamp, original_dp.timestamp)

    def test_append(self):
        with TemporaryDirectory() as tmp_dir:
            persistence_manager = CSVPersistenceManager(
                repository_id="ppth", base_path=tmp_dir
            )
            for i, dp in enumerate(self.datapoints):
                persistence_manager.append(DatapointList([dp]))

                file_path = f"{tmp_dir}/ppth_EditDatapoint.csv"
                with open(file_path, mode="r", newline="") as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = list(reader)
                    appended_row = rows[-1]
                    self.assertEqual(len(rows), i + 1)
                    self.assertEqual(appended_row["author_name"], dp.author.name)
                    self.assertEqual(appended_row["author_email"], dp.author.email)
                    self.assertEqual(
                        appended_row["prev_author_name"], dp.prev_author.name
                    )
                    self.assertEqual(
                        appended_row["prev_author_email"], dp.prev_author.email
                    )
                    self.assertEqual(appended_row["commit_hash"], dp.commit.hash)
                    self.assertEqual(appended_row["new_path"], dp.new_path)
                    self.assertEqual(appended_row["old_path"], dp.old_path)
                    self.assertEqual(int(appended_row["line_number"]), dp.line_number)
                    self.assertEqual(appended_row["new_line"], dp.new_line)
                    self.assertEqual(appended_row["edit_type"], dp.edit_type)
