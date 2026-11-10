import unittest
from datetime import datetime
from typing import List, Set
from zoneinfo import ZoneInfo

from git2kg.datapoint import (
    ActorInfo,
    CommitDatapoint,
    CommitInfo,
    EditDatapoint,
)


class TestGitDatapoints(unittest.TestCase):
    def test_commit_datapoint_dtypes(self):
        dtypes = CommitDatapoint.dtypes(flatten=False)
        expected = {
            "commit": CommitInfo,
            "author": ActorInfo,
            "committer": ActorInfo,
            "author_date": datetime,
            "author_timezone": int,
            "committer_date": datetime,
            "committer_timezone": int,
            "n_modified_files": int,
            "message": str,
            "is_merge": bool,
            "is_in_main_branch": bool,
            "is_bugfix": bool,
            "n_modified_documentation": int,
            "insertions": int,
            "deletions": int,
            "message_polarity": float,
            "message_subjectivity": float,
            "commit_parents": List[CommitInfo],
            "branches": Set[str],
            "sig_type": str,
            "sig": str,
        }
        self.assertDictEqual(dtypes, expected)
    def test_commit_flat_datapoint_dtypes(self):
        dtypes = CommitDatapoint.dtypes(flatten=True)
        expected = {
            "commit_hash": str,
            "commit_id": int,

            "author_name": str,
            "author_email": str,
            "author_type": str,
            "author_github_id": int,
            "author_id": int,

            "committer_name": str,
            "committer_email": str,
            "committer_type": str,
            "committer_github_id": int,
            "committer_id": int,
            
            "author_date": datetime,
            "author_timezone": int,
            "committer_date": datetime,
            "committer_timezone": int,
            "n_modified_files": int,
            "message": str,
            "is_merge": bool,
            "is_in_main_branch": bool,
            "is_bugfix": bool,
            "n_modified_documentation": int,
            "insertions": int,
            "deletions": int,
            "message_polarity": float,
            "message_subjectivity": float,
            "commit_parents_hash": List[str],
            "commit_parents_id": List[int],
            "branches": Set[str],
            "sig_type": str,
            "sig": str,
        }
        self.assertDictEqual(dtypes, expected)

    def test_edit_datapoint_dtypes(self):
        dtypes = EditDatapoint.dtypes(flatten=False)
        expected = {
            "author": ActorInfo,
            "prev_author": ActorInfo,
            "prev_commit": CommitInfo,
            "commit": CommitInfo,
            "new_path": str,
            "old_path": str,
            "line_number": int,
            "new_line": str,
            "edit_type": str,
            "timestamp": datetime,
        }
        self.assertEqual(dtypes, expected)
        
    def test_edit_flat_datapoint_dtypes(self):
        dtypes = EditDatapoint.dtypes(flatten=True)
        expected = {
            "author_name": str,
            "author_email": str,
            "author_type": str,
            "author_github_id": int,
            "author_id": int,           
            "prev_author_name": str,
            "prev_author_email": str,
            "prev_author_type": str,
            "prev_author_github_id": int,
            "prev_author_id": int,
            "prev_commit_hash": str,
            "prev_commit_id": int,
            "commit_hash": str,
            "commit_id": int,
            "new_path": str,
            "old_path": str,
            "line_number": int,
            "new_line": str,
            "edit_type": str,
            "timestamp": datetime,
        }
        self.assertEqual(dtypes, expected)

    def test_commit_datapoint_to_dict(self):
        tz = ZoneInfo("America/New_York")
        commit_dp = CommitDatapoint(
            commit=CommitInfo(hash="abc123", id=4),
            author=ActorInfo(name="Greg House", email="diagnostics@plainsborough.com", type="git", id=1),
            committer=ActorInfo(
                name="Greg House", email="diagnostics@plainsborough.com", type="git", id=1
            ),
            author_date=datetime(2024, 1, 1, 12, 0, 0, tzinfo=tz),
            author_timezone=-18000,
            committer_date=datetime(2024, 1, 1, 12, 0, 0, tzinfo=tz),
            committer_timezone=-18000,
            n_modified_files=3,
            message="Fixed a critical bug in the patient",
            is_merge=False,
            is_in_main_branch=True,
            is_bugfix=True,
            n_modified_documentation=1,
            insertions=10,
            deletions=2,
            message_polarity=0.5,
            message_subjectivity=0.6,
            commit_parents=[CommitInfo(hash="def456", id=3)],
            branches={"main"},
            sig_type="pgp",
            sig="-----BEGIN PGP SIGNATURE-----\n...\n-----END PGP SIGNATURE-----",
        )
        dp_dict = commit_dp.to_dict()
        expected_dict = {
            "commit_hash": "abc123",
            "commit_id": 4,
            "author_name": "Greg House",
            "author_email": "diagnostics@plainsborough.com",
            "author_type": "git",
            "author_github_id": None,
            "author_id": 1,
            "committer_name": "Greg House",
            "committer_email": "diagnostics@plainsborough.com",
            "committer_type": "git",
            "committer_github_id": None,
            "committer_id": 1,
            "author_date": datetime(2024, 1, 1, 12, 0, 0, tzinfo=tz),
            "author_timezone": -18000,
            "committer_date": datetime(2024, 1, 1, 12, 0, 0, tzinfo=tz),
            "committer_timezone": -18000,
            "n_modified_files": 3,
            "message": "Fixed a critical bug in the patient",
            "is_merge": False,
            "is_in_main_branch": True,
            "is_bugfix": True,
            "n_modified_documentation": 1,
            "insertions": 10,
            "deletions": 2,
            "message_polarity": 0.5,
            "message_subjectivity": 0.6,
            "commit_parents_id": [3],
            "commit_parents_hash": ["def456"],
            "branches": {"main"},
            "sig_type": "pgp",
            "sig": "-----BEGIN PGP SIGNATURE-----\n...\n-----END PGP SIGNATURE-----",
        }
        self.assertEqual(dp_dict, expected_dict)
    
    def test_edit_datapoint_to_dict(self):
        edit_dp = EditDatapoint(
            author=ActorInfo(name="Greg House", email="diagnostics@plainsborough.com", type="git", id=1),
            prev_author=ActorInfo(name="Dr. Wilson", email="wilson@plainsborough.com", type="git", id=2),
            commit=CommitInfo("abc123", id=4),
            prev_commit=CommitInfo("cba321", id=3),
            new_path="src/patient.py",
            old_path="src/patient_old.py",
            line_number=42,
            new_line="def diagnose(patient):",
            edit_type="modification",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )
        dp_dict = edit_dp.to_dict()
        expected_dict = {
            "author_name": "Greg House",
            "author_email": "diagnostics@plainsborough.com",
            "author_type": "git",
            "author_github_id": None,
            "author_id": 1,
            "prev_author_name": "Dr. Wilson",
            "prev_author_email": "wilson@plainsborough.com",
            "prev_author_type": "git",  
            "prev_author_github_id": None,
            "prev_author_id": 2,
            "prev_commit_hash": "cba321",
            "prev_commit_id": 3,
            "commit_hash": "abc123",
            "commit_id": 4,
            "new_path": "src/patient.py",
            "old_path": "src/patient_old.py",
            "line_number": 42,
            "new_line": "def diagnose(patient):",
            "edit_type": "modification",
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
        }
        self.assertEqual(dp_dict, expected_dict)
