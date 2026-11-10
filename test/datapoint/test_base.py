import unittest
from dataclasses import dataclass

from git2kg.datapoint.base import (
    ActorInfo,
    CommitInfo,
    Datapoint,
    DatapointList,
    FileInfo,
    IssueInfo,
)


class TestDatapointBase(unittest.TestCase):
    def test_actor_info(self):
        actor_info = ActorInfo("Actor", email="actor@actor.ac", type="user", github_id=1)
        self.assertDictEqual(
            {"name": "Actor", "email": "actor@actor.ac", "github_id": 1, "id": None, "type": "user"},
            actor_info.to_dict(),
        )
        self.assertEqual(
            ActorInfo("Actor", email="actor@actor.ac", github_id=1, type="user"), actor_info
        )
        self.assertNotEqual(ActorInfo("Actor", email="actor@actor.ac", type="user"), actor_info)

    def test_issue_info(self):
        issue_info = IssueInfo(number=42)
        self.assertDictEqual({"number": 42}, issue_info.to_dict())
        self.assertEqual(IssueInfo(number=42), issue_info)
        self.assertNotEqual(IssueInfo(number=24), issue_info)

    def test_commit_info(self):
        commit_info = CommitInfo("555482044c681d52df8c6fa8ee34344797004164", id=1)
        self.assertDictEqual(
            {"hash": "555482044c681d52df8c6fa8ee34344797004164", "id": 1},
            commit_info.to_dict(),
        )
        self.assertEqual(
            CommitInfo("555482044c681d52df8c6fa8ee34344797004164", id=1), commit_info
        )
        self.assertNotEqual(
            CommitInfo("82d332dfb12708fb6a95dbd8fda34768935993ed", id=1), commit_info
        )
        self.assertNotEqual(
            CommitInfo("555482044c681d52df8c6fa8ee34344797004164", id=2), commit_info
        )

    def test_file_info(self):
        file_info = FileInfo(
            "555482044c681d52df8c6fa8ee34344797004164", "server/modbus/proxy.go", id=1
        )
        self.assertEqual(
            {
                "commit_hash": "555482044c681d52df8c6fa8ee34344797004164",
                "path": "server/modbus/proxy.go",
                "id": 1,
            },
            file_info.to_dict(),
        )
        self.assertEqual(
            FileInfo(
                "555482044c681d52df8c6fa8ee34344797004164", "server/modbus/proxy.go", id=1
            ),
            file_info,
        )

        self.assertEqual(
            FileInfo(
                "82d332dfb12708fb6a95dbd8fda34768935993ed", "server/modbus/proxy.go", id=1
            ),
            file_info,
        )
        self.assertNotEqual(
            FileInfo(
                "555482044c681d52df8c6fa8ee34344797004164", "server/modbus/handler.go", id=2
            ),
            file_info,
        )


    def test_datapointlist_types(self):
        @dataclass
        class DummyDatapoint(Datapoint):
            string_property: str
            def from_field(self):
                return self.string_property
            @classmethod
            def appearance_key(cls):
                return lambda dp: dp.string_property
        @dataclass
        class BummyDatapoint(Datapoint):
            float_property: float
            def from_field(self):
                return self.float_property
            @classmethod
            def appearance_key(cls):
                return lambda dp: dp.float_property

        self.assertEqual(DummyDatapoint, DatapointList([], DummyDatapoint).dp_type)
        self.assertEqual(DummyDatapoint, DatapointList([DummyDatapoint("abc")]).dp_type)    
        self.assertRaises(ValueError, lambda: DatapointList([DummyDatapoint("abc")], BummyDatapoint))        
        self.assertRaises(ValueError, lambda: DatapointList([DummyDatapoint("abc"), BummyDatapoint(2.3)]))

    def test_mining_key_defaults_to_appearance_key(self):
        @dataclass
        class DummyDatapoint(Datapoint):
            string_property: str

            def from_field(self):
                return self.string_property

            @classmethod
            def appearance_key(cls):
                return lambda dp: dp.string_property

        latest = DatapointList(
            [DummyDatapoint("b"), DummyDatapoint("c"), DummyDatapoint("a")]
        ).find_latest()

        self.assertEqual("c", latest.string_property)
