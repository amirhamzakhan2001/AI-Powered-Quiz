# backend/student_data.py
# it will store connect app with MongoDB cloud and save the quiz data into database.

from typing import List, Dict, Optional
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.collection import Collection

class DataStore:
    def __init__(self, mongo_uri: str):
        """
        Initialize the connection to MongoDB.
        Connect with ServerApi version 1 for forward compatibility

        Args:
            mongo_uri (str): MongoDB connection URI.
        
        Database: quiz_app
        Collection : student_performance
        """
        self.client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        self.db = self.client["quiz_app"]
        self.performance = self.db["student_performance"]

    def get_student_performance(self, student_id: str) -> Optional[Dict]:
        """
        Returns student performance document for the given student_id,
        or None if not found.
        """
        return self.performance.find_one({"student_id": student_id})

    def update_student_performance(self, student_id: str, class_name: str, evaluation_results: List[Dict]) -> None:
        """
        Update or insert aggregate student performance data based on a new quiz attempt.

        Params:
            student_id: unique learner ID.
            evaluation_results: list of dicts, each with keys:
                - 'subject': str
                - 'topic': str
                - 'is_correct': bool
        """

        total_attempts = len(evaluation_results)
        correct_answers = sum(1 for r in evaluation_results if r["is_correct"])
        incorrect_answers = total_attempts - correct_answers

        # Aggregate per subject and topic
        subjects_agg = {}
        for res in evaluation_results:
            subj = res.get("subject", "Unknown")
            topic = res.get("topic", "Unknown")
            is_correct = res["is_correct"]

            if subj not in subjects_agg:
                subjects_agg[subj] = {"total_attempts": 0, "correct_count": 0, "topics": {}}

            subjects_agg[subj]["total_attempts"] += 1
            if is_correct:
                subjects_agg[subj]["correct_count"] += 1

            if topic not in subjects_agg[subj]["topics"]:
                subjects_agg[subj]["topics"][topic] = {"total_attempts": 0, "correct_count": 0}

            subjects_agg[subj]["topics"][topic]["total_attempts"] += 1
            if is_correct:
                subjects_agg[subj]["topics"][topic]["correct_count"] += 1

        # Build MongoDB increment document for atomic update
        inc_fields = {
            "total_questions_attempted": total_attempts,
            "total_correct_answers": correct_answers,
            "total_incorrect_answers": incorrect_answers,
        }

        for subj, stats in subjects_agg.items():
            subj_prefix = f"subjects.{subj}"
            inc_fields[f"{subj_prefix}.total_attempts"] = stats["total_attempts"]
            inc_fields[f"{subj_prefix}.correct_count"] = stats["correct_count"]

            for topic, t_stats in stats["topics"].items():
                inc_fields[f"{subj_prefix}.topics.{topic}.total_attempts"] = t_stats["total_attempts"]
                inc_fields[f"{subj_prefix}.topics.{topic}.correct_count"] = t_stats["correct_count"]

        # Perform atomic upsert
        self.performance.update_one(
            {"student_id": student_id},
            {
            "$inc": inc_fields,
            "$setOnInsert": {"class": class_name}
            },
            upsert=True
        )
