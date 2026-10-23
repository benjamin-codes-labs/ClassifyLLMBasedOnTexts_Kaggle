"""Meaningful checks for split isolation, batch invariance, and submissions."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from attribution import Predictor, PromptRegularizedRidge, numeric_features
from dataio import embedding_rows, split_prompts, submit, text_hashes
from workflow import choose


class PipelineTests(unittest.TestCase):
    def test_group_split_is_reproducible_and_disjoint(self):
        data = pd.DataFrame([
            {"id": f"{g}-{c}", "prompt_id": f"p{g}", "text": "text", "model": str(c)}
            for g in range(45) for c in range(3)
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"split.csv"
            first = split_prompts(data, path)
            second = split_prompts(data, path)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.groupby("prompt_id").fold.nunique().max(),1)
        self.assertEqual(first.loc[first.fold<0,"prompt_id"].nunique(),30)
        self.assertEqual(set(first.fold),{-1,0,1,2,3,4})

    def test_submission_contract_and_leading_zeroes(self):
        data = pd.DataFrame({"id":["00000abc0001","00000abc0002"],"text":["one","two"]})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"submission.parquet"
            p=np.array([[.8,.2],[.1,.9]])
            result=submit(data,p,["alpha","beta"],output)
            self.assertEqual(result.id.tolist(),data.id.tolist())
            self.assertEqual(result.model.tolist(),["alpha","beta"])
            self.assertEqual(pd.read_parquet(output).columns.tolist(),["id","model"])
            for invalid in [np.array([[np.nan,.2],[.1,.9]]),np.ones((2,2)),np.ones((1,2))]:
                with self.assertRaises(ValueError):
                    submit(data,invalid,["alpha","beta"],output)

    def test_encoder_cache_checks_identity_and_reorders(self):
        frame=pd.DataFrame({"id":["b","a"],"text":["second","first"]})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"features.npz"
            np.savez(path,ids=["a","b"],hashes=text_hashes(["first","second"]),
                     vectors=np.array([[1,2],[3,4]]))
            np.testing.assert_array_equal(embedding_rows(path,frame),[[3,4],[1,2]])
            frame.loc[0,"text"]="changed"
            with self.assertRaises(ValueError):
                embedding_rows(path,frame)

    def test_predictions_do_not_depend_on_other_test_rows(self):
        texts = [
            f"The calm sea is blue. A quiet afternoon follows {i}." for i in range(8)
        ] + [
            f"Yet—why? We ask; we wonder, and we try again {i}!" for i in range(8)
        ] + [
            f"Physics describes matter through precise equations ({i})." for i in range(8)
        ]
        model=Predictor(["a","b","c"],selected=["structure_svm"]).fit(texts,np.repeat(np.arange(3),8))
        samples=["The waves were calm and blue.","Why—indeed—does this matter?"]
        full=model.predict_members(samples)["structure_svm"]
        alone=model.predict_members(samples[:1])["structure_svm"]
        reversed_batch=model.predict_members(samples[::-1])["structure_svm"]
        np.testing.assert_allclose(full[:1],alone,atol=1e-7)
        np.testing.assert_allclose(full,reversed_batch[::-1],atol=1e-7)

    def test_numeric_features_are_finite_for_short_text(self):
        x=numeric_features(["A.","1","“Hello”—world!","Short. Much longer sentence follows here."])
        self.assertEqual(len(x),4)
        self.assertTrue(np.isfinite(x).all())

    def test_selection_considers_single_models(self):
        y=np.tile(np.arange(3),5)
        correct=np.eye(3)[y]
        wrong=np.roll(correct,1,axis=1)
        result,_,_=choose({"good":correct,"bad":wrong},y,np.repeat(np.arange(5),3))
        self.assertEqual(result["score"],1)
        self.assertEqual(result["weights"],{"good":1.0})

    def test_prompt_regularization_uses_training_groups_only(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(45,8)).astype(np.float32)
        y=np.tile(np.arange(3),15)
        groups=np.repeat(np.arange(15),3)
        model=PromptRegularizedRidge(alpha=10,strength=5).fit(x,y,groups)
        full=model.predict_proba(x[:3])
        alone=model.predict_proba(x[:1])
        np.testing.assert_allclose(full[:1],alone,atol=1e-7)
        np.testing.assert_allclose(full.sum(1),1)
        with self.assertRaises(ValueError):
            PromptRegularizedRidge().fit(x,y,None)


if __name__=="__main__":
    unittest.main()
