"""CPU-only tests for the isolated reconstruction Test entry."""
import importlib
import importlib.util
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from tests.test_causal_recurrent_modes import TinyEncoder, config, model

torch.set_num_threads(1)
OUTPUT_BASE = "/data/linux/wkx/IntPhys/references_code/jepa_onn/output"

class CopyReconstructor(nn.Module):
    feature_source = "shared_target"
    objective = "reconstruct_current"
    chunk_tokens = 2
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
    def forward(self, features):
        return features + self.anchor

class ToyDataset(torch.utils.data.Dataset):
    tasks = ["O1_test_a", "O2_test_b", "O3_test_c"]
    def __len__(self): return 3
    def __getitem__(self, index):
        return torch.ones(3,49,2,2)*(index+1), torch.tensor([index],dtype=torch.float32)

class ReconstructionTestTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec("evals.intphys_reconstruction_test")
        self.assertIsNotNone(spec, "missing independent reconstruction Test package")
        self.entry = importlib.import_module("evals.intphys_reconstruction_test.eval")

    def checkpoint(self):
        cfg = config()
        cfg["data"].update(frames_per_clip=16,resolution=2,stride_sliding_window=2)
        cfg["pretrain"].update(patch_size=1)
        return {"mode":"onn_causal_recurrent","experiment_mode":"onn_causal_recurrent",
                "config":cfg,"feature_source":"shared_target","objective":"reconstruct_current",
                "predictor":CopyReconstructor().state_dict(),"epoch":25}

    def output_dir(self):
        return Path(tempfile.mkdtemp(prefix=".reconstruction_test_checks_",dir=OUTPUT_BASE))

    def test_same_chunk_alignment_and_spatial_mean(self):
        z=torch.randn(2,8,2,4)
        offsets=torch.arange(1,8).view(1,7,1,1)/10.
        p=z[:,:-1]+offsets
        scores=self.entry.reconstruction_step_scores(p,z)
        torch.testing.assert_close(scores,offsets[:, :, 0, 0].expand(2,-1))

    def test_chunk_eight_is_not_a_target(self):
        z=torch.randn(1,8,2,4);p=z[:,:-1].clone()
        z[:,-1]=10000.
        torch.testing.assert_close(self.entry.reconstruction_step_scores(p,z),torch.zeros(1,7))

    def test_scores_are_float32_and_no_grad(self):
        z=torch.randn(1,8,2,4,dtype=torch.bfloat16)
        p=(z[:,:-1]+1).requires_grad_()
        score=self.entry.reconstruction_step_scores(p,z)
        self.assertEqual(score.dtype,torch.float32)
        self.assertFalse(score.requires_grad)

    def test_windows_preserve_video_window_and_step_axes(self):
        clips=torch.randn(2,3,20,2,2)
        m=CopyReconstructor()
        scores=self.entry.score_video_batch(clips,TinyEncoder(),TinyEncoder(),m,window_batch_size=2)
        self.assertEqual(tuple(scores.shape),(2,3,7))
        torch.testing.assert_close(scores,torch.zeros_like(scores))

    def test_real_onns_microbatch_matches_single_video_scoring(self):
        clips=torch.randn(2,3,20,2,2);m=model().eval()
        enc,target=TinyEncoder(),TinyEncoder()
        score=self.entry.score_video_batch(clips,enc,target,m,window_batch_size=4)
        expected=torch.cat([self.entry.score_video_batch(c[None],enc,target,m,window_batch_size=1)
                            for c in clips],dim=0)
        torch.testing.assert_close(score,expected,rtol=1e-5,atol=1e-5)

    def test_video_final_unscored_chunk_does_not_change_reconstruction(self):
        clips=torch.randn(1,3,16,2,2);enc,target=TinyEncoder(),TinyEncoder();m=model().eval()
        a=self.entry.score_video_batch(clips,enc,target,m)
        changed=clips.clone();changed[:,:,14:]=500.
        b=self.entry.score_video_batch(changed,enc,target,m)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_prediction_model_is_rejected(self):
        m=CopyReconstructor();m.objective="next_chunk"
        with self.assertRaisesRegex(ValueError,"reconstruct"):
            self.entry.score_video_batch(torch.zeros(1,3,16,2,2),TinyEncoder(),TinyEncoder(),m)

    def test_bad_window_arguments_rejected(self):
        for kw in ({"window_batch_size":0},{"stride":0}):
            with self.assertRaises(ValueError):
                self.entry.score_video_batch(torch.zeros(1,3,16,2,2),TinyEncoder(),TinyEncoder(),
                                             CopyReconstructor(),**kw)

    def test_checkpoint_requires_reconstruction_and_full_config(self):
        cp=self.checkpoint()
        self.entry.validate_test_checkpoint(cp)
        bad=copy.deepcopy(cp);bad["objective"]="next_chunk"
        with self.assertRaises(ValueError):self.entry.validate_test_checkpoint(bad)
        bad=copy.deepcopy(cp);bad.pop("config")
        with self.assertRaises(ValueError):self.entry.validate_test_checkpoint(bad)

    def test_resume_rejects_prediction_cache_and_changed_settings(self):
        metadata={"evaluation_kind":"reconstruction_test","batch_size":15}
        valid={"metadata":metadata,"next_batch":1,"scores":[torch.zeros(2,3,7)],
               "indices":[torch.tensor([0,1])]}
        self.entry.validate_progress(valid,metadata,total_batches=3,total_tasks=5)
        for field,value in (("evaluation_kind","prediction_test"),("batch_size",10)):
            changed=dict(metadata);changed[field]=value
            with self.assertRaises(ValueError):self.entry.validate_progress(valid,changed,3,5)

    def test_resume_rejects_missing_batches_or_duplicate_ids(self):
        metadata={"evaluation_kind":"reconstruction_test"}
        state={"metadata":metadata,"next_batch":1,"scores":[torch.zeros(2,3,7)],
               "indices":[torch.tensor([0,0])]}
        with self.assertRaises(ValueError):self.entry.validate_progress(state,metadata,3,5)

    def test_mean_max_aggregate_errors_before_mapping(self):
        scores=torch.tensor([[[.2]*7,[2.0]*7],[[.5]*7,[.5]*7]])
        metrics=self.entry.answer_metrics(scores)
        torch.testing.assert_close(metrics["average_surprise"],torch.tensor([1/2.1,1/1.5]))
        torch.testing.assert_close(metrics["maximum_surprise"],torch.tensor([1/3.,1/1.5]))
        self.assertLess(metrics["maximum_surprise"][0],metrics["maximum_surprise"][1])

    def test_top_k_is_optional_and_mean_max_remain(self):
        metrics=self.entry.answer_metrics(torch.tensor([[[1.,2.,3.,4.,5.,6.,7.]]]),
                                         temporal_reduction="top_k",temporal_top_n=2)
        self.assertIn("average_surprise",metrics)
        self.assertIn("maximum_surprise",metrics)
        torch.testing.assert_close(metrics["top_2_surprise"],torch.tensor([1/7.5]))

    def test_answer_rows_follow_task_ids_not_batch_order(self):
        rows=self.entry.answer_rows(["a","b","c"],torch.tensor([2,0,1]),torch.tensor([.3,.1,.2]))
        self.assertEqual([row.split()[0] for row in rows],["a","b","c"])
        self.assertAlmostEqual(float(rows[0].split()[1]),.1,places=6)
        with self.assertRaises(ValueError):
            self.entry.answer_rows(["a","b"],torch.tensor([0,0]),torch.tensor([.1,.2]))

    def test_prediction_test_guard_is_still_active(self):
        from evals.intphys_test import eval as prediction
        with self.assertRaisesRegex(ValueError,"reconstruction"):
            prediction.main(config())

    def test_full_cpu_runner_exports_all_tasks_and_independent_metadata(self):
        folder=self.output_dir(); cp=self.checkpoint()
        loader=torch.utils.data.DataLoader(ToyDataset(),batch_size=2)
        with patch.object(self.entry.torch,"load",return_value=cp), \
             patch.object(self.entry,"prepare_models",return_value=(TinyEncoder(),TinyEncoder(),CopyReconstructor())), \
             patch.object(self.entry,"make_test_loader",return_value=loader), \
             patch.object(self.entry,"select_device",return_value=torch.device("cpu")):
            result=self.entry.run_test("/data/linux/wkx/test.pt",output_dir=folder,batch_size=2,
                                       window_batch_size=2,num_workers=1,save_every_batches=1)
        self.assertTrue(result["complete"])
        self.assertEqual(result["processed_videos"],3)
        self.assertEqual(result["windows_per_video"],17)
        answer=folder/"intphys-test-reconstruction"/"maximum_surprise_answer.txt"
        self.assertEqual([row.split()[0] for row in answer.read_text().splitlines()],ToyDataset.tasks)
        self.assertTrue((folder/"reconstruction_resume.pt").is_file())
        self.assertFalse((folder/"test_resume.pt").exists())
        metadata=json.loads((folder/"test_metadata.json").read_text())
        self.assertEqual(metadata["evaluation_kind"],"reconstruction_test")

    def test_partial_cpu_runner_does_not_export_official_answers(self):
        folder=self.output_dir();cp=self.checkpoint()
        loader=torch.utils.data.DataLoader(ToyDataset(),batch_size=2)
        with patch.object(self.entry.torch,"load",return_value=cp), \
             patch.object(self.entry,"prepare_models",return_value=(TinyEncoder(),TinyEncoder(),CopyReconstructor())), \
             patch.object(self.entry,"make_test_loader",return_value=loader), \
             patch.object(self.entry,"select_device",return_value=torch.device("cpu")):
            result=self.entry.run_test("/data/linux/wkx/test.pt",output_dir=folder,batch_size=2,
                                       max_batches=1,num_workers=1)
        self.assertFalse(result["complete"])
        self.assertFalse(list(folder.rglob("*_answer.txt")))

    def test_existing_prediction_output_directory_is_refused(self):
        folder=self.output_dir(); (folder/"test_resume.pt").write_text("prediction")
        with self.assertRaisesRegex(ValueError,"independent|metadata|prediction|empty"):
            self.entry.prepare_output(folder,{"evaluation_kind":"reconstruction_test"},resume=True)

if __name__=="__main__":unittest.main()
