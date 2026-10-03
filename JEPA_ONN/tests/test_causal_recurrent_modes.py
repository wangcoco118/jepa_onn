"""Small CPU-only tests for explicit causal feature source and objective."""
import copy
import inspect
import logging
import unittest
import torch
from torch import nn
import evals.causal_recurrent as cr
from src.models.causal_recurrent_onn import CausalRecurrentONNPredictor
from tests.test_causal_recurrent_onn import small_onn_config

torch.set_num_threads(1)

class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1., -0.3, 0.7, 1.9]))
        self.calls = 0
    def forward(self, clips, masks):
        self.calls += 1
        base = clips.mean((1, 2, 3, 4))[:, None, None]
        return [(base * self.weight[None, None] +
                 torch.arange(4, device=clips.device)[None, None]).expand(-1, 2, -1)]

def config(source="shared_target", objective="reconstruct_current"):
    return {"predictor_type": "onn_causal_recurrent",
            "predictor": {"feature_source": source, "objective": objective,
                          "chunk_frames": 2, "chunk_tokens": 2},
            "loss": {"loss_exp": 1., "motion_beta": 0. if objective=="reconstruct_current" else 1.},
            "data": {"sampling_rate": 2, "frame_steps": 2, "use_bfloat16": False},
            "pretrain": {"folder": "/weights", "checkpoint": "vjepa.pt"},
            "training": {"experiment_mode": "onn_causal_recurrent",
                         "checkpoint_selection": "last_epoch"}}

def model(source="shared_target", objective="reconstruct_current"):
    return CausalRecurrentONNPredictor(
        embed_dim=4, predictor_embed_dim=4, num_context_chunks=7,
        chunk_tokens=2, memory_lambda=.4, onn_config=small_onn_config(4, 2),
        feature_source=source, objective=objective)

class CausalModeTests(unittest.TestCase):
    def helper(self, name):
        self.assertTrue(hasattr(cr, name), "missing causal mode helper: "+name)
        return getattr(cr, name)

    def test_old_defaults_are_explicit_v1(self):
        self.assertEqual(self.helper("resolve_causal_mode")({}),
                         ("legacy_dual", "next_chunk"))

    def test_invalid_modes_rejected(self):
        resolve=self.helper("resolve_causal_mode")
        for field in ("feature_source", "objective"):
            with self.assertRaises(ValueError):
                resolve({field:"bad"})

    def test_v2_requires_matching_explicit_sampling(self):
        validate=self.helper("validate_causal_config")
        cfg=config()
        validate(cfg)
        for field in ("sampling_rate", "frame_steps"):
            bad=copy.deepcopy(cfg); bad["data"].pop(field)
            with self.assertRaises(ValueError): validate(bad)
        bad=config(); bad["data"]["sampling_rate"]=4
        with self.assertRaises(ValueError): validate(bad)
        # Legacy defaults historically differ, and remain unchanged.
        validate({"predictor":{}, "data":{"frame_steps":2}})

    def test_reconstruction_rejects_motion_or_non_l1(self):
        validate=self.helper("validate_causal_config")
        for field,value in (("motion_beta",1.),("loss_exp",2.)):
            bad=config();bad["loss"][field]=value
            with self.assertRaises(ValueError): validate(bad)

    def test_shared_target_encoded_once_normalized_reused_and_frozen(self):
        self.assertIn("feature_source", inspect.signature(cr.encode_independent_chunks).parameters)
        enc,target=TinyEncoder(),TinyEncoder()
        enc.train();target.train()
        clips=torch.randn(2,3,16,2,2,requires_grad=True)
        x,z=cr.encode_independent_chunks(clips,enc,target,chunk_tokens=2,
                                        feature_source="shared_target")
        self.assertEqual((enc.calls,target.calls),(0,1))
        torch.testing.assert_close(x,z[:,:-1],rtol=0,atol=0)
        self.assertFalse(x.requires_grad);self.assertFalse(z.requires_grad)
        self.assertFalse(enc.training);self.assertFalse(target.training)
        self.assertTrue(all(not p.requires_grad for e in (enc,target) for p in e.parameters()))
        self.assertEqual(tuple(x.shape),(2,7,2,4))

    def test_legacy_encoding_unchanged(self):
        self.assertIn("feature_source", inspect.signature(cr.encode_independent_chunks).parameters)
        enc,target=TinyEncoder(),TinyEncoder()
        clips=torch.randn(2,3,16,2,2)
        a=cr.encode_independent_chunks(clips,enc,target,chunk_tokens=2)
        b=cr.encode_independent_chunks(clips,enc,target,chunk_tokens=2,feature_source="legacy_dual")
        for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_objective_alignment_and_chunk8_excluded(self):
        self.assertIn("objective", inspect.signature(cr.compute_causal_recurrent_loss).parameters)
        z=torch.randn(2,8,2,4)
        p=z[:,:-1].clone().requires_grad_()
        recon=cr.compute_causal_recurrent_loss(p,z,objective="reconstruct_current",motion_beta=0.)
        self.assertEqual(float(recon["loss"]),0.)
        pred=cr.compute_causal_recurrent_loss(z[:,1:].clone(),z)
        self.assertEqual(float(pred["loss"]),0.)
        z[:,-1]+=1000
        self.assertEqual(float(cr.compute_causal_recurrent_loss(
            p,z,objective="reconstruct_current",motion_beta=0.)["loss"]),0.)

    def test_recon_metrics_alignment_variation_and_no_copy_claim(self):
        metrics=self.helper("compute_reconstruction_metrics")
        z=torch.randn(2,8,2,4)
        result=metrics(z[:,:-1],z)
        self.assertEqual(float(result["reconstruction_l1"]),0.)
        self.assertEqual(len(result["per_step_l1"]),7)
        self.assertEqual(float(result["first_step_l1"]),0.)
        self.assertGreater(float(result["target_variation_std"]),0.)
        self.assertFalse(any("copy" in k for k in result))

    def test_recon_last_loss_reaches_earlier_state_and_all_onns(self):
        self.assertIn("objective", inspect.signature(CausalRecurrentONNPredictor).parameters)
        torch.manual_seed(47)
        m=model();x=torch.randn(2,7,2,4)
        pred,states=m(x,return_states=True)
        last=pred[:,-1].abs().mean()
        last.backward()
        for branch in (m.input_onn,m.memory_onn,m.prediction_onn):
            self.assertGreater(float(branch.slm_layers[0].phase_logits.grad.abs().sum()),0.)
        self.assertGreater(float(m.output_linear.weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.predictor_embed.parameters()))
        self.assertFalse(m.input_onn.feedback_gain_raw.requires_grad)
        # Each input time can affect the final reconstruction, through BPTT.
        x=x.detach().requires_grad_()
        m(x)[:,-1].square().mean().backward()
        self.assertGreater(float(x.grad[:,0].abs().sum()),0.)

    def test_model_state_dict_remains_compatible_with_v1(self):
        self.assertIn("objective", inspect.signature(CausalRecurrentONNPredictor).parameters)
        old=model("legacy_dual","next_chunk")
        new=model()
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        new.load_state_dict(old.state_dict(),strict=True)

    def test_checkpoint_defaults_and_conflicts(self):
        validate=self.helper("validate_causal_checkpoint")
        old={}
        validate(old,config("legacy_dual","next_chunk"))
        with self.assertRaises(ValueError):validate(old,config())
        cp={"feature_source":"shared_target","objective":"reconstruct_current"}
        validate(cp,config())
        with self.assertRaises(ValueError):validate(cp,config("shared_target","next_chunk"))

    def test_reconstruction_cannot_enter_official_test(self):
        validate=self.helper("validate_causal_config")
        with self.assertRaisesRegex(ValueError,"reconstruct|reconstruction"):
            validate(config(),ordinary_test=True)

    def test_epoch_one_update_and_frozen_encoders(self):
        self.assertIn("objective", inspect.signature(CausalRecurrentONNPredictor).parameters)
        from evals.intuitive_physics import train_optical
        enc,target=TinyEncoder(),TinyEncoder();m=model()
        train_optical._set_predictor_trainability(m)
        class CountingSGD(torch.optim.SGD):
            def __init__(self,params):super().__init__(params,lr=.01);self.zeros=self.steps=0
            def zero_grad(self,*a,**kw):self.zeros+=1;return super().zero_grad(*a,**kw)
            def step(self,*a,**kw):self.steps+=1;return super().step(*a,**kw)
        opt=CountingSGD([p for p in m.parameters() if p.requires_grad])
        before=[p.detach().clone() for e in (enc,target) for p in e.parameters()]
        metrics=train_optical._run_causal_recurrent_epoch(
            [(torch.randn(2,3,16,2,2),)],config(),enc,target,m,opt,
            torch.device("cpu"),logging.getLogger("v2-test"),1,True)
        self.assertEqual((opt.zeros,opt.steps),(1,1))
        self.assertIn("per_step_l1",metrics)
        self.assertIn("first_step_l1",metrics)
        for p,v in zip([p for e in (enc,target) for p in e.parameters()],before):
            self.assertIsNone(p.grad);torch.testing.assert_close(p,v,rtol=0,atol=0)

    def test_checkpoint_records_mode_without_redefining_architecture_version(self):
        self.assertIn("objective", inspect.signature(CausalRecurrentONNPredictor).parameters)
        from evals.intuitive_physics import train_optical
        m=model();opt=torch.optim.SGD(m.parameters(),lr=.01)
        cp=train_optical._end_to_end_checkpoint(
            m,opt,None,1,1,None,{}, "/data/linux/wkx/split.json",config(),
            "last",experiment_mode="onn_causal_recurrent")
        self.assertEqual(cp["architecture_version"],2)
        self.assertEqual(cp["feature_source"],"shared_target")
        self.assertEqual(cp["objective"],"reconstruct_current")
        self.assertTrue(cp["input_linear_frozen"])
        self.assertEqual(cp["training_sampling_rate"],2)
        self.assertIn("encoder_weight_source",cp)
        self.assertIn("normalization_rule",cp)


    def test_resume_without_explicit_config_still_rejects_mode_conflict(self):
        from unittest.mock import patch
        from evals.intuitive_physics import train_optical
        old=model("legacy_dual","next_chunk")
        old_opt=torch.optim.SGD(old.parameters(),lr=.01)
        cp=train_optical._end_to_end_checkpoint(
            old,old_opt,None,1,1,None,{}, "/data/linux/wkx/split.json",
            config("legacy_dual","next_chunk"),"last",
            experiment_mode="onn_causal_recurrent")
        new=model(); new_opt=torch.optim.SGD(new.parameters(),lr=.01)
        with patch.object(train_optical.torch,"load",return_value=cp):
            with self.assertRaisesRegex(ValueError,"mode|objective|feature"):
                train_optical._load_end_to_end_checkpoint(
                    "/data/linux/wkx/fake.pt",new,new_opt,None,
                    expected_mode="onn_causal_recurrent")

    def test_ordinary_evaluator_rejects_reconstruction_before_initialization(self):
        from evals.intphys_test import eval as test_eval
        from evals.intuitive_physics import eval as dev_eval
        for module in (test_eval,dev_eval):
            with self.assertRaisesRegex(ValueError,"reconstruction"):
                module.main(config())

    def test_trained_loader_rejects_same_shape_wrong_mode(self):
        from unittest.mock import patch
        from evals.intphys_test import eval as test_eval
        m=model("legacy_dual","next_chunk")
        cp={"mode":"onn_causal_recurrent", "predictor":m.state_dict(),
            "feature_source":"shared_target","objective":"reconstruct_current"}
        with patch.object(test_eval.torch,"load",return_value=cp):
            with self.assertRaises(ValueError):
                test_eval._load_trained_predictor(m,"/data/linux/wkx/not-read.pt")

    def test_v2_strict_target_loading_rejects_missing_pretrained_weights(self):
        from unittest.mock import patch
        from evals.intphys_test import eval as test_eval
        enc,target=nn.Linear(4,4),nn.Linear(4,4)
        cp={"encoder":enc.state_dict(),"target_encoder":{}}
        with patch.object(test_eval.torch,"load",return_value=cp):
            with self.assertRaises(RuntimeError):
                test_eval.load_pretrained(enc,target,None,"/weights/test.pt",
                                         load_predictor=False,strict_encoders=True)

    def test_distinct_target_checkpoint_weights_are_used(self):
        from unittest.mock import patch
        from evals.intuitive_physics import eval as dev_eval
        enc,target=nn.Linear(4,4),nn.Linear(4,4)
        context_state={k:torch.ones_like(v) for k,v in enc.state_dict().items()}
        target_state={k:torch.full_like(v,2.) for k,v in target.state_dict().items()}
        with patch.object(dev_eval.torch,"load",return_value={
                "encoder":context_state,"target_encoder":target_state}):
            dev_eval.load_pretrained(enc,target,None,"/weights/test.pt",load_predictor=False)
        torch.testing.assert_close(target.weight,torch.full_like(target.weight,2.))
        torch.testing.assert_close(enc.weight,torch.ones_like(enc.weight))

    def test_prediction_checkpoint_cannot_be_relabeled_reconstruction_validation(self):
        from unittest.mock import patch
        from evals.intuitive_physics.validate_causal_reconstruction import run_validation
        cp={"mode":"onn_causal_recurrent","config":config("legacy_dual","next_chunk")}
        with patch("torch.load",return_value=cp):
            with self.assertRaisesRegex(ValueError,"not trained"):
                run_validation("/data/linux/wkx/not-read.pt")

    def test_legacy_state_roundtrip_preserves_outputs(self):
        # Default model adds mode attributes only, not trainable weights or buffers.
        self.assertIn("objective", inspect.signature(CausalRecurrentONNPredictor).parameters)
        a=model("legacy_dual","next_chunk")
        b=model("legacy_dual","next_chunk")
        b.load_state_dict(a.state_dict(),strict=True)
        x=torch.randn(2,7,2,4)
        torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)

if __name__=="__main__":
    unittest.main()
