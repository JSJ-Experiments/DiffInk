from copy import deepcopy
import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
import yaml
from iam_tools.autopsy import check_geometry_config,coordinate_forward,freeze_auxiliaries,train_geometry
from iam_tools.trajectory_diagnostics import diagnostics
from test_vae_ctc import VAE,small_config

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT

class AutopsyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_modal_entrypoint_self_contained_and_guarded(self):
        path=ROOT/'modal_autopsy.py'
        tree=ast.parse(path.read_text())
        imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        modules={node.module if isinstance(node,ast.ImportFrom) else alias.name for node in imports for alias in (node.names if isinstance(node,ast.Import) else [None])}
        self.assertEqual(modules,{'modal','pathlib'})
        spec=importlib.util.spec_from_file_location('modal_geometry_guard',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertFalse(module.require_opt_in(False,False))
        with self.assertRaises(ValueError):module.require_opt_in(True,False)
        self.assertTrue(module.require_opt_in(True,True))

    def test_geometry_caps_and_objective_guard(self):
        cfg=yaml.safe_load((REPO/'configs/vae_iam_autopsy_geometry.yaml').read_text());check_geometry_config(cfg)
        for key,value in [('max_steps',1001),('max_wall_seconds',601),('ctc_weight',1),('pen_weight',1),('style_weight',.1),('kl_weight',1e-7),('train_batch_size',2)]:
            changed=deepcopy(cfg);changed[key]=value
            with self.assertRaises(ValueError):check_geometry_config(changed)
        with self.assertRaises(ValueError):train_geometry('missing','missing')

    def test_coordinate_backward_does_not_use_auxiliary_or_pen_objectives(self):
        torch.manual_seed(42);model=VAE(small_config()).train();freeze_auxiliaries(model)
        data=torch.randn(1,24,5);data[:,:,2:]=torch.tensor([1.,0.,0.]);data[:,-1,2:]=torch.tensor([0.,0.,1.])
        batch=(data,torch.ones(1,24,dtype=torch.bool),torch.tensor([[0,1,2]]),[],torch.tensor([0]))
        with patch.object(model,'get_ocr_loss',side_effect=AssertionError('CTC must be off')),patch.object(model,'get_style_loss',side_effect=AssertionError('style must be off')):
            loss,output,kl=coordinate_forward(model,batch,{'model_input_scale':.01},'cpu')
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for module in [model.ocr_model,model.style_classifier]:self.assertTrue(all(p.grad is None for p in module.parameters()))
        self.assertEqual(int(model.transformer_decoder.fc.weight.grad[:3].count_nonzero()),0)
        self.assertGreater(int(model.transformer_decoder.fc.weight.grad[3:].count_nonzero()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_diagnostic_counts_confusion_rmse_and_constant_correlation(self):
        truth=np.array([[0,0,1,0,0],[1,1,1,0,0],[2,0,0,1,0],[3,1,0,0,1]],dtype=float)
        prediction=truth.copy();prediction[:,:2]*=.01
        prediction[1,2:]=[0,1,0]
        row=diagnostics(prediction,truth,.01)
        self.assertEqual(row['true_counts'],[2,1,1]);self.assertEqual(row['predicted_counts'],[1,2,1])
        self.assertEqual(row['per_class_recall'],[.5,1.,1.]);self.assertEqual(row['axes']['y']['rmse_model_units'],0)
        self.assertAlmostEqual(row['axes']['x']['correlation'],1)
        prediction[:,1]=0
        self.assertIsNone(diagnostics(prediction,truth,.01)['axes']['y']['correlation'])
        with self.assertRaises(ValueError):diagnostics(prediction[:-1],truth,.01)

if __name__=='__main__':unittest.main()
