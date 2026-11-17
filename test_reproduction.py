import copy
import unittest
import numpy as np
import torch
from model import PDUModel, indicator_edges, objective
from metrics import evaluate


class ReproductionTests(unittest.TestCase):
    def test_checkpointed_gradients_match_batched_graph(self):
        torch.manual_seed(102)
        model = PDUModel(12)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        other = copy.deepcopy(model)
        other.memory_efficient = False
        x = torch.randn(12, 4, 9)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        for m in [model, other]:
            m.train()
            torch.manual_seed(103)
            pred, contrast = m(x, edges)
            (pred.square().mean() + contrast).backward()
        for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-5, msg=name)

    def test_memory_efficient_inference_matches_batched_graph(self):
        torch.manual_seed(101)
        model = PDUModel(12)
        model.eval()
        x = torch.randn(12, 4, 9)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        with torch.no_grad():
            sequential, _ = model(x, edges, contrast=False)
            model.memory_efficient = False
            batched, _ = model(x, edges, contrast=False)
        torch.testing.assert_close(sequential, batched, rtol=1e-5, atol=1e-6)

    def test_temporal_split_boundaries(self):
        from train import target_days
        train, valid, test = target_days(0, 756), target_days(756, 1008), target_days(1008, 1237)
        self.assertEqual((train[0], train[-1], len(train)), (4, 755, 752))
        self.assertEqual((valid[0], valid[-1], len(valid)), (756, 1007, 252))
        self.assertEqual(test[0], 1008)
        self.assertFalse(set(train) & set(valid))
        self.assertFalse(set(valid) & set(test))

    def test_graph_copies_never_cross_indicators(self):
        edges = torch.tensor([[0, 2, 3], [2, 3, 0]])
        lifted = indicator_edges(edges, 12)
        self.assertEqual(lifted.shape, (2, 27))
        self.assertTrue(torch.equal(lifted[0] // 12, lifted[1] // 12))
        for i in range(9):
            self.assertTrue(torch.equal(lifted[:, 3*i:3*(i+1)] - 12*i, edges))

    def test_metrics_invariant_to_stock_ids_and_masks(self):
        rng = np.random.default_rng(44)
        pred, y = rng.normal(size=(20, 8)), rng.normal(0, .02, size=(20, 8))
        m = np.ones_like(y); m[0] = 0
        a = evaluate(pred, y, m)
        perm = rng.permutation(20)
        b = evaluate(pred[perm], y[perm], m[perm])
        for key in a:
            np.testing.assert_allclose(a[key], b[key])
        pred[0], y[0] = 1e10, -1e10
        self.assertEqual(a, evaluate(pred, y, m))

    def test_perfect_and_reversed_rankings(self):
        y = np.linspace(-.04, .04, 20)[:, None] * np.array([[1, 2, 1.5]])
        mask = np.ones_like(y)
        perfect, reverse = evaluate(y, y, mask), evaluate(-y, y, mask)
        self.assertAlmostEqual(perfect["mrr"], 1)
        self.assertAlmostEqual(perfect["ndcg5_percentile"], 1)
        self.assertLess(reverse["ndcg5_percentile"], .3)

    def test_backward_ema_and_inference(self):
        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = PDUModel(12, device=device)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], device=device)
        x = torch.randn(12, 4, 9, device=device)
        x_before = x.clone()
        m.train()
        p, c = m(x, edges)
        loss, _ = objective(p, torch.ones(12, device=device), torch.zeros(12, device=device),
                            torch.ones(12, device=device), c)
        opt = torch.optim.Adam((p for p in m.parameters() if p.requires_grad), lr=.001)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        self.assertTrue(all(p.grad is None for p in m.target.parameters()))
        before = next(m.target.parameters()).clone()
        opt.step(); m.update_moving_average()
        torch.testing.assert_close(next(m.target.parameters()), before*.99 + next(m.online.parameters())*.01)
        torch.testing.assert_close(x, x_before)
        m.eval()
        with torch.no_grad():
            a, _ = m(x, edges, contrast=False)
            b, _ = m(x, edges, contrast=False)
        self.assertEqual(a.shape, (12, 1))
        torch.testing.assert_close(a, b)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
