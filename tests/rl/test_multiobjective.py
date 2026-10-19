"""Vector rewards, pluggable update rules, rollout tags and the GATv2 actor in ``swarp.rl``."""

from __future__ import annotations

import gc
from dataclasses import replace

import pytest
import torch

from swarp.rl import MAPPO, Critic, PolicyNet, StateCritic, UpdateRule, write_grads
from swarp.rl.networks import DenseGATv2, pair_features, tanh_normal_log_prob
from swarp.rl.ppo import LEGACY_STAT_NAMES, gae, standardize_advantage
from swarp.rl.recipes import SPECS, build_env

TAG = "team"


def _patch_tag(env) -> None:
    """A fixed-address integer ``[E, A]`` tag in ``info()``: agent ``a`` of env ``e`` is in
    group ``(e + a) % 2``."""
    scen, E, A = env.scenario, env.n_envs, env.n_agents
    tag = (torch.arange(E).unsqueeze(-1) + torch.arange(A)).remainder(2).to(env.device)
    info = scen.info

    def info_with_tag():
        out = dict(info())
        out[TAG] = tag
        return out

    scen.info = info_with_tag


def _algo(
    graphs: bool,
    device: str = "cuda:0",
    rule: UpdateRule | None = None,
    tags: tuple[str, ...] = (),
    seed: int = 0,
    **spec_kw,
) -> MAPPO:
    """Give-way at test size, trained on its five-term ``multiobj_reward``."""
    gc.collect()
    spec = replace(
        SPECS["giveway"], n_envs=32, steps_per_batch=8, epochs=2, minibatches=4, num_cells=32,
        reward_key="multiobj_reward", **spec_kw,
    )
    env = build_env(
        "giveway", n_envs=spec.n_envs, n_agents=spec.n_agents, device=device, dt=spec.dt,
        substeps=spec.substeps, max_steps=spec.max_steps, seed=seed,
        env_kwargs={"auto_reset": True},
    )
    env.reset()  # info() is populated
    if tags:
        _patch_tag(env)
    torch.manual_seed(seed)
    K = env.scenario.info()["multiobj_reward"].shape[-1]
    actor = PolicyNet(env.obs_dim, env.act_dim, spec.num_cells)
    critic = Critic(env.obs_dim, env.n_agents, spec.num_cells, n_values=K)
    algo = MAPPO(
        env, actor.to(device), critic.to(device), spec, graphs=graphs, update_rule=rule,
        tags=tags,
    )
    algo.reset()
    return algo


# ------------------------------------------------------------------------- the maths


def test_vector_gae_is_one_scalar_gae_per_objective():
    gen = torch.Generator().manual_seed(0)
    T, E, A, K = 12, 5, 3, 4
    rew = torch.randn(T, E, A, K, generator=gen)
    value = torch.randn(T + 1, E, A, K, generator=gen)
    term = (torch.rand(T, E, generator=gen) < 0.15).float()
    trunc = (torch.rand(T, E, generator=gen) < 0.15).float()
    vec = gae(rew, value, term, trunc, 0.99, 0.95)
    for k in range(K):
        one = gae(rew[..., k], value[..., k], term, trunc, 0.99, 0.95)
        assert torch.allclose(vec[..., k], one, atol=1e-6)


def test_vector_advantages_are_standardized_per_objective_and_role_slot():
    gen = torch.Generator().manual_seed(1)
    counts = (1, 2)
    roles = torch.stack([torch.tensor([0, 1, 1])[torch.randperm(3, generator=gen)]
                         for _ in range(64)])
    adv = torch.randn(64, 3, 2, generator=gen)
    vec = standardize_advantage(adv, roles, counts)
    for k in range(2):
        assert torch.allclose(vec[..., k], standardize_advantage(adv[..., k], roles, counts))
    plain = standardize_advantage(adv)
    for k in range(2):
        assert torch.allclose(plain[..., k], standardize_advantage(adv[..., k]))


def test_vector_critics_output_one_value_per_objective():
    torch.manual_seed(0)
    obs, roles = torch.randn(8, 3, 5), torch.tensor([[0, 1, 1]] * 8)
    assert Critic(5, 3, 16, n_values=4)(obs).shape == (8, 3, 4)
    team = StateCritic(0, 5, 3, 16, use_obs=True, n_values=4)
    assert team(obs).shape == (8, 3, 4)
    assert torch.equal(team(obs)[:, 0], team(obs)[:, 2])  # broadcast to every agent
    per = StateCritic(0, 5, 3, 16, use_obs=True, role_counts=(1, 2), per_agent=True, n_values=4)
    v = per(obs, roles=roles)
    assert v.shape == (8, 3, 4)
    # Slot-ordered rows of the head, mapped back to agent order.
    head = per.net(obs.flatten(-2)).view(8, 3, 4)
    assert torch.allclose(v, head)  # roles [0, 1, 1] are already role-sorted
    assert "n_values" in per.config
    assert "n_values" not in StateCritic(0, 5, 3, 16, use_obs=True).config


# ------------------------------------------------------------------ the graph network


def _naive_gatv2(layer: DenseGATv2, x, allowed, edge):
    """PyG's edge-list semantics, one receiver and one edge at a time."""
    B, A, _ = x.shape
    H, C = layer.heads, layer.channels
    x_l = layer.lin_l(x).view(B, A, H, C)
    x_r = layer.lin_r(x).view(B, A, H, C)
    out = torch.zeros(B, A, H, C)
    for b in range(B):
        for i in range(A):
            senders = [j for j in range(A) if bool(allowed[b, i, j])]
            if not senders:
                continue
            scores = []
            for j in senders:
                s = x_r[b, i] + x_l[b, j]
                if layer.lin_edge is not None:
                    s = s + layer.lin_edge(edge[b, i, j]).view(H, C)
                scores.append((torch.nn.functional.leaky_relu(s, 0.2) * layer.att).sum(-1))
            alpha = torch.softmax(torch.stack(scores), 0)  # [n_senders, H]
            for w, j in zip(alpha, senders, strict=True):
                out[b, i] += w.unsqueeze(-1) * x_l[b, j]
    return out.view(B, A, H * C) + layer.bias


@pytest.mark.parametrize("heads", [1, 2])
def test_dense_gatv2_matches_a_per_edge_loop(heads):
    torch.manual_seed(0)
    B, A, D = 4, 5, 8
    layer = DenseGATv2(D, heads=heads, edge_dim=3)
    with torch.no_grad():
        layer.bias.normal_()  # zero at init; a non-zero bias checks isolated agents get it
    x = torch.randn(B, A, D)
    pos = torch.randn(B, A, 2)
    allowed = (torch.rand(B, A, A) < 0.6) & ~torch.eye(A, dtype=torch.bool)
    allowed[0, 2] = False  # an isolated receiver
    edge = pair_features(pos, (0, 1), ())
    dense, _ = layer(x, allowed, edge)
    assert torch.allclose(dense, _naive_gatv2(layer, x, allowed, edge), atol=1e-6)
    assert torch.allclose(dense[0, 2], layer.bias)
    dense.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


def test_pair_features_are_sender_minus_receiver():
    obs = torch.randn(3, 4, 6)
    f = pair_features(obs, (0, 1), (2, 3))
    assert f.shape == (3, 4, 4, 5)
    i, j = 1, 3
    assert torch.allclose(f[:, i, j, :2], obs[:, j, :2] - obs[:, i, :2])
    assert torch.allclose(f[:, i, j, 2], (obs[:, j, :2] - obs[:, i, :2]).norm(dim=-1))
    assert torch.allclose(f[:, i, j, 3:], obs[:, j, 2:4] - obs[:, i, 2:4])


def test_gatv2_policy_config_round_trips():
    net = PolicyNet(
        6, 2, 32, comm="gatv2", comm_dim=16, comm_layers=2, comm_heads=4, edge_pos=(0, 1),
        edge_rel=(2, 3), trunk_depth=1,
    )
    again = PolicyNet(**net.config)
    again.load_state_dict(net.state_dict())
    obs = torch.randn(5, 3, 6)
    mask = torch.rand(5, 3, 3) < 0.5
    a = net.step(obs, comm_mask=mask)[0]
    b = again.step(obs, comm_mask=mask)[0]
    assert torch.equal(a, b)
    assert net.comm_weights(obs, comm_mask=mask).shape == (5, 3, 3)


@pytest.mark.gpu
def test_gatv2_policy_forward_and_backward_capture():
    """The GATv2 actor replays under a CUDA graph and matches its eager self."""
    torch.manual_seed(0)
    dev = "cuda:0"
    net = PolicyNet(
        6, 2, 32, comm="gatv2", comm_dim=16, comm_heads=2, edge_pos=(0, 1), edge_rel=(2, 3),
    ).to(dev)
    obs = torch.randn(64, 3, 6, device=dev)
    mask = torch.rand(64, 3, 3, device=dev) < 0.5

    def body():
        for p in net.parameters():
            p.grad = None if p.grad is None else p.grad.zero_()
        loc, scale, _, _ = net.step(obs, comm_mask=mask)
        (loc.square().sum() + scale.sum()).backward()
        return loc

    eager = body().detach().clone()
    eager_grads = [p.grad.clone() for p in net.parameters()]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        body()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = body()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, eager, atol=1e-6)
    for p, g in zip(net.parameters(), eager_grads, strict=True):
        assert torch.allclose(p.grad, g, atol=1e-5)


# ------------------------------------------------------------- the trainer, end to end


def _copy_rollout(dst: MAPPO, src: MAPPO) -> None:
    for name in ("obs_buf", "u_buf", "logp_buf", "rew_buf", "term_buf", "trunc_buf"):
        getattr(dst, name).copy_(getattr(src, name))
    for k, b in src.tag_bufs.items():
        dst.tag_bufs[k].copy_(b)


def _legacy_minibatch(algo: MAPPO) -> None:
    """The minibatch body as it was before update rules existed (scalar reward, no
    roles/aux/memory): the reference the default rule must reproduce bit for bit."""
    A = algo.A
    idx = algo.perms.index_select(0, algo.k).view(-1)
    obs = algo.obs_buf[: algo.T].reshape(algo.N, A, -1).index_select(0, idx)
    u = algo.u_buf.reshape(algo.N, A, -1).index_select(0, idx)
    logp_old = algo.logp_buf.reshape(algo.N, A).index_select(0, idx)
    adv = algo.adv_buf.reshape(algo.N, A).index_select(0, idx)
    ret = algo.ret_buf.reshape(algo.N, A).index_select(0, idx)
    noise = torch.randn_like(u)
    losses = algo.rule._loss(
        algo.actor, algo.critic, obs, u, logp_old, adv, ret, noise,
        algo.spec.normalize_advantage,
    )
    loss_obj, loss_crit, entropy = losses[:3]
    loss_ent = -algo.ent_coef * entropy
    algo.optim.zero_grad(set_to_none=False)
    total = loss_obj + loss_crit + loss_ent
    total.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(algo.params, 1.0, foreach=algo.cuda or None)
    found = (~torch.isfinite(gnorm)).float()
    if algo.cuda:
        algo.found_inf.copy_(torch.maximum(found, algo.force_skip))
        algo.optim.step()
    elif not found.item():
        algo.optim.step()
    kept = torch.stack(losses).detach()
    algo.mb_stats[:3] += torch.where(found > 0, torch.zeros_like(kept), kept)
    algo.mb_stats[3] += found
    algo.k.add_(1)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_default_rule_is_bit_identical_to_the_pre_rule_update(device):
    """Same weights, rollout, permutations and noise draws: the default rule lands on the
    same bits as the minibatch body it replaced."""
    from test_ppo import _algo as nav_algo

    new = nav_algo(graphs=False, device=device)
    old = nav_algo(graphs=False, device=device)
    old.actor.load_state_dict(new.actor.state_dict())
    old.critic.load_state_dict(new.critic.state_dict())
    assert new.stat_names[:7] == list(LEGACY_STAT_NAMES)
    new.rollout()
    _copy_rollout(old, new)
    for algo in (new, old):
        algo._prepare_body()
    old.perms.copy_(new.perms)
    for algo, step in ((new, new._minibatch_body), (old, lambda: _legacy_minibatch(old))):
        torch.manual_seed(123)
        for _ in range(algo.spec.epochs * algo.M):
            step()
    for a, b in zip(new.params, old.params, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(new.mb_stats, old.mb_stats)


class _ProjectedRule(UpdateRule):
    """A rule with the shape of a lexicographic update: one shared forward, one PPO loss
    per objective and per tag group, a separate ``autograd.grad`` for each, a combination
    of the per-objective gradients, then the critic's own backward."""

    stat_names = ("loss_objective", "loss_critic", "tag_share")

    def setup(self, algo):
        self.actor_params = list(algo.actor.parameters())
        self.K = algo.K

    def backward(self, mb, actor, critic, ent_coef):
        loc, scale, _, _ = actor.step(mb.obs)
        log_w = tanh_normal_log_prob(mb.u, loc, scale) - mb.logp_old
        ratio = log_w.exp()
        adv = (mb.adv - mb.adv.mean((0, 1))) / mb.adv.std((0, 1)).clamp_min(1e-6)
        group = mb.tags[TAG]
        losses = []
        for g in range(2):
            m = (group == g).float()
            for k in range(self.K):
                gain = torch.minimum(ratio * adv[..., k], ratio.clamp(0.8, 1.2) * adv[..., k])
                losses.append(-(gain * m).sum() / m.sum().clamp_min(1.0))
        grads = [
            torch.autograd.grad(loss, self.actor_params, retain_graph=True) for loss in losses
        ]
        # Drop each objective's component against the first (a toy projection).
        flat = torch.stack([torch.cat([g.reshape(-1) for g in gs]) for gs in grads])
        first = flat[0]
        coef = (flat @ first).clamp_max(0.0) / first.dot(first).clamp_min(1e-12)
        combined = (flat - coef.unsqueeze(-1) * first).sum(0)
        out, off = [], 0
        for p in self.actor_params:
            out.append(combined[off : off + p.numel()].view_as(p))
            off += p.numel()
        write_grads(self.actor_params, out)
        value_loss = (critic(mb.obs) - mb.ret).square().mean()
        value_loss.backward()
        return torch.stack([torch.stack(losses).sum(), value_loss, group.float().mean()]).detach()


def test_vector_reward_fills_k_wide_buffers_and_tags():
    algo = _algo(graphs=False, device="cpu", rule=_ProjectedRule(), tags=(TAG,))
    assert algo.K == 5 and algo.rew_buf.shape[-1] == 5 and algo.v_buf.shape[-1] == 5
    algo.rollout()
    info_rew = algo.env.scenario.info()["multiobj_reward"]
    assert torch.equal(algo.rew_buf[-1], info_rew)
    E, A = algo.E, algo.A
    want = (torch.arange(E).unsqueeze(-1) + torch.arange(A)).remainder(2)
    assert torch.equal(algo.tag_bufs[TAG], want.expand(algo.T, E, A))
    stats = algo.update(0.0)
    assert list(stats) == [*LEGACY_STAT_NAMES[:3], *_ProjectedRule.stat_names, "grad_skips"]
    assert stats["tag_share"] == pytest.approx(0.5 * algo.spec.epochs * algo.M)


def test_default_rule_trains_on_a_vector_reward():
    algo = _algo(graphs=False, device="cpu", objective_weights=(1.0, 0.5, 0.5, 1.0, 2.0))
    before = [p.detach().clone() for p in algo.params]
    algo.rollout()
    stats = algo.update(algo.spec.entropy_coeff)
    assert stats["grad_skips"] == 0
    assert any(not torch.equal(p, q) for p, q in zip(algo.params, before, strict=True))


@pytest.mark.gpu
@pytest.mark.parametrize("clip_per_network", [False, True])
def test_a_custom_rule_with_several_autograd_grads_captures(clip_per_network):
    """Same weights, rollout and permutations: the captured custom rule lands where the
    eager one does, ``k`` ``autograd.grad`` calls over one retained graph included."""
    kw = {"rule": None, "tags": (TAG,), "clip_per_network": clip_per_network}
    eager = _algo(graphs=False, **{**kw, "rule": _ProjectedRule()})
    graphed = _algo(graphs=True, **{**kw, "rule": _ProjectedRule()})
    graphed.actor.load_state_dict(eager.actor.state_dict())
    graphed.critic.load_state_dict(eager.critic.state_dict())
    eager.rollout()
    _copy_rollout(graphed, eager)
    for algo in (eager, graphed):
        algo._prepare_body()
    graphed.perms.copy_(eager.perms)
    for _ in range(eager.spec.epochs * eager.M):
        eager._minibatch()
        graphed._minibatch()
    assert graphed._minibatch.graph is not None
    for a, b in zip(eager.params, graphed.params, strict=True):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
    assert torch.allclose(eager.mb_stats, graphed.mb_stats, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_a_rule_that_is_not_graph_safe_runs_its_minibatches_eagerly():
    class Eager(_ProjectedRule):
        graph_safe = False

    algo = _algo(graphs=True, rule=Eager(), tags=(TAG,))
    algo.rollout()
    algo.update(0.0)
    assert algo._minibatch.graph is None and algo._act.graph is not None
