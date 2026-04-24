"""Correctness analysis of FrankaReachAPGEnv vs FrankaReachVecEnv.

Forward analysis identifies two root issues:

Issue 1 – VecEnv staleness (major)
    VecEnv.step() updates state_0.joint_q in-place and then calls _get_obs()
    and _get_rewards() *before* calling eval_fk.  Therefore state_0.body_q is
    stale: it still reflects the body positions from the most recent eval_fk
    call, which happened at the end of the previous reset/step.  The EEF in
    the returned obs and the reward both measure the pre-action pose, not the
    post-action pose.

    APGEnv is correct: _NewtonStepFunc runs eval_fk on the new joint_q inside
    the tape and builds obs/reward from the resulting body_q.

Issue 2 – APGEnv unclamped FK vs clamped persistent state (minor)
    _set_joint_targets_kernel computes new_joint_q = state_joint_q + action *
    scale without applying joint-limit clamping.  After forward(), the
    persistent state_0.joint_q is updated *with* clamping.  When joints are
    near their limits an action that would violate them produces:
      • obs/reward evaluated at an out-of-limit joint configuration, and
      • a persistent state that starts the next step at the clamped position.
    The two states differ, so there is a gap between what the current step
    evaluates and where the next step begins.

    Consequence: _check_success() (and thus `terminated`) is computed after
    _render_current_state() which uses the clamped joint_q, while reward/obs
    use the unclamped FK result.

Backward analysis:
    Warp-tape gradients are expected to flow through
        action → new_joint_q → eval_fk → body_q → reward
    and through the direct action-rate term.  Tests verify:
      • gradients are finite and non-zero,
      • analytic gradients agree with central finite differences, and
      • rewards are independent across environments (cross-env grad is zero).
"""

import unittest

import numpy as np
import torch
import warp as wp

import newton
from franka_reach_env import (
    DEFAULT_ACTION_SCALE,
    DEFAULT_JOINT_Q,
    FRANKA_NUM_ARM_JOINTS,
    FRANKA_NUM_JOINTS,
    FrankaReachAPGEnv,
    FrankaReachVecEnv,
    _compute_reward,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _capture(env):
    """Return a snapshot of the env state needed for a full restore."""
    return (
        wp.to_torch(env.state_0.joint_q).detach().clone(),
        env.target_pos.clone(),
        env.target_quat.clone(),
        env.last_action.clone(),
    )


def _restore_env(env, state):
    """Restore env to a previously captured state without randomization."""
    jq_flat, tpos, tquat, la = state
    # Use wp.copy to bypass PyTorch autograd in-place restriction on leaf tensors
    # that have requires_grad=True (as is the case for APGEnv).
    jq_np = jq_flat.detach().cpu().numpy().astype(np.float32)
    tmp_wp = wp.array(jq_np, dtype=wp.float32, device=env.model.device)
    wp.copy(env.state_0.joint_q, tmp_wp)
    env.target_pos.copy_(tpos)
    env.target_quat.copy_(tquat)
    env.last_action.copy_(la)
    env.step_count.zero_()
    newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)


def _manual_fk_body_q(env, joint_q_flat_torch):
    """Run FK on a flat joint_q tensor and return body_q as a torch tensor.

    Uses a temporary state so the env's persistent state_0 is untouched.
    """
    tmp = env.model.state()
    jq_np = joint_q_flat_torch.detach().cpu().numpy().astype(np.float32)
    jq_wp = wp.array(jq_np, dtype=wp.float32, device=env.model.device)
    newton.eval_fk(env.model, jq_wp, tmp.joint_qd, tmp)
    return wp.to_torch(tmp.body_q).detach().clone()


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


class TestForwardCorrectness(unittest.TestCase):
    """APGEnv and VecEnv forward pass: obs and reward match manual FK."""

    N = 2  # num_envs

    @classmethod
    def setUpClass(cls):
        cls.apg = FrankaReachAPGEnv(
            num_envs=cls.N,
            headless=True,
            device="cpu",
            max_episode_steps=1000,
        )
        cls.vec = FrankaReachVecEnv(
            num_envs=cls.N,
            headless=True,
            device="cpu",
            max_episode_steps=1000,
        )

    @classmethod
    def tearDownClass(cls):
        cls.apg.close()
        cls.vec.close()

    # ------------------------------------------------------------------
    # APGEnv forward correctness
    # ------------------------------------------------------------------

    def test_apg_joint_q_in_obs_equals_clamped_new_joint_q(self):
        """Arm joint_q in APGEnv obs must equal clamp(init_q + action*scale, lo, hi)."""
        self.apg.reset(seed=10)
        init_jq = wp.to_torch(self.apg.state_0.joint_q).detach().clone().view(self.N, -1)

        # Use actions well inside [-1, 1] to avoid the action-clamp distraction.
        action = torch.rand(self.N, FRANKA_NUM_ARM_JOINTS) * 0.4 - 0.2
        obs, _, _, _, _ = self.apg.step(action)

        expected = torch.clamp(
            init_jq[:, :FRANKA_NUM_ARM_JOINTS] + action.detach() * DEFAULT_ACTION_SCALE,
            self.apg.arm_joint_limit_lower,
            self.apg.arm_joint_limit_upper,
        )
        torch.testing.assert_close(
            obs[:, :FRANKA_NUM_ARM_JOINTS],
            expected,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_apg_eef_in_obs_matches_fk_of_new_joint_q(self):
        """EEF position component of APGEnv obs must equal FK(clamp(init_q + action*scale))."""
        self.apg.reset(seed=11)
        init_jq = wp.to_torch(self.apg.state_0.joint_q).detach().clone().view(self.N, -1)

        action = torch.rand(self.N, FRANKA_NUM_ARM_JOINTS) * 0.4 - 0.2
        obs, _, _, _, _ = self.apg.step(action)

        new_jq = init_jq.clone()
        new_jq[:, :FRANKA_NUM_ARM_JOINTS] = torch.clamp(
            new_jq[:, :FRANKA_NUM_ARM_JOINTS] + action.detach() * DEFAULT_ACTION_SCALE,
            self.apg.arm_joint_limit_lower,
            self.apg.arm_joint_limit_upper,
        )
        body_q_ref = _manual_fk_body_q(self.apg, new_jq.reshape(-1))

        for i in range(self.N):
            ee_g = int(self.apg._ee_global[i])
            eef_ref = body_q_ref[ee_g, :3]
            eef_obs = obs[i, FRANKA_NUM_ARM_JOINTS : FRANKA_NUM_ARM_JOINTS + 3]
            torch.testing.assert_close(
                eef_obs,
                eef_ref,
                atol=1e-5,
                rtol=1e-5,
                msg=f"EEF pos mismatch for env {i}",
            )

    def test_apg_reward_matches_manual_compute_reward(self):
        """APGEnv reward must equal _compute_reward applied to post-action FK EEF."""
        self.apg.reset(seed=12)
        saved = _capture(self.apg)
        init_jq = saved[0].view(self.N, -1).clone()

        action = torch.rand(self.N, FRANKA_NUM_ARM_JOINTS) * 0.4 - 0.2
        _, reward_apg, _, _, _ = self.apg.step(action)

        new_jq = init_jq.clone()
        new_jq[:, :FRANKA_NUM_ARM_JOINTS] = torch.clamp(
            new_jq[:, :FRANKA_NUM_ARM_JOINTS] + action.detach() * DEFAULT_ACTION_SCALE,
            self.apg.arm_joint_limit_lower,
            self.apg.arm_joint_limit_upper,
        )
        body_q_ref = _manual_fk_body_q(self.apg, new_jq.reshape(-1))

        ee_idx = self.apg._ee_global
        eef_pos = torch.stack([body_q_ref[ee_idx[i], :3] for i in range(self.N)])
        eef_quat = torch.stack([body_q_ref[ee_idx[i], 3:7] for i in range(self.N)])

        reward_ref = _compute_reward(
            eef_pos,
            eef_quat,
            saved[1],
            saved[2],
            action=action.detach(),
            last_action=saved[3],
        )
        torch.testing.assert_close(reward_apg, reward_ref, atol=1e-5, rtol=1e-5)

    # ------------------------------------------------------------------
    # Issue 1: VecEnv staleness
    # ------------------------------------------------------------------

    def test_vec_obs_eef_equals_post_step_position(self):
        """VecEnv obs EEF must equal the post-step FK body_q.

        VecEnv.step() calls eval_fk before _get_obs(), so obs EEF reflects
        the new joint_q after applying and clamping the action.
        """
        self.vec.reset(seed=20)
        saved = _capture(self.vec)
        init_jq = saved[0].view(self.N, -1).clone()

        # Large action to produce a clearly visible EEF change.
        action = torch.ones(self.N, FRANKA_NUM_ARM_JOINTS) * 0.5
        obs, _, _, _, _ = self.vec.step(action)

        # Expected: FK of clamped new joint_q.
        new_jq = init_jq.clone()
        new_jq[:, :FRANKA_NUM_ARM_JOINTS] = torch.clamp(
            new_jq[:, :FRANKA_NUM_ARM_JOINTS] + action * self.vec.action_scale,
            self.vec.arm_joint_limit_lower,
            self.vec.arm_joint_limit_upper,
        )
        body_q_ref = _manual_fk_body_q(self.vec, new_jq.reshape(-1))
        ee_local = int(self.vec.ee_body_indices[0])
        n_bodies_per_env = body_q_ref.shape[0] // self.N
        eef_ref = body_q_ref.view(self.N, n_bodies_per_env, 7)[0, ee_local, :3]

        eef_in_obs = obs[0, FRANKA_NUM_ARM_JOINTS : FRANKA_NUM_ARM_JOINTS + 3]
        torch.testing.assert_close(
            eef_in_obs,
            eef_ref,
            atol=1e-5,
            rtol=1e-5,
            msg="VecEnv obs must contain the post-step EEF",
        )

    def test_vec_reward_computed_from_post_step_eef(self):
        """VecEnv reward must be computed from the post-step FK EEF.

        _get_rewards() reads state_0.body_q which is fresh after the eval_fk
        fix.  The returned reward therefore measures the distance from the EEF
        *after* the action was applied.
        """
        self.vec.reset(seed=21)
        saved = _capture(self.vec)
        init_jq = saved[0].view(self.N, -1).clone()

        action = torch.rand(self.N, FRANKA_NUM_ARM_JOINTS) * 0.6 - 0.3
        _, reward_vec, _, _, _ = self.vec.step(action)

        # Post-step body_q: FK of clamped new joint_q.
        new_jq = init_jq.clone()
        new_jq[:, :FRANKA_NUM_ARM_JOINTS] = torch.clamp(
            new_jq[:, :FRANKA_NUM_ARM_JOINTS] + action * self.vec.action_scale,
            self.vec.arm_joint_limit_lower,
            self.vec.arm_joint_limit_upper,
        )
        body_q_ref = _manual_fk_body_q(self.vec, new_jq.reshape(-1))
        ee_local = int(self.vec.ee_body_indices[0])
        n_bodies_per_env = body_q_ref.shape[0] // self.N
        body_q_3d = body_q_ref.view(self.N, n_bodies_per_env, 7)
        eef_pos = body_q_3d[:, ee_local, :3]
        eef_quat = body_q_3d[:, ee_local, 3:7]

        reward_ref = _compute_reward(
            eef_pos,
            eef_quat,
            saved[1],
            saved[2],
            action=action,
            last_action=saved[3],
        )
        torch.testing.assert_close(reward_vec, reward_ref, atol=1e-5, rtol=1e-5)

    def test_apg_eef_is_not_stale(self):
        """APGEnv obs EEF must equal the post-action EEF, not the pre-step EEF.

        After a large action the post-step EEF must differ from the pre-step EEF.
        """
        self.apg.reset(seed=30)
        # Pre-step EEF from global flat body_q (env 0's EE body).
        eef_pre = wp.to_torch(self.apg.state_0.body_q).detach().clone()[
            self.apg._ee_global[0], :3
        ]

        action = torch.ones(self.N, FRANKA_NUM_ARM_JOINTS) * 0.5
        obs, _, _, _, _ = self.apg.step(action)
        eef_in_obs = obs[0, FRANKA_NUM_ARM_JOINTS : FRANKA_NUM_ARM_JOINTS + 3]

        # Post-action EEF must differ from the pre-step EEF.
        self.assertFalse(
            torch.allclose(eef_in_obs, eef_pre, atol=1e-5),
            "APGEnv should return the post-action EEF, not the stale pre-step one",
        )

    # ------------------------------------------------------------------
    # Issue 2: APGEnv unclamped FK vs clamped persistent state
    # ------------------------------------------------------------------

    def test_apg_fk_and_state_are_consistent_at_joint_limits(self):
        """APGEnv obs joint_q and persistent state must agree at joint limits.

        When an action pushes a joint past its upper limit, both the FK path
        (obs joint_q) and state_0.joint_q must clamp to the same limit value,
        ensuring step-to-step continuity.
        """
        self.apg.reset(seed=40)

        # Force env-0 joint-0 just below its upper limit using wp.copy to avoid
        # in-place ops on a requires_grad leaf tensor.
        upper = self.apg.arm_joint_limit_upper[0].item()
        init_jq_np = (
            wp.to_torch(self.apg.state_0.joint_q).detach().cpu().numpy().copy()
        )
        init_jq_np = init_jq_np.reshape(self.N, -1)
        init_jq_np[0, 0] = upper - 0.01
        tmp_wp = wp.array(
            init_jq_np.reshape(-1).astype(np.float32),
            dtype=wp.float32,
            device=self.apg.model.device,
        )
        wp.copy(self.apg.state_0.joint_q, tmp_wp)
        # Refresh body_q after manual joint_q edit.
        newton.eval_fk(
            self.apg.model,
            self.apg.state_0.joint_q,
            self.apg.state_0.joint_qd,
            self.apg.state_0,
        )

        # +1 action on joint-0 with scale 0.2 → new_q = upper - 0.01 + 0.2 > upper.
        action = torch.zeros(self.N, FRANKA_NUM_ARM_JOINTS)
        action[0, 0] = 1.0
        obs, _, _, _, _ = self.apg.step(action)

        # Both the FK path (obs) and persistent state must clamp to upper.
        obs_joint_q_0_0 = obs[0, 0].item()
        stored_joint_q_0_0 = (
            wp.to_torch(self.apg.state_0.joint_q).detach().view(self.N, -1)[0, 0].item()
        )
        self.assertAlmostEqual(
            obs_joint_q_0_0,
            upper,
            places=4,
            msg="obs joint_q must be clamped to the upper limit",
        )
        self.assertAlmostEqual(
            stored_joint_q_0_0,
            upper,
            places=4,
            msg="stored state joint_q must be clamped to the upper limit",
        )


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------


class TestBackwardCorrectness(unittest.TestCase):
    """APGEnv backward pass: Warp-tape gradients match finite differences."""

    N = 2

    def setUp(self):
        self.env = FrankaReachAPGEnv(
            num_envs=self.N,
            headless=True,
            device="cpu",
            max_episode_steps=1000,
        )
        self.env.reset(seed=99)
        # Force joints to the home configuration so all joints are well within
        # their limits.  This avoids FD test failures caused by stradling a
        # clamp boundary (analytic grad = 0 at limit, FD grad ≠ 0).
        home_np = np.array(DEFAULT_JOINT_Q * self.N, dtype=np.float32)
        home_wp = wp.array(home_np, dtype=wp.float32, device=self.env.model.device)
        wp.copy(self.env.state_0.joint_q, home_wp)
        newton.eval_fk(
            self.env.model,
            self.env.state_0.joint_q,
            self.env.state_0.joint_qd,
            self.env.state_0,
        )
        # Fix targets far from initial EEF to avoid accidental termination.
        self.env.target_pos[:] = torch.tensor([0.6, 0.3, 0.5])
        self.env.target_quat[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self._state = _capture(self.env)

    def tearDown(self):
        self.env.close()

    def _restore(self):
        _restore_env(self.env, self._state)

    # ------------------------------------------------------------------

    def test_gradients_are_finite_and_nonzero(self):
        """d(sum(reward)) / d(action) must be finite and not identically zero."""
        action = torch.randn(self.N, FRANKA_NUM_ARM_JOINTS, requires_grad=True)
        _, reward, _, _, _ = self.env.step(action)
        reward.sum().backward()

        self.assertIsNotNone(action.grad)
        self.assertFalse(torch.isnan(action.grad).any(), "NaN in gradient")
        self.assertFalse(torch.isinf(action.grad).any(), "Inf in gradient")
        self.assertFalse((action.grad == 0).all(), "Gradient is all zero")

    def test_analytic_gradient_matches_finite_difference(self):
        """Warp-tape analytic gradient must agree with central FD (atol 1e-2).

        The gradient flows through:
          action → new_joint_q → eval_fk → body_q → reward  (FK path)
          action → action_rate term                           (direct path)

        Float32 central-difference roundoff is O(eps^-1 * machine_eps) ≈ 1e-3
        at eps=1e-4, so a tolerance of 1e-2 is generous but appropriate.
        """
        eps = 1e-4
        action_base = torch.zeros(self.N, FRANKA_NUM_ARM_JOINTS)

        # ---- analytic gradient ----
        self._restore()
        action_ad = action_base.clone().requires_grad_(True)
        _, reward_ad, _, _, _ = self.env.step(action_ad)
        reward_ad.sum().backward()
        grad_analytic = action_ad.grad.clone()

        # ---- central finite-difference gradient ----
        grad_fd = torch.zeros_like(action_base)
        for i in range(self.N):
            for j in range(FRANKA_NUM_ARM_JOINTS):
                self._restore()
                a_plus = action_base.clone()
                a_plus[i, j] += eps
                _, r_plus, _, _, _ = self.env.step(a_plus)

                self._restore()
                a_minus = action_base.clone()
                a_minus[i, j] -= eps
                _, r_minus, _, _, _ = self.env.step(a_minus)

                grad_fd[i, j] = (r_plus.sum() - r_minus.sum()) / (2.0 * eps)

        torch.testing.assert_close(grad_analytic, grad_fd, atol=1e-2, rtol=5e-2)

    def test_reward_independence_across_envs(self):
        """Reward of env-i must not depend on the action of env-j (i ≠ j).

        The Warp kernel loops only over own-env indices, and FK for each
        robot only reads its own joint_q slice.  The cross-env gradient
        must therefore be exactly zero.
        """
        action = torch.zeros(self.N, FRANKA_NUM_ARM_JOINTS, requires_grad=True)
        _, reward, _, _, _ = self.env.step(action)

        # d(reward[0]) / d(action[1, :]) should be zero.
        reward[0].backward(retain_graph=True)
        grad0 = action.grad.clone()
        action.grad.zero_()

        # d(reward[1]) / d(action[0, :]) should be zero.
        reward[1].backward()
        grad1 = action.grad.clone()

        torch.testing.assert_close(
            grad0[1],
            torch.zeros(FRANKA_NUM_ARM_JOINTS),
            atol=1e-7,
            rtol=0.0,
            msg="env-0 reward must not depend on env-1 action",
        )
        torch.testing.assert_close(
            grad1[0],
            torch.zeros(FRANKA_NUM_ARM_JOINTS),
            atol=1e-7,
            rtol=0.0,
            msg="env-1 reward must not depend on env-0 action",
        )


if __name__ == "__main__":
    unittest.main()
